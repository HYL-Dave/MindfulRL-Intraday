"""Run the offline lifecycle matrix with truthful final-unconfirmed History."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
from types import ModuleType
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright


APP_URL = os.environ.get("ARKSCOPE_HONESTY_APP_URL", "http://127.0.0.1:4206/")
PACKET = Path(__file__).resolve().parent
OUTPUT = PACKET / "browser"
MATRIX = PACKET / "browser-matrix.json"
ROOT = PACKET.parents[3]
PREVIOUS_RUNNER = (
    ROOT
    / "docs/superpowers/evidence/2026-08-26-lifecycle-automated-disposition"
    / "run_browser_matrix.py"
)


def _load_previous() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "automated_disposition_fixture", PREVIOUS_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("fixture_loader_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREVIOUS = _load_previous()
SCENARIOS = deepcopy(PREVIOUS.SCENARIOS)
SCENARIOS["final-unconfirmed"] = {
    "case_id": "case-final-unconfirmed",
    "ticker": "UNCHECKED",
    "issuer": "Unconfirmed Completion Corp.",
    "queue": "history",
    "disposition": "not_confirmed_yet",
    "reason": "not_confirmed_as_of",
    "tier": None,
    "readiness": None,
    "statuses": {
        "regulator": "present",
        "market_infrastructure": "present",
        "publisher": "present",
    },
}
LABELS = deepcopy(PREVIOUS.LABELS)
LABELS["en"]["reasons"]["not_confirmed_as_of"] = (
    "Not confirmed as of 2026-08-27; active checking stopped."
)
LABELS["zh-Hant"]["reasons"]["not_confirmed_as_of"] = (
    "\u622a\u81f3 2026-08-27 \u5c1a\u672a\u78ba\u8a8d\uff1b"
    "\u5df2\u505c\u6b62\u4e3b\u52d5\u8ffd\u67e5\u3002"
)

PREVIOUS.APP_URL = APP_URL
PREVIOUS.OUTPUT = OUTPUT
PREVIOUS.SCENARIOS = SCENARIOS
PREVIOUS.LABELS = LABELS
_ORIGINAL_SUMMARY = PREVIOUS._summary
_ORIGINAL_DETAIL = PREVIOUS._detail


def _summary(name: str) -> dict:
    row = _ORIGINAL_SUMMARY(name)
    row["disposition_as_of"] = None
    if name == "final-unconfirmed":
        row.update(
            {
                "workflow_state": "resolved",
                "disposition_as_of": "2026-08-27",
                "last_checked_at": "2026-08-27T12:00:00Z",
                "next_check_at": None,
                "acknowledgement_count": 0,
            }
        )
    return row


def _detail(name: str) -> dict:
    row = _ORIGINAL_DETAIL(name)
    if name == "final-unconfirmed":
        row["current_assessment"] = None
        row["assessment_history"] = []
        row["acknowledgement_history"] = []
        row["ticker_transition"] = None
    return row


def _case_list(query: dict[str, list[str]]) -> list[dict]:
    if query.get("source_presence") == ["source_missing"]:
        return [_summary("source-missing")]
    queue = query.get("queue_bucket", [None])[0]
    names = {
        "attention": ("attention-conflict",),
        "monitoring": ("pending-frozen", "future-effective"),
        "history": ("settled-history", "final-unconfirmed"),
        None: (
            "attention-conflict",
            "pending-frozen",
            "future-effective",
            "settled-history",
            "final-unconfirmed",
        ),
    }[queue]
    return [_summary(name) for name in names]


PREVIOUS._summary = _summary
PREVIOUS._detail = _detail
PREVIOUS._case_list = _case_list


def _run_entry(browser, name: str, locale: str, width: int, height: int) -> dict:
    scenario = SCENARIOS[name]
    labels = LABELS[locale]
    state: dict[str, list] = {"requests": [], "external": []}
    context = browser.new_context(viewport={"width": width, "height": height})
    context.add_init_script(
        f"localStorage.setItem('arkscope.ui.locale.v1', {json.dumps(locale)});"
    )
    page = context.new_page()
    page.set_default_timeout(10_000)
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))

    def handler(route) -> None:
        request = route.request
        parsed = urlparse(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            state["external"].append(request.url)
            route.abort()
            return
        if parsed.port != 8420:
            route.continue_()
            return
        state["requests"].append({"method": request.method, "path": parsed.path})
        if request.method == "OPTIONS":
            PREVIOUS.STAGE5._response(route, {}, 204)
        elif parsed.path == "/status":
            PREVIOUS.STAGE5._response(
                route,
                {
                    "status": "ok",
                    "timestamp": "2026-08-27T12:00:00Z",
                    "tools_registered": 50,
                    "tool_categories": {},
                    "data_sources": {},
                },
            )
        elif parsed.path == "/config/runtime":
            PREVIOUS.STAGE5._response(route, PREVIOUS.STAGE5._runtime_config())
        elif parsed.path == "/profile/settings/ui-locale":
            PREVIOUS.STAGE5._response(
                route, {"locale": locale, "source": "stored"}
            )
        elif parsed.path == "/profile/universe":
            PREVIOUS.STAGE5._response(
                route,
                {
                    "as_of": "2026-08-27",
                    "generated_at": "2026-08-27T12:00:00Z",
                    "total": 0,
                    "shown": 0,
                    "archived_count": 0,
                    "summarized": 0,
                    "rows": [],
                },
            )
        elif parsed.path == "/profile/lists":
            PREVIOUS.STAGE5._response(route, {"lists": []})
        elif parsed.path == "/analysis/cards":
            PREVIOUS.STAGE5._response(route, {"cards": []})
        elif parsed.path == "/research/threads":
            PREVIOUS.STAGE5._response(route, {"threads": []})
        elif parsed.path == "/security-lifecycle/cases":
            cases = _case_list(parse_qs(parsed.query))
            PREVIOUS.STAGE5._response(
                route,
                {
                    "cases": cases,
                    "count": len(cases),
                    "queue_counts": {"attention": 1, "monitoring": 2, "history": 2},
                    "data_integrity": {"source_missing_count": 1},
                },
            )
        elif parsed.path.startswith("/security-lifecycle/cases/"):
            case_id = parsed.path.rsplit("/", 1)[-1]
            scenario_name = next(
                key for key, value in SCENARIOS.items() if value["case_id"] == case_id
            )
            PREVIOUS.STAGE5._response(route, _detail(scenario_name))
        elif parsed.path == "/security-lifecycle/transition-activity":
            PREVIOUS.STAGE5._response(
                route,
                {"items": [], "count": 0, "unacknowledged_count": 0},
            )
        else:
            PREVIOUS.STAGE5._response(
                route, {"detail": {"code": "fixture_unavailable"}}, 503
            )

    page.route("**/*", handler)
    page.goto(APP_URL, wait_until="networkidle", timeout=20_000)
    universe = page.get_by_role("button", name=labels["universe"], exact=True)
    if not universe.is_visible():
        page.get_by_role("button", name=labels["open_nav"], exact=True).click()
    universe.click()
    page.get_by_role("tab", name=labels["lifecycle"], exact=True).click()

    if name == "source-missing":
        page.get_by_role(
            "button", name=re.compile(rf"^{re.escape(labels['integrity'])}")
        ).click()
    elif scenario["queue"] != "attention":
        page.locator(f"[data-queue-view='{scenario['queue']}']").click()

    trigger = page.get_by_role("button", name=re.compile(rf"^{scenario['ticker']}\b"))
    trigger.wait_for(state="visible")
    row = trigger.locator("xpath=ancestor::tr")
    assert row.locator(f"[data-disposition='{scenario['disposition']}']").count() == 1
    row_text = row.inner_text()
    trigger.click()
    drawer = page.locator(".ui-drawer")
    drawer.wait_for(state="visible")
    drawer_text = drawer.inner_text()
    assert labels["dispositions"][scenario["disposition"]] in drawer_text
    assert labels["reasons"][scenario["reason"]] in drawer_text
    for source_state in set(scenario["statuses"].values()):
        assert labels["states"][source_state] in drawer_text
    if name == "pending-frozen":
        assert "frozen" in drawer_text
        assert "2026-09-02T08:00:00Z" in drawer_text
    if name == "future-effective":
        assert "2026-09-30" in drawer_text
    if name in {"settled-history", "source-missing"}:
        assert labels["revalidation"] in drawer_text
    final_copy: str | None = None
    if name == "final-unconfirmed":
        expected_final_copy = labels["reasons"]["not_confirmed_as_of"]
        final_copy = row_text + "\n" + drawer_text
        assert expected_final_copy in row_text
        assert expected_final_copy in drawer_text
        if locale == "en":
            assert "Confirmed complete" not in final_copy
        else:
            assert "\u5df2\u78ba\u8a8d\u5b8c\u6210" not in final_copy

    metrics = PREVIOUS.STAGE5._geometry(page)
    PREVIOUS.STAGE5._assert_geometry(metrics)
    screenshot = OUTPUT / f"{width}x{height}-{locale}-{name}.png"
    page.screenshot(path=str(screenshot))
    pixels = PREVIOUS.STAGE5._pixel_check(screenshot, width, height)
    write_requests = [
        item
        for item in state["requests"]
        if item["method"] in {"POST", "PUT", "PATCH", "DELETE"}
    ]
    render_acknowledgements = sum(
        1 for item in write_requests if item["path"].endswith("/acknowledge")
    )
    assert state["external"] == [], state["external"]
    assert write_requests == [], write_requests
    assert render_acknowledgements == 0
    assert console_errors == [], console_errors
    assert page_errors == [], page_errors
    result = {
        "scenario": name,
        "locale": locale,
        "viewport": [width, height],
        "screenshot": str(screenshot.relative_to(PACKET)),
        "screenshot_sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
        "pixels": pixels,
        "request_count": len(state["requests"]),
        "external_requests": state["external"],
        "write_requests": write_requests,
        "render_acknowledgements": render_acknowledgements,
        "console_errors": console_errors,
        "page_errors": page_errors,
        "overlap_count": len(metrics["overlaps"]),
        "clipped_text_count": len(metrics["textOverflow"]),
        "final_unconfirmed_text": final_copy,
    }
    context.close()
    return result


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        entries = [
            _run_entry(browser, name, locale, width, height)
            for width, height in ((1440, 900), (390, 844))
            for locale in ("en", "zh-Hant")
            for name in SCENARIOS
        ]
        browser.close()
    summary = {
        "entry_count": len(entries),
        "screenshot_count": len(entries),
        "external_requests": sum(len(item["external_requests"]) for item in entries),
        "write_requests": sum(len(item["write_requests"]) for item in entries),
        "render_acknowledgements": sum(
            item["render_acknowledgements"] for item in entries
        ),
        "console_errors": sum(len(item["console_errors"]) for item in entries),
        "page_errors": sum(len(item["page_errors"]) for item in entries),
        "overlap_count": sum(item["overlap_count"] for item in entries),
        "clipped_text_count": sum(item["clipped_text_count"] for item in entries),
    }
    final_entries = [
        item for item in entries if item["scenario"] == "final-unconfirmed"
    ]
    assert summary == {
        "entry_count": 24,
        "screenshot_count": 24,
        "external_requests": 0,
        "write_requests": 0,
        "render_acknowledgements": 0,
        "console_errors": 0,
        "page_errors": 0,
        "overlap_count": 0,
        "clipped_text_count": 0,
    }
    assert all(
        "Not confirmed as of 2026-08-27; active checking stopped."
        in item["final_unconfirmed_text"]
        for item in final_entries
        if item["locale"] == "en"
    )
    assert all(
        "\u622a\u81f3 2026-08-27 \u5c1a\u672a\u78ba\u8a8d\uff1b"
        "\u5df2\u505c\u6b62\u4e3b\u52d5\u8ffd\u67e5\u3002"
        in item["final_unconfirmed_text"]
        for item in final_entries
        if item["locale"] == "zh-Hant"
    )
    payload = {
        "schema_version": 1,
        "app_url": APP_URL,
        "fixture_only": True,
        "transient_frontend_fixture_server": True,
        "production_app_restart": False,
        "authority_semantics": (
            "declared_fixture_execution_boundary_not_instrumented_measurement"
        ),
        "provider_calls": 0,
        "production_backend_started": False,
        "production_database_operations": 0,
        "summary": summary,
        "entries": entries,
    }
    MATRIX.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
