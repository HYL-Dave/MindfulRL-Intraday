"""Run the fixture-only automated-disposition browser matrix."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from types import ModuleType
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright


APP_URL = os.environ.get("ARKSCOPE_DISPOSITION_APP_URL", "http://127.0.0.1:4201/")
PACKET = Path(__file__).resolve().parent
OUTPUT = PACKET / "browser"
OUTPUT.mkdir(parents=True, exist_ok=True)
ROOT = PACKET.parents[3]
STAGE5_RUNNER = (
    ROOT
    / "docs/superpowers/evidence/2026-08-25-trusted-lifecycle-automation-stage-5-repair"
    / "run_browser_matrix.py"
)


def _load_stage5_fixture() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage5_repair_fixture", STAGE5_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("stage5_fixture_loader_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE5 = _load_stage5_fixture()
SCENARIOS = {
    "attention-conflict": {
        "case_id": "case-conflict",
        "ticker": "CONFLICT",
        "issuer": "Conflicting Sources Corp.",
        "queue": "attention",
        "disposition": "exception_required",
        "reason": "source_conflict",
        "tier": "review_suggested",
        "readiness": "action_blocked",
        "statuses": {
            "regulator": "conflict",
            "market_infrastructure": "present",
            "publisher": "unavailable",
        },
    },
    "pending-frozen": {
        "case_id": "case-pending",
        "ticker": "PENDING",
        "issuer": "Pending Combination Corp.",
        "queue": "monitoring",
        "disposition": "not_confirmed_yet",
        "reason": "event_completion_not_confirmed",
        "tier": "verified_automatic",
        "readiness": "waiting_market_confirmation",
        "statuses": {
            "regulator": "present",
            "market_infrastructure": "missing",
            "publisher": "unavailable",
        },
    },
    "future-effective": {
        "case_id": "case-future",
        "ticker": "FUTURE",
        "issuer": "Future Effective Corp.",
        "queue": "monitoring",
        "disposition": "confirmed_monitoring",
        "reason": "waiting_effective_date",
        "tier": "verified_automatic",
        "readiness": "waiting_effective_date",
        "statuses": {
            "regulator": "confirmed",
            "market_infrastructure": "present",
            "publisher": "present",
        },
    },
    "settled-history": {
        "case_id": "case-history",
        "ticker": "HAPN",
        "issuer": "Happify Network, Inc.",
        "queue": "history",
        "disposition": "confirmed_effective",
        "reason": "transition_applied",
        "tier": "verified_automatic",
        "readiness": "not_applicable",
        "statuses": {
            "regulator": "confirmed",
            "market_infrastructure": "confirmed",
            "publisher": "confirmed",
        },
    },
    "source-missing": {
        "case_id": "case-source-missing",
        "ticker": "MISSING",
        "issuer": "Missing Source Corp.",
        "queue": "attention",
        "disposition": "exception_required",
        "reason": "source_missing",
        "tier": None,
        "readiness": None,
        "statuses": {
            "regulator": "missing",
            "market_infrastructure": "missing",
            "publisher": "missing",
        },
        "source_presence": "source_missing",
    },
}
LABELS = {
    "en": {
        "universe": "Universe",
        "open_nav": "Open navigation",
        "lifecycle": "Security event investigation",
        "integrity": "Data integrity",
        "dispositions": {
            "confirmed_monitoring": "Confirmed, monitoring",
            "confirmed_effective": "Confirmed complete",
            "not_confirmed_yet": "Not yet confirmed",
            "exception_required": "Needs review",
        },
        "reasons": {
            "source_conflict": "Source facts conflict",
            "event_completion_not_confirmed": "Event completion has not been confirmed",
            "waiting_effective_date": "Waiting for the effective date",
            "transition_applied": "The tracking transition was applied",
            "source_missing": "Source observation is missing",
        },
        "states": {
            "confirmed": "Confirmed",
            "present": "Available",
            "missing": "Missing",
            "unavailable": "Temporarily unavailable",
            "conflict": "Conflicting",
        },
        "revalidation": "Revalidation required",
    },
    "zh-Hant": {
        "universe": "全部標的",
        "open_nav": "開啟導覽",
        "lifecycle": "標的事件調查",
        "integrity": "資料完整性",
        "dispositions": {
            "confirmed_monitoring": "已確認，持續監看",
            "confirmed_effective": "已確認完成",
            "not_confirmed_yet": "尚未確認發生",
            "exception_required": "需要複查",
        },
        "reasons": {
            "source_conflict": "來源事實互相衝突",
            "event_completion_not_confirmed": "尚未確認事件已完成",
            "waiting_effective_date": "等待生效日",
            "transition_applied": "已套用追蹤轉移",
            "source_missing": "原始觀察缺失",
        },
        "states": {
            "confirmed": "已確認",
            "present": "可用",
            "missing": "缺失",
            "unavailable": "暫時無法取得",
            "conflict": "互相衝突",
        },
        "revalidation": "需要重新驗證",
    },
}


def _accepted_assessment(
    *,
    effective_date: str,
    source_ticker: str = "LC",
    successor_ticker: str = "HAPN",
) -> dict:
    assessment = deepcopy(STAGE5.ACCEPTED)
    assessment["effective_date"] = effective_date
    assessment["automation_run_id"] = "run-v3"
    assessment["decision_provenance_sha256"] = "9" * 64
    assessment["successor_ticker"] = successor_ticker
    assessment["conclusion"] = (
        f"The tracked security continued from {source_ticker} to "
        f"{successor_ticker}."
    )
    assessment["impact_summary"] = (
        f"Tracking continues under {successor_ticker} from {effective_date}."
    )
    return assessment


def _summary(name: str) -> dict:
    scenario = SCENARIOS[name]
    source_presence = scenario.get("source_presence", "present")
    current = (
        _accepted_assessment(
            effective_date="2026-09-30",
            source_ticker="FUTURE",
            successor_ticker="FUTR2",
        )
        if name == "future-effective"
        else _accepted_assessment(effective_date="2026-06-27")
        if name == "settled-history"
        else None
    )
    summary = deepcopy(STAGE5._summary())
    summary.update(
        {
            "case_id": scenario["case_id"],
            "source_ref": f"fixture-{name}",
            "ticker": scenario["ticker"],
            "source_presence": source_presence,
            "workflow_state": "resolved" if name == "settled-history" else "unresolved",
            "issuer_name": scenario["issuer"],
            "filing_date": "2026-08-20",
            "kinds": [
                {
                    "event_type": "listing_status_review",
                    "effective_date": (
                        "2026-09-30" if name == "future-effective" else "2026-08-20"
                    ),
                }
            ],
            "current_assessment": current,
            "current_acknowledgement": None,
            "active_sources": ["manual_lists"],
            "source_context": "unavailable" if source_presence == "source_missing" else "available",
            "components": {},
            "investigation_run_count": 0,
            "automation_run_count": 1 if scenario["tier"] else 0,
            "automation_fact_count": 5 if scenario["tier"] else 0,
            "automation_tier": scenario["tier"],
            "action_readiness": scenario["readiness"],
            "disposition": scenario["disposition"],
            "queue_bucket": scenario["queue"],
            "disposition_reason": scenario["reason"],
            "last_checked_at": "2026-08-26T08:00:00Z",
            "next_check_at": (
                "2026-09-30T00:00:00Z"
                if name == "future-effective"
                else "2026-09-02T08:00:00Z"
                if name == "pending-frozen"
                else None
            ),
            "source_family_status": scenario["statuses"],
            "evidence_count": 3 if source_presence == "present" else 0,
            "assessment_count": 1 if current else 0,
            "acknowledgement_count": 1 if name in {"settled-history", "source-missing"} else 0,
            "proposal_count": 2 if current else 0,
        }
    )
    return summary


def _detail(name: str) -> dict:
    scenario = SCENARIOS[name]
    detail = deepcopy(STAGE5._detail())
    detail.update(_summary(name))
    if detail["observation"] is not None:
        detail["observation"].update(
            {
                "issuer_name": scenario["issuer"],
                "filing_date": "2026-08-20",
                "description": f"Fixture evidence for {scenario['ticker']}.",
                "kinds": detail["kinds"],
            }
        )
    detail["observation_fingerprint_sha256"] = "8" * 64
    detail["investigation_runs"] = []
    detail["automation_runs"] = []
    detail["automation_facts"] = []
    detail["proposals"] = []
    detail["ticker_transition"] = None
    detail["assessment_history"] = []
    detail["acknowledgement_history"] = []

    for evidence in detail["evidence"]:
        family = evidence["source_family"]
        evidence["title"] = f"{family.replace('_', ' ').title()} fixture: {scenario['ticker']}"
        evidence["excerpt"] = (
            f"Reviewed {family.replace('_', ' ')} fixture evidence for "
            f"{scenario['ticker']} in the {name} scenario."
        )
        evidence["content_sha256"] = hashlib.sha256(
            evidence["excerpt"].encode("utf-8")
        ).hexdigest()
        evidence["translations"] = []

    if name == "pending-frozen":
        market = next(
            item for item in detail["evidence"]
            if item["source_family"] == "market_infrastructure"
        )
        market["excerpt"] = (
            "IBKR market data status is frozen; last price 9.50 was observed "
            "after the event date and is not a fresh-market confirmation."
        )
        market["content_sha256"] = hashlib.sha256(
            market["excerpt"].encode("utf-8")
        ).hexdigest()
        market["source_locator"] = {
            "contract_status": "found",
            "market_data": {
                "status": "frozen",
                "last": "9.50",
                "provider_time": "2026-08-26T07:30:00Z",
                "retrieved_at": "2026-08-26T08:00:00Z",
                "fresh": False,
            },
        }
    elif name == "future-effective":
        assessment = _accepted_assessment(
            effective_date="2026-09-30",
            source_ticker="FUTURE",
            successor_ticker="FUTR2",
        )
        detail["current_assessment"] = assessment
        detail["assessment_history"] = [assessment]
        transition = deepcopy(STAGE5._detail()["ticker_transition"])
        transition.update(
            {
                "transition_id": "transition-future",
                "case_id": scenario["case_id"],
                "status": "approved",
                "execute_on": "2026-09-30",
                "latest_attempt": None,
                "activity_history": [],
                "activity_count": 0,
                "unacknowledged_activity_count": 0,
            }
        )
        detail["ticker_transition"] = transition
    elif name == "settled-history":
        accepted = _accepted_assessment(effective_date="2026-06-27")
        stale = deepcopy(STAGE5.DRAFT)
        stale.update(
            {
                "assessment_id": "assessment-stale",
                "status": "draft",
                "stale": True,
            }
        )
        detail["current_assessment"] = accepted
        detail["assessment_history"] = [stale, accepted]
        detail["acknowledgement_history"] = [
            {
                "acknowledgement_id": "ack-reopened",
                "reason": "evidence_insufficient",
                "note": None,
                "stale": True,
                "acknowledged_at": "2026-08-20T08:00:00Z",
                "reopened_at": "2026-08-21T08:00:00Z",
            }
        ]
        detail["ticker_transition"] = deepcopy(STAGE5._detail()["ticker_transition"])
    elif name == "source-missing":
        stale = deepcopy(STAGE5.ACCEPTED)
        stale.update(
            {
                "assessment_id": "assessment-missing-stale",
                "stale": True,
                "rule_id": "lifecycle.insufficient_identity_facts",
                "conclusion": "The prior MISSING assessment requires revalidation.",
                "impact_summary": "No tracking change is authorized from stale evidence.",
                "outcomes": ["undetermined"],
                "successor_ticker": None,
                "destination_venue": None,
                "effective_date": None,
            }
        )
        detail["observation"] = None
        detail["observation_fingerprint_sha256"] = None
        detail["evidence"] = []
        detail["current_assessment"] = None
        detail["assessment_history"] = [stale]
        detail["acknowledgement_history"] = [
            {
                "acknowledgement_id": "ack-missing",
                "reason": "evidence_insufficient",
                "note": None,
                "stale": True,
                "acknowledged_at": "2026-08-20T08:00:00Z",
                "reopened_at": "2026-08-21T08:00:00Z",
            }
        ]
    return detail


def _case_list(query: dict[str, list[str]]) -> list[dict]:
    if query.get("source_presence") == ["source_missing"]:
        return [_summary("source-missing")]
    queue = query.get("queue_bucket", [None])[0]
    names = {
        "attention": ("attention-conflict",),
        "monitoring": ("pending-frozen", "future-effective"),
        "history": ("settled-history",),
        None: (
            "attention-conflict",
            "pending-frozen",
            "future-effective",
            "settled-history",
        ),
    }[queue]
    return [_summary(name) for name in names]


def _fixture_inventory() -> dict:
    dispositions = {value["disposition"] for value in SCENARIOS.values()}
    states = {
        state
        for value in SCENARIOS.values()
        for state in value["statuses"].values()
    }
    pending = _detail("pending-frozen")
    pending_market = next(
        item for item in pending["evidence"]
        if item["source_family"] == "market_infrastructure"
    )
    inventory = {
        "dispositions": sorted(dispositions),
        "source_family_states": sorted(states),
        "source_conflict": _summary("attention-conflict")["disposition_reason"],
        "frozen_quote": pending_market["source_locator"]["market_data"],
        "future_effective_date": _summary("future-effective")["next_check_at"],
        "post_date_market_wait": _summary("pending-frozen")["action_readiness"],
        "settled_history": _summary("settled-history")["disposition_reason"],
        "stale_reopened_history": _detail("settled-history")["acknowledgement_history"],
        "source_missing_integrity": _summary("source-missing")["source_presence"],
    }
    assert dispositions == {
        "confirmed_monitoring",
        "confirmed_effective",
        "not_confirmed_yet",
        "exception_required",
    }
    assert states == {"confirmed", "present", "missing", "unavailable", "conflict"}
    assert inventory["frozen_quote"]["fresh"] is False
    return inventory


def _run_entry(browser, name: str, locale: str, width: int, height: int) -> dict:
    scenario = SCENARIOS[name]
    labels = LABELS[locale]
    state = {"requests": [], "external": []}
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
            STAGE5._response(route, {}, 204)
        elif parsed.path == "/status":
            STAGE5._response(
                route,
                {
                    "status": "ok",
                    "timestamp": "2026-08-26T08:00:00Z",
                    "tools_registered": 50,
                    "tool_categories": {},
                    "data_sources": {},
                },
            )
        elif parsed.path == "/config/runtime":
            STAGE5._response(route, STAGE5._runtime_config())
        elif parsed.path == "/profile/settings/ui-locale":
            STAGE5._response(route, {"locale": locale, "source": "stored"})
        elif parsed.path == "/profile/universe":
            STAGE5._response(
                route,
                {
                    "as_of": "2026-08-26",
                    "generated_at": "2026-08-26T08:00:00Z",
                    "total": 0,
                    "shown": 0,
                    "archived_count": 0,
                    "summarized": 0,
                    "rows": [],
                },
            )
        elif parsed.path == "/profile/lists":
            STAGE5._response(route, {"lists": []})
        elif parsed.path == "/analysis/cards":
            STAGE5._response(route, {"cards": []})
        elif parsed.path == "/research/threads":
            STAGE5._response(route, {"threads": []})
        elif parsed.path == "/security-lifecycle/cases":
            cases = _case_list(parse_qs(parsed.query))
            STAGE5._response(
                route,
                {
                    "cases": cases,
                    "count": len(cases),
                    "queue_counts": {"attention": 1, "monitoring": 2, "history": 1},
                    "data_integrity": {"source_missing_count": 1},
                },
            )
        elif parsed.path.startswith("/security-lifecycle/cases/"):
            case_id = parsed.path.rsplit("/", 1)[-1]
            scenario_name = next(
                key for key, value in SCENARIOS.items() if value["case_id"] == case_id
            )
            STAGE5._response(route, _detail(scenario_name))
        elif parsed.path == "/security-lifecycle/transition-activity":
            STAGE5._response(
                route,
                {"items": [], "count": 0, "unacknowledged_count": 0},
            )
        else:
            STAGE5._response(route, {"detail": {"code": "fixture_unavailable"}}, 503)

    page.route("**/*", handler)
    page.goto(APP_URL, wait_until="networkidle", timeout=20_000)
    universe = page.get_by_role("button", name=labels["universe"], exact=True)
    if not universe.is_visible():
        page.get_by_role("button", name=labels["open_nav"], exact=True).click()
    universe.click()
    page.get_by_role("tab", name=labels["lifecycle"], exact=True).click()

    if name == "source-missing":
        page.get_by_role("button", name=re.compile(rf"^{re.escape(labels['integrity'])}")) .click()
    elif scenario["queue"] != "attention":
        page.locator(f"[data-queue-view='{scenario['queue']}']").click()

    trigger = page.get_by_role("button", name=re.compile(rf"^{scenario['ticker']}\b"))
    trigger.wait_for(state="visible")
    row = trigger.locator("xpath=ancestor::tr")
    assert row.locator(
        f"[data-disposition='{scenario['disposition']}']"
    ).count() == 1
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

    metrics = STAGE5._geometry(page)
    STAGE5._assert_geometry(metrics)
    screenshot = OUTPUT / f"{width}x{height}-{locale}-{name}.png"
    page.screenshot(path=str(screenshot))
    pixels = STAGE5._pixel_check(screenshot, width, height)
    writes = [
        item for item in state["requests"]
        if item["method"] in {"POST", "PUT", "PATCH", "DELETE"}
    ]
    render_acknowledgements = sum(
        1 for item in writes if item["path"].endswith("/acknowledge")
    )
    assert state["external"] == [], state["external"]
    assert writes == [], writes
    assert render_acknowledgements == 0
    assert console_errors == [], console_errors
    assert page_errors == [], page_errors
    result = {
        "scenario": name,
        "locale": locale,
        "viewport": [width, height],
        "screenshot": screenshot.name,
        "pixels": pixels,
        "request_count": len(state["requests"]),
        "external_requests": state["external"],
        "writes": writes,
        "render_acknowledgements": render_acknowledgements,
        "console_errors": console_errors,
        "page_errors": page_errors,
        "overlap_count": len(metrics["overlaps"]),
        "clipped_text_count": len(metrics["textOverflow"]),
    }
    context.close()
    return result


def main() -> int:
    inventory = _fixture_inventory()
    (OUTPUT / "fixture-inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        entries = [
            _run_entry(browser, name, locale, width, height)
            for width, height in ((1440, 900), (390, 844))
            for locale in ("en", "zh-Hant")
            for name in SCENARIOS
        ]
        browser.close()
    payload = {
        "schema_version": 1,
        "app_url": APP_URL,
        "fixture_only": True,
        "provider_calls": 0,
        "production_backend_started": False,
        "production_database_operations": 0,
        "entries": entries,
    }
    (OUTPUT / "matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "entries": len(entries),
                "screenshots": len(entries),
                "external_requests": sum(len(item["external_requests"]) for item in entries),
                "writes": sum(len(item["writes"]) for item in entries),
                "render_acknowledgements": sum(
                    item["render_acknowledgements"] for item in entries
                ),
                "console_errors": sum(len(item["console_errors"]) for item in entries),
                "page_errors": sum(len(item["page_errors"]) for item in entries),
                "overlap_count": sum(item["overlap_count"] for item in entries),
                "clipped_text_count": sum(
                    item["clipped_text_count"] for item in entries
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
