"""Run the fixture-only lifecycle content-translation browser matrix."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
from types import ModuleType
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


APP_URL = os.environ.get("ARKSCOPE_TRANSLATION_APP_URL", "http://127.0.0.1:4198/")
PACKET = Path(__file__).resolve().parent
OUTPUT = PACKET / "browser"
OUTPUT.mkdir(parents=True, exist_ok=True)
ROOT = PACKET.parents[3]
STAGE5_RUNNER = (
    ROOT
    / "docs/superpowers/evidence/2026-08-25-trusted-lifecycle-automation-stage-5"
    / "run_browser_matrix.py"
)


def _load_stage5_fixture() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage5_browser_fixture", STAGE5_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("stage5_fixture_loader_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE5 = _load_stage5_fixture()
ORIGINAL = "The same common stock will trade as HAPN on Nasdaq beginning June 27, 2026."
TRANSLATED = {
    "en": "The same common stock will continue trading as HAPN on Nasdaq from June 27, 2026.",
    "zh-Hant": "同一普通股將自 2026 年 6 月 27 日起以 HAPN 在 Nasdaq 繼續交易。",
}
UI = {
    "en": {
        "universe": "Universe",
        "open_nav": "Open navigation",
        "lifecycle": "Security event investigation",
        "translate": "Translate evidence",
        "machine_translation": "Machine translation",
        "retry": "Retry translation",
        "settings": "Open Content Translation settings",
        "messages": {
            "translation_timeout": "Translation timed out. Try again.",
            "translation_auth_rejected": (
                "Content translation authentication was rejected. Sign in again or adjust "
                "Content Translation settings."
            ),
            "translation_quota_exhausted": (
                "The selected content translation account has no remaining quota."
            ),
            "translation_output_invalid": (
                "The model returned an invalid translation output. Try again."
            ),
            "evidence_changed": (
                "The source evidence changed. Refresh the case before translating again."
            ),
        },
    },
    "zh-Hant": {
        "universe": "全部標的",
        "open_nav": "開啟導覽",
        "lifecycle": "標的事件調查",
        "translate": "翻譯證據",
        "machine_translation": "機器翻譯",
        "retry": "重試翻譯",
        "settings": "前往內容翻譯設定",
        "messages": {
            "translation_timeout": "翻譯逾時，請重試。",
            "translation_auth_rejected": "內容翻譯認證遭拒，請重新登入或調整內容翻譯設定。",
            "translation_quota_exhausted": "所選內容翻譯帳戶的可用額度已用盡。",
            "translation_output_invalid": "模型回傳的翻譯格式無效，請重試。",
            "evidence_changed": "來源證據已變更，請重新整理案件後再翻譯。",
        },
    },
}
SCENARIOS = {
    "success": {"kind": "success"},
    "cached": {"kind": "cached"},
    "timeout": {
        "kind": "failure",
        "code": "translation_timeout",
        "status": 504,
        "retryable": True,
        "action": "retry",
    },
    "auth-rejected": {
        "kind": "failure",
        "code": "translation_auth_rejected",
        "status": 401,
        "retryable": False,
        "action": "settings",
    },
    "quota-exhausted": {
        "kind": "failure",
        "code": "translation_quota_exhausted",
        "status": 429,
        "retryable": False,
        "action": "settings",
    },
    "invalid-output": {
        "kind": "failure",
        "code": "translation_output_invalid",
        "status": 502,
        "retryable": False,
        "action": "retry",
    },
    "evidence-changed": {
        "kind": "failure",
        "code": "evidence_changed",
        "status": 409,
        "retryable": False,
        "action": None,
    },
}


def _translation(locale: str) -> dict:
    return {
        "evidence_id": "evidence-regulator",
        "evidence_content_sha256": "a" * 64,
        "locale": locale,
        "translated_text": TRANSLATED[locale],
        "provider": "fixture-provider",
        "model": "fixture-model",
        "harness": "fixture-harness",
        "translated_at": "2026-08-26T06:00:00Z",
    }


def _fixture_detail(locale: str, *, cached: bool) -> dict:
    detail = deepcopy(STAGE5._detail())
    target = next(
        item for item in detail["evidence"] if item["evidence_id"] == "evidence-regulator"
    )
    target["excerpt"] = ORIGINAL
    target["translations"] = [_translation(locale)] if cached else []
    detail["evidence"] = [target]
    detail["evidence_count"] = 1
    return detail


def _scroll_into_drawer(page, target) -> int:
    value = target.evaluate(
        """(node) => {
          const scroller = node.closest('.ui-drawer-body');
          if (!scroller) throw new Error('drawer_scroll_container_missing');
          const delta = node.getBoundingClientRect().top
            - scroller.getBoundingClientRect().top - 8;
          scroller.scrollTop = Math.max(0, scroller.scrollTop + delta);
          return Math.round(scroller.scrollTop);
        }"""
    )
    page.wait_for_timeout(50)
    return value


def _run_entry(browser, scenario_name: str, locale: str, width: int, height: int) -> dict:
    scenario = SCENARIOS[scenario_name]
    labels = UI[locale]
    current_detail = _fixture_detail(locale, cached=scenario["kind"] == "cached")
    state = {"requests": [], "external": [], "translation_posts": 0}
    context = browser.new_context(viewport={"width": width, "height": height})
    context.add_init_script(
        f"localStorage.setItem('arkscope.ui.locale.v1', {json.dumps(locale)});"
    )
    if scenario["kind"] == "failure":
        failure_body = {
            "detail": {
                "code": scenario["code"],
                "message": "fixture-secret-must-not-render",
                "provider": "openai",
                "model": "fixture-model",
                "harness": "fixture-harness",
                "retryable": scenario["retryable"],
            }
        }
        context.add_init_script(
            f"""(() => {{
              const nativeFetch = window.fetch.bind(window);
              window.__translationFixturePosts = 0;
              window.fetch = (input, init) => {{
                const url = typeof input === 'string' ? input : input.url;
                const method = (init?.method || (typeof input === 'string' ? 'GET' : input.method))
                  .toUpperCase();
                if (method === 'POST'
                    && url.includes('/security-lifecycle/evidence/evidence-regulator/translations')) {{
                  window.__translationFixturePosts += 1;
                  return Promise.resolve(new Response(
                    {json.dumps(json.dumps(failure_body, separators=(",", ":")))},
                    {{status: {scenario["status"]}, headers: {{'content-type': 'application/json'}}}},
                  ));
                }}
                return nativeFetch(input, init);
              }};
            }})();"""
        )
    page = context.new_page()
    page.set_default_timeout(10_000)
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
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
                    "timestamp": "2026-08-26T06:00:00Z",
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
                    "generated_at": "2026-08-26T06:00:00Z",
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
            summary = deepcopy(STAGE5._summary())
            summary["evidence_count"] = 1
            STAGE5._response(
                route,
                {"cases": [summary], "count": 1, "data_integrity": {"source_missing_count": 0}},
            )
        elif parsed.path == "/security-lifecycle/cases/case-hapn":
            STAGE5._response(route, current_detail)
        elif parsed.path == "/security-lifecycle/transition-activity":
            STAGE5._response(route, {"items": [], "count": 0, "unacknowledged_count": 0})
        elif (
            parsed.path == "/security-lifecycle/evidence/evidence-regulator/translations"
            and request.method == "POST"
        ):
            state["translation_posts"] += 1
            if scenario["kind"] == "success":
                translation = _translation(locale)
                current_detail["evidence"][0]["translations"] = [translation]
                STAGE5._response(route, translation)
            else:
                raise AssertionError("failure_fixture_reached_network_route")
        else:
            STAGE5._response(route, {"detail": {"code": "fixture_unavailable"}}, 503)

    page.route("**/*", handler)
    page.goto(APP_URL, wait_until="domcontentloaded", timeout=20_000)
    universe = page.get_by_role("button", name=labels["universe"], exact=True)
    if not universe.is_visible():
        page.get_by_role("button", name=labels["open_nav"], exact=True).click()
    universe.click()
    page.get_by_role("tab", name=labels["lifecycle"], exact=True).click()
    page.get_by_role("button", name=re.compile(r"^HAPN\b")).click()

    drawer = page.locator(".ui-drawer")
    drawer.wait_for(state="visible")
    evidence = drawer.locator("article.lifecycle-evidence-item").filter(
        has_text="SEC current report"
    )
    evidence.wait_for(state="visible")
    assert ORIGINAL in evidence.inner_text()

    if scenario["kind"] == "cached":
        evidence.get_by_text(TRANSLATED[locale], exact=True).wait_for(state="visible")
        assert state["translation_posts"] == 0
    else:
        evidence.get_by_role("button", name=labels["translate"], exact=True).click()
        if scenario["kind"] == "success":
            evidence.get_by_text(TRANSLATED[locale], exact=True).wait_for(state="visible")
        else:
            message = labels["messages"][scenario["code"]]
            evidence.get_by_text(message, exact=True).wait_for(state="visible")
            state["translation_posts"] = page.evaluate(
                "() => window.__translationFixturePosts"
            )
            assert "OpenAI · fixture-model · fixture-harness" in evidence.inner_text()
            if scenario["action"] == "retry":
                assert evidence.get_by_role("button", name=labels["retry"], exact=True).is_visible()
            elif scenario["action"] == "settings":
                assert evidence.get_by_role(
                    "button", name=labels["settings"], exact=True
                ).is_visible()
            else:
                assert evidence.get_by_role("button", name=labels["retry"], exact=True).count() == 0
                assert evidence.locator("[data-action='open-content-translation-settings']").count() == 0
        assert state["translation_posts"] == 1

    _scroll_into_drawer(page, evidence)
    visible = evidence.inner_text()
    assert ORIGINAL in visible
    assert "fixture-secret-must-not-render" not in visible
    if scenario["kind"] in {"success", "cached"}:
        assert labels["machine_translation"] in visible
        assert TRANSLATED[locale] in visible

    metrics = STAGE5._geometry(page)
    STAGE5._assert_geometry(metrics)
    screenshot = OUTPUT / f"{scenario_name}-{locale}-{width}x{height}.png"
    page.screenshot(path=str(screenshot))
    pixels = STAGE5._pixel_check(screenshot, width, height)

    unexpected_posts = [
        item
        for item in state["requests"]
        if item["method"] in {"POST", "PUT", "PATCH", "DELETE"}
        and item["path"] != "/security-lifecycle/evidence/evidence-regulator/translations"
    ]
    assert unexpected_posts == [], unexpected_posts
    assert state["external"] == [], state["external"]
    assert console_errors == [], console_errors
    assert page_errors == [], page_errors
    result = {
        "scenario": scenario_name,
        "locale": locale,
        "viewport": [width, height],
        "screenshot": screenshot.name,
        "pixels": pixels,
        "request_count": len(state["requests"]),
        "translation_posts": state["translation_posts"],
        "unexpected_mutations": unexpected_posts,
        "external_requests": state["external"],
        "console_errors": console_errors,
        "page_errors": page_errors,
        "original_visible": ORIGINAL in visible,
        "translation_visible": (
            TRANSLATED[locale] in visible if scenario["kind"] in {"success", "cached"} else None
        ),
    }
    context.close()
    return result


def main() -> int:
    selected = tuple(
        value.strip()
        for value in os.environ.get(
            "ARKSCOPE_TRANSLATION_SCENARIOS", ",".join(SCENARIOS)
        ).split(",")
        if value.strip()
    )
    unknown = set(selected) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"unknown_scenarios:{sorted(unknown)}")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        entries = []
        for scenario in selected:
            for locale in ("en", "zh-Hant"):
                for width, height in ((1440, 900), (390, 844)):
                    label = f"{scenario}:{locale}:{width}x{height}"
                    print(json.dumps({"started": label}), flush=True)
                    entries.append(_run_entry(browser, scenario, locale, width, height))
                    print(json.dumps({"completed": label}), flush=True)
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
                "console_errors": sum(len(item["console_errors"]) for item in entries),
                "page_errors": sum(len(item["page_errors"]) for item in entries),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
