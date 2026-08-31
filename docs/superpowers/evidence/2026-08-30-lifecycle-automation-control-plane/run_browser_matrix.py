"""Capture the fixture-only lifecycle automation control-plane browser matrix."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from types import ModuleType
from urllib.parse import urlparse

from PIL import Image
from playwright.sync_api import sync_playwright


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
OUTPUT = PACKET / "browser"
APP_URL = os.environ.get("ARKSCOPE_AUTOMATION_APP_URL", "http://127.0.0.1:4210/")
BASE_RUNNER = (
    ROOT
    / "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority"
    / "run_browser_matrix.py"
)
VIEWPORTS = ((1440, 900), (390, 844))
LOCALES = ("en", "zh-Hant")


def load_base() -> ModuleType:
    spec = importlib.util.spec_from_file_location("listing_browser_fixture", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("browser_fixture_loader_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()
LABELS = {
    "en": {
        "settings_title": "Deterministic automation",
        "run_due": "Run due cases now",
        "run_case": "Run this case",
        "close": "Close",
        "stage": "Listing directories",
    },
    "zh-Hant": {
        "settings_title": "確定性自動化",
        "run_due": "立即檢查到期案件",
        "run_case": "執行此案件",
        "close": "關閉",
        "stage": "上市名錄",
    },
}


def response(route, value: object, status: int = 200) -> None:
    BASE.STAGE5._response(route, value, status)


def result_payload() -> dict[str, object]:
    return {
        "status": "succeeded",
        "reason": None,
        "selected": 1,
        "processed": 1,
        "accepted": 1,
        "drafted": 0,
        "blocked": 0,
        "failed": 0,
        "skipped_current": 0,
        "case_ids": ["case-active"],
        "result_version": 2,
        "case_outcomes": {"case-active": "accepted"},
    }


def status_payload(state: dict[str, object]) -> dict[str, object]:
    progress: list[dict[str, object]] = []
    if state.get("run_request_id") is not None:
        state["post_dispatch_status_calls"] = int(
            state.get("post_dispatch_status_calls", 0)
        ) + 1
        if int(state["post_dispatch_status_calls"]) == 1:
            progress = [{
                "trigger": state["trigger"],
                "request_id": state["run_request_id"],
                "case_id": state.get("run_case_id", "case-active"),
                "started_at": "2026-08-31T01:00:00Z",
                "current_stage": "listing",
                "completed_stages": ["preparing", "sec"],
                "skipped_stages": [],
            }]
    elif state.get("show_initial_progress"):
        progress = [{
            "trigger": "scheduler",
            "request_id": "sla_fixture_initial",
            "case_id": "case-active",
            "started_at": "2026-08-31T00:55:00Z",
            "current_stage": "evaluate",
            "completed_stages": ["preparing", "sec", "listing"],
            "skipped_stages": ["ibkr"],
        }]
    return {
        "config_status": "valid",
        "config": state.get("config", {
            "enabled": True,
            "interval_minutes": 5,
            "batch_limit": 2,
            "apply_profile_transitions": False,
        }),
        "schedule": {
            "status": "scheduled",
            "last_attempt_at": "2026-08-31T00:55:00Z",
            "next_scheduled_at": "2026-08-31T01:00:00Z",
        },
        "telemetry_status": "valid",
        "last_status": "succeeded",
        "last_result": result_payload(),
        "active_incident": None,
        "latest_failed_runs": [],
        "current_progress": progress,
    }


def geometry(page) -> dict[str, object]:
    metrics = page.evaluate(
        """() => {
          const visible = (node) => {
            const style = getComputedStyle(node);
            const rect = node.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden'
              && Number(style.opacity) !== 0 && rect.width > 0 && rect.height > 0;
          };
          const roots = [...document.querySelectorAll(
            '.lifecycle-automation-settings, .lifecycle-case-automation, .ui-drawer'
          )];
          const controls = [...new Set(roots.flatMap((root) =>
            [...root.querySelectorAll('button,input,select,a[href]')]
          ))].filter(visible).map((node) => {
            const rect = node.getBoundingClientRect();
            return {
              tag: node.tagName,
              text: (node.textContent || node.getAttribute('aria-label') || '').trim(),
              className: node.className || '',
              detailsClassName: node.closest('details')?.className || '',
              detailsOpen: node.closest('details')?.open ?? null,
              left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom,
              width: rect.width, height: rect.height,
              viewportClipped: rect.left < -1 || rect.right > innerWidth + 1,
            };
          });
          const textOverflow = [...new Set(roots.flatMap((root) =>
            [...root.querySelectorAll('h2,h3,h4,strong,p,dt,dd,label,button,option')]
          ))].filter(visible).flatMap((node) => {
            const horizontal = node.scrollWidth > node.clientWidth + 1;
            const vertical = node.scrollHeight > node.clientHeight + 1;
            return horizontal || vertical ? [{
              tag: node.tagName,
              text: (node.textContent || '').trim(), horizontal, vertical,
              scrollWidth: node.scrollWidth, clientWidth: node.clientWidth,
              scrollHeight: node.scrollHeight, clientHeight: node.clientHeight,
            }] : [];
          });
          return {controls, textOverflow};
        }"""
    )
    controls = metrics["controls"]
    overlaps = []
    for index, left in enumerate(controls):
        for right in controls[index + 1 :]:
            width = min(left["right"], right["right"]) - max(
                left["left"], right["left"]
            )
            height = min(left["bottom"], right["bottom"]) - max(
                left["top"], right["top"]
            )
            if width > 1 and height > 1:
                overlaps.append([left, right])
    return {
        **metrics,
        "overlaps": overlaps,
        "viewport_clipped": [row for row in controls if row["viewportClipped"]],
    }


def calibrate_geometry(browser) -> dict[str, int]:
    page = browser.new_page(viewport={"width": 500, "height": 300})
    page.set_content(
        "<div class='ui-drawer' style='position:relative;width:300px'>"
        "<button style='position:absolute;left:0;top:0;width:100px'>A</button>"
        "<button style='position:absolute;left:40px;top:0;width:100px'>B</button>"
        "<p style='position:absolute;top:60px;width:80px;height:12px;overflow:hidden'>"
        "This text is deliberately too tall and too wide for its box.</p></div>"
    )
    measured = geometry(page)
    page.close()
    if not measured["overlaps"] or not measured["textOverflow"]:
        raise AssertionError("geometry_observer_inactive")
    return {
        "known_overlap_count": len(measured["overlaps"]),
        "known_clipped_text_count": len(measured["textOverflow"]),
    }


def pixel_check(path: Path, width: int, height: int) -> dict[str, int]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            raise AssertionError("screenshot_dimensions")
        colors = image.getcolors(maxcolors=width * height)
        color_count = 0 if colors is None else len(colors)
        extrema = image.getextrema()
        dynamic_channels = sum(low != high for low, high in extrema)
    if color_count < 20 or dynamic_channels < 2:
        raise AssertionError("screenshot_blank")
    return {"color_count": color_count, "dynamic_channels": dynamic_channels}


def run_entry(browser, *, surface: str, locale: str, width: int, height: int) -> dict:
    labels = LABELS[locale]
    state: dict[str, object] = {
        "requests": [],
        "external": [],
        "writes": [],
        "show_initial_progress": False,
    }
    context = browser.new_context(viewport={"width": width, "height": height})
    context.add_init_script(
        f"localStorage.setItem('arkscope.ui.locale.v1', {json.dumps(locale)});"
        "localStorage.setItem('arkscope.settings.activeGroup.v1', 'data_sync');"
    )
    page = context.new_page()
    page.set_default_timeout(12_000)
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on("console", lambda message: console_errors.append(message.text)
            if message.type == "error" else None)
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
        row = {"method": request.method, "path": parsed.path}
        state["requests"].append(row)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            state["writes"].append(row)
        if request.method == "OPTIONS":
            response(route, {}, 204)
            return
        if parsed.path == "/security-lifecycle/automation" and request.method == "GET":
            response(route, status_payload(state))
            return
        if parsed.path == "/security-lifecycle/automation" and request.method == "PUT":
            body = request.post_data_json
            state["config"] = body
            response(route, {"config_status": "valid", "config": body})
            return
        if parsed.path == "/security-lifecycle/automation/run" and request.method == "POST":
            state.update({
                "run_request_id": "sla_fixture_due",
                "run_case_id": "case-active",
                "trigger": "manual_due",
                "post_dispatch_status_calls": 0,
            })
            response(route, {
                "scope": "due", "status": "started", "request_id": "sla_fixture_due",
            })
            return
        if parsed.path.endswith("/automation/run") and request.method == "POST":
            case_id = parsed.path.split("/")[-3]
            state.update({
                "run_request_id": "sla_fixture_case",
                "run_case_id": case_id,
                "trigger": "manual_case",
                "post_dispatch_status_calls": 0,
                "show_initial_progress": False,
            })
            response(route, {
                "scope": "case", "status": "started",
                "request_id": "sla_fixture_case", "case_id": case_id,
            })
            return
        if request.method != "GET":
            response(route, {"detail": {"code": "fixture_write_forbidden"}}, 405)
            return
        BASE._api_response(route, parsed.path, parsed.query, locale)

    page.route("**/*", handler)
    page.goto(APP_URL, wait_until="networkidle", timeout=20_000)
    if surface == "settings":
        nav_labels = BASE.LABELS[locale]
        target = page.get_by_role(
            "button", name=nav_labels["settings"], exact=True
        )
        if not target.is_visible():
            page.get_by_role(
                "button", name=nav_labels["open_nav"], exact=True
            ).click()
        target.click()
        title = page.get_by_text(labels["settings_title"], exact=True)
        title.wait_for(state="visible")
        title.scroll_into_view_if_needed()
        page.get_by_role("button", name=labels["run_due"], exact=True).click()
        page.wait_for_timeout(250)
        assert any(row["path"] == "/security-lifecycle/automation/run" for row in state["writes"])
    else:
        BASE._navigate(page, "active", locale)
        drawer = page.locator(".ui-drawer")
        drawer.get_by_role("button", name=labels["run_case"], exact=True).click()
        page.wait_for_timeout(150)
        page.keyboard.press("Escape")
        page.locator("[data-queue-view='history']").click()
        page.get_by_role("button", name=re.compile(r"^TERM\b")).click()
        drawer.wait_for(state="visible")
        page.wait_for_timeout(1_250)
        assert "TERM" in drawer.inner_text()
        assert "HAPN" not in drawer.locator("h2,h3").first.inner_text()
        assert any(row["path"].endswith("/automation/run") for row in state["writes"])

    page.wait_for_timeout(200)
    measured = geometry(page)
    if measured["overlaps"] or measured["textOverflow"] or measured["viewport_clipped"]:
        raise AssertionError(json.dumps(measured, ensure_ascii=True))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    screenshot = OUTPUT / f"{width}x{height}-{locale}-{surface}.png"
    page.screenshot(path=str(screenshot), full_page=False)
    pixels = pixel_check(screenshot, width, height)
    body = page.locator("body").inner_text()
    if console_errors or page_errors or state["external"]:
        raise AssertionError({
            "console_errors": console_errors,
            "page_errors": page_errors,
            "external": state["external"],
        })
    result = {
        "surface": surface,
        "locale": locale,
        "viewport": [width, height],
        "screenshot": screenshot.name,
        "screenshot_sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
        "visible_text_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "request_count": len(state["requests"]),
        "fixture_write_count": len(state["writes"]),
        "external_requests": state["external"],
        "console_errors": console_errors,
        "page_errors": page_errors,
        "overlap_count": 0,
        "clipped_text_count": 0,
        "viewport_clipped_control_count": 0,
        "pixels": pixels,
        "latest_case_refresh_witness": surface != "lifecycle" or "TERM" in body,
    }
    context.close()
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob("*.png"):
        old.unlink()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        calibration = calibrate_geometry(browser)
        entries = [
            run_entry(browser, surface=surface, locale=locale, width=width, height=height)
            for width, height in VIEWPORTS
            for locale in LOCALES
            for surface in ("settings", "lifecycle")
        ]
        browser.close()
    payload = {
        "schema_version": 1,
        "fixture_only": True,
        "app_url": APP_URL,
        "geometry_positive_calibration": calibration,
        "provider_calls": {"value": 0, "basis": "declared_not_authorized"},
        "production_database_operations": {"value": 0, "basis": "declared_not_authorized"},
        "entries": entries,
    }
    (OUTPUT / "matrix.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "entries": len(entries),
        "screenshots": len(entries),
        "external_requests": sum(len(row["external_requests"]) for row in entries),
        "overlaps": sum(row["overlap_count"] for row in entries),
        "clipped_text": sum(row["clipped_text_count"] for row in entries),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
