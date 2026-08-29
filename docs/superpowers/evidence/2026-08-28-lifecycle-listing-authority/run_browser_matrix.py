"""Capture the fixture-only listing-authority browser admission matrix."""

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


APP_URL = os.environ.get("ARKSCOPE_LISTING_APP_URL", "http://127.0.0.1:4208/")
PACKET = Path(__file__).resolve().parent
OUTPUT = PACKET / "browser"
ROOT = PACKET.parents[3]
BASE_RUNNER = (
    ROOT
    / "docs/superpowers/evidence/2026-08-26-lifecycle-automated-disposition"
    / "run_browser_matrix.py"
)


def _load_base() -> ModuleType:
    spec = importlib.util.spec_from_file_location("disposition_browser_fixture", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("browser_fixture_loader_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
STAGE5 = BASE.STAGE5
VIEWPORTS = ((1440, 900), (390, 844))
LOCALES = ("en", "zh-Hant")
DECLARED_ZERO = {"value": 0, "basis": "declared_not_authorized"}
SCENARIOS = {
    "active": {
        "case_id": "case-active",
        "ticker": "HAPN",
        "issuer": "Happify Network, Inc.",
        "queue": "history",
        "disposition": "confirmed_effective",
        "reason": "resolved_no_change",
        "tier": "verified_automatic",
        "readiness": "not_applicable",
        "regulator_issuer_cik": "0001409970",
        "listings": [
            {
                "authority": "nasdaq_trader",
                "listing_status": "active",
                "market": "stocks",
                "primary_exchange": "XNAS",
                "candidate_ticker": "HAPN",
                "issuer_cik": None,
            }
        ],
    },
    "not-found-monitoring": {
        "case_id": "case-not-found",
        "ticker": "MISS",
        "issuer": "Missing Confirmation Corp.",
        "queue": "monitoring",
        "disposition": "not_confirmed_yet",
        "reason": "event_completion_not_confirmed",
        "tier": "verified_automatic",
        "readiness": "waiting_market_confirmation",
        "regulator_issuer_cik": "0001409970",
        "listings": [
            {
                "authority": "nasdaq_trader",
                "listing_status": "not_found",
                "market": "stocks",
                "primary_exchange": None,
                "candidate_ticker": "OLD",
                "issuer_cik": None,
            }
        ],
    },
    "inactive-history": {
        "case_id": "case-inactive",
        "ticker": "TERM",
        "issuer": "Terminal Listing Corp.",
        "queue": "history",
        "disposition": "confirmed_effective",
        "reason": "transition_applied",
        "tier": "verified_automatic",
        "readiness": "transition_eligible",
        "regulator_issuer_cik": "0001409970",
        "synthetic_post_apply_projection": True,
        "transition": {
            "kind": "terminal_delisting",
            "source_ticker": "TERM",
            "successor_ticker": None,
            "rule_id": "lifecycle.terminal_delisting",
            "outcomes": ["listing_ended"],
        },
        "listings": [
            {
                "authority": "massive",
                "listing_status": "inactive",
                "market": "stocks",
                "primary_exchange": "XNAS",
                "candidate_ticker": "TERM",
                "issuer_cik": "0001409970",
            }
        ],
    },
    "conflict-attention": {
        "case_id": "case-conflict",
        "ticker": "CONFLICT",
        "issuer": "Conflicting Listing Corp.",
        "queue": "attention",
        "disposition": "exception_required",
        "reason": "source_conflict",
        "tier": "review_suggested",
        "readiness": "action_blocked",
        "regulator_issuer_cik": "0001409970",
        "listings": [
            {
                "authority": "nasdaq_trader",
                "listing_status": "active",
                "market": "stocks",
                "primary_exchange": "XNYS",
                "candidate_ticker": "NEW",
                "issuer_cik": None,
            },
            {
                "authority": "massive",
                "listing_status": "active",
                "market": "stocks",
                "primary_exchange": "XNYS",
                "candidate_ticker": "NEW",
                "issuer_cik": "0000000001",
            },
        ],
    },
    "otc-continuation": {
        "case_id": "case-otc",
        "ticker": "OTC-A",
        "issuer": "OTC Continuation Corp.",
        "queue": "history",
        "disposition": "confirmed_effective",
        "reason": "transition_applied",
        "tier": "verified_automatic",
        "readiness": "transition_eligible",
        "regulator_issuer_cik": "0001409970",
        "synthetic_post_apply_projection": True,
        "transition": {
            "kind": "symbol_continuation",
            "source_ticker": "OTC-A",
            "successor_ticker": "NEW",
            "rule_id": "lifecycle.simple_symbol_continuation",
            "outcomes": ["symbol_changed", "venue_transfer"],
        },
        "listings": [
            {
                "authority": "massive",
                "listing_status": "active",
                "market": "otc",
                "primary_exchange": "OTC",
                "candidate_ticker": "NEW",
                "issuer_cik": "0001409970",
            }
        ],
    },
    "settings-massive-key": {"settings": True},
}
LABELS = {
    "en": {
        "universe": "Universe",
        "settings": "Settings",
        "open_nav": "Open navigation",
        "lifecycle": "Security event investigation",
        "listing_family": "Listing authority",
        "acknowledge": "Acknowledge",
        "reverse_transition": "Reverse transition",
        "reverse_activity": "Reverse tracking change",
        "translate_evidence": "Translate evidence",
        "statuses": {
            "active": "Active",
            "inactive": "Inactive",
            "not_found": "Not found in this completed snapshot",
        },
    },
    "zh-Hant": {
        "universe": "全部標的",
        "settings": "設定",
        "open_nav": "開啟導覽",
        "lifecycle": "標的事件調查",
        "listing_family": "上市主管機關",
        "acknowledge": "知道了",
        "reverse_transition": "反轉代號轉移",
        "reverse_activity": "還原追蹤變更",
        "translate_evidence": "翻譯證據",
        "statuses": {
            "active": "有效",
            "inactive": "非有效",
            "not_found": "在這份完整快照中找不到",
        },
    },
}


def _assert_evidence_surface_counts(
    *,
    expected_listing_count: int,
    regulator_evidence_count: int,
    expanded_regulator_evidence_count: int,
    regulator_translation_button_count: int,
    listing_evidence_count: int,
    expanded_listing_evidence_count: int,
    listing_translation_button_count: int,
) -> None:
    expected = {
        "regulator_evidence_count": 1,
        "expanded_regulator_evidence_count": 1,
        "regulator_translation_button_count": 1,
        "listing_evidence_count": expected_listing_count,
        "expanded_listing_evidence_count": expected_listing_count,
        "listing_translation_button_count": 0,
    }
    actual = {
        "regulator_evidence_count": regulator_evidence_count,
        "expanded_regulator_evidence_count": expanded_regulator_evidence_count,
        "regulator_translation_button_count": regulator_translation_button_count,
        "listing_evidence_count": listing_evidence_count,
        "expanded_listing_evidence_count": expanded_listing_evidence_count,
        "listing_translation_button_count": listing_translation_button_count,
    }
    if actual != expected:
        raise AssertionError(
            "browser_evidence_surface_mismatch:"
            + json.dumps(actual, sort_keys=True, separators=(",", ":"))
        )


def _assert_post_apply_surface_counts(
    *,
    acknowledgement_count: int,
    reverse_transition_count: int,
    reverse_activity_count: int,
) -> None:
    actual = {
        "acknowledgement_count": acknowledgement_count,
        "reverse_transition_count": reverse_transition_count,
        "reverse_activity_count": reverse_activity_count,
    }
    expected = {key: 1 for key in actual}
    if actual != expected:
        raise AssertionError(
            "browser_post_apply_surface_mismatch:"
            + json.dumps(actual, sort_keys=True, separators=(",", ":"))
        )


def _summary(name: str) -> dict:
    scenario = SCENARIOS[name]
    summary = deepcopy(BASE._summary("settled-history"))
    summary.update(
        {
            "case_id": scenario["case_id"],
            "source_ref": f"fixture-{name}",
            "ticker": scenario["ticker"],
            "issuer_name": scenario["issuer"],
            "workflow_state": "resolved" if scenario["queue"] == "history" else "unresolved",
            "kinds": [{"event_type": "listing_status_review", "effective_date": "2026-08-28"}],
            "current_assessment": None,
            "current_acknowledgement": None,
            "investigation_run_count": 0,
            "automation_run_count": 1,
            "automation_fact_count": len(scenario["listings"]),
            "automation_tier": scenario["tier"],
            "action_readiness": scenario["readiness"],
            "disposition": scenario["disposition"],
            "queue_bucket": scenario["queue"],
            "disposition_reason": scenario["reason"],
            "last_checked_at": "2026-08-28T08:00:00Z",
            "next_check_at": (
                "2026-09-04T08:00:00Z" if scenario["queue"] == "monitoring" else None
            ),
            "source_family_status": {
                "regulator": "confirmed",
                "listing_authority": (
                    "conflict" if name == "conflict-attention" else "confirmed"
                ),
                "manual": "missing",
            },
            "evidence_count": len(scenario["listings"]) + 1,
            "assessment_count": 0,
            "acknowledgement_count": 0,
            "proposal_count": 0,
        }
    )
    return summary


def _listing_evidence(name: str) -> list[dict]:
    scenario = SCENARIOS[name]
    rows = []
    for index, listing in enumerate(scenario["listings"], 1):
        authority = listing["authority"]
        rows.append(
            {
                "evidence_id": f"listing-{name}-{index}",
                "source_family": "listing_authority",
                "kind": "listing_directory_snapshot",
                "source_url": (
                    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
                    if authority == "nasdaq_trader"
                    else "https://api.massive.com/v3/reference/tickers"
                ),
                "created_at": "2026-08-28T08:00:00Z",
                "listing": {
                    "authority": authority,
                    "directory": "nasdaq_listed" if authority == "nasdaq_trader" else None,
                    "candidate_ticker": listing["candidate_ticker"],
                    "listing_status": listing["listing_status"],
                    "market": listing["market"],
                    "primary_exchange": listing["primary_exchange"],
                    "source_as_of": "2026-08-28",
                },
            }
        )
    return rows


def _preview_digest(preview: dict) -> str:
    payload = {key: value for key, value in preview.items() if key != "preview_sha256"}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _terminal_activity_changes(preview: dict) -> list[dict]:
    effects = preview["effects"]
    counts = {
        "source_hidden": int(bool(effects["suppression"]["hide_source"])),
        "watchlist_membership_archived": len(effects["watchlists"]["archive"]),
    }
    return [
        {"change_type": change_type, "count": count}
        for change_type, count in sorted(counts.items())
        if count
    ]


def _assert_terminal_product_invariants(scenario: dict, transition: dict) -> None:
    source_ticker = scenario["transition"]["source_ticker"]
    preview = transition["approved_preview"]
    activity = transition["activity_history"][0]
    expected_changes = [
        {"change_type": "source_hidden", "count": 1},
        {"change_type": "watchlist_membership_archived", "count": 1},
    ]
    assert [row["candidate_ticker"] for row in scenario["listings"]] == [
        source_ticker
    ]
    assert transition["source_ticker"] == source_ticker
    assert transition["successor_ticker"] is None
    assert preview["source_ticker"] == source_ticker
    assert preview["successor_ticker"] is None
    assert "portfolio_open" not in preview["active_sources"]
    assert "portfolio_open" not in preview["provider_owned_sources"]
    assert "portfolio_position_retained" not in preview["caveats"]
    assert preview["effects"]["watchlists"] == {
        "add": [],
        "archive": [
            {
                "list_id": 1,
                "list_name": "Core",
                "position": 3,
                "ticker": source_ticker,
            }
        ],
        "reactivate": [],
        "unchanged": [],
    }
    assert preview["effects"]["suppression"] == {
        "hide_source": True,
        "source_hidden": False,
        "successor_hidden": False,
        "unhide_successor": False,
    }
    assert activity["source_ticker"] == source_ticker
    assert activity["successor_ticker"] is None
    assert activity["user_owned_changes"] == expected_changes


def _applied_transition(name: str) -> dict:
    scenario = SCENARIOS[name]
    projection = scenario["transition"]
    transition = deepcopy(STAGE5._detail()["ticker_transition"])
    transition_id = f"transition-{name}"
    preview = deepcopy(transition["approved_preview"])
    preview.update(
        {
            "case_id": scenario["case_id"],
            "execute_on": "2026-08-28",
            "outcomes": projection["outcomes"],
            "source_ticker": projection["source_ticker"],
            "successor_ticker": projection["successor_ticker"],
            "transition_kind": projection["kind"],
        }
    )
    if projection["kind"] == "terminal_delisting":
        preview.update(
            {
                "active_sources": ["manual_lists"],
                "provider_owned_sources": [],
                "caveats": [],
            }
        )
        preview["effects"]["watchlists"] = {
            "add": [],
            "archive": [
                {
                    "list_id": 1,
                    "list_name": "Core",
                    "position": 3,
                    "ticker": projection["source_ticker"],
                }
            ],
            "reactivate": [],
            "unchanged": [],
        }
        preview["effects"]["suppression"]["hide_source"] = True
    preview["preview_sha256"] = _preview_digest(preview)
    activity = deepcopy(transition["activity_history"][0])
    activity.update(
        {
            "activity_id": f"activity-{name}",
            "transition_id": transition_id,
            "case_id": scenario["case_id"],
            "activity_type": "applied",
            "source_ticker": projection["source_ticker"],
            "successor_ticker": projection["successor_ticker"],
            "effective_date": "2026-08-28",
            "rule_id": projection["rule_id"],
            "occurred_at": "2026-08-28T08:00:00Z",
            "created_at": "2026-08-28T08:00:00Z",
            "acknowledged_at": None,
            "reverse_readiness": {"reversible": True, "block_reasons": []},
        }
    )
    if projection["kind"] == "terminal_delisting":
        activity["provider_owned_retained"] = []
        activity["user_owned_changes"] = _terminal_activity_changes(preview)
    transition.update(
        {
            "transition_id": transition_id,
            "case_id": scenario["case_id"],
            "kind": projection["kind"],
            "status": "applied",
            "source_ticker": projection["source_ticker"],
            "successor_ticker": projection["successor_ticker"],
            "execute_on": "2026-08-28",
            "approved_preview_sha256": preview["preview_sha256"],
            "approved_preview": preview,
            "rule_id": projection["rule_id"],
            "rule_version": "1",
            "updated_at": "2026-08-28T08:00:00Z",
            "latest_attempt": {
                "status": "applied",
                "block_reasons": [],
                "attempted_at": "2026-08-28T08:00:00Z",
            },
            "reverse_readiness": {"reversible": True, "block_reasons": []},
            "activity_history": [activity],
            "activity_count": 1,
            "unacknowledged_activity_count": 1,
        }
    )
    if projection["kind"] == "terminal_delisting":
        _assert_terminal_product_invariants(scenario, transition)
    return transition


def _detail(name: str) -> dict:
    scenario = SCENARIOS[name]
    detail = deepcopy(BASE._detail("settled-history"))
    detail.update(_summary(name))
    detail["observation"].update(
        {
            "ticker": scenario["ticker"],
            "cik": scenario["regulator_issuer_cik"],
            "issuer_name": scenario["issuer"],
            "filing_date": "2026-08-28",
            "source_ref": f"fixture-{name}",
            "description": f"Offline listing-authority fixture for {scenario['ticker']}.",
            "kinds": detail["kinds"],
        }
    )
    regulator = next(
        item for item in detail["evidence"] if item["source_family"] == "regulator"
    )
    regulator_excerpt = f"Offline SEC helper fixture for {scenario['ticker']}."
    regulator.update(
        {
            "title": f"Regulatory fixture: {scenario['ticker']}",
            "excerpt": regulator_excerpt,
            "content_sha256": hashlib.sha256(regulator_excerpt.encode()).hexdigest(),
            "source_published_at": "2026-08-28T08:00:00Z",
        }
    )
    regulator["translations"] = []
    detail["evidence"] = [regulator, *_listing_evidence(name)]
    detail["investigation_runs"] = []
    detail["automation_runs"] = []
    detail["automation_facts"] = []
    detail["proposals"] = []
    detail["ticker_transition"] = (
        _applied_transition(name)
        if scenario.get("synthetic_post_apply_projection") is True
        else None
    )
    detail["assessment_history"] = []
    detail["acknowledgement_history"] = []
    detail["current_assessment"] = None
    if scenario.get("synthetic_post_apply_projection") is True:
        transition = detail["ticker_transition"]
        assert transition is not None and transition["status"] == "applied"
        assert transition["activity_history"]
        assert transition["reverse_readiness"] == {
            "reversible": True,
            "block_reasons": [],
        }
    return detail


def _case_list(query: dict[str, list[str]]) -> list[dict]:
    queue = query.get("queue_bucket", ["attention"])[0]
    return [
        _summary(name)
        for name, scenario in SCENARIOS.items()
        if not scenario.get("settings") and scenario["queue"] == queue
    ]


def _provider_config() -> dict:
    return {
        "providers": {
            "polygon": {
                "fields": [
                    {
                        "field": "api_key",
                        "label": "API key",
                        "secret": True,
                        "env_var": "MASSIVE_API_KEY",
                        "app_value_set": True,
                        "app_value_masked": "••••fixture",
                        "effective_source": "app",
                        "needs_import": False,
                        "import_source": None,
                        "importable_env_vars": ["MASSIVE_API_KEY", "POLYGON_API_KEY"],
                        "defaulted": False,
                        "guarded": False,
                        "guard_reason": None,
                    }
                ],
                "testable": True,
                "default_available": False,
            }
        },
        "setup": {"required": False, "code": None, "reason": None},
        "env_fallback": {"enabled": False, "source": "default"},
    }


def _api_response(route, path: str, query: str, locale: str) -> None:
    if path == "/status":
        payload = {
            "status": "ok",
            "timestamp": "2026-08-28T08:00:00Z",
            "tools_registered": 50,
            "tool_categories": {},
            "data_sources": {},
        }
    elif path == "/config/runtime":
        payload = STAGE5._runtime_config()
    elif path == "/profile/settings/ui-locale":
        payload = {"locale": locale, "source": "stored"}
    elif path == "/profile/universe":
        payload = {
            "as_of": "2026-08-28",
            "generated_at": "2026-08-28T08:00:00Z",
            "total": 0,
            "shown": 0,
            "archived_count": 0,
            "summarized": 0,
            "rows": [],
        }
    elif path == "/profile/lists":
        payload = {"lists": []}
    elif path == "/analysis/cards":
        payload = {"cards": []}
    elif path == "/research/threads":
        payload = {"threads": []}
    elif path == "/security-lifecycle/cases":
        cases = _case_list(parse_qs(query))
        payload = {
            "cases": cases,
            "count": len(cases),
            "queue_counts": {"attention": 1, "monitoring": 1, "history": 3},
            "data_integrity": {"source_missing_count": 0},
        }
    elif path.startswith("/security-lifecycle/cases/"):
        case_id = path.rsplit("/", 1)[-1]
        name = next(
            key for key, value in SCENARIOS.items() if value.get("case_id") == case_id
        )
        payload = _detail(name)
    elif path == "/security-lifecycle/transition-activity":
        payload = {"items": [], "count": 0, "unacknowledged_count": 0}
    elif path == "/providers/config":
        payload = _provider_config()
    elif path == "/providers/health":
        payload = {
            "generated_at": "2026-08-28T08:00:00Z",
            "providers": [],
            "jobs": {},
            "local_market": {"db_exists": False, "sync": {}},
            "notes": [],
        }
    elif path == "/schedule":
        payload = {"sources": {}}
    elif path == "/sa/extension-health":
        payload = {
            "chain_state": "interrupted",
            "generated_at": "2026-08-28T08:00:00Z",
            "segments": [],
        }
    elif path == "/config/model-catalog":
        payload = {
            "providers": ["anthropic", "openai"],
            "tasks": [],
            "models": [],
            "effort_options": {"anthropic": [], "openai": []},
            "routes": {},
            "credentials": {"anthropic": [], "openai": []},
            "custom_allowed": True,
        }
    elif path == "/market-data/status":
        payload = {
            "market_db": "fixture",
            "exists": False,
            "prices": {"row_count": 0, "ticker_count": 0, "latest_datetime": None},
            "news": {"row_count": 0, "source_count": 0, "latest_published": None},
            "fundamentals": {"row_count": 0, "ticker_count": 0, "latest_date": None},
            "financial_cache": {
                "row_count": 0,
                "valid_count": 0,
                "expired_count": 0,
                "latest_fetched_at": None,
            },
            "sync": {"prices": None, "news": None, "fundamentals": None},
            "prices_authority": "local",
            "fundamentals_mode": "local_cache_refetch",
            "use_local_market_setting": False,
            "env_override": False,
            "local_market_strict_setting": False,
            "strict_env_override": False,
            "strict_enabled": False,
            "routing_enabled": False,
        }
    elif path == "/market-data/trading-days":
        payload = {
            "version": 2,
            "market_scope": "us_listed_equity_proxy",
            "coverage_session": "rth",
            "interval": "15min",
            "lookback_days": 10,
            "universe_count": 0,
            "generated_at_et": "2026-08-28T04:00:00-04:00",
            "calendar_health": {
                "status": "unavailable",
                "reason_codes": [],
                "reviewed_through": "2026-08-28",
                "forward_horizon_months": 0,
            },
            "observation_health": {"status": "unavailable", "reason_code": None},
            "days": [],
            "provider_errors": [],
        }
    elif path == "/news/status":
        payload = {
            "market_db": "fixture",
            "exists": False,
            "news": {"row_count": 0, "source_count": 0, "latest_published": None},
            "use_local_news_setting": False,
            "setting_explicit": False,
            "env_override": False,
            "env_value": None,
            "direct_active": False,
            "normalized_writes_setting": False,
            "normalized_writes_setting_explicit": False,
            "normalized_writes_env_override": False,
            "normalized_writes_env_value": None,
            "write_route": "blocked",
            "write_route_reason": "fixture_only",
            "sync": None,
        }
    elif path == "/macro/status":
        payload = {
            "macro_db": "fixture",
            "exists": False,
            "tables": {},
            "use_local_macro_setting": False,
            "env_override": False,
            "local_first_active": False,
        }
    elif path == "/macro/snapshot":
        payload = {
            "available": False,
            "macro_db": "fixture",
            "series_count": 0,
            "observation_count": 0,
            "release_dates_count": 0,
            "latest_fetched_at": None,
            "items": [],
            "missing_series": [],
        }
    else:
        STAGE5._response(route, {"detail": {"code": "fixture_unavailable"}}, 503)
        return
    STAGE5._response(route, payload)


def _navigate(page, scenario_name: str, locale: str) -> tuple[str, str, dict[str, int]]:
    scenario = SCENARIOS[scenario_name]
    labels = LABELS[locale]
    if scenario.get("settings"):
        target = page.get_by_role("button", name=labels["settings"], exact=True)
        if not target.is_visible():
            page.get_by_role("button", name=labels["open_nav"], exact=True).click()
        target.click()
        massive = page.get_by_text("Massive", exact=True)
        try:
            massive.wait_for(state="visible")
        except Exception:
            print(page.locator("body").inner_text()[:6000])
            raise
        massive.scroll_into_view_if_needed()
        row = massive.locator("xpath=ancestor::*[self::tr or @data-testid][1]")
        if row.count() == 0:
            row = massive.locator("xpath=ancestor::tr")
        assert page.locator('input[type="password"]').count() == 1
        return page.locator("body").inner_text(), "settings", {
            "regulator_evidence_count": 0,
            "expanded_regulator_evidence_count": 0,
            "regulator_translation_button_count": 0,
            "listing_evidence_count": 0,
            "expanded_listing_evidence_count": 0,
            "listing_translation_button_count": 0,
            "acknowledgement_count": 0,
            "reverse_transition_count": 0,
            "reverse_activity_count": 0,
        }

    universe = page.get_by_role("button", name=labels["universe"], exact=True)
    if not universe.is_visible():
        page.get_by_role("button", name=labels["open_nav"], exact=True).click()
    universe.click()
    page.get_by_role("tab", name=labels["lifecycle"], exact=True).click()
    if scenario["queue"] != "attention":
        page.locator(f"[data-queue-view='{scenario['queue']}']").click()
    trigger = page.get_by_role("button", name=re.compile(rf"^{re.escape(scenario['ticker'])}\b"))
    trigger.wait_for(state="visible")
    trigger.click()
    drawer = page.locator(".ui-drawer")
    drawer.wait_for(state="visible")
    text = drawer.inner_text()
    assert labels["listing_family"] in text
    for listing in scenario["listings"]:
        assert labels["statuses"][listing["listing_status"]] in text
    if scenario_name == "otc-continuation":
        assert "OTC" in text

    evidence_items = drawer.locator("details.lifecycle-evidence-item")
    regulator_items = evidence_items.filter(
        has_text=f"Regulatory fixture: {scenario['ticker']}"
    )
    regulator_evidence_count = regulator_items.count()
    expanded_regulator_evidence_count = 0
    regulator_translation_button_count = 0
    if regulator_evidence_count == 1:
        regulator = regulator_items.first
        regulator.locator("summary").click()
        expanded_regulator_evidence_count = int(
            regulator.get_attribute("open") is not None
        )
        translate = regulator.get_by_role(
            "button", name=labels["translate_evidence"], exact=True
        )
        regulator_translation_button_count = int(
            translate.count() == 1 and translate.is_visible()
        )

    listing_items = drawer.locator(
        "details.lifecycle-evidence-item:has(.lifecycle-assessment-facts)"
    )
    listing_evidence_count = listing_items.count()
    expanded_listing_evidence_count = 0
    listing_translation_button_count = 0
    for index in range(listing_evidence_count):
        listing_item = listing_items.nth(index)
        listing_item.locator("summary").click()
        expanded_listing_evidence_count += int(
            listing_item.get_attribute("open") is not None
        )
        listing_translation_button_count += listing_item.get_by_role(
            "button", name=labels["translate_evidence"], exact=True
        ).count()
    _assert_evidence_surface_counts(
        expected_listing_count=len(scenario["listings"]),
        regulator_evidence_count=regulator_evidence_count,
        expanded_regulator_evidence_count=expanded_regulator_evidence_count,
        regulator_translation_button_count=regulator_translation_button_count,
        listing_evidence_count=listing_evidence_count,
        expanded_listing_evidence_count=expanded_listing_evidence_count,
        listing_translation_button_count=listing_translation_button_count,
    )

    acknowledgement_count = 0
    reverse_transition_count = 0
    reverse_activity_count = 0
    if scenario.get("synthetic_post_apply_projection") is True:
        assert labels["reverse_transition"] in text
        assert labels["reverse_activity"] in text
        acknowledgement = drawer.get_by_role(
            "button", name=labels["acknowledge"], exact=True
        )
        reverse_transition = drawer.get_by_role(
            "button", name=labels["reverse_transition"], exact=True
        )
        reverse_activity = drawer.get_by_role(
            "button", name=labels["reverse_activity"], exact=True
        )
        acknowledgement_count = int(
            acknowledgement.count() == 1 and acknowledgement.is_visible()
        )
        reverse_transition_count = int(
            reverse_transition.count() == 1 and reverse_transition.is_visible()
        )
        reverse_activity_count = int(
            reverse_activity.count() == 1 and reverse_activity.is_visible()
        )
        _assert_post_apply_surface_counts(
            acknowledgement_count=acknowledgement_count,
            reverse_transition_count=reverse_transition_count,
            reverse_activity_count=reverse_activity_count,
        )
        reverse_activity.scroll_into_view_if_needed()
    return drawer.inner_text(), "lifecycle", {
        "regulator_evidence_count": regulator_evidence_count,
        "expanded_regulator_evidence_count": expanded_regulator_evidence_count,
        "regulator_translation_button_count": regulator_translation_button_count,
        "listing_evidence_count": listing_evidence_count,
        "expanded_listing_evidence_count": expanded_listing_evidence_count,
        "listing_translation_button_count": listing_translation_button_count,
        "acknowledgement_count": acknowledgement_count,
        "reverse_transition_count": reverse_transition_count,
        "reverse_activity_count": reverse_activity_count,
    }


def _geometry(page) -> dict:
    metrics = STAGE5._geometry(page)
    controls = page.evaluate(
        """() => {
          const clips = (value) => ['auto', 'clip', 'hidden', 'scroll'].includes(value);
          const visibleRect = (node) => {
            const style = getComputedStyle(node);
            const original = node.getBoundingClientRect();
            if (style.visibility === 'hidden' || style.display === 'none'
              || original.width <= 0 || original.height <= 0) return null;
            const rect = {
              left: Math.max(0, original.left),
              right: Math.min(innerWidth, original.right),
              top: Math.max(0, original.top),
              bottom: Math.min(innerHeight, original.bottom),
            };
            for (let parent = node.parentElement; parent; parent = parent.parentElement) {
              const parentStyle = getComputedStyle(parent);
              const parentRect = parent.getBoundingClientRect();
              if (clips(parentStyle.overflowX)) {
                rect.left = Math.max(rect.left, parentRect.left);
                rect.right = Math.min(rect.right, parentRect.right);
              }
              if (clips(parentStyle.overflowY)) {
                rect.top = Math.max(rect.top, parentRect.top);
                rect.bottom = Math.min(rect.bottom, parentRect.bottom);
              }
            }
            if (rect.right <= rect.left || rect.bottom <= rect.top) return null;
            const hit = document.elementFromPoint(
              (rect.left + rect.right) / 2,
              (rect.top + rect.bottom) / 2,
            );
            if (!hit || (!node.contains(hit) && !hit.contains(node))) return null;
            return rect;
          };
          return [...document.querySelectorAll(
            '.lifecycle-activity-band button, .ui-drawer button, .ui-drawer input, '
            + '.ui-drawer select, .ui-drawer textarea, .ui-drawer a'
          )].flatMap((node) => {
            const rect = visibleRect(node);
            if (!rect) return [];
            return [{
              tag: node.tagName,
              text: (node.textContent || node.getAttribute('aria-label') || '').trim(),
              ...rect,
            }];
          });
        }"""
    )
    overlaps = []
    for index, left in enumerate(controls):
        for right in controls[index + 1 :]:
            width = min(left["right"], right["right"]) - max(left["left"], right["left"])
            height = min(left["bottom"], right["bottom"]) - max(left["top"], right["top"])
            if width > 1 and height > 1:
                overlaps.append([left, right])
    metrics["controls"] = controls
    metrics["overlaps"] = overlaps
    return metrics


def _run_entry(browser, scenario_name: str, locale: str, width: int, height: int) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = SCENARIOS[scenario_name]
    state = {"requests": [], "external": []}
    context = browser.new_context(viewport={"width": width, "height": height})
    context.add_init_script(
        "\n".join(
            [
                f"localStorage.setItem('arkscope.ui.locale.v1', {json.dumps(locale)});",
                "localStorage.setItem('arkscope.settings.activeGroup.v1', 'data_sync');",
            ]
        )
    )
    page = context.new_page()
    page.set_default_timeout(12_000)
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
        elif request.method != "GET":
            STAGE5._response(route, {"detail": {"code": "fixture_write_forbidden"}}, 405)
        else:
            _api_response(route, parsed.path, parsed.query, locale)

    page.route("**/*", handler)
    page.goto(APP_URL, wait_until="networkidle", timeout=20_000)
    try:
        visible_text, surface, surface_counts = _navigate(
            page, scenario_name, locale
        )
    except Exception:
        print(json.dumps({"console_errors": console_errors, "page_errors": page_errors}))
        raise
    page.wait_for_timeout(250)
    body_text = page.locator("body").inner_text()
    forbidden_publisher_labels = ("Publisher reporting", "新聞出版來源")
    assert not any(label in body_text for label in forbidden_publisher_labels)
    if scenario.get("settings"):
        assert page.get_by_role("button", name=re.compile(r"Translate|翻譯")).count() == 0
    assert page.locator("[data-action='open-content-translation-settings']").count() == 0

    metrics = _geometry(page)
    STAGE5._assert_geometry(metrics)
    screenshot = OUTPUT / f"{width}x{height}-{locale}-{scenario_name}.png"
    page.screenshot(path=str(screenshot), full_page=False)
    pixels = STAGE5._pixel_check(screenshot, width, height)
    writes = [
        item for item in state["requests"]
        if item["method"] in {"POST", "PUT", "PATCH", "DELETE"}
    ]
    command_calls = [
        item for item in state["requests"]
        if item["path"].endswith(("/run", "/accept", "/execute", "/reverse", "/acknowledge"))
    ]
    render_acknowledgements = [
        item for item in state["requests"] if item["path"].endswith("/acknowledge")
    ]
    assert state["external"] == [], state["external"]
    assert writes == [], writes
    assert command_calls == [], command_calls
    assert render_acknowledgements == [], render_acknowledgements
    assert console_errors == [], console_errors
    assert page_errors == [], page_errors
    synthetic_projection = scenario.get("synthetic_post_apply_projection") is True
    transition_surface_witnesses = (
        [
            label
            for label in (
                LABELS[locale]["reverse_transition"],
                LABELS[locale]["reverse_activity"],
            )
            if label in visible_text
        ]
        if synthetic_projection
        else []
    )
    if synthetic_projection:
        assert len(transition_surface_witnesses) == 2
    acknowledgement_surface_witnesses = (
        [LABELS[locale]["acknowledge"]]
        if surface_counts["acknowledgement_count"] == 1
        else []
    )
    if synthetic_projection:
        assert len(acknowledgement_surface_witnesses) == 1
    result = {
        "scenario": scenario_name,
        "surface": surface,
        "locale": locale,
        "viewport": [width, height],
        "screenshot": screenshot.name,
        "pixels": pixels,
        "visible_text_sha256": hashlib.sha256(visible_text.encode()).hexdigest(),
        "request_count": len(state["requests"]),
        "external_requests": state["external"],
        "writes": writes,
        "command_calls": command_calls,
        "render_acknowledgements": render_acknowledgements,
        "console_errors": console_errors,
        "page_errors": page_errors,
        "publisher_family_text_count": sum(body_text.count(label) for label in forbidden_publisher_labels),
        "regulator_evidence_count": surface_counts["regulator_evidence_count"],
        "expanded_regulator_evidence_count": surface_counts[
            "expanded_regulator_evidence_count"
        ],
        "regulator_translation_button_count": surface_counts[
            "regulator_translation_button_count"
        ],
        "expected_listing_evidence_count": len(scenario.get("listings", [])),
        "listing_evidence_count": surface_counts["listing_evidence_count"],
        "expanded_listing_evidence_count": surface_counts[
            "expanded_listing_evidence_count"
        ],
        "listing_translation_button_count": surface_counts[
            "listing_translation_button_count"
        ],
        "overlap_count": len(metrics["overlaps"]),
        "clipped_text_count": len(metrics["textOverflow"]),
        "synthetic_post_apply_projection": synthetic_projection,
        "produced_by_shadow_execution": False,
        "fixture_provenance": (
            "synthetic_post_apply_ui_projection_not_produced_by_shadow"
            if synthetic_projection
            else "local_browser_fixture"
        ),
        "transition_surface_witnesses": transition_surface_witnesses,
        "acknowledgement_surface_witnesses": acknowledgement_surface_witnesses,
        "fixture_cik_shape": (
            None
            if scenario.get("settings")
            else {
                "regulator_issuer_cik": scenario["regulator_issuer_cik"],
                "listing_issuer_ciks": [
                    listing["issuer_cik"] for listing in scenario["listings"]
                ],
            }
        ),
    }
    context.close()
    return result


def main() -> int:
    if OUTPUT.exists():
        for path in OUTPUT.glob("*.png"):
            path.unlink()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        entries = [
            _run_entry(browser, name, locale, width, height)
            for width, height in VIEWPORTS
            for locale in LOCALES
            for name in SCENARIOS
        ]
        browser.close()
    payload = {
        "schema_version": 3,
        "app_url": APP_URL,
        "fixture_only": True,
        "provider_calls": dict(DECLARED_ZERO),
        "production_backend_starts": dict(DECLARED_ZERO),
        "production_database_operations": dict(DECLARED_ZERO),
        "merges": dict(DECLARED_ZERO),
        "pushes": dict(DECLARED_ZERO),
        "fixture_metadata": {
            "post_apply_scenarios": ["inactive-history", "otc-continuation"],
            "post_apply_basis": "synthetic_ui_projection",
            "post_apply_produced_by_shadow_execution": False,
            "conflict_basis": "active_nasdaq_and_active_massive_with_sec_listing_cik_disagreement",
        },
        "entries": entries,
    }
    (OUTPUT / "matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    totals = {
        "entries": len(entries),
        "screenshots": len(entries),
        "external_requests": sum(len(item["external_requests"]) for item in entries),
        "writes": sum(len(item["writes"]) for item in entries),
        "command_calls": sum(len(item["command_calls"]) for item in entries),
        "render_acknowledgements": sum(len(item["render_acknowledgements"]) for item in entries),
        "console_errors": sum(len(item["console_errors"]) for item in entries),
        "page_errors": sum(len(item["page_errors"]) for item in entries),
        "publisher_family_text_count": sum(item["publisher_family_text_count"] for item in entries),
        "listing_translation_button_count": sum(item["listing_translation_button_count"] for item in entries),
        "regulator_translation_button_count": sum(
            item["regulator_translation_button_count"] for item in entries
        ),
        "listing_evidence_count": sum(item["listing_evidence_count"] for item in entries),
        "expanded_listing_evidence_count": sum(
            item["expanded_listing_evidence_count"] for item in entries
        ),
        "overlap_count": sum(item["overlap_count"] for item in entries),
        "clipped_text_count": sum(item["clipped_text_count"] for item in entries),
        "synthetic_post_apply_projection_entries": sum(
            item["synthetic_post_apply_projection"] for item in entries
        ),
        "transition_surface_witnesses": sum(
            len(item["transition_surface_witnesses"]) for item in entries
        ),
        "acknowledgement_surface_witnesses": sum(
            len(item["acknowledgement_surface_witnesses"]) for item in entries
        ),
    }
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
