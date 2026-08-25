"""Run the repaired Stage 5 fixture-only bilingual browser matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse

from PIL import Image, ImageStat
from playwright.sync_api import sync_playwright


APP_URL = os.environ.get("ARKSCOPE_STAGE5_APP_URL", "http://127.0.0.1:4197/")
OUTPUT = Path(__file__).resolve().parent / "browser"
OUTPUT.mkdir(parents=True, exist_ok=True)


LABELS = {
    "en": {
        "universe": "Universe",
        "open_nav": "Open navigation",
        "lifecycle": "Security event investigation",
        "activity": "Automatic tracking changes",
        "reverse": "Reverse tracking change",
        "acknowledge": "Acknowledge",
        "case": "HAPN",
        "close": "Close",
        "original": "Original source evidence",
        "translation": "Machine translation",
        "evidence_section": "Evidence and searches",
        "automation": "Automation decision",
        "accepted": "Accepted by automation policy",
        "suggested": "Review suggested",
        "blocked": "Action blocked",
        "revalidation": "Waiting to revalidate tracking transition",
        "approval_changed": "Transition approval inputs changed; revalidation is scheduled",
        "conflict": "Conflicting source facts",
        "market_missing": "Market confirmation is missing",
        "save_revision": "Save as human revision",
        "accept_suggestion": "Accept unchanged suggestion",
        "families": ("Regulatory filing", "Market infrastructure", "Publisher reporting"),
        "facts": "Extracted facts",
        "transition": "Ticker tracking transition",
        "transaction_summary": "Asset acquisition · Terms not extracted",
    },
    "zh-Hant": {
        "universe": "全部標的",
        "open_nav": "開啟導覽",
        "lifecycle": "標的事件調查",
        "activity": "自動追蹤變更",
        "reverse": "還原追蹤變更",
        "acknowledge": "知道了",
        "case": "HAPN",
        "close": "關閉",
        "original": "來源原文證據",
        "translation": "機器翻譯",
        "evidence_section": "證據與搜尋",
        "automation": "自動化判定",
        "accepted": "由自動化政策接受",
        "suggested": "建議複查",
        "blocked": "動作受阻",
        "revalidation": "等待重新驗證追蹤轉移",
        "approval_changed": "追蹤轉移核准輸入已變更；已排程重新驗證",
        "conflict": "來源事實互相衝突",
        "market_missing": "缺少市場確認",
        "save_revision": "另存為人工修訂",
        "accept_suggestion": "接受未修改的建議",
        "families": ("監管申報", "市場基礎設施", "新聞出版來源"),
        "facts": "已擷取事實",
        "transition": "標的追蹤轉移",
        "transaction_summary": "資產收購 · 尚未抽取交易條件",
    },
}


def _assessment(*, draft: bool) -> dict:
    return {
        "assessment_id": "assessment-suggestion" if draft else "assessment-accepted",
        "status": "draft" if draft else "accepted",
        "author": "automation",
        "automation_method": "model_assisted" if draft else "deterministic_rule",
        "acceptance_authority": None if draft else "automation_policy",
        "automation_run_id": "run-review" if draft else "run-accepted",
        "rule_id": "lifecycle.m-and-a-review" if draft else "lifecycle.simple_symbol_continuation",
        "rule_version": "2" if draft else "1",
        "decision_provenance_sha256": ("d" if draft else "e") * 64,
        "relevance": "direct_tracked_security",
        "confidence": "medium" if draft else "high",
        "conclusion": (
            "The transaction terms require review before changing tracking."
            if draft
            else "The tracked security continued from LC to HAPN."
        ),
        "impact_summary": (
            "Confirm the successor security, venue, effective date, and consideration terms."
            if draft
            else "The historical LC tracking identity continued as HAPN on Nasdaq."
        ),
        "outcomes": ["acquisition_mixed"] if draft else ["symbol_changed", "venue_transfer"],
        "citations": [
            {
                "reference_kind": "observation",
                "evidence_id": None,
                "cited_content_sha256": "f" * 64,
            },
            {
                "reference_kind": "evidence",
                "evidence_id": "evidence-regulator",
                "cited_content_sha256": "a" * 64,
            },
        ],
        "stale": False,
        "created_at": "2026-08-25T10:00:00Z" if draft else "2026-08-25T09:00:00Z",
        "counterparty_name": "Acquirer Corp." if draft else None,
        "counterparty_ticker": "ACQ" if draft else None,
        "counterparty_cik": "0000123456" if draft else None,
        "successor_ticker": "HAPN",
        "destination_venue": "NASDAQ",
        "effective_date": "2026-09-30" if draft else "2026-06-27",
        "consideration_currency": "USD" if draft else None,
        "cash_per_security_decimal": "10.50" if draft else None,
        "exchange_ratio_decimal": "0.25" if draft else None,
    }


ACCEPTED = _assessment(draft=False)
DRAFT = _assessment(draft=True)


def _summary() -> dict:
    return {
        "case_id": "case-hapn",
        "source": "sec_edgar",
        "source_ref": "0001409970-26-000131",
        "ticker": "HAPN",
        "source_presence": "present",
        "workflow_state": "resolved",
        "issuer_name": "Happify Network, Inc.",
        "filing_date": "2026-06-27",
        "kinds": [{"event_type": "listing_status_review", "effective_date": "2026-06-27"}],
        "current_assessment": ACCEPTED,
        "current_acknowledgement": None,
        "active_sources": ["manual_lists", "portfolio_open"],
        "source_context": "available",
        "components": {},
        "investigation_run_count": 0,
        "automation_run_count": 3,
        "automation_fact_count": 5,
        "automation_tier": "verified_automatic",
        "action_readiness": "waiting_transition_revalidation",
        "evidence_count": 3,
        "assessment_count": 2,
        "acknowledgement_count": 0,
        "proposal_count": 2,
    }


def _translation(evidence_id: str, locale: str, text: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "evidence_content_sha256": {
            "evidence-regulator": "a",
            "evidence-market": "b",
            "evidence-publisher": "c",
        }[evidence_id] * 64,
        "locale": locale,
        "translated_text": text,
        "provider": "fixture-provider",
        "model": "fixture-model",
        "harness": "fixture-harness",
        "translated_at": "2026-08-25T11:00:00Z",
    }


def _evidence() -> list[dict]:
    rows = [
        (
            "evidence-regulator",
            "regulator",
            "regulator_excerpt",
            "SEC current report",
            "U.S. Securities and Exchange Commission",
            "The same common stock will trade as HAPN on Nasdaq beginning June 27, 2026.",
            "SEC source confirms LC continued as HAPN on Nasdaq on June 27, 2026.",
            "SEC 來源確認 LC 自 2026 年 6 月 27 日起以 HAPN 在 Nasdaq 延續交易。",
            "https://www.sec.gov/Archives/edgar/data/1409970/fixture.htm",
            "a",
        ),
        (
            "evidence-market",
            "market_infrastructure",
            "market_infrastructure_snapshot",
            "IBKR contract snapshot",
            "Interactive Brokers",
            "HAPN · STK · NASDAQ · conId 112233",
            "IBKR identifies HAPN as the Nasdaq stock contract with conId 112233.",
            "IBKR 將 HAPN 識別為 Nasdaq 股票合約，conId 為 112233。",
            None,
            "b",
        ),
        (
            "evidence-publisher",
            "publisher",
            "publisher_excerpt",
            "Issuer transition report",
            "Reviewed publisher fixture",
            "旧銘柄 LC は新銘柄 HAPN として取引を継続します。",
            "The former LC ticker continues trading as HAPN.",
            "原 LC 代號將以 HAPN 繼續交易。",
            "https://publisher.invalid/reviewed-fixture",
            "c",
        ),
    ]
    return [
        {
            "evidence_id": evidence_id,
            "source_family": family,
            "kind": kind,
            "excerpt": original,
            "source_url": source_url,
            "content_sha256": digest * 64,
            "title": title,
            "publisher": publisher,
            "source_published_at": "2026-06-27T12:00:00Z",
            "translations": [
                _translation(evidence_id, "en", translated_en),
                _translation(evidence_id, "zh-Hant", translated_zh),
            ],
            "created_at": "2026-08-25T10:00:00Z",
        }
        for (
            evidence_id,
            family,
            kind,
            title,
            publisher,
            original,
            translated_en,
            translated_zh,
            source_url,
            digest,
        ) in rows
    ]


def _activity() -> dict:
    return {
        "activity_id": "activity-hapn",
        "transition_id": "transition-hapn",
        "case_id": "case-hapn",
        "activity_type": "applied",
        "source_ticker": "LC",
        "successor_ticker": "HAPN",
        "effective_date": "2026-06-27",
        "user_owned_changes": [
            {"change_type": "watchlist_membership_added", "count": 1},
            {"change_type": "watchlist_membership_archived", "count": 1},
        ],
        "provider_owned_retained": ["portfolio_open"],
        "state_sha256": "4" * 64,
        "rule_id": "lifecycle.simple_symbol_continuation",
        "rule_version": "1",
        "decision_provenance_sha256": "e" * 64,
        "occurred_at": "2026-08-25T12:00:00Z",
        "acknowledged_at": None,
        "created_at": "2026-08-25T12:00:00Z",
        "reverse_readiness": {"reversible": True, "block_reasons": []},
    }


def _empty_effects() -> dict:
    return {
        "editable_tags_to_copy": [],
        "legacy_config_seed": {"add": [], "archive": [], "reactivate": [], "unchanged": []},
        "priority": {
            "resolution": None,
            "result_value": None,
            "source_value": None,
            "successor_value": None,
            "write_successor": False,
        },
        "suppression": {
            "hide_source": False,
            "source_hidden": False,
            "successor_hidden": False,
            "unhide_successor": False,
        },
        "watchlists": {"add": [], "archive": [], "reactivate": [], "unchanged": []},
    }


def _preview() -> dict:
    return {
        "active_sources": ["manual_lists", "portfolio_open"],
        "assessment_fingerprint_sha256": "1" * 64,
        "assessment_id": "assessment-accepted",
        "block_reasons": [],
        "case_id": "case-hapn",
        "caveats": ["portfolio_position_retained"],
        "effects": _empty_effects(),
        "eligible": True,
        "evidence_set_sha256": "2" * 64,
        "execute_on": "2026-06-27",
        "observation_fingerprint_sha256": "3" * 64,
        "outcomes": ["symbol_changed", "venue_transfer"],
        "preview_sha256": "5" * 64,
        "profile_state_sha256": "4" * 64,
        "proposal_ids": ["proposal-remap"],
        "provider_owned_sources": ["portfolio_open"],
        "source_ticker": "LC",
        "successor_ticker": "HAPN",
        "transition_kind": "symbol_continuation",
    }


def _detail() -> dict:
    activity = _activity()
    return {
        **_summary(),
        "observation_fingerprint_sha256": "f" * 64,
        "observation": {
            "ticker": "HAPN",
            "cik": "0001409970",
            "issuer_name": "Happify Network, Inc.",
            "filing_date": "2026-06-27",
            "source": "sec_edgar",
            "source_ref": "0001409970-26-000131",
            "filing_form": "8-K",
            "filing_items": ["3.01"],
            "evidence_url": "https://www.sec.gov/Archives/edgar/data/1409970/fixture.htm",
            "description": "The issuer reported a symbol and venue transition.",
            "first_observed_at": "2026-08-17T07:26:14Z",
            "last_observed_at": "2026-08-17T07:26:14Z",
            "kinds": [{"event_type": "listing_status_review", "effective_date": "2026-06-27"}],
        },
        "investigation_runs": [],
        "automation_runs": [
            {
                "run_id": "run-revalidation",
                "case_id": "case-hapn",
                "mode": "historical",
                "status": "blocked",
                "policy_version": "lifecycle-automation-v1",
                "decision_tier": "verified_automatic",
                "action_readiness": "waiting_transition_revalidation",
                "failure_code": None,
                "blockers": [
                    {"blocker_code": "transition_approval_changed", "retryable": True},
                ],
                "created_at": "2026-08-25T11:00:00Z",
            },
            {
                "run_id": "run-review",
                "case_id": "case-hapn",
                "mode": "historical",
                "status": "blocked",
                "policy_version": "lifecycle-automation-v1",
                "decision_tier": "review_suggested",
                "action_readiness": "action_blocked",
                "failure_code": None,
                "blockers": [
                    {"blocker_code": "source_conflict", "retryable": False},
                    {"blocker_code": "market_confirmation_missing", "retryable": True},
                ],
                "created_at": "2026-08-25T10:00:00Z",
            },
            {
                "run_id": "run-accepted",
                "case_id": "case-hapn",
                "mode": "historical",
                "status": "succeeded",
                "policy_version": "lifecycle-automation-v1",
                "decision_tier": "verified_automatic",
                "action_readiness": "not_applicable",
                "failure_code": None,
                "blockers": [],
                "created_at": "2026-08-25T09:00:00Z",
            },
        ],
        "automation_facts": [
            {
                "fact_id": "fact-transaction",
                "automation_run_id": "run-review",
                "evidence_id": "evidence-regulator",
                "source_family": "regulator",
                "fact_type": "transaction_structure",
                "normalized_value": {
                    "kind": "asset_acquisition",
                    "terms_status": "not_extracted",
                },
                "source_span_start": 0,
                "source_span_end": 16,
                "cited_text_sha256": "5" * 64,
                "extractor_rule_id": "sec.explicit_asset_acquisition",
                "extractor_rule_version": "1",
                "created_at": "2026-08-25T10:00:00Z",
            },
            {
                "fact_id": "fact-source",
                "automation_run_id": "run-review",
                "evidence_id": "evidence-regulator",
                "source_family": "regulator",
                "fact_type": "source_ticker",
                "normalized_value": "LC",
                "source_span_start": 0,
                "source_span_end": 2,
                "cited_text_sha256": "1" * 64,
                "extractor_rule_id": "sec.explicit_symbol_change",
                "extractor_rule_version": "1",
                "created_at": "2026-08-25T10:00:00Z",
            },
            {
                "fact_id": "fact-successor",
                "automation_run_id": "run-review",
                "evidence_id": "evidence-regulator",
                "source_family": "regulator",
                "fact_type": "successor_ticker",
                "normalized_value": "HAPN",
                "source_span_start": 3,
                "source_span_end": 7,
                "cited_text_sha256": "2" * 64,
                "extractor_rule_id": "sec.explicit_symbol_change",
                "extractor_rule_version": "1",
                "created_at": "2026-08-25T10:00:00Z",
            },
            {
                "fact_id": "fact-market",
                "automation_run_id": "run-review",
                "evidence_id": "evidence-market",
                "source_family": "market_infrastructure",
                "fact_type": "destination_venue",
                "normalized_value": "NASDAQ",
                "source_span_start": 0,
                "source_span_end": 6,
                "cited_text_sha256": "3" * 64,
                "extractor_rule_id": "ibkr.primary_exchange",
                "extractor_rule_version": "1",
                "created_at": "2026-08-25T10:00:00Z",
            },
            {
                "fact_id": "fact-publisher",
                "automation_run_id": "run-review",
                "evidence_id": "evidence-publisher",
                "source_family": "publisher",
                "fact_type": "tracked_security_effect",
                "normalized_value": "symbol_and_venue_change",
                "source_span_start": 0,
                "source_span_end": 16,
                "cited_text_sha256": "4" * 64,
                "extractor_rule_id": "publisher.reviewed_fixture",
                "extractor_rule_version": "1",
                "created_at": "2026-08-25T10:00:00Z",
            },
        ],
        "evidence": _evidence(),
        "assessment_history": [DRAFT, ACCEPTED],
        "acknowledgement_history": [],
        "proposals": [
            {
                "proposal_id": "proposal-notify",
                "action_type": "notify",
                "status": "proposed",
                "block_reason": None,
                "source_snapshot": ["manual_lists", "portfolio_open"],
                "created_at": "2026-08-25T10:00:00Z",
            },
            {
                "proposal_id": "proposal-remap",
                "action_type": "remap_symbol",
                "status": "proposed",
                "block_reason": None,
                "source_snapshot": ["manual_lists"],
                "replacement_ticker": "HAPN",
                "created_at": "2026-08-25T10:00:00Z",
            },
        ],
        "ticker_transition": {
            "transition_id": "transition-hapn",
            "kind": "symbol_continuation",
            "status": "applied",
            "source_ticker": "LC",
            "successor_ticker": "HAPN",
            "execute_on": "2026-06-27",
            "approved_preview_sha256": "5" * 64,
            "approved_preview": _preview(),
            "approval_authority": "automation_policy",
            "automation_policy_version": "lifecycle-automation-v1",
            "rule_id": "lifecycle.simple_symbol_continuation",
            "rule_version": "1",
            "decision_provenance_sha256": "e" * 64,
            "updated_at": "2026-08-25T12:00:00Z",
            "latest_attempt": {
                "status": "applied",
                "block_reasons": [],
                "attempted_at": "2026-08-25T12:00:00Z",
            },
            "reverse_readiness": {"reversible": True, "block_reasons": []},
            "activity_history": [activity],
            "activity_count": 1,
            "unacknowledged_activity_count": 1,
        },
        "truncation": {},
    }


def _response(route, payload: object, status: int = 200) -> None:
    route.fulfill(
        status=status,
        content_type="application/json",
        headers={
            "access-control-allow-origin": "*",
            "access-control-allow-methods": "GET,POST,PUT,OPTIONS",
            "access-control-allow-headers": "content-type,x-arkscope-token",
        },
        body=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )


def _runtime_config() -> dict:
    route = {
        "task": "fixture",
        "provider": "openai",
        "model": "fixture",
        "effort": "none",
        "source": "default",
        "custom": False,
        "warning": None,
    }
    return {
        "anthropic": {
            "model": "fixture",
            "model_advanced": "fixture",
            "effort": None,
            "thinking": False,
            "key_set": False,
            "credentials": [],
        },
        "openai": {
            "model": "fixture",
            "model_advanced": "fixture",
            "reasoning_effort": "none",
            "key_set": False,
            "credentials": [],
        },
        "card_synthesis": {**route, "task": "card_synthesis"},
        "card_translation": {**route, "task": "card_translation"},
        "ai_research": {**route, "task": "ai_research"},
        "research_runtime": {
            "max_tool_calls": 1,
            "session_timeout_s": 60,
            "per_tool_timeout_s": 30,
            "source": "default",
            "db_saved": False,
            "warning": None,
        },
        "data_keys": {},
    }


def _geometry(page) -> dict:
    return page.evaluate(
        """() => {
          const visible = (node) => {
            const style = getComputedStyle(node);
            const rect = node.getBoundingClientRect();
            if (style.visibility === 'hidden' || style.display === 'none'
              || rect.width <= 0 || rect.height <= 0
              || rect.bottom <= 0 || rect.top >= innerHeight) return false;
            const x = Math.max(0, Math.min(innerWidth - 1, (rect.left + rect.right) / 2));
            const y = Math.max(0, Math.min(innerHeight - 1, (rect.top + rect.bottom) / 2));
            const hit = document.elementFromPoint(x, y);
            return Boolean(hit && (node.contains(hit) || hit.contains(node)))
              && rect.width > 0 && rect.height > 0
              && rect.bottom > 0 && rect.top < innerHeight;
          };
          const controls = [...document.querySelectorAll(
            '.lifecycle-activity-band button, .ui-drawer button, .ui-drawer input, '
            + '.ui-drawer select, .ui-drawer textarea, .ui-drawer a'
          )].filter(visible).map((node) => {
            const rect = node.getBoundingClientRect();
            return {tag: node.tagName, text: (node.textContent || node.getAttribute('aria-label') || '').trim(),
              left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom};
          });
          const overlaps = [];
          for (let i = 0; i < controls.length; i += 1) {
            for (let j = i + 1; j < controls.length; j += 1) {
              const a = controls[i], b = controls[j];
              const width = Math.min(a.right, b.right) - Math.max(a.left, b.left);
              const height = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
              if (width > 1 && height > 1) overlaps.push([a, b]);
            }
          }
          const textOverflow = [...document.querySelectorAll(
            '.lifecycle-activity-band strong, .lifecycle-activity-band p, '
            + '.ui-drawer h3, .ui-drawer h4, .ui-drawer h5, .ui-drawer strong, '
            + '.ui-drawer p, .ui-drawer dt, .ui-drawer dd, .ui-drawer label, .ui-drawer button'
          )].filter(visible).filter((node) => node.scrollWidth > node.clientWidth + 1)
            .map((node) => ({tag: node.tagName, text: (node.textContent || '').trim(),
              scrollWidth: node.scrollWidth, clientWidth: node.clientWidth}));
          const drawer = document.querySelector('.ui-drawer')?.getBoundingClientRect();
          return {
            viewport: {width: innerWidth, height: innerHeight},
            documentScrollWidth: document.documentElement.scrollWidth,
            bodyScrollWidth: document.body.scrollWidth,
            drawer: drawer ? {left: drawer.left, right: drawer.right, width: drawer.width} : null,
            controls,
            overlaps,
            textOverflow,
          };
        }"""
    )


def _assert_geometry(metrics: dict) -> None:
    width = metrics["viewport"]["width"]
    assert metrics["documentScrollWidth"] <= width + 1, metrics
    assert metrics["bodyScrollWidth"] <= width + 1, metrics
    if metrics["drawer"]:
        assert metrics["drawer"]["left"] >= -1, metrics["drawer"]
        assert metrics["drawer"]["right"] <= width + 1, metrics["drawer"]
    assert metrics["overlaps"] == [], metrics["overlaps"]
    assert metrics["textOverflow"] == [], metrics["textOverflow"]


def _pixel_check(path: Path, width: int, height: int) -> dict:
    with Image.open(path) as image:
        assert image.size == (width, height), (path, image.size, width, height)
        rgb = image.convert("RGB")
        stats = ImageStat.Stat(rgb)
        extrema = rgb.getextrema()
        assert max(stats.stddev) > 8.0, (path, stats.stddev)
        assert any(high - low > 40 for low, high in extrema), (path, extrema)
        return {
            "size": list(image.size),
            "channel_stddev": [round(value, 3) for value in stats.stddev],
            "extrema": [list(value) for value in extrema],
        }


def _run_entry(browser, width: int, height: int, locale: str) -> dict:
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
            _response(route, {}, 204)
        elif parsed.path == "/status":
            _response(
                route,
                {
                    "status": "ok",
                    "timestamp": "2026-08-25T00:00:00Z",
                    "tools_registered": 50,
                    "tool_categories": {},
                    "data_sources": {},
                },
            )
        elif parsed.path == "/config/runtime":
            _response(route, _runtime_config())
        elif parsed.path == "/profile/settings/ui-locale":
            _response(route, {"locale": locale, "source": "stored"})
        elif parsed.path == "/profile/universe":
            _response(
                route,
                {
                    "as_of": "2026-08-25",
                    "generated_at": "2026-08-25T00:00:00Z",
                    "total": 0,
                    "shown": 0,
                    "archived_count": 0,
                    "summarized": 0,
                    "rows": [],
                },
            )
        elif parsed.path == "/profile/lists":
            _response(route, {"lists": []})
        elif parsed.path == "/analysis/cards":
            _response(route, {"cards": []})
        elif parsed.path == "/research/threads":
            _response(route, {"threads": []})
        elif parsed.path == "/security-lifecycle/cases":
            _response(
                route,
                {"cases": [_summary()], "count": 1, "data_integrity": {"source_missing_count": 0}},
            )
        elif parsed.path == "/security-lifecycle/cases/case-hapn":
            _response(route, _detail())
        elif parsed.path == "/security-lifecycle/transition-activity":
            _response(
                route,
                {"items": [_activity()], "count": 1, "unacknowledged_count": 1},
            )
        else:
            _response(route, {"detail": {"code": "fixture_unavailable"}}, 503)

    page.route("**/*", handler)
    page.goto(APP_URL, wait_until="networkidle", timeout=20_000)
    universe = page.get_by_role("button", name=labels["universe"], exact=True)
    if not universe.is_visible():
        page.get_by_role("button", name=labels["open_nav"], exact=True).click()
    universe.click()
    page.get_by_role("tab", name=labels["lifecycle"], exact=True).click()
    page.get_by_role("heading", name=labels["activity"], exact=True).wait_for(state="visible")
    page.get_by_role("button", name=re.compile(r"^HAPN\b")).wait_for(state="visible")

    captures: dict[str, dict] = {}

    def capture(name: str) -> None:
        metrics = _geometry(page)
        _assert_geometry(metrics)
        path = OUTPUT / f"{width}x{height}-{locale}-{name}.png"
        page.screenshot(path=str(path))
        captures[name] = {
            "file": path.name,
            "geometry": metrics,
            "pixels": _pixel_check(path, width, height),
        }

    def scroll_drawer_to(target) -> int:
        scroll_top = target.evaluate(
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
        return scroll_top

    body = page.locator("body")
    for text in (labels["activity"], "LC -> HAPN", labels["reverse"], labels["acknowledge"]):
        assert text in body.inner_text(), text
    capture("activity")

    page.get_by_role("button", name=re.compile(r"^HAPN\b")).click()
    drawer = page.locator(".ui-drawer")
    drawer.wait_for(state="visible")
    drawer_text = drawer.inner_text()
    for text in (
        labels["accepted"],
        labels["revalidation"],
        labels["approval_changed"],
        labels["original"],
        labels["translation"],
        labels["facts"],
        labels["transition"],
        labels["transaction_summary"],
        *labels["families"],
    ):
        assert text in drawer_text, (text, drawer_text)
    for superseded in (labels["conflict"], labels["market_missing"]):
        assert superseded not in drawer_text, (superseded, drawer_text)
    assert drawer.get_by_label(
        "Successor ticker" if locale == "en" else "承接標的代號"
    ).input_value() == "HAPN"
    assert drawer.get_by_label(
        "Cash per security" if locale == "en" else "每單位現金對價"
    ).input_value() == "10.50"
    assert drawer.get_by_label(
        "Exchange ratio" if locale == "en" else "換股比例"
    ).input_value() == "0.25"
    assert drawer.get_by_role("button", name=labels["save_revision"], exact=True).is_visible()
    assert drawer.get_by_role("button", name=labels["accept_suggestion"], exact=True).is_visible()

    evidence_heading = drawer.get_by_role(
        "heading", name=labels["evidence_section"], exact=True
    )
    evidence_scroll_top = scroll_drawer_to(evidence_heading)
    capture("drawer-evidence")
    automation_scroll_top = scroll_drawer_to(
        drawer.get_by_role("heading", name=labels["automation"], exact=True)
    )
    assert automation_scroll_top > evidence_scroll_top, (
        evidence_scroll_top,
        automation_scroll_top,
    )
    capture("drawer-automation")
    facts_scroll_top = scroll_drawer_to(drawer.get_by_text(labels["facts"], exact=True))
    assert facts_scroll_top > automation_scroll_top, (
        automation_scroll_top,
        facts_scroll_top,
    )
    capture("drawer-facts-review")
    transition_scroll_top = scroll_drawer_to(
        drawer.get_by_text(labels["transition"], exact=True)
    )
    assert transition_scroll_top > facts_scroll_top, (facts_scroll_top, transition_scroll_top)
    drawer.get_by_role("button", name=(
        "Reverse transition" if locale == "en" else "反轉代號轉移"
    ), exact=True).wait_for(state="visible")
    capture("drawer-transition")

    visible_text = page.evaluate(
        """() => {
          const body = document.body.cloneNode(true);
          body.querySelectorAll('.mono').forEach((node) => node.remove());
          return body.innerText;
        }"""
    )
    for raw in (
        "verified_automatic",
        "review_suggested",
        "action_blocked",
        "waiting_transition_revalidation",
        "transition_approval_changed",
        "automation_policy",
        "deterministic_rule",
        "source_conflict",
        "market_confirmation_missing",
        "market_infrastructure",
        "manual_lists",
        "portfolio_open",
        "common_stock",
        "asset_acquisition",
        "not_extracted",
        "corporate_unification",
        "terminal_delisting",
        "symbol_change",
        "venue_change_only",
        "symbol_and_venue_change",
        "no_identity_change",
        "asset_acquisition_no_registrant_change",
    ):
        assert raw not in visible_text, (
            raw,
            [line for line in visible_text.splitlines() if raw in line],
        )
    writes = [
        item
        for item in state["requests"]
        if item["method"] in {"POST", "PUT", "PATCH", "DELETE"}
    ]
    assert writes == [], writes
    assert state["external"] == [], state["external"]
    assert console_errors == [], console_errors
    assert page_errors == [], page_errors
    result = {
        "locale": locale,
        "viewport": [width, height],
        "request_count": len(state["requests"]),
        "writes": writes,
        "external_requests": state["external"],
        "console_errors": console_errors,
        "page_errors": page_errors,
        "render_acknowledgements": 0,
        "captures": captures,
    }
    context.close()
    return result


def main() -> int:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        results = [
            _run_entry(browser, width, height, locale)
            for width, height in ((1440, 900), (390, 844))
            for locale in ("en", "zh-Hant")
        ]
        browser.close()
    payload = {
        "schema_version": 1,
        "app_url": APP_URL,
        "fixture_only": True,
        "production_backend_started": False,
        "entries": results,
    }
    (OUTPUT / "matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "entries": len(results),
        "screenshots": sum(len(item["captures"]) for item in results),
        "writes": sum(len(item["writes"]) for item in results),
        "external_requests": sum(len(item["external_requests"]) for item in results),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
