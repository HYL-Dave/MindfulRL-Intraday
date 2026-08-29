"""Contract tests for the Task 8 admission packet itself."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from playwright.sync_api import sync_playwright


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
FIXTURES = ROOT / "tests/fixtures/listing_authority"


def _load(name: str) -> ModuleType:
    path = PACKET / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"task8_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shadow_cases_bind_exact_repository_payload_bytes() -> None:
    shadow = _load("run_shadow")
    authority = json.loads(
        (FIXTURES / "shadow-cases.json").read_text(encoding="utf-8")
    )

    assert len(authority["cases"]) == 9
    for case in authority["cases"]:
        assert case["listing_payloads"]
        for binding in case["listing_payloads"]:
            path = FIXTURES / binding["filename"]
            expected = binding["sha256"]
            body = shadow._read_bound_payload(binding)
            assert body == path.read_bytes()
            assert hashlib.sha256(body).hexdigest() == expected


def test_shadow_executes_real_listing_session_transport_contract() -> None:
    shadow = _load("run_shadow")
    from src.security_lifecycle_listing_evidence import ListingAuthoritySession

    real_lookup = ListingAuthoritySession.lookup
    observed = []

    def observed_lookup(self, **kwargs):
        observed.append(
            {
                "candidate_tickers": kwargs["candidate_tickers"],
                "require_explicit_inactive": kwargs["require_explicit_inactive"],
            }
        )
        return real_lookup(self, **kwargs)

    with patch.object(ListingAuthoritySession, "lookup", observed_lookup):
        result = shadow.run()

    assert observed, "shadow bypassed ListingAuthoritySession.lookup"
    contract = result["session_contract"]
    assert contract["transport"] == "fake_production_interface_exact_repository_bytes"
    assert contract["session"] == "real_listing_authority_session"
    assert contract["real_session_lookup_calls"] == len(observed)
    assert contract["provider_calls"] == 0
    assert contract["terminal_requiredness"] == {
        "candidate_ticker": "OLD",
        "massive_expected_active": False,
        "massive_market": "stocks",
        "require_explicit_inactive": True,
    }
    assert contract["otc_fallback_order"] == [
        "nasdaq:nasdaq_listed",
        "nasdaq:other_listed",
        "massive:NEW:true:stocks",
        "massive:NEW:true:otc",
    ]
    assert contract["deduplication"] == {
        "case": "OTC-A",
        "repeated_lookup_additional_requests": 0,
        "repeated_lookup_byte_identical": True,
    }
    assert contract["blocker_normalization"] == {
        "missing_credential": "massive_credential_missing",
        "parser_failure": "massive_reference_unavailable",
    }
    assert contract["request_budgets"]["all_within_limits"] is True


def test_preexisting_product_test_fixture_authorities_are_preserved() -> None:
    expected = {
        "massive-active.json": "f8ab57e07d82eb4dbec4fa254730540931ac1ec432e3ce30befee0219ceed3cc",
        "massive-inactive.json": "98a75198cd690614146d9d2ec3a61c3308c17b9178c49768e0647863ca3a653e",
        "massive-otc.json": "39d402f6f8c0e80abcd52f2a24de7d1bbc00ecab2fd24912a3bee7e870bca679",
        "nasdaqlisted.txt": "09c5739cb35b5318d62cbb539acdd109bf07569bd0c9a1fa08cf335189a10b4a",
        "otherlisted.txt": "71fb4b1f445be5f86ea622d7aff89fab47aaa772eac2562d774288326f67a8bd",
    }

    assert {
        name: hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        for name in expected
    } == expected


def test_every_mutation_has_baseline_probe_and_stable_signatures() -> None:
    mutations = _load("run_mutations")

    assert len(mutations.MUTATIONS) == 39
    for mutation in mutations.MUTATIONS:
        assert mutation.failure_signatures
        assert mutation.command[:4] == (
            "pytest",
            "-p",
            "mutation_pytest_probe",
            "-vv",
        )


def test_m27_ledger_proves_semantic_blocker_misclassification() -> None:
    ledger = json.loads((PACKET / "mutation-ledger.json").read_text(encoding="utf-8"))
    m27 = next(row for row in ledger["mutations"] if row["id"] == "M27")
    signature = (
        "At index 0 diff: 'listing_status_unresolved' != "
        "'massive_reference_unavailable'"
    )
    output_tail = "\n".join(m27["mutant"]["output_tail"])

    assert m27["required_failure_signatures"] == [signature]
    assert m27["failure_signature_matches"] == {signature: True}
    assert m27["all_required_failure_signatures_observed"] is True
    assert m27["actual_failed_node_ids"] == m27["expected_failed_node_ids"]
    assert "NameError" not in output_tail
    assert "UnboundLocalError" not in output_tail


def test_frontend_presentation_maps_every_v3_listing_blocker() -> None:
    presentation = (
        ROOT / "apps/arkscope-web/src/lifecycle/lifecyclePresentation.ts"
    ).read_text(encoding="utf-8")
    api = (ROOT / "apps/arkscope-web/src/api.ts").read_text(encoding="utf-8")
    en = (ROOT / "apps/arkscope-web/src/i18n/resources/en/explore.ts").read_text(
        encoding="utf-8"
    )
    zh_hant = (
        ROOT / "apps/arkscope-web/src/i18n/resources/zh-Hant/explore.ts"
    ).read_text(encoding="utf-8")
    runtime_owner = (
        ROOT / "apps/arkscope-web/src/lifecycle/lifecyclePresentation.test.ts"
    ).read_text(encoding="utf-8")
    blockers = {
        "listing_directory_unavailable": "listingDirectoryUnavailable",
        "listing_directory_schema_mismatch": "listingDirectorySchemaMismatch",
        "listing_directory_stale": "listingDirectoryStale",
        "listing_status_unresolved": "listingStatusUnresolved",
        "listing_authority_conflict": "listingAuthorityConflict",
        "massive_credential_missing": "massiveCredentialMissing",
        "massive_access_denied": "massiveAccessDenied",
        "massive_rate_limited": "massiveRateLimited",
        "massive_reference_unavailable": "massiveReferenceUnavailable",
    }

    assert len(blockers) == 9
    for code, copy_key in blockers.items():
        assert f'  | "{code}"' in api, code
        assert f"    {code}: copy.{copy_key}," in presentation, code
        assert f"      {copy_key}: " in en, code
        assert f"      {copy_key}: " in zh_hant, code
        assert f'"{code}"' in runtime_owner, code


def test_old_code_path_resolution_follows_file_uri_symlinks(tmp_path: Path) -> None:
    old_code = _load("verify_old_code")
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.db"
    link = allowed / "escape.db"
    link.symlink_to(outside)

    assert old_code._resolve_database_path(f"file:{link}?mode=rw") == outside.resolve()
    assert old_code._is_within(outside.resolve(), allowed.resolve()) is False


def test_browser_terminal_projection_is_preflight_valid() -> None:
    browser = _load("run_browser_matrix")

    detail = browser._detail("inactive-history")
    transition = detail["ticker_transition"]
    preview = transition["approved_preview"]
    activity = transition["activity_history"][0]
    listings = [
        row["listing"]
        for row in detail["evidence"]
        if row["source_family"] == "listing_authority"
    ]

    assert transition["kind"] == "terminal_delisting"
    assert {listing["candidate_ticker"] for listing in listings} == {"TERM"}
    assert {
        (listing["authority"], listing["directory"], listing["listing_status"])
        for listing in listings
    } == {
        ("nasdaq_trader", "nasdaq_listed", "not_found"),
        ("nasdaq_trader", "other_listed", "not_found"),
        ("massive", None, "inactive"),
    }
    assert preview["source_ticker"] == "TERM"
    assert preview["successor_ticker"] is None
    assert preview["proposal_ids"] == []
    assert preview["eligible"] is True
    assert preview["block_reasons"] == []
    assert preview["effects"]["watchlists"] == {
        "add": [],
        "archive": [
            {
                "list_id": 1,
                "list_name": "Core",
                "position": 3,
                "ticker": "TERM",
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
    assert "portfolio_open" not in preview["active_sources"]
    assert "portfolio_open" not in preview["provider_owned_sources"]
    assert "portfolio_position_retained" not in preview["caveats"]
    assert activity["source_ticker"] == "TERM"
    assert activity["successor_ticker"] is None
    assert activity["user_owned_changes"] == [
        {"change_type": "source_hidden", "count": 1},
        {"change_type": "watchlist_membership_archived", "count": 1},
    ]
    assert "portfolio_open" not in activity["provider_owned_retained"]


def test_browser_massive_settings_fixture_uses_one_primary_field_with_legacy_alias() -> None:
    browser = _load("run_browser_matrix")
    providers = browser._provider_config()["providers"]

    assert set(providers) == {"polygon"}
    assert len(providers["polygon"]["fields"]) == 1
    assert providers["polygon"]["fields"][0] == {
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


def test_browser_applied_projections_have_visible_command_and_evidence_witnesses() -> None:
    browser = _load("run_browser_matrix")

    conflict = browser.SCENARIOS["conflict-attention"]
    assert {row["listing_status"] for row in conflict["listings"]} == {"active"}
    assert len({row["issuer_cik"] for row in conflict["listings"]}) == 2

    for labels in browser.LABELS.values():
        assert labels["acknowledge"]
        assert labels["translate_evidence"]

    for name in ("inactive-history", "otc-continuation"):
        scenario = browser.SCENARIOS[name]
        detail = browser._detail(name)
        transition = detail["ticker_transition"]
        assert scenario["synthetic_post_apply_projection"] is True
        assert transition["status"] == "applied"
        assert transition["activity_history"]
        assert transition["reverse_readiness"]["reversible"] is True

    for name, scenario in browser.SCENARIOS.items():
        if scenario.get("settings"):
            continue
        detail = browser._detail(name)
        listing_evidence = [
            item
            for item in detail["evidence"]
            if item["source_family"] == "listing_authority"
        ]
        regulator_evidence = [
            item for item in detail["evidence"] if item["source_family"] == "regulator"
        ]
        assert len(listing_evidence) == len(scenario["listings"])
        assert len(regulator_evidence) == 1
        assert regulator_evidence[0]["translations"] == []
        assert all("translations" not in item for item in listing_evidence)

    assert browser.DECLARED_ZERO == {
        "value": 0,
        "basis": "declared_not_authorized",
    }


def test_browser_evidence_surface_validator_fails_closed() -> None:
    browser = _load("run_browser_matrix")
    valid = {
        "expected_listing_count": 2,
        "regulator_evidence_count": 1,
        "expanded_regulator_evidence_count": 1,
        "regulator_translation_button_count": 1,
        "listing_evidence_count": 2,
        "expanded_listing_evidence_count": 2,
        "listing_translation_button_count": 0,
    }
    browser._assert_evidence_surface_counts(**valid)

    invalid = (
        {**valid, "regulator_translation_button_count": 0},
        {**valid, "listing_evidence_count": 1},
        {**valid, "expanded_listing_evidence_count": 1},
        {**valid, "listing_translation_button_count": 1},
    )
    for counts in invalid:
        with pytest.raises(AssertionError, match="^browser_evidence_surface_mismatch:"):
            browser._assert_evidence_surface_counts(**counts)


def test_browser_post_apply_surface_validator_fails_closed() -> None:
    browser = _load("run_browser_matrix")
    valid = {
        "acknowledgement_count": 1,
        "reverse_transition_count": 1,
        "reverse_activity_count": 1,
    }
    browser._assert_post_apply_surface_counts(**valid)

    for field in valid:
        with pytest.raises(AssertionError, match="^browser_post_apply_surface_mismatch:"):
            browser._assert_post_apply_surface_counts(**{**valid, field: 0})


def test_browser_geometry_uses_the_visible_overflow_clipped_control_rect() -> None:
    browser_fixture = _load("run_browser_matrix")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.set_content(
            """
            <style>
              * { box-sizing: border-box; }
              body { margin: 0; }
              .ui-drawer {
                width: 390px;
                height: 844px;
                display: grid;
                grid-template-rows: 48px minmax(0, 1fr);
              }
              .ui-overlay-head {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding: 8px 12px;
              }
              .ui-drawer-body { overflow: auto; }
              button { height: 28px; }
            </style>
            <aside class="ui-drawer">
              <header class="ui-overlay-head"><button>Close</button></header>
              <div class="ui-drawer-body">
                <div style="height: 12px"></div>
                <button id="save">Save draft</button>
                <div style="height: 900px"></div>
              </div>
            </aside>
            """
        )
        page.locator(".ui-drawer-body").evaluate("node => { node.scrollTop = 25; }")

        metrics = browser_fixture._geometry(page)
        browser_fixture.STAGE5._assert_geometry(metrics)
        save = next(control for control in metrics["controls"] if control["text"] == "Save draft")
        body_top = page.locator(".ui-drawer-body").bounding_box()["y"]

        assert save["top"] == body_top, "visible_overflow_clip"
        browser.close()


def test_browser_geometry_clips_other_edges_and_omits_fully_hidden_controls() -> None:
    browser_fixture = _load("run_browser_matrix")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})

        def render(content: str) -> None:
            page.set_content(
                f"""
                <style>
                  * {{ box-sizing: border-box; }}
                  body {{ margin: 0; }}
                  .ui-drawer {{
                    width: 200px;
                    height: 200px;
                    display: grid;
                    grid-template-rows: 48px minmax(0, 1fr);
                  }}
                  .ui-overlay-head {{ display: flex; justify-content: flex-end; }}
                  .ui-drawer-body {{ overflow: auto; }}
                  button {{ width: 40px; height: 28px; }}
                </style>
                <aside class="ui-drawer">
                  <header class="ui-overlay-head"><button>Close</button></header>
                  <div class="ui-drawer-body">{content}</div>
                </aside>
                """
            )

        render(
            '<div style="width:400px">'
            '<button id="left" style="margin-left:12px">Left</button></div>'
        )
        page.locator(".ui-drawer-body").evaluate("node => { node.scrollLeft = 25; }")
        left = next(
            control
            for control in browser_fixture._geometry(page)["controls"]
            if control["text"] == "Left"
        )
        body = page.locator(".ui-drawer-body").bounding_box()
        assert left["left"] == body["x"], "visible_overflow_left_clip"

        render('<div style="height:140px"></div><button id="bottom">Bottom</button>')
        bottom = next(
            control
            for control in browser_fixture._geometry(page)["controls"]
            if control["text"] == "Bottom"
        )
        body = page.locator(".ui-drawer-body").bounding_box()
        assert bottom["bottom"] == body["y"] + body["height"], (
            "visible_overflow_bottom_clip"
        )

        render(
            '<div style="height:12px"></div><button id="hidden">Hidden</button>'
            '<div style="height:900px"></div>'
        )
        page.locator(".ui-drawer-body").evaluate("node => { node.scrollTop = 100; }")
        controls = browser_fixture._geometry(page)["controls"]
        assert all(control["text"] != "Hidden" for control in controls), (
            "fully_clipped_control_hidden"
        )
        browser.close()


def test_log_normalization_removes_machine_paths_and_trailing_blank_lines(
    tmp_path: Path,
) -> None:
    normalizer = _load("normalize_packet_logs")
    path = tmp_path / "gate.txt"
    path.write_text(f"{ROOT}/src\n{normalizer.PYTHON_ENV}/bin/python\n\n", encoding="utf-8")

    counts = normalizer._normalize_file(path)

    assert path.read_text(encoding="utf-8") == (
        "<REPO_ROOT>/src\n<PYTHON_ENV>/bin/python\n"
    )
    assert counts["repo_root_replacements"] == 1
    assert counts["python_env_replacements"] == 1
    assert counts["trailing_blank_lines_removed"] == 1
