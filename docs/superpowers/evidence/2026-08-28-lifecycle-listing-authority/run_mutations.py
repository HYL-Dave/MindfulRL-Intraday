"""Apply the listing-authority admission mutations independently."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    description: str
    path: str
    old: str
    new: str
    owners: tuple[str, ...]
    command: tuple[str, ...]
    extra_replacements: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.command[:2] != ("pytest", "-q"):
            raise ValueError(f"mutation_command_shape:{self.mutation_id}")
        object.__setattr__(
            self,
            "command",
            (
                "pytest",
                "-p",
                "mutation_pytest_probe",
                "-vv",
                "--tb=short",
                *self.command[2:],
            ),
        )

    @property
    def failure_signatures(self) -> tuple[str, ...]:
        return FAILURE_SIGNATURES[self.mutation_id]


TRANSPORT = "data_sources/listing_authority_transport.py"
LISTING = "src/security_lifecycle_listing_evidence.py"
KERNEL = "src/security_lifecycle_fact_kernel.py"
POLICY = "src/security_lifecycle_decision_policy.py"
SCHEDULER = "src/service/security_lifecycle_automation_scheduler.py"
TOOLS = "src/tools/security_lifecycle_tools.py"
TRANSLATION = "src/security_lifecycle_translation.py"
PROVIDERS = "src/data_provider_config.py"
MIGRATION = "src/security_lifecycle_listing_migration.py"
SCHEMA = "src/security_lifecycle_schema.py"
FRONTEND_PRESENTATION = "apps/arkscope-web/src/lifecycle/lifecyclePresentation.ts"
PACKET_SHADOW = (
    "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/run_shadow.py"
)
PACKET_BROWSER = (
    "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
    "run_browser_matrix.py"
)


MUTATIONS = (
    Mutation(
        "M01",
        "remove nasdaq host/path allowlist",
        TRANSPORT,
        "        if str(source_url) not in _NASDAQ_URLS:\n",
        "        if False and str(source_url) not in _NASDAQ_URLS:\n",
        tuple(
            "tests/test_listing_authority_transport.py::"
            "test_nasdaq_rejects_urls_outside_the_two_exact_files[" + value + "]"
            for value in (
                "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                "https://www.nasdaqtrader.com/dynamic/SymDir/unknown.txt",
                "https://evil.example/dynamic/SymDir/nasdaqlisted.txt",
                "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt?x=1",
            )
        ),
        (
            "pytest",
            "-q",
            "tests/test_listing_authority_transport.py::"
            "test_nasdaq_rejects_urls_outside_the_two_exact_files",
        ),
    ),
    Mutation(
        "M02",
        "turn Nasdaq not_found into inactive",
        LISTING,
        '        status = "active" if row is not None else "not_found"\n',
        '        status = "active" if row is not None else "inactive"\n',
        (
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_nasdaq_parser_preserves_matching_component_and_per_file_hashes",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_nasdaq_parser_preserves_matching_component_and_per_file_hashes",
        ),
    ),
    Mutation(
        "M03",
        "accept a missing Nasdaq footer",
        LISTING,
        "    lines = _decode_nasdaq(body)\n"
        "    expected_header = _NASDAQ_HEADER if directory == \"nasdaq_listed\" else _OTHER_HEADER\n",
        "    lines = _decode_nasdaq(body)\n"
        "    if _FOOTER.fullmatch(lines[-1]) is None:\n"
        '        lines.append("File Creation Time: 08282026|120000")\n'
        "    expected_header = _NASDAQ_HEADER if directory == \"nasdaq_listed\" else _OTHER_HEADER\n",
        (
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_nasdaq_parser_rejects_incomplete_stale_or_drifted_files"
            "[missing_footer-<lambda>-listing_directory_schema_mismatch]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_nasdaq_parser_rejects_incomplete_stale_or_drifted_files"
            "[missing_footer-<lambda>-listing_directory_schema_mismatch]",
        ),
    ),
    Mutation(
        "M04",
        "ignore Nasdaq file freshness",
        LISTING,
        "        or created.date() < latest\n",
        "        or False\n",
        (
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_nasdaq_parser_rejects_incomplete_stale_or_drifted_files"
            "[stale_file-<lambda>-listing_directory_stale]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_nasdaq_parser_rejects_incomplete_stale_or_drifted_files"
            "[stale_file-<lambda>-listing_directory_stale]",
        ),
    ),
    Mutation(
        "M05",
        "follow Massive next_url",
        LISTING,
        '        or "next_url" in payload\n',
        "        or False\n",
        (
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_massive_parser_rejects_ambiguous_or_contradictory_inactive_rows"
            "[<lambda>-listing_status_unresolved3]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_massive_parser_rejects_ambiguous_or_contradictory_inactive_rows"
            "[<lambda>-listing_status_unresolved3]",
        ),
    ),
    Mutation(
        "M06",
        "log Massive API key",
        TRANSPORT,
        '        source_url = f"{MASSIVE_TICKERS_URL}?{urlencode(canonical_params)}"\n',
        '        source_url = f"{MASSIVE_TICKERS_URL}?{urlencode(canonical_params)}&apiKey={key}"\n',
        (
            "tests/test_listing_authority_transport.py::"
            "test_massive_query_secret_never_leaves_the_request_boundary",
        ),
        (
            "pytest",
            "-q",
            "tests/test_listing_authority_transport.py::"
            "test_massive_query_secret_never_leaves_the_request_boundary",
        ),
    ),
    Mutation(
        "M07",
        "accept Massive inactive without delisted_utc",
        LISTING,
        "        if delisted_value is None:\n"
        "            raise _massive_failure()\n"
        '        status = "inactive"\n'
        "        delisted_utc = _delisted_date(delisted_value, lookup_date=retrieved.date())\n",
        '        status = "inactive"\n'
        "        delisted_utc = (\n"
        "            None if delisted_value is None\n"
        "            else _delisted_date(delisted_value, lookup_date=retrieved.date())\n"
        "        )\n",
        (
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_massive_parser_rejects_ambiguous_or_contradictory_inactive_rows"
            "[<lambda>-listing_status_unresolved0]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_massive_parser_rejects_ambiguous_or_contradictory_inactive_rows"
            "[<lambda>-listing_status_unresolved0]",
        ),
    ),
    Mutation(
        "M08",
        "remove producer-to-kernel hash validation",
        KERNEL,
        "        if hashlib.sha256(excerpt.encode()).hexdigest() != content_digest:\n",
        "        if False and hashlib.sha256(excerpt.encode()).hexdigest() != content_digest:\n",
        tuple(
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_listing_producer_mutations_fail_at_real_kernel_validator["
            f"{adapter}-excerpt-evidence_content_sha256]"
            for adapter in ("nasdaq_symbol_directory", "massive_reference")
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_listing_producer_mutations_fail_at_real_kernel_validator"
            "[nasdaq_symbol_directory-excerpt-evidence_content_sha256]",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_listing_producer_mutations_fail_at_real_kernel_validator"
            "[massive_reference-excerpt-evidence_content_sha256]",
        ),
    ),
    Mutation(
        "M09",
        "remove producer-to-kernel citation validation",
        KERNEL,
        "    if hashlib.sha256(cited).hexdigest() != digest:\n"
        "        raise ValueError(error_name)\n",
        "    if False and hashlib.sha256(cited).hexdigest() != digest:\n"
        "        raise ValueError(error_name)\n",
        tuple(
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_listing_producer_mutations_fail_at_real_kernel_validator["
            f"{adapter}-span-fact_citation]"
            for adapter in ("nasdaq_symbol_directory", "massive_reference")
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_listing_producer_mutations_fail_at_real_kernel_validator"
            "[nasdaq_symbol_directory-span-fact_citation]",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_listing_producer_mutations_fail_at_real_kernel_validator"
            "[massive_reference-span-fact_citation]",
        ),
    ),
    Mutation(
        "M10",
        "select one latest listing record and discard the rest",
        POLICY,
        "        if len(rows) == 1:\n"
        "            return tuple(rows)\n",
        "        if rows:\n"
        "            return (sorted(rows, key=lambda row: row.evidence_id)[-1],)\n",
        (
            "tests/test_security_lifecycle_decision_policy.py::"
            "test_equal_time_disagreement_inside_one_listing_component_fails_closed",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_decision_policy.py::"
            "test_equal_time_disagreement_inside_one_listing_component_fails_closed",
        ),
    ),
    Mutation(
        "M11",
        "allow IBKR missing to prove delisting",
        POLICY,
        "        elif not _listing_explicit_inactive(evidence_rows, regulator_source, today):\n",
        "        elif not (\n"
        "            _listing_explicit_inactive(evidence_rows, regulator_source, today)\n"
        "            or any(\n"
        '                row.source_family == "market_infrastructure"\n'
        '                and row.source_locator.get("contract_status") == "missing"\n'
        "                for row in evidence_rows\n"
        "            )\n"
        "        ):\n",
        (
            "tests/test_security_lifecycle_decision_policy.py::"
            "test_ibkr_missing_never_proves_terminal",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_decision_policy.py::"
            "test_ibkr_missing_never_proves_terminal",
        ),
        extra_replacements=(
            (
                '        and row.source_locator.get("contract_status") == "found"\n',
                '        and row.source_locator.get("contract_status") in {"found", "missing"}\n',
            ),
        ),
    ),
    Mutation(
        "M12",
        "require a fresh quote for listing acceptance",
        POLICY,
        "    listing = _selected_listing_rows(evidence)\n"
        "    market = tuple(\n",
        "    listing = (\n"
        "        _selected_listing_rows(evidence)\n"
        "        if any(\n"
        '            row.source_family == "market_infrastructure"\n'
        '            and isinstance(row.source_locator.get("market_data"), Mapping)\n'
        '            and row.source_locator["market_data"].get("fresh") is True\n'
        "            for row in evidence\n"
        "        )\n"
        "        else ()\n"
        "    )\n"
        "    market = tuple(\n",
        (
            "tests/test_security_lifecycle_decision_policy.py::"
            "test_quote_freshness_is_inert_for_v4_acceptance",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_decision_policy.py::"
            "test_quote_freshness_is_inert_for_v4_acceptance",
        ),
    ),
    Mutation(
        "M13",
        "allow publisher evidence into v4 material",
        POLICY,
        "    selected_ids = {\n"
        '        row.evidence_id for row in evidence if row.source_family == "regulator"\n'
        "    }\n",
        "    selected_ids = {\n"
        "        row.evidence_id\n"
        "        for row in evidence\n"
        '        if row.source_family in {"regulator", "publisher"}\n'
        "    }\n",
        (
            "tests/test_security_lifecycle_decision_policy.py::"
            "test_publisher_evidence_cannot_change_v4_decision",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_decision_policy.py::"
            "test_publisher_evidence_cannot_change_v4_decision",
        ),
    ),
    Mutation(
        "M14",
        "restore publisher as pending-monitoring required family",
        SCHEDULER,
        '        "regulator",\n'
        '        "listing_authority",\n',
        '        "regulator",\n'
        '        "listing_authority",\n'
        '        "publisher",\n',
        (
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_pending_event_monitoring_uses_explicit_dates_and_final_source_check",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_pending_event_monitoring_uses_explicit_dates_and_final_source_check",
        ),
    ),
    Mutation(
        "M15",
        "expose publisher evidence through active detail",
        TOOLS,
        '    "listing_authority",\n'
        '    "market_infrastructure",\n',
        '    "listing_authority",\n'
        '    "market_infrastructure",\n'
        '    "publisher",\n',
        (
            "tests/test_security_lifecycle_tools.py::"
            "test_active_case_projection_uses_closed_families_but_preserves_storage",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_tools.py::"
            "test_active_case_projection_uses_closed_families_but_preserves_storage",
        ),
    ),
    Mutation(
        "M16",
        "translate a listing snapshot",
        TRANSLATION,
        '    if evidence.get("kind") == "listing_directory_snapshot":\n'
        '        raise ValueError("unsupported_content")\n',
        "",
        (
            "tests/test_security_lifecycle_translation.py::"
            "test_listing_snapshot_translation_rejects_before_every_downstream_boundary",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_translation.py::"
            "test_listing_snapshot_translation_rejects_before_every_downstream_boundary",
        ),
    ),
    Mutation(
        "M17",
        "add a second Massive secret field",
        PROVIDERS,
        '    "polygon": [FieldDef("api_key", "POLYGON_API_KEY", True, "API key")],\n',
        '    "polygon": [FieldDef("api_key", "POLYGON_API_KEY", True, "API key")],\n'
        '    "massive": [FieldDef("api_key", "MASSIVE_API_KEY", True, "API key")],\n',
        (
            "tests/test_data_provider_config.py::"
            "test_massive_reuses_the_polygon_credential_authority",
        ),
        (
            "pytest",
            "-q",
            "tests/test_data_provider_config.py::"
            "test_massive_reuses_the_polygon_credential_authority",
        ),
    ),
    Mutation(
        "M18",
        "fail to preserve one v2 publisher row during migration",
        MIGRATION,
        "    conn.executemany(\n"
        "        f\"INSERT INTO {_quote_identifier(table)} ({projection}) \"\n"
        "        f\"VALUES ({placeholders})\",\n"
        "        snapshot.rows,\n"
        "    )\n",
        "    rows = snapshot.rows\n"
        '    if table == "security_lifecycle_evidence":\n'
        '        rows = tuple(row for row in rows if "sle_publisher" not in row)\n'
        "    conn.executemany(\n"
        "        f\"INSERT INTO {_quote_identifier(table)} ({projection}) \"\n"
        "        f\"VALUES ({placeholders})\",\n"
        "        rows,\n"
        "    )\n",
        (
            "tests/test_security_lifecycle_listing_migration.py::"
            "test_v2_to_v3_preserves_every_existing_cell_and_adds_no_listing_rows",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_migration.py::"
            "test_v2_to_v3_preserves_every_existing_cell_and_adds_no_listing_rows",
        ),
    ),
    Mutation(
        "M19",
        "change one v2 translated-text byte during migration",
        MIGRATION,
        "    conn.executemany(\n"
        "        f\"INSERT INTO {_quote_identifier(table)} ({projection}) \"\n"
        "        f\"VALUES ({placeholders})\",\n"
        "        snapshot.rows,\n"
        "    )\n",
        "    rows = snapshot.rows\n"
        '    if table == "security_lifecycle_evidence_translations":\n'
        "        rows = tuple(\n"
        "            tuple(\n"
        '                "Translated publisher evidence!"\n'
        '                if cell == "Translated publisher evidence." else cell\n'
        "                for cell in row\n"
        "            )\n"
        "            for row in rows\n"
        "        )\n"
        "    conn.executemany(\n"
        "        f\"INSERT INTO {_quote_identifier(table)} ({projection}) \"\n"
        "        f\"VALUES ({placeholders})\",\n"
        "        rows,\n"
        "    )\n",
        (
            "tests/test_security_lifecycle_listing_migration.py::"
            "test_v2_to_v3_preserves_every_existing_cell_and_adds_no_listing_rows",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_migration.py::"
            "test_v2_to_v3_preserves_every_existing_cell_and_adds_no_listing_rows",
        ),
    ),
    Mutation(
        "M20",
        "allow a v3 binary to verify a v2 database without migration",
        SCHEMA,
        "def verify_profile_connection(conn: sqlite3.Connection) -> None:\n"
        "    _verify_connection(conn, PROFILE_TABLE_SQL, PROFILE_INDEX_SQL)\n",
        "def verify_profile_connection(conn: sqlite3.Connection) -> None:\n"
        "    try:\n"
        "        _verify_connection(conn, PROFILE_TABLE_SQL, PROFILE_INDEX_SQL)\n"
        "    except LifecycleSchemaMismatch:\n"
        "        _verify_connection(conn, V2_PROFILE_TABLE_SQL, V2_PROFILE_INDEX_SQL)\n",
        (
            "tests/test_security_lifecycle_listing_migration.py::"
            "test_v2_to_v3_preserves_every_existing_cell_and_adds_no_listing_rows",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_migration.py::"
            "test_v2_to_v3_preserves_every_existing_cell_and_adds_no_listing_rows",
        ),
    ),
    Mutation(
        "M21",
        "drop identity facts from exact inactive Massive evidence",
        LISTING,
        "    inactive_massive = (\n"
        "        record.adapter == \"massive_reference\"\n"
        "        and record.listing_status == \"inactive\"\n"
        "    )\n",
        "    inactive_massive = False\n",
        (
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_real_listing_session_terminal_output_drives_exact_terminal_policy",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_real_listing_session_terminal_output_drives_exact_terminal_policy",
        ),
    ),
    Mutation(
        "M22",
        "bypass the exact terminal Massive locator gate",
        POLICY,
        "        if not (\n"
        "            snapshot.get(\"locator_kind\") == \"listing_directory_snapshot\"\n",
        "        if False and not (\n"
        "            snapshot.get(\"locator_kind\") == \"listing_directory_snapshot\"\n",
        tuple(
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_real_terminal_pipeline_requires_exact_inactive_massive_locator["
            + value
            + "]"
            for value in (
                "wrong_expected_intent",
                "wrong_market",
                "incomplete_snapshot",
            )
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_real_terminal_pipeline_requires_exact_inactive_massive_locator",
        ),
    ),
    Mutation(
        "M23",
        "remove lifecycle dependency-log secret redaction",
        TRANSPORT,
        "        with dependency_log_redaction(secret_values):\n",
        "        if True:\n",
        (
            "tests/test_listing_authority_transport.py::"
            "test_lifecycle_transport_redacts_massive_key_from_urllib3_debug_logs",
        ),
        (
            "pytest",
            "-q",
            "tests/test_listing_authority_transport.py::"
            "test_lifecycle_transport_redacts_massive_key_from_urllib3_debug_logs",
        ),
    ),
    Mutation(
        "M24",
        "require explicit inactive Massive before the SEC effective date",
        SCHEDULER,
        "    explicit_inactive_required = terminal and (\n"
        "        effective is None or today >= effective\n"
        "    )\n",
        "    explicit_inactive_required = terminal\n",
        (
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_terminal_massive_requiredness_changes_on_effective_date_through_scheduler",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_terminal_massive_requiredness_changes_on_effective_date_through_scheduler",
        ),
    ),
    Mutation(
        "M25",
        "retain Massive as a required component before the SEC effective date",
        SCHEDULER,
        "    if terminal and not explicit_inactive_required:\n"
        "        required_listing_components = required_listing_components - {\"massive\"}\n",
        "    if False and terminal and not explicit_inactive_required:\n"
        "        required_listing_components = required_listing_components - {\"massive\"}\n",
        (
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_terminal_massive_requiredness_changes_on_effective_date_through_scheduler",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_terminal_massive_requiredness_changes_on_effective_date_through_scheduler",
        ),
    ),
    Mutation(
        "M26",
        "persist listing body-byte diagnostics without content-safe naming",
        SCHEDULER,
        '            key.replace("_body_bytes", "_payload_bytes"): value\n',
        "            key: value\n",
        (
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_terminal_massive_requiredness_changes_on_effective_date_through_scheduler",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_terminal_massive_requiredness_changes_on_effective_date_through_scheduler",
        ),
    ),
    Mutation(
        "M27",
        "misclassify Massive parser failures as listing status unresolved",
        LISTING,
        '            result = (None, "massive_reference_unavailable")\n',
        '            result = (None, "listing_status_unresolved")\n',
        (
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_listing_session_maps_massive_parser_failures_to_component_blocker["
            "source-candidate]",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_listing_session_maps_massive_parser_failures_to_component_blocker["
            "successor-candidate]",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_real_session_filters_optional_massive_parser_failure_for_nms_successor",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_listing_evidence.py::"
            "test_listing_session_maps_massive_parser_failures_to_component_blocker",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_real_session_filters_optional_massive_parser_failure_for_nms_successor",
        ),
    ),
    Mutation(
        "M28",
        "overwrite a completed worker result after listing cleanup failure",
        SCHEDULER,
        "        except Exception as close_exc:\n"
        "            logger.warning(\n"
        '                "security lifecycle listing transport cleanup failed code=%s",\n'
        "                type(close_exc).__name__,\n"
        "            )\n"
        "    return result\n",
        "        except Exception as close_exc:\n"
        "            logger.warning(\n"
        '                "security lifecycle listing transport cleanup failed code=%s",\n'
        "                type(close_exc).__name__,\n"
        "            )\n"
        "        return security_lifecycle_automation_failure(\n"
        '            "automation_scheduler_failed"\n'
        "        )\n"
        "    return result\n",
        (
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_listing_session_close_failure_retains_result_with_sanitized_cleanup_witness",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_listing_session_close_failure_retains_result_with_sanitized_cleanup_witness",
        ),
    ),
    Mutation(
        "M29",
        "drop all v3 listing blocker presentation mappings",
        FRONTEND_PRESENTATION,
        "    listing_directory_unavailable: copy.listingDirectoryUnavailable,\n"
        "    listing_directory_schema_mismatch: copy.listingDirectorySchemaMismatch,\n"
        "    listing_directory_stale: copy.listingDirectoryStale,\n"
        "    listing_status_unresolved: copy.listingStatusUnresolved,\n"
        "    listing_authority_conflict: copy.listingAuthorityConflict,\n"
        "    massive_credential_missing: copy.massiveCredentialMissing,\n"
        "    massive_access_denied: copy.massiveAccessDenied,\n"
        "    massive_rate_limited: copy.massiveRateLimited,\n"
        "    massive_reference_unavailable: copy.massiveReferenceUnavailable,\n",
        "",
        (
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_frontend_presentation_maps_every_v3_listing_blocker",
        ),
        (
            "pytest",
            "-q",
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_frontend_presentation_maps_every_v3_listing_blocker",
        ),
    ),
    Mutation(
        "M30",
        "replace the real shadow listing session with an inert object",
        PACKET_SHADOW,
        "    session = ListingAuthoritySession(\n",
        "    session = object()\n"
        "    if False:\n"
        "        session = ListingAuthoritySession(\n",
        (
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_shadow_executes_real_listing_session_transport_contract",
        ),
        (
            "pytest",
            "-q",
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_shadow_executes_real_listing_session_transport_contract",
        ),
    ),
    Mutation(
        "M31",
        "retain an open portfolio source in the terminal browser projection",
        PACKET_BROWSER,
        '    if projection["kind"] == "terminal_delisting":\n'
        "        preview.update(\n",
        '    if False and projection["kind"] == "terminal_delisting":\n'
        "        preview.update(\n",
        (
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_browser_terminal_projection_is_preflight_valid",
        ),
        (
            "pytest",
            "-q",
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_browser_terminal_projection_is_preflight_valid",
        ),
    ),
    Mutation(
        "M32",
        "bypass measured browser evidence-surface validation",
        PACKET_BROWSER,
        "    if actual != expected:\n"
        "        raise AssertionError(\n"
        '            "browser_evidence_surface_mismatch:"\n',
        "    if False and actual != expected:\n"
        "        raise AssertionError(\n"
        '            "browser_evidence_surface_mismatch:"\n',
        (
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_browser_evidence_surface_validator_fails_closed",
        ),
        (
            "pytest",
            "-q",
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_browser_evidence_surface_validator_fails_closed",
        ),
    ),
    Mutation(
        "M33",
        "bypass visible post-apply command-surface validation",
        PACKET_BROWSER,
        "    if actual != expected:\n"
        "        raise AssertionError(\n"
        '            "browser_post_apply_surface_mismatch:"\n',
        "    if False and actual != expected:\n"
        "        raise AssertionError(\n"
        '            "browser_post_apply_surface_mismatch:"\n',
        (
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_browser_post_apply_surface_validator_fails_closed",
        ),
        (
            "pytest",
            "-q",
            "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/"
            "test_packet_contracts.py::"
            "test_browser_post_apply_surface_validator_fails_closed",
        ),
    ),
)


FAILURE_SIGNATURES = {
    "M01": ("IndexError: pop from empty list",),
    "M02": ("('nasdaq_listed', 'inactive'",),
    "M03": ("DID NOT RAISE", "listing_directory_schema_mismatch"),
    "M04": ("DID NOT RAISE", "listing_directory_stale"),
    "M05": ("DID NOT RAISE", "listing_status_unresolved3"),
    "M06": ("&apiKey=secret-value",),
    "M07": ("DID NOT RAISE", "listing_status_unresolved0"),
    "M08": ("ValueError: fact_citation", "+ evidence_content_sha256"),
    "M09": ("DID NOT RAISE", "fact_citation"),
    "M10": ("listing_authority_conflict",),
    "M11": ("IBKR missing must not preview terminal action",),
    "M12": ("transition_eligible", "action_blocked"),
    "M13": ("publisher", "Differing attributes"),
    "M14": ("final.retryable", "assert True is False"),
    "M15": ("Extra items in the left set", "'publisher'"),
    "M16": ("listing translation reached cache lookup",),
    "M17": ('assert "massive" not in dpc.PROVIDER_FIELDS',),
    "M18": ("lifecycle foreign key mismatch",),
    "M19": ("lifecycle_rows_changed",),
    "M20": ("assert 'v3' == 'v2'",),
    "M21": ("Right contains 2 more items",),
    "M22": ("transition_eligible", "waiting_market_confirmation"),
    "M23": ("lifecycle massive key+/%",),
    "M24": ("case_processing_blocked",),
    "M25": ("case_processing_blocked",),
    "M26": ("case_processing_failed",),
    "M27": (
        "At index 0 diff: 'listing_status_unresolved' != "
        "'massive_reference_unavailable'",
    ),
    "M28": ("automation_scheduler_failed",),
    "M29": ("listing_directory_unavailable",),
    "M30": ("AttributeError", "object", "lookup"),
    "M31": ("portfolio_open",),
    "M32": ("DID NOT RAISE",),
    "M33": ("DID NOT RAISE",),
}
assert set(FAILURE_SIGNATURES) == {mutation.mutation_id for mutation in MUTATIONS}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"mutation_target_count:{path.relative_to(ROOT)}:{count}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _clear_python_bytecode(path: Path) -> None:
    cache = path.parent / "__pycache__"
    if not cache.is_dir():
        return
    for compiled in cache.glob(f"{path.stem}.*.pyc"):
        compiled.unlink()


def _pytest_failures(output: str) -> tuple[list[str], int, int]:
    failures = sorted(
        set(re.findall(r"^FAILED (\S+?)(?:\s+-|$)", output, flags=re.MULTILINE))
    )
    passed = re.search(r"(\d+) passed", output)
    skipped = re.search(r"(\d+) skipped", output)
    return (
        failures,
        int(passed.group(1)) if passed else 0,
        int(skipped.group(1)) if skipped else 0,
    )


def _normalize_output(output: str) -> str:
    normalized = output.replace(str(ROOT), "<REPO_ROOT>")
    if sys.prefix != sys.base_prefix:
        normalized = normalized.replace(str(Path(sys.prefix).resolve()), "<PYTHON_ENV>")
    return normalized


def _probe_nodes(output: str, marker: str) -> tuple[list[str], bool]:
    values = re.findall(rf"^{re.escape(marker)} (.+)$", output, flags=re.MULTILINE)
    return sorted(set(values)), len(values) == len(set(values))


def _run_declared_command(mutation: Mutation) -> dict:
    pythonpath = os.environ.get("PYTHONPATH", "")
    process = subprocess.run(
        mutation.command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": (
                str(PACKET) if not pythonpath else f"{PACKET}{os.pathsep}{pythonpath}"
            ),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = _normalize_output(process.stdout)
    failures, passed, skipped = _pytest_failures(output)
    collected, collection_markers_unique = _probe_nodes(
        output, "TASK8_COLLECTED_NODE"
    )
    executed, execution_markers_unique = _probe_nodes(
        output, "TASK8_EXECUTED_NODE"
    )
    return {
        "command": list(mutation.command),
        "exit_code": process.returncode,
        "collected_node_ids": collected,
        "executed_node_ids": executed,
        "failed_node_ids": failures,
        "passed_node_count": passed,
        "skipped_node_count": skipped,
        "collection_markers_unique": collection_markers_unique,
        "execution_markers_unique": execution_markers_unique,
        "output_sha256": _sha256(output.encode("utf-8")),
        "output_tail": output.splitlines()[-24:],
        "_output": output,
    }


def _exact_scope(result: dict, expected: list[str]) -> bool:
    return (
        result["collection_markers_unique"]
        and result["execution_markers_unique"]
        and result["collected_node_ids"] == expected
        and result["executed_node_ids"] == expected
    )


def _public_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "_output"}


def _run(mutation: Mutation) -> dict:
    path = ROOT / mutation.path
    original = path.read_bytes()
    error: str | None = None
    expected = sorted(mutation.owners)
    baseline: dict | None = None
    mutant: dict | None = None
    baseline_admitted = False
    mutation_applied = False
    try:
        _clear_python_bytecode(path)
        baseline = _run_declared_command(mutation)
        baseline_admitted = (
            baseline["exit_code"] == 0
            and baseline["failed_node_ids"] == []
            and _exact_scope(baseline, expected)
            and path.read_bytes() == original
        )
        if not baseline_admitted:
            raise RuntimeError("baseline_admission_failed")
        _replace_once(path, mutation.old, mutation.new)
        for old, new in mutation.extra_replacements:
            _replace_once(path, old, new)
        mutation_applied = True
        mutant = _run_declared_command(mutation)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    finally:
        path.write_bytes(original)
        _clear_python_bytecode(path)

    restored = path.read_bytes() == original
    failures = [] if mutant is None else mutant["failed_node_ids"]
    unexpected = sorted(set(failures) - set(expected))
    missing = sorted(set(expected) - set(failures))
    signature_matches = {
        signature: mutant is not None and signature in mutant["_output"]
        for signature in mutation.failure_signatures
    }
    commands_identical = (
        baseline is not None
        and mutant is not None
        and baseline["command"] == mutant["command"] == list(mutation.command)
    )
    killed = (
        baseline_admitted
        and mutation_applied
        and mutant is not None
        and mutant["exit_code"] == 1
        and _exact_scope(mutant, expected)
        and error is None
        and not unexpected
        and not missing
        and all(signature_matches.values())
        and commands_identical
        and restored
    )
    return {
        "id": mutation.mutation_id,
        "mutation": mutation.description,
        "product_files": [mutation.path],
        "declared_command": list(mutation.command),
        "declared_scope_node_ids": expected,
        "baseline": None if baseline is None else _public_result(baseline),
        "baseline_admitted": baseline_admitted,
        "mutation_applied": mutation_applied,
        "mutant": None if mutant is None else _public_result(mutant),
        "commands_identical": commands_identical,
        "expected_failed_node_ids": expected,
        "actual_failed_node_ids": failures,
        "unexpected_failures_inside_declared_scope": unexpected,
        "missing_expected_failures_inside_declared_scope": missing,
        "expected_failure_count": len(expected),
        "actual_failure_count": len(failures),
        "required_failure_signatures": list(mutation.failure_signatures),
        "failure_signature_matches": signature_matches,
        "all_required_failure_signatures_observed": all(signature_matches.values()),
        "runner_error": error,
        "killed": killed,
        "restored_files": [
            {
                "path": mutation.path,
                "before_sha256": _sha256(original),
                "after_sha256": _sha256(path.read_bytes()),
                "byte_identical": restored,
            }
        ],
    }


def _product_paths(mutations: Iterable[Mutation]) -> list[str]:
    return sorted({mutation.path for mutation in mutations})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    initial = {path: (ROOT / path).read_bytes() for path in _product_paths(MUTATIONS)}
    results = [_run(mutation) for mutation in MUTATIONS]
    final_restore = [
        {
            "path": path,
            "sha256": _sha256(initial[path]),
            "byte_identical": (ROOT / path).read_bytes() == initial[path],
        }
        for path in sorted(initial)
    ]
    scope_anomalies = [
        {
            "mutation_id": result["id"],
            "baseline_admitted": result["baseline_admitted"],
            "unexpected_failures_inside_declared_scope": result[
                "unexpected_failures_inside_declared_scope"
            ],
            "missing_expected_failures_inside_declared_scope": result[
                "missing_expected_failures_inside_declared_scope"
            ],
            "all_required_failure_signatures_observed": result[
                "all_required_failure_signatures_observed"
            ],
            "runner_error": result["runner_error"],
        }
        for result in results
        if not result["baseline_admitted"]
        or result["unexpected_failures_inside_declared_scope"]
        or result["missing_expected_failures_inside_declared_scope"]
        or not result["all_required_failure_signatures_observed"]
        or result["runner_error"]
    ]
    payload = {
        "schema_version": 2,
        "mutation_count": len(results),
        "killed_count": sum(result["killed"] for result in results),
        "all_mutations_killed": all(result["killed"] for result in results),
        "all_baselines_admitted": all(
            result["baseline_admitted"] for result in results
        ),
        "all_declared_commands_identical_between_baseline_and_mutant": all(
            result["commands_identical"] for result in results
        ),
        "unexpected_failures_inside_declared_mutation_scopes": sum(
            len(result["unexpected_failures_inside_declared_scope"])
            for result in results
        ),
        "all_product_files_restored_byte_identically": all(
            row["byte_identical"] for row in final_restore
        ),
        "mutation_scope_anomalies": scope_anomalies,
        "final_product_file_restore": final_restore,
        "mutations": results,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "mutations": payload["mutation_count"],
                "killed": payload["killed_count"],
                "restored": payload["all_product_files_restored_byte_identically"],
                "scope_anomalies": len(scope_anomalies),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["all_mutations_killed"] and payload[
        "all_product_files_restored_byte_identically"
    ] else 1


if __name__ == "__main__":
    raise SystemExit(main())
