"""Reproduce the Stage 4 transition lifecycle against disposable databases."""

from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


AT = "2026-08-25T13:00:00Z"
ACK_AT = "2026-08-26T08:00:00Z"
REVERSE_AT = "2026-08-26T14:00:00Z"
SOURCE_REF = "0000000000-26-000001"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

USER_OWNED_TABLES = (
    "ticker_meta",
    "ticker_tags",
    "universe_source_memberships",
    "watchlist_memberships",
)
PROFILE_HISTORY_TABLES = (
    "security_lifecycle_cases",
    "security_lifecycle_automation_runs",
    "security_lifecycle_automation_run_blockers",
    "security_lifecycle_evidence",
    "security_lifecycle_automation_facts",
    "security_lifecycle_assessments",
    "security_lifecycle_assessment_outcomes",
    "security_lifecycle_assessment_evidence",
    "security_lifecycle_action_proposals",
)
MARKET_HISTORY_TABLES = (
    "security_lifecycle_observations",
    "security_lifecycle_observation_kinds",
)


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _table_rows(conn: sqlite3.Connection, tables: tuple[str, ...]) -> dict:
    return {
        table: [
            list(row)
            for row in conn.execute(
                f'SELECT * FROM "{table}" ORDER BY rowid'
            ).fetchall()
        ]
        for table in tables
    }


def _database_rows(path: Path, tables: tuple[str, ...]) -> dict:
    with sqlite3.connect(path) as conn:
        return _table_rows(conn, tables)


def _id_factory():
    counters: dict[str, int] = {}

    def generate(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}_{counters[prefix]}"

    return generate


def _seed_databases(root: Path) -> tuple[Path, Path, str]:
    from src.profile_state import ProfileStateStore
    from src.security_lifecycle import (
        LifecycleObservation,
        ObservationKind,
        SecurityLifecycleStore,
    )
    from src.security_lifecycle_investigation import (
        SecurityLifecycleInvestigationStore,
        case_id_for,
    )
    from src.ticker_identity_schema import create_ticker_identity_schema

    root.mkdir(parents=True, exist_ok=True)
    market_path = root / "market_data.db"
    profile_path = root / "profile_state.db"
    for path in (market_path, profile_path):
        if PROJECT_ROOT in path.resolve().parents:
            raise AssertionError("scratch_path_inside_project")

    ProfileStateStore(profile_path)
    with sqlite3.connect(profile_path) as conn:
        SecurityLifecycleInvestigationStore(conn)
        create_ticker_identity_schema(conn)
        conn.execute(
            "INSERT INTO watchlists "
            "(id,name,kind,position,archived_at,created_at,updated_at) "
            "VALUES (1,'Core','custom',0,NULL,?,?)",
            (AT, AT),
        )
        conn.execute(
            "INSERT INTO watchlist_memberships "
            "(list_id,ticker,position,archived_at,created_at,updated_at) "
            "VALUES (1,'OLD',0,NULL,?,?)",
            (AT, AT),
        )
        conn.execute(
            "INSERT INTO universe_source_memberships "
            "(source_key,ticker,created_at,archived_at) VALUES (?,?,?,NULL)",
            ("legacy_config_seed", "OLD", AT),
        )
        conn.executemany(
            "INSERT INTO ticker_tags (ticker,facet,value,source,created_at) "
            "VALUES (?,?,?,?,?)",
            [
                ("OLD", "theme", "AI", "user", AT),
                ("OLD", "category", "Core", "legacy", AT),
                (
                    "OLD",
                    "sector",
                    "Technology",
                    "provider:fundamentals",
                    AT,
                ),
            ],
        )
        conn.execute(
            "INSERT INTO ticker_meta (ticker,priority,hidden_at,updated_at) "
            "VALUES ('OLD','high',NULL,?)",
            (AT,),
        )
        conn.commit()

    with sqlite3.connect(market_path) as conn:
        store = SecurityLifecycleStore(conn)
        store.upsert_observation(
            LifecycleObservation(
                ticker="OLD",
                cik="0000000000",
                issuer_name="Old Issuer Inc.",
                filing_date="2026-08-22",
                source="sec_edgar",
                source_ref=SOURCE_REF,
                filing_form="8-K",
                filing_items=("3.01",),
                evidence_url="https://www.sec.gov/Archives/example/old-8k.htm",
                description="Issuer reports a listing identity transition.",
                observed_at=AT,
                kinds=(
                    ObservationKind("listing_status_review", "2026-08-25"),
                ),
            )
        )
    return (
        market_path,
        profile_path,
        case_id_for("sec_edgar", SOURCE_REF, "OLD"),
    )


def _evidence(case: Mapping[str, object], *, family: str, payload: dict):
    from src.security_lifecycle_fact_kernel import AutomationEvidence

    excerpt = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return AutomationEvidence(
        evidence_id=f"{family}-{str(case['case_id'])[-8:]}",
        source_family=family,
        adapter="sec_edgar" if family == "regulator" else "ibkr_contract",
        kind=(
            "regulator_excerpt"
            if family == "regulator"
            else "market_infrastructure_snapshot"
        ),
        source_url=(
            str(case["observation"]["evidence_url"])
            if family == "regulator"
            else None
        ),
        title=f"{family} evidence",
        publisher="SEC EDGAR" if family == "regulator" else "Interactive Brokers",
        domain="sec.gov" if family == "regulator" else None,
        source_published_at="2026-08-22" if family == "regulator" else None,
        retrieved_at=AT,
        excerpt=excerpt,
        content_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        source_document_sha256="d" * 64 if family == "regulator" else None,
        source_locator=(
            {"filing_chain_complete": True}
            if family == "regulator"
            else {"snapshot": payload}
        ),
        evidence_dedupe_key=f"{family}:{case['case_id']}",
    )


def _fact(evidence: object, payload: dict, key: str):
    from src.security_lifecycle_fact_kernel import AutomationFact

    token = json.dumps(
        payload[key],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    excerpt = str(getattr(evidence, "excerpt")).encode("utf-8")
    start = excerpt.index(token)
    return AutomationFact(
        evidence_id=str(getattr(evidence, "evidence_id")),
        fact_type=key,
        normalized_value=payload[key],
        source_span_start=start,
        source_span_end=start + len(token),
        cited_text_sha256=hashlib.sha256(token).hexdigest(),
        extractor_rule_id=f"scratch.{key}",
        extractor_rule_version="1",
    )


def _bundle(case: Mapping[str, object]):
    from src.security_lifecycle_automation_worker import (
        LifecycleAutomationEvidenceBundle,
    )

    regulator_payload = {
        "destination_venue": "NASDAQ",
        "effective_date": "2026-08-25",
        "issuer_cik": "0000000000",
        "security_class": "common_stock",
        "source_ticker": "OLD",
        "source_venue": "NYSE",
        "successor_ticker": "NEW",
    }
    market_payload = {
        "destination_venue": "NASDAQ",
        "security_class": "common_stock",
        "successor_ticker": "NEW",
    }
    regulator = _evidence(case, family="regulator", payload=regulator_payload)
    market = _evidence(
        case,
        family="market_infrastructure",
        payload=market_payload,
    )
    return LifecycleAutomationEvidenceBundle(
        evidence=(regulator, market),
        facts=(
            *(_fact(regulator, regulator_payload, key) for key in regulator_payload),
            *(_fact(market, market_payload, key) for key in market_payload),
        ),
        blockers=(),
        diagnostics={"ibkr_requests": 0, "sec_attempts": 0},
        retry_at=None,
    )


@contextmanager
def _connection(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def _build_runtime(root: Path):
    import src.security_lifecycle_investigation as investigation_module
    from src.security_lifecycle_automation_worker import LifecycleAutomationWorker
    from src.ticker_identity_service import TickerIdentityService
    from src.ticker_identity_transition import (
        build_automation_transition_preflight,
    )
    from src.tools.security_lifecycle_tools import SecurityLifecycleReadService

    uuid_counter = {"value": 0}

    def deterministic_uuid4() -> uuid.UUID:
        uuid_counter["value"] += 1
        return uuid.UUID(int=uuid_counter["value"])

    investigation_module.uuid.uuid4 = deterministic_uuid4
    market_path, profile_path, case_id = _seed_databases(root)
    sources = {
        "OLD": (
            "legacy_config_seed",
            "manual_lists",
            "sa_alpha_picks_current",
        )
    }
    current_time = {"value": AT}
    service = TickerIdentityService(
        market_db_path=str(market_path),
        profile_db_path=str(profile_path),
        source_loader=lambda: sources,
        clock=lambda: current_time["value"],
        id_factory=_id_factory(),
    )
    read_service = SecurityLifecycleReadService(
        market_db_path=str(market_path),
        profile_db_path=str(profile_path),
        source_loader=lambda: sources,
    )
    case = read_service.get_case(case_id)
    provider_calls = {"count": 0}

    def evidence_loader(loaded_case, *, mode: str, at: str):
        if mode != "historical" or at != current_time["value"]:
            raise AssertionError("scratch_worker_context")
        return _bundle(loaded_case)

    def transition_preview(*, case, request, sources):
        with sqlite3.connect(profile_path) as conn:
            return build_automation_transition_preflight(
                conn,
                case=case,
                request=request,
                sources=sources,
            )

    def transition_approver(*, case, request, sources):
        if tuple(sources) != tuple(sorted(sources)):
            raise AssertionError("source_order")
        return service.approve_automation_case(
            str(case["case_id"]),
            request=request,
        )

    worker = LifecycleAutomationWorker(
        case_loader=lambda: [case],
        profile_connection=lambda: _connection(profile_path),
        evidence_loader=evidence_loader,
        source_loader=lambda: sources,
        transition_preview=transition_preview,
        transition_approver=transition_approver,
        clock=lambda: current_time["value"],
    )
    return {
        "case_id": case_id,
        "current_time": current_time,
        "market_path": market_path,
        "profile_path": profile_path,
        "provider_calls": provider_calls,
        "service": service,
        "worker": worker,
    }


def _run_scheduler(service: object) -> tuple[dict, list[tuple[str, dict]]]:
    import src.service.ticker_identity_scheduler as scheduler

    permission_calls: list[tuple[str, dict]] = []
    scheduler._service = lambda: service
    scheduler.require_profile_state_write = (
        lambda action, detail: permission_calls.append((action, dict(detail)))
    )
    result = scheduler.run_due_ticker_identity_transitions(
        now=datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc),
    )
    return result, permission_calls


def _prepare_applied(root: Path) -> dict:
    runtime = _build_runtime(root)
    worker_result = runtime["worker"].run(limit=1, mode="historical")
    if worker_result != {
        "case_ids": [runtime["case_id"]],
        "selected": 1,
        "processed": 1,
        "accepted": 1,
        "drafted": 0,
        "blocked": 0,
        "failed": 0,
        "skipped_current": 0,
    }:
        raise AssertionError("automation_worker_result")

    profile_path = runtime["profile_path"]
    market_path = runtime["market_path"]
    before_user_rows = _database_rows(profile_path, USER_OWNED_TABLES)
    before_history = {
        "market": _database_rows(market_path, MARKET_HISTORY_TABLES),
        "profile": _database_rows(profile_path, PROFILE_HISTORY_TABLES),
    }
    scheduler_result, permission_calls = _run_scheduler(runtime["service"])
    if scheduler_result["status"] != "succeeded" or scheduler_result["applied"] != 1:
        raise AssertionError("due_scheduler_apply")
    if len(permission_calls) != 1:
        raise AssertionError("scheduler_permission")
    transition_id = str(scheduler_result["transition_ids"][0])
    return {
        **runtime,
        "before_history": before_history,
        "before_user_rows": before_user_rows,
        "permission_calls": permission_calls,
        "scheduler_result": scheduler_result,
        "transition_id": transition_id,
        "worker_result": worker_result,
    }


def _happy_path(root: Path) -> dict:
    runtime = _prepare_applied(root)
    service = runtime["service"]
    initial = service.list_transition_activity(limit=10)
    if initial["count"] != 1 or initial["unacknowledged_count"] != 1:
        raise AssertionError("initial_activity_count")
    applied_activity = initial["items"][0]
    if applied_activity["acknowledged_at"] is not None:
        raise AssertionError("render_side_acknowledgement")

    runtime["current_time"]["value"] = ACK_AT
    acknowledged = service.acknowledge_transition_activity(
        str(applied_activity["activity_id"]),
        before_write=lambda: None,
    )
    if acknowledged["acknowledged_at"] != ACK_AT:
        raise AssertionError("explicit_acknowledgement")

    runtime["current_time"]["value"] = REVERSE_AT
    reversed_result = service.reverse_transition(
        runtime["transition_id"],
        before_write=lambda: None,
    )
    if reversed_result["status"] != "reversed":
        raise AssertionError("reverse_status")

    after_user_rows = _database_rows(runtime["profile_path"], USER_OWNED_TABLES)
    after_history = {
        "market": _database_rows(runtime["market_path"], MARKET_HISTORY_TABLES),
        "profile": _database_rows(
            runtime["profile_path"],
            PROFILE_HISTORY_TABLES,
        ),
    }
    activity = service.list_transition_activity(limit=10)
    if [item["activity_type"] for item in activity["items"]] != [
        "reversed",
        "applied",
    ]:
        raise AssertionError("activity_history")
    if activity["items"][1]["acknowledged_at"] != ACK_AT:
        raise AssertionError("acknowledgement_history")

    before_user_sha = _sha256(runtime["before_user_rows"])
    after_user_sha = _sha256(after_user_rows)
    before_history_sha = _sha256(runtime["before_history"])
    after_history_sha = _sha256(after_history)
    if before_user_sha != after_user_sha:
        raise AssertionError("user_state_not_restored")
    if before_history_sha != after_history_sha:
        raise AssertionError("provider_history_changed")

    return {
        "activity": {
            "acknowledged_applied_survives": True,
            "activity_count_after_apply": int(initial["count"]),
            "activity_count_after_reverse": int(activity["count"]),
            "activity_types_after_reverse": [
                item["activity_type"] for item in activity["items"]
            ],
            "initial_acknowledged_at_is_null": True,
            "unacknowledged_count_after_reverse": int(
                activity["unacknowledged_count"]
            ),
        },
        "case_id": runtime["case_id"],
        "history_rows_after_reverse_sha256": after_history_sha,
        "history_rows_before_apply_sha256": before_history_sha,
        "history_rows_unchanged": True,
        "permission_action_count": len(runtime["permission_calls"]),
        "scheduler": {
            key: runtime["scheduler_result"][key]
            for key in (
                "status",
                "due",
                "applied",
                "needs_review",
                "already_applied",
            )
        },
        "transition_id": runtime["transition_id"],
        "transition_status_after_reverse": str(reversed_result["status"]),
        "user_owned_rows_after_reverse_sha256": after_user_sha,
        "user_owned_rows_before_apply_sha256": before_user_sha,
        "user_owned_rows_exactly_restored": True,
        "worker": {
            key: runtime["worker_result"][key]
            for key in (
                "selected",
                "processed",
                "accepted",
                "drafted",
                "blocked",
                "failed",
            )
        },
    }


def _drift_path(root: Path) -> dict:
    runtime = _prepare_applied(root)
    with sqlite3.connect(runtime["profile_path"]) as conn:
        conn.execute(
            "UPDATE ticker_meta SET priority='low',updated_at=? "
            "WHERE ticker='NEW'",
            ("2026-08-26T13:59:59Z",),
        )
        conn.commit()
        edited_rows = _table_rows(conn, USER_OWNED_TABLES)

    runtime["current_time"]["value"] = REVERSE_AT
    result = runtime["service"].reverse_transition(
        runtime["transition_id"],
        before_write=lambda: None,
    )
    after_rows = _database_rows(runtime["profile_path"], USER_OWNED_TABLES)
    with sqlite3.connect(runtime["profile_path"]) as conn:
        priority = conn.execute(
            "SELECT priority FROM ticker_meta WHERE ticker='NEW'"
        ).fetchone()
        status = conn.execute(
            "SELECT status FROM ticker_identity_transitions "
            "WHERE transition_id=?",
            (runtime["transition_id"],),
        ).fetchone()
    if result.get("block_reasons") != ["reverse_state_changed"]:
        raise AssertionError("reverse_drift_reason")
    if edited_rows != after_rows or priority != ("low",) or status != ("applied",):
        raise AssertionError("reverse_drift_overwrite")
    return {
        "block_reasons": list(result["block_reasons"]),
        "later_edit_preserved": True,
        "post_attempt_user_rows_sha256": _sha256(after_rows),
        "pre_attempt_user_rows_sha256": _sha256(edited_rows),
        "transition_id": runtime["transition_id"],
        "transition_status_after_block": "applied",
    }


def main() -> None:
    network_attempts = {"count": 0}

    def deny_network(*_args, **_kwargs):
        network_attempts["count"] += 1
        raise AssertionError("network_not_authorized")

    socket.getaddrinfo = deny_network
    socket.create_connection = deny_network

    with tempfile.TemporaryDirectory(prefix="arkscope-stage4-scratch-") as raw:
        root = Path(raw)
        happy = _happy_path(root / "happy")
        drift = _drift_path(root / "drift")
        scratch_database_count = len(list(root.rglob("*.db")))
    if network_attempts["count"] != 0:
        raise AssertionError("network_attempted")
    if root.exists():
        raise AssertionError("scratch_cleanup")

    report = {
        "authority": {
            "network_attempts": 0,
            "production_database_operations": 0,
            "provider_calls": 0,
            "scope": "scratch_only",
        },
        "drift_path": drift,
        "happy_path": happy,
        "report_schema_version": 1,
        "scratch_cleanup_complete": True,
        "scratch_database_count": scratch_database_count,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
