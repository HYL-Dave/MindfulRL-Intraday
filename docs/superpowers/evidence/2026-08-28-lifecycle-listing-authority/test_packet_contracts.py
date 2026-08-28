"""Contract tests for the Task 8 admission packet itself."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


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

    assert len(mutations.MUTATIONS) == 20
    for mutation in mutations.MUTATIONS:
        assert mutation.failure_signatures
        assert mutation.command[:4] == (
            "pytest",
            "-p",
            "mutation_pytest_probe",
            "-vv",
        )


def test_old_code_path_resolution_follows_file_uri_symlinks(tmp_path: Path) -> None:
    old_code = _load("verify_old_code")
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.db"
    link = allowed / "escape.db"
    link.symlink_to(outside)

    assert old_code._resolve_database_path(f"file:{link}?mode=rw") == outside.resolve()
    assert old_code._is_within(outside.resolve(), allowed.resolve()) is False


def test_browser_applied_projections_have_transition_and_reverse_witnesses() -> None:
    browser = _load("run_browser_matrix")

    conflict = browser.SCENARIOS["conflict-attention"]
    assert {row["listing_status"] for row in conflict["listings"]} == {"active"}
    assert len({row["issuer_cik"] for row in conflict["listings"]}) == 2

    for name in ("inactive-history", "otc-continuation"):
        scenario = browser.SCENARIOS[name]
        detail = browser._detail(name)
        transition = detail["ticker_transition"]
        assert scenario["synthetic_post_apply_projection"] is True
        assert transition["status"] == "applied"
        assert transition["activity_history"]
        assert transition["reverse_readiness"]["reversible"] is True

    assert browser.DECLARED_ZERO == {
        "value": 0,
        "basis": "declared_not_authorized",
    }


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
