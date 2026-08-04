from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
from collections import Counter
from pathlib import Path


EXPECTED_CENSUS_ROWS = 128
EXPECTED_CENSUS_SHA256 = "a08e7f683b426c10090f1cb7f6e4f4104f22678147a46639ceaece0bcb088c64"
EXPECTED_BEHAVIOR_SHA256 = "613024acc6568296cb798a2832fb8ca1e67fba05a9857c4e4bd5629755c556ba"
TASK8_AUTHORITY_PATHS = frozenset(
    {
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/README.md",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/cache-classification.tsv",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/consumer-census.tsv",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/controller_probe.py",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/db-result.json",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/destructive_controller.py",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/legacy-price-files.tsv",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/old-cache-rows.tsv",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/task8_db_row_manifest.py",
        "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/task8_price_manifest.py",
    }
)
BEHAVIOR_ROWS = (
    (
        "src/agents/anthropic_agent/tools.py",
        "execute_tool.<lambda>",
        "get_fundamentals_analysis",
        "tests/test_agents.py::TestAnthropicToolSchemas::test_tool_names,tests/test_tools.py::TestAnalysisTools::test_get_fundamentals_analysis",
        "bridge remains reachable; callee adopts current annual-analysis truth",
    ),
    (
        "src/agents/openai_agent/tools.py",
        "create_openai_tools.tool_get_fundamentals_analysis",
        "get_fundamentals_analysis",
        "tests/test_agents.py::TestOpenAIToolCreation::test_tools_have_names,tests/test_tools.py::TestAnalysisTools::test_get_fundamentals_analysis",
        "bridge remains reachable; callee adopts current annual-analysis truth",
    ),
    (
        "src/api/routes/fundamentals.py",
        "fundamentals",
        "get_fundamentals_analysis",
        "tests/test_api.py::TestFundamentalsEndpoints::test_fundamentals",
        "default route preserves typed current annual-analysis response",
    ),
    (
        "src/evidence_packet.py",
        "gather_evidence",
        "get_fundamentals_analysis",
        "tests/test_evidence_packet.py::test_one_failing_source_degrades_to_coverage,tests/test_evidence_packet.py::test_packet_has_expected_sources_and_tags",
        "institutional evidence preserves success and degraded coverage behavior",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return _sha256(data)


def _safe(value: str) -> str:
    if any(separator in value for separator in ("\t", "\n", "\r")):
        raise ValueError("unsafe census field")
    return value


def _run(repo_root: Path, output_root: Path) -> None:
    repo_root = repo_root.resolve(strict=True)
    if output_root.exists():
        raise FileExistsError(f"single-use output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    source_path = Path(__file__).resolve(strict=True)
    test_path = repo_root / "tests" / "test_eir006_retired_data_boundaries.py"
    namespace = runpy.run_path(str(test_path))

    initial_discovery = namespace["_discover_consumers"]()
    observed_task8_authorities = {
        path
        for path, _ in initial_discovery
        if path.startswith(
            "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/"
        )
    }
    if observed_task8_authorities != TASK8_AUTHORITY_PATHS:
        raise AssertionError(
            "Task 8 self-authority paths changed: "
            f"{sorted(observed_task8_authorities)}"
        )
    namespace["_EIR006_AUTHORITIES"].update(TASK8_AUTHORITY_PATHS)

    namespace["test_current_docs_training_and_tool_copy_name_only_current_authorities"]()
    namespace["test_current_runtime_consumer_census_is_closed_and_exact"]()
    discovered = [
        row
        for row in initial_discovery
        if row[0] not in namespace["_EIR006_AUTHORITIES"]
    ]
    classified = [
        (path, match, namespace["_verdict"](path))
        for path, match in discovered
    ]
    classified.sort(key=lambda row: tuple(value.encode("utf-8") for value in row))
    if len(classified) != len(set(classified)):
        raise AssertionError("consumer census contains duplicate rows")
    census_bytes = (
        "\n".join("\t".join(_safe(value) for value in row) for row in classified) + "\n"
    ).encode("utf-8")
    census_sha = _write(output_root / "consumer-census.tsv", census_bytes)
    if (len(classified), census_sha) != (EXPECTED_CENSUS_ROWS, EXPECTED_CENSUS_SHA256):
        raise AssertionError(f"consumer census drifted: rows={len(classified)} sha256={census_sha}")

    behavior_bytes = (
        "\n".join("\t".join(_safe(value) for value in row) for row in BEHAVIOR_ROWS) + "\n"
    ).encode("utf-8")
    behavior_sha = _write(output_root / "behavior-propagation.tsv", behavior_bytes)
    if behavior_sha != EXPECTED_BEHAVIOR_SHA256:
        raise AssertionError(f"behavior ledger identity drifted: {behavior_sha}")
    for caller_path, _, callee, owner_nodes, _ in BEHAVIOR_ROWS:
        caller_text = (repo_root / caller_path).read_text(encoding="utf-8")
        if callee not in caller_text:
            raise AssertionError(f"callee absent from caller: {caller_path}")
        for owner_node in owner_nodes.split(","):
            owner_path = repo_root / owner_node.split("::", 1)[0]
            if not owner_path.is_file():
                raise AssertionError(f"owner test file absent: {owner_path}")

    verdict_counts = Counter(row[2] for row in classified)
    result = {
        "schema_version": 1,
        "repo_head": subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_identity": {
            "path": str(source_path),
            "lines": len(source_path.read_bytes().splitlines()),
            "bytes": source_path.stat().st_size,
            "sha256": _sha256(source_path.read_bytes()),
        },
        "census_owner_identity": {
            "path": "tests/test_eir006_retired_data_boundaries.py",
            "bytes": test_path.stat().st_size,
            "sha256": _sha256(test_path.read_bytes()),
        },
        "consumer_census": {
            "path": "consumer-census.tsv",
            "rows": len(classified),
            "sha256": census_sha,
            "verdict_counts": dict(sorted(verdict_counts.items())),
        },
        "behavior_propagation": {
            "path": "behavior-propagation.tsv",
            "rows": len(BEHAVIOR_ROWS),
            "sha256": behavior_sha,
        },
        "old_current_consumers": 0,
        "old_current_writers": 0,
        "unknown_verdicts": 0,
    }
    result_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    result_sha = _write(output_root / "result.json", result_bytes)
    identities = {
        "behavior-propagation.tsv": behavior_sha,
        "consumer-census.tsv": census_sha,
        "result.json": result_sha,
    }
    sums = "\n".join(f"{digest}  {name}" for name, digest in sorted(identities.items())) + "\n"
    _write(output_root / "SHA256SUMS", sums.encode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    _run(arguments.repo_root, arguments.output_root)


if __name__ == "__main__":
    main()
