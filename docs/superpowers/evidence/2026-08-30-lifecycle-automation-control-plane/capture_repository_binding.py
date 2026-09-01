"""Capture the exact repository and no-DDL-drift binding for this packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
BASE_COMMIT = "947a51fca2f078e750bef64cad4817682141ea8f"
PRODUCT_HEAD = "65c5fa65bb34857e945437a15bf3660d56741232"
PACKET_PREFIX = str(PACKET.relative_to(ROOT)) + "/"
SCHEMA_AUTHORITIES = (
    "src/security_lifecycle_schema.py",
    "src/ticker_identity_schema.py",
)
BROWSER_FIXTURE_AUTHORITIES = (
    "docs/superpowers/evidence/2026-08-28-lifecycle-listing-authority/run_browser_matrix.py",
    "docs/superpowers/evidence/2026-08-26-lifecycle-automated-disposition/run_browser_matrix.py",
    "docs/superpowers/evidence/2026-08-25-trusted-lifecycle-automation-stage-5-repair/run_browser_matrix.py",
)


def git(*args: str, text: bool = True):
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    ).stdout


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    head = git("rev-parse", "HEAD").strip()
    base = BASE_COMMIT
    post_product_paths = tuple(
        line
        for line in git(
            "diff", "--name-only", f"{PRODUCT_HEAD}..{head}"
        ).splitlines()
        if line
    )
    changed = tuple(
        line
        for line in git(
            "diff", "--name-only", f"{base}..{PRODUCT_HEAD}"
        ).splitlines()
        if line
    )
    current_hashes = {
        path: sha256((ROOT / path).read_bytes())
        for path in changed
        if (ROOT / path).is_file()
    }
    schema = {}
    for path in SCHEMA_AUTHORITIES:
        base_bytes = git("show", f"{base}:{path}", text=False)
        current_bytes = (ROOT / path).read_bytes()
        schema[path] = {
            "base_sha256": sha256(base_bytes),
            "head_sha256": sha256(current_bytes),
            "byte_identical": base_bytes == current_bytes,
        }
    browser_fixtures = {}
    for path in BROWSER_FIXTURE_AUTHORITIES:
        head_bytes = git("show", f"{PRODUCT_HEAD}:{path}", text=False)
        current_bytes = (ROOT / path).read_bytes()
        browser_fixtures[path] = {
            "git_blob": git("rev-parse", f"{head}:{path}").strip(),
            "sha256": sha256(current_bytes),
            "matches_tested_head": current_bytes == head_bytes,
        }
    payload = {
        "schema_version": 1,
        "base_commit": base,
        "head_commit": head,
        "product_head_commit": PRODUCT_HEAD,
        "branch": git("branch", "--show-current").strip(),
        "base_is_ancestor": subprocess.run(
            ("git", "merge-base", "--is-ancestor", base, "HEAD"),
            cwd=ROOT,
            check=False,
        ).returncode == 0,
        "product_head_is_ancestor": subprocess.run(
            ("git", "merge-base", "--is-ancestor", PRODUCT_HEAD, "HEAD"),
            cwd=ROOT,
            check=False,
        ).returncode == 0,
        "post_product_paths": list(post_product_paths),
        "post_product_scope_only_packet": all(
            path.startswith(PACKET_PREFIX) for path in post_product_paths
        ),
        "merge_commits_since_base": git(
            "rev-list", "--merges", f"{base}..{head}"
        ).splitlines(),
        "changed_paths": list(changed),
        "changed_path_sha256": current_hashes,
        "schema_authorities": schema,
        "all_schema_authorities_byte_identical": all(
            row["byte_identical"] for row in schema.values()
        ),
        "browser_fixture_authorities": browser_fixtures,
        "all_browser_fixture_authorities_match_tested_head": all(
            row["matches_tested_head"] for row in browser_fixtures.values()
        ),
    }
    (PACKET / "repository-binding.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "base": base,
        "head": head,
        "product_head": PRODUCT_HEAD,
        "changed_paths": len(changed),
        "schema_unchanged": payload["all_schema_authorities_byte_identical"],
    }, sort_keys=True))
    return 0 if (
        payload["base_is_ancestor"]
        and payload["product_head_is_ancestor"]
        and payload["post_product_scope_only_packet"]
        and not payload["merge_commits_since_base"]
        and payload["all_schema_authorities_byte_identical"]
        and payload["all_browser_fixture_authorities_match_tested_head"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
