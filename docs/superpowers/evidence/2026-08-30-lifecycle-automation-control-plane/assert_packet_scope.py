"""Fail unless the replay is bound to the expected product head and packet scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
PACKET_PREFIX = str(PACKET.relative_to(ROOT)) + "/"
BASE = "947a51fca2f078e750bef64cad4817682141ea8f"
PRIORITY_MAP = "docs/design/PROJECT_PRIORITY_MAP.md"


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def paths(*args: str) -> set[str]:
    return {line for line in git(*args).splitlines() if line}


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=ROOT,
        check=False,
    ).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    head = git("rev-parse", "HEAD").strip()
    if not is_ancestor(BASE, args.expected_head):
        raise AssertionError("product_head_not_based_on_authority")
    if not is_ancestor(args.expected_head, head):
        raise AssertionError(f"product_head_not_ancestor:{head}")
    committed_after_product = paths(
        "diff", "--name-only", f"{args.expected_head}..{head}"
    )
    outside_committed = sorted(
        path
        for path in committed_after_product
        if not path.startswith(PACKET_PREFIX)
    )
    if outside_committed:
        raise AssertionError("post_product_scope:" + ",".join(outside_committed))
    if paths(
        "diff", "--name-only", f"{BASE}..{args.expected_head}", "--", PRIORITY_MAP
    ):
        raise AssertionError("priority_map_branch_drift")
    changed = (
        paths("diff", "--name-only")
        | paths("diff", "--cached", "--name-only")
        | paths("ls-files", "--others", "--exclude-standard")
    )
    outside = sorted(path for path in changed if not path.startswith(PACKET_PREFIX))
    if outside:
        raise AssertionError("packet_scope:" + ",".join(outside))
    print(json.dumps({
        "product_head": args.expected_head,
        "replay_head": head,
        "committed_packet_paths": len(committed_after_product),
        "packet_worktree_paths": len(changed),
        "priority_map_matches_base": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
