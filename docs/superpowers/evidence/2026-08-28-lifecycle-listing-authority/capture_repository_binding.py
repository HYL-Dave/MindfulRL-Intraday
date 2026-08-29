"""Bind one packet replay to a clean Git tree and packet-only output changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


class RepositoryBindingError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RepositoryBindingError(f"repository_git_failed:{args[0]}")
    return result.stdout.strip()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _changed_paths(root: Path) -> set[str]:
    changed: set[str] = set()
    for args in (
        ("diff", "--name-only", "--no-renames", "HEAD"),
        ("diff", "--cached", "--name-only", "--no-renames", "HEAD"),
        ("ls-files", "--others", "--exclude-standard"),
    ):
        changed.update(line for line in _git(root, *args).splitlines() if line)
    return changed


def _allowed_packet_paths(root: Path, packet_relative: str) -> set[str]:
    packet = root / packet_relative
    namespace = runpy.run_path(str(packet / "seal_packet.py"))
    names = set(namespace["GENERATED"]) | set(namespace["SCREENSHOTS"])
    names.add("SHA256SUMS")
    return {f"{packet_relative}/{name}" for name in names}


def _dependency(root: Path, head: str, relative: str) -> dict[str, str]:
    path = root / relative
    if not path.is_file():
        raise RepositoryBindingError(f"repository_dependency_missing:{relative}")
    blob = _git(root, "rev-parse", f"{head}:{relative}")
    committed = subprocess.run(
        ["git", "show", f"{head}:{relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if committed.returncode != 0 or committed.stdout != path.read_bytes():
        raise RepositoryBindingError(f"repository_dependency_changed:{relative}")
    return {
        "path": relative,
        "git_blob": blob,
        "sha256": hashlib.sha256(committed.stdout).hexdigest(),
    }


def _start(root: Path, output: Path) -> None:
    if _changed_paths(root):
        raise RepositoryBindingError("repository_replay_start_not_clean")
    head = _git(root, "rev-parse", "HEAD")
    _write(
        output,
        {
            "schema_version": 1,
            "tested_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "tested_git_head": head,
            "tested_git_tree": _git(root, "rev-parse", f"{head}^{{tree}}"),
            "start_worktree_clean": True,
        },
    )


def _finish(
    root: Path,
    packet_relative: str,
    start_path: Path,
    output: Path,
    dependencies: list[str],
) -> None:
    start = json.loads(start_path.read_text(encoding="utf-8"))
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", f"{head}^{{tree}}")
    if head != start.get("tested_git_head") or tree != start.get("tested_git_tree"):
        raise RepositoryBindingError("repository_replay_head_changed")
    changed = _changed_paths(root)
    allowed = _allowed_packet_paths(root, packet_relative)
    unexpected = sorted(changed - allowed)
    if unexpected:
        raise RepositoryBindingError(
            "repository_replay_scope_changed:" + ",".join(unexpected)
        )
    output_relative = str(output.resolve().relative_to(root.resolve()))
    if output_relative not in allowed:
        raise RepositoryBindingError("repository_binding_output_not_allowlisted")
    dependency_rows = [
        _dependency(root, head, relative) for relative in sorted(set(dependencies))
    ]
    _write(
        output,
        {
            **start,
            "replay_head_unchanged": True,
            "product_code_modified_during_replay": False,
            "allowed_packet_changes_only": True,
            "observed_packet_change_paths": sorted(changed | {output_relative}),
            "dependency_paths": [row["path"] for row in dependency_rows],
            "dependencies": dependency_rows,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--root", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    finish = subparsers.add_parser("finish")
    finish.add_argument("--root", type=Path, required=True)
    finish.add_argument("--packet-relative", required=True)
    finish.add_argument("--start", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    finish.add_argument("--dependency", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.command == "start":
            _start(args.root.resolve(), args.output.resolve())
        else:
            _finish(
                args.root.resolve(),
                args.packet_relative.strip("/"),
                args.start.resolve(),
                args.output.resolve(),
                args.dependency,
            )
    except (OSError, ValueError, json.JSONDecodeError, RepositoryBindingError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
