"""Compare two captured lifecycle-owned profile schema authorities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--startup-ddl-changes", type=int, required=True)
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    head = json.loads(Path(args.head).read_text(encoding="utf-8"))
    object_equal = base["objects"] == head["objects"]
    table_info_equal = base["table_info"] == head["table_info"]
    prohibited = sorted(
        set(base["prohibited_projection_columns"])
        | set(head["prohibited_projection_columns"])
    )
    report = {
        "sqlite_master_owned_object_diff": "empty" if object_equal else "changed",
        "pragma_table_info_diff": "empty" if table_info_equal else "changed",
        "prohibited_projection_columns": prohibited,
        "prohibited_projection_column_count": len(prohibited),
        "startup_ddl_changes": args.startup_ddl_changes,
        "base_owned_object_count": len(base["objects"]),
        "head_owned_object_count": len(head["objects"]),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not object_equal or not table_info_equal or prohibited or args.startup_ddl_changes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
