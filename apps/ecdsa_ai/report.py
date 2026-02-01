"""Summarize sweep runs stored under a runs/ directory."""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _gather_runs(run_root: str) -> List[Dict[str, Any]]:
    runs_dir = os.path.join(run_root, "runs")
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")

    records: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".json"):
            continue
        records.append(_load_json(os.path.join(runs_dir, name)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sweep runs")
    parser.add_argument("run_root", help="Path to a sweep run directory")
    args = parser.parse_args()

    records = _gather_runs(args.run_root)
    if not records:
        print("No run records found.")
        return

    by_score = sorted(records, key=lambda r: r["best_score"], reverse=True)
    by_hnp = sorted(records, key=lambda r: (r["hnp_low_error"] is None, r["hnp_low_error"]))

    summary = {
        "run_root": args.run_root,
        "total_runs": len(records),
        "top_by_score": by_score[:10],
        "top_by_hnp_error": by_hnp[:10],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
