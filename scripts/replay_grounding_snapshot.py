#!/usr/bin/env python3
"""Replay SmoothNav grounding-stage snapshots without Habitat.

The online hook writes JSONL records under ``grounding_snapshots/``.  Each record
contains the frontier-value inputs at the semantic grounding boundary plus an
online replay verdict.  This script reloads those records and recomputes the
verdict with the current code so a target-anchor fix can be checked without
rerunning a full episode.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoothnav.frontier_scoring import replay_frontier_value_snapshot  # noqa: E402


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        records: List[Dict[str, Any]] = []
        for child in sorted(path.glob("**/*.jsonl")):
            records.extend(_load_json_records(child))
        for child in sorted(path.glob("*.json")):
            records.extend(_load_json_records(child))
        return records
    if path.suffix == ".jsonl":
        items = []
        for line in path.read_text().splitlines():
            if line.strip():
                items.append(json.loads(line))
        return items
    if path.suffix == ".json":
        loaded = json.loads(path.read_text())
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
        if isinstance(loaded, dict):
            return [loaded]
    return []


def _snapshot_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if "snapshot" in record and isinstance(record["snapshot"], dict):
        return record["snapshot"]
    return record


def replay_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for idx, record in enumerate(records):
        snapshot = _snapshot_from_record(record)
        replay = replay_frontier_value_snapshot(snapshot)
        checks = replay.get("checks") or {}
        results.append(
            {
                "index": idx,
                "step_idx": record.get("step_idx"),
                "trigger": record.get("trigger"),
                "active_target_region": snapshot.get("active_target_region"),
                "status": replay.get("status"),
                "checks": checks,
                "semantic_term": replay.get("semantic_term"),
                "semantic_term_share": replay.get("semantic_term_share"),
                "target_progress_score": (
                    replay.get("selected_score_breakdown") or {}
                ).get("target_progress_score"),
                "selected_frontier": replay.get("selected_frontier"),
                "selected_branch_id": replay.get("selected_branch_id"),
                "replay_result": replay,
            }
        )

    target_results = [
        item for item in results
        if str(item.get("active_target_region") or "").startswith("unexplored target:")
    ]
    failing = [item for item in results if item.get("status") != "pass"]
    target_progress_zero = [
        item for item in target_results
        if float(item.get("target_progress_score") or 0.0) <= 0.0
    ]
    semantic_zero = [
        item for item in target_results
        if float(item.get("semantic_term") or 0.0) <= 0.0
    ]
    summary = {
        "schema_version": "smoothnav.grounding_snapshot_replay.summary.v1",
        "record_count": len(results),
        "target_record_count": len(target_results),
        "pass_count": len(results) - len(failing),
        "fail_count": len(failing),
        "target_progress_zero_count": len(target_progress_zero),
        "target_semantic_term_zero_count": len(semantic_zero),
        "status": "pass" if not failing and not target_progress_zero and not semantic_zero else "fail",
        "results": results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Snapshot JSON/JSONL files or directories")
    parser.add_argument("--output", help="Write summary JSON to this path")
    parser.add_argument("--json", action="store_true", help="Print full JSON summary")
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    for path in args.paths:
        records.extend(_load_json_records(Path(path)))
    summary = replay_records(records)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"status={summary['status']} records={summary['record_count']} "
            f"target_records={summary['target_record_count']} "
            f"failures={summary['fail_count']} "
            f"target_progress_zero={summary['target_progress_zero_count']} "
            f"target_semantic_zero={summary['target_semantic_term_zero_count']}"
        )
        if args.output:
            print(f"wrote {args.output}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
