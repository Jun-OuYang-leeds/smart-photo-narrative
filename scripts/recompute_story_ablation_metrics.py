"""Recompute derived metrics without rerunning or changing any model output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from story_agent import ContextAggregator  # noqa: E402
from story_evaluation import load_story_cases, score_experiment_output  # noqa: E402
from scripts.run_story_ablation import _write_public_manifest  # noqa: E402


PRIVATE = ROOT / "evaluation" / "private"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=PRIVATE / "story_ablation_n0_n3_v1.jsonl")
    parser.add_argument("--cases", type=Path, default=PRIVATE / "story_cases_v1.json")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {item.case_id: item for item in load_story_cases(args.cases)}
    aggregator = ContextAggregator()
    contexts = {
        case_id: aggregator.aggregate_by_photo_ids(
            case.photo_ids, label=case.label, event_id=case.event_id, source_kind=case.source_kind,
        ) for case_id, case in cases.items()
    }
    immutable_before = [(item["case_id"], item["variant_id"], item["output_hash"], item["raw_attempts"]) for item in records]
    for record in records:
        variant = record["variant_id"]
        old = record.get("automatic_metrics", {})
        record["automatic_metrics"] = score_experiment_output(
            variant, record.get("normalized_output"), contexts[record["case_id"]],
            latency_ms=float(record["latency_ms"]), status=record["status"],
            json_valid=(None if variant == "N0" else bool(old.get("json_valid"))),
            language=record["language"],
        )
    immutable_after = [(item["case_id"], item["variant_id"], item["output_hash"], item["raw_attempts"]) for item in records]
    if immutable_after != immutable_before:
        raise RuntimeError("Metric recomputation attempted to alter a formal model output")
    temporary = args.results.with_suffix(args.results.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(args.results)
    _write_public_manifest(records, args.results)
    print(json.dumps({"records_recomputed": len(records), "model_outputs_changed": False}, indent=2))


if __name__ == "__main__":
    main()
