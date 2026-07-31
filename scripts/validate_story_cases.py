"""Offline validation of the frozen Story cases and v3 evidence plans."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH, STORY_MAX_EVIDENCE_GROUPS  # noqa: E402
from retrieval_engine import get_multimodal_retriever  # noqa: E402
from story_agent import ContextAggregator  # noqa: E402
from story_evaluation import load_story_cases  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, nargs="?", default=ROOT / "evaluation" / "private" / "story_cases_v1.json")
    args = parser.parse_args()
    cases = load_story_cases(args.cases)
    retriever = get_multimodal_retriever()
    aggregator = ContextAggregator(retriever, retriever.storage)
    existing = set(retriever.storage.metadata_eligible_ids())
    summaries = []
    for case in cases:
        unknown = set(case.photo_ids) - existing
        if unknown:
            raise ValueError(f"{case.case_id} contains unknown photo IDs")
        source_kind = "date" if case.source_kind == "date" else ("event" if case.source_kind == "event" else "basket")
        context = aggregator.aggregate_by_photo_ids(
            case.photo_ids, label=case.label, event_id=case.event_id, source_kind=source_kind,
        )
        if not context.groups or len(context.groups) > STORY_MAX_EVIDENCE_GROUPS:
            raise ValueError(f"{case.case_id} has invalid evidence-group count {len(context.groups)}")
        summaries.append({
            "case_id": case.case_id, "source_kind": case.source_kind,
            "photos": context.photo_count, "groups": len(context.groups),
            "conflict_groups": sum(bool(group.conflicts) for group in context.groups),
            "near_duplicates": sum(len(group.near_duplicate_evidence_ids) for group in context.groups),
        })
    print(json.dumps({
        "case_count": len(cases),
        "source_counts": dict(Counter(case.source_kind for case in cases)),
        "all_photo_ids_exist": True,
        "all_group_counts_valid": True,
        "cases": summaries,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
