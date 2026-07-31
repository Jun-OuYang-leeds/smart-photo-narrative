"""Publish aggregate qrels statistics and a private-file hash only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import load_qrels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, default=ROOT / "evaluation" / "private" / "qrels_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "qrels_v1_manifest.json")
    args = parser.parse_args()
    judgments = load_qrels(args.qrels)
    raw = args.qrels.read_bytes()
    payload = {
        "protocol_version": "qrels-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "private_qrels_sha256": hashlib.sha256(raw).hexdigest(),
        "query_count": len(judgments),
        "category_counts": dict(sorted(Counter(item.category for item in judgments).items())),
        "language_counts": dict(sorted(Counter(item.language for item in judgments).items())),
        "split_counts": dict(sorted(Counter(item.split for item in judgments).items())),
        "judgment_source_counts": dict(sorted(Counter(item.judgment_source for item in judgments).items())),
        "grade_counts": dict(sorted(Counter(
            str(int(grade)) if float(grade).is_integer() else str(grade)
            for item in judgments for grade in item.relevance.values()
        ).items())),
        "positive_judgment_count": sum(
            1 for item in judgments for grade in item.relevance.values() if grade > 0
        ),
        "private_content_committed": False,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
