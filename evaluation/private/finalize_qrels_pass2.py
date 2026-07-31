"""Record explicit pooled negatives after the second system-blind pass."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
QRELS = HERE / "qrels_v1.json"
POOL = HERE / "candidate_pool" / "pool.json"


def main() -> None:
    payload = json.loads(QRELS.read_text(encoding="utf-8"))
    pools = {
        item["query_id"]: item["candidate_ids"]
        for item in json.loads(POOL.read_text(encoding="utf-8"))["queries"]
    }
    for record in payload["queries"]:
        relevance = record["relevance"]
        for photo_id in pools[record["query_id"]]:
            relevance.setdefault(photo_id, 0)
        record["judgment_source"] = "agent_pass2"
        record["notes"] = (
            "two independent deterministic shuffles reviewed against original images; "
            "positive/ambiguous human terminal review still required"
        )
    QRELS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {len(payload['queries'])} queries with explicit pooled negatives")


if __name__ == "__main__":
    main()
