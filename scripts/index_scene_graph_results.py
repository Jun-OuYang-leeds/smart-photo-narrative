from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene_graph_service import apply_indexable_scene_graphs


def main() -> int:
    parser = argparse.ArgumentParser(description="Index an audited Scene Graph JSONL into the main project")
    parser.add_argument("indexable_jsonl", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = apply_indexable_scene_graphs(args.indexable_jsonl, dry_run=args.dry_run)
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0 if not summary.checksum_conflicts and not summary.malformed else 2


if __name__ == "__main__":
    raise SystemExit(main())

