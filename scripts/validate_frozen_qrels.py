"""Validate the private 48-query qrels set without printing private content."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH  # noqa: E402
from evaluation import load_qrels, validate_frozen_qrels  # noqa: E402
from storage import PhotoStorage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("qrels", type=Path, nargs="?", default=ROOT / "evaluation" / "private" / "qrels_v1.json")
    args = parser.parse_args()
    judgments = load_qrels(args.qrels)
    storage = PhotoStorage(APP_DB_PATH)
    with storage.transaction(write=False) as connection:
        rows = connection.execute(
            "SELECT photo_id, date_local, location, timestamp_confidence FROM photos"
        ).fetchall()
    metadata = {str(row["photo_id"]): dict(row) for row in rows}
    summary = validate_frozen_qrels(
        judgments, existing_photo_ids=metadata, photo_metadata=metadata,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
