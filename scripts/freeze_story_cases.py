"""Select the deterministic private 12-case Story evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH  # noqa: E402
from storage import PhotoStorage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "private" / "story_cases_v1.json")
    args = parser.parse_args()
    storage = PhotoStorage(APP_DB_PATH)
    with storage.transaction(write=False) as connection:
        event_rows = connection.execute(
            """
            SELECT e.event_id, e.title, e.date_local, e.start_at_sort, COUNT(ep.photo_id) AS photo_count
            FROM events e JOIN event_photos ep ON ep.event_id = e.event_id
            GROUP BY e.event_id
            ORDER BY e.start_at_sort IS NULL, e.start_at_sort, e.event_id
            """
        ).fetchall()
        events = [dict(row) for row in event_rows]
        event_photos = {
            event["event_id"]: [row[0] for row in connection.execute(
                "SELECT photo_id FROM event_photos WHERE event_id = ? ORDER BY position", (event["event_id"],),
            ).fetchall()]
            for event in events
        }
        all_photo_ids = [row[0] for row in connection.execute(
            "SELECT photo_id FROM photos ORDER BY captured_at_sort IS NULL, captured_at_sort, relative_path"
        ).fetchall()]
        date_rows = connection.execute(
            """
            SELECT e.date_local, COUNT(DISTINCT e.event_id) AS event_count, COUNT(ep.photo_id) AS photo_count
            FROM events e JOIN event_photos ep ON ep.event_id = e.event_id
            WHERE e.date_local IS NOT NULL
            GROUP BY e.date_local HAVING COUNT(DISTINCT e.event_id) >= 2
            ORDER BY e.date_local
            """
        ).fetchall()

    selected_events = []
    regression = max(
        (event for event in events if event["date_local"] == "2026-07-12"),
        key=lambda event: event["photo_count"],
    )
    selected_events.append(regression)
    buckets = [
        ("2-3", lambda count: 2 <= count <= 3, 3),
        ("4-7", lambda count: 4 <= count <= 7, 3),
        ("8+", lambda count: count >= 8, 2),
    ]
    bucket_names = {regression["event_id"]: "8+"}
    for name, predicate, target in buckets:
        existing = sum(bucket_names.get(event["event_id"]) == name for event in selected_events)
        for event in events:
            if existing >= target or event in selected_events:
                continue
            if predicate(int(event["photo_count"])):
                selected_events.append(event)
                bucket_names[event["event_id"]] = name
                existing += 1
    if len(selected_events) != 8:
        raise RuntimeError(f"Could not select eight event cases; got {len(selected_events)}")

    preferred_dates = ["2026-07-10", "2025-09-21"]
    available_dates = {row["date_local"]: dict(row) for row in date_rows}
    selected_dates = [date for date in preferred_dates if date in available_dates]
    selected_dates.extend(
        row["date_local"] for row in date_rows
        if row["date_local"] not in selected_dates and len(selected_dates) < 2
    )

    cases = [
        {"case_id": "single_01", "source_kind": "single", "label": "single photo note 1", "photo_ids": [all_photo_ids[0]], "size_bucket": "1"},
        {"case_id": "single_02", "source_kind": "single", "label": "single photo note 2", "photo_ids": [all_photo_ids[-1]], "size_bucket": "1"},
    ]
    for index, event in enumerate(selected_events, 1):
        label = "2026-07-12" if event["event_id"] == regression["event_id"] else str(event["title"] or event["date_local"])
        cases.append({
            "case_id": f"event_{index:02d}", "source_kind": "event", "label": label,
            "event_id": event["event_id"], "photo_ids": event_photos[event["event_id"]],
            "size_bucket": bucket_names[event["event_id"]],
        })
    for index, date in enumerate(selected_dates, 1):
        cases.append({
            "case_id": f"date_{index:02d}", "source_kind": "date", "label": date,
            "photo_ids": storage.metadata_eligible_ids(date_from=date, date_to=date), "size_bucket": "multi-event",
        })
    for index, case in enumerate(cases):
        case["language"] = "zh" if index % 2 == 0 else "en"
    payload = {
        "protocol_version": "story-cases-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    public = {
        "protocol_version": payload["protocol_version"],
        "private_cases_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "case_count": 12, "languages": {"zh": 6, "en": 6},
        "source_kinds": {"single": 2, "event": 8, "date": 2},
        "event_size_buckets": {
            name: sum(case["source_kind"] == "event" and case["size_bucket"] == name for case in cases)
            for name in ("2-3", "4-7", "8+")
        },
        "includes_2026_07_12_regression": True,
        "private_content_committed": False,
    }
    (ROOT / "evaluation" / "story_cases_v1_manifest.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
