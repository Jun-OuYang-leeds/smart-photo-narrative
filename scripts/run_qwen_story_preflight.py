"""Two-case real Qwen preflight using development-only photos."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH  # noqa: E402
from qwen_story_evaluation import (  # noqa: E402
    MOOD_EVENT_ID,
    OLD_CASES_PATH,
    PRIVATE_DIR,
    QwenCreativeRunner,
    QwenFaithfulRunner,
    QwenOllamaBackend,
    atomic_write_json,
    canonical_hash,
)
from story_evaluation import StoryCase  # noqa: E402


def _photo_ids_for_event(connection: sqlite3.Connection, event_id: str) -> tuple[str, ...]:
    return tuple(row[0] for row in connection.execute(
        "SELECT photo_id FROM event_photos WHERE event_id=? ORDER BY position", (event_id,),
    ))


def _photo_ids_for_date(connection: sqlite3.Connection, date: str) -> tuple[str, ...]:
    return tuple(row[0] for row in connection.execute(
        "SELECT photo_id FROM photos WHERE date_local=? ORDER BY captured_at_sort, photo_id", (date,),
    ))


def main() -> None:
    connection = sqlite3.connect(APP_DB_PATH)
    try:
        before = int(connection.execute("SELECT COUNT(*) FROM stories").fetchone()[0])
        faithful_ids = _photo_ids_for_event(connection, MOOD_EVENT_ID)
        # Four-photo 2026-07-11 development event used during the v4/v5 audit;
        # it is explicitly excluded from the new Creative formal manifest.
        creative_event_id = "702ac65a-74ac-5780-bb19-ebcc0b686f1e"
        creative_ids = _photo_ids_for_event(connection, creative_event_id)
    finally:
        connection.close()
    frozen = json.loads(OLD_CASES_PATH.read_text(encoding="utf-8"))
    frozen_ids = {photo_id for case in frozen["cases"] for photo_id in case["photo_ids"]}
    if set(faithful_ids) & frozen_ids or set(creative_ids) & frozen_ids:
        raise RuntimeError("Preflight photos overlap the historical formal Faithful cases")
    if not faithful_ids or not creative_ids:
        raise RuntimeError("Development preflight cases are unavailable")

    backend = QwenOllamaBackend()
    faithful_case = StoryCase(
        "preflight_faithful", "event", "Development Faithful preflight",
        faithful_ids, "zh", event_id=MOOD_EVENT_ID, size_bucket="4-7",
    )
    creative_case = StoryCase(
        "preflight_creative", "event", "Development Creative preflight",
        creative_ids, "zh", event_id=creative_event_id, size_bucket="4-7",
    )
    faithful_path = PRIVATE_DIR / "story_qwen_preflight_faithful_v1.jsonl"
    creative_path = PRIVATE_DIR / "story_qwen_preflight_creative_v1.jsonl"
    faithful = QwenFaithfulRunner(backend, faithful_path).run([faithful_case])
    creative = QwenCreativeRunner(backend, creative_path).run_cases([creative_case])

    connection = sqlite3.connect(f"file:{Path(APP_DB_PATH).resolve().as_posix()}?mode=ro", uri=True)
    try:
        after = int(connection.execute("SELECT COUNT(*) FROM stories").fetchone()[0])
    finally:
        connection.close()
    if before != after:
        raise RuntimeError(f"Production Story count changed during preflight: {before} -> {after}")
    report = {
        "protocol_version": "story-qwen-preflight-v1",
        "case_count": 2,
        "formal_case_overlap": False,
        "faithful": [{
            "variant_id": item["variant_id"], "status": item["status"],
            "latency_ms": item["latency_ms"], "prompt_hash": item["prompt_hash"],
            "output_hash": item["output_hash"],
        } for item in faithful],
        "creative": [{
            "variant_id": item["variant_id"], "status": item["status"],
            "latency_ms": item["latency_ms"], "prompt_hash": item["prompt_hash"],
            "output_hash": item["output_hash"],
        } for item in creative],
        "production_story_count_before": before,
        "production_story_count_after": after,
    }
    report["report_sha256"] = canonical_hash(report)
    atomic_write_json(PRIVATE_DIR / "story_qwen_preflight_v1.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
