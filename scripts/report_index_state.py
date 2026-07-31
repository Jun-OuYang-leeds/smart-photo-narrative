"""Print a machine-readable consistency report for the local derived indexes."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APP_DB_PATH, CHROMA_PERSIST_DIR  # noqa: E402
from storage import PhotoStorage  # noqa: E402
from vector_store import ChromaVectorStore  # noqa: E402


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        return round((datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds(), 3)
    except ValueError:
        return None


def main() -> int:
    storage = PhotoStorage(APP_DB_PATH)
    vectors = ChromaVectorStore(CHROMA_PERSIST_DIR)
    try:
        with storage.transaction(write=False) as connection:
            count_queries = {
                "photos": "SELECT COUNT(*) FROM photos",
                "captions_ok": "SELECT COUNT(*) FROM captions WHERE status='ok'",
                "fts_rows": "SELECT COUNT(*) FROM photo_fts",
                # Non-semantic CLIP completion / model-version identity markers.
                # The 72 preset zero-shot tags were removed; this counts photos
                # whose image vector is marked current, not photos with labels.
                "clip_marked_photos": "SELECT COUNT(DISTINCT photo_id) FROM tags",
                "scene_graphs": "SELECT COUNT(*) FROM scene_graphs",
                "scene_graph_triples": "SELECT COUNT(*) FROM scene_graph_triples",
                "events": "SELECT COUNT(*) FROM events",
                "event_photos": "SELECT COUNT(*) FROM event_photos",
                "stories": "SELECT COUNT(*) FROM stories",
                "failures": "SELECT COUNT(*) FROM index_failures",
                "gps_photos": (
                    "SELECT COUNT(*) FROM photos "
                    "WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL"
                ),
            }
            counts = {
                name: connection.execute(query).fetchone()[0]
                for name, query in count_queries.items()
            }
            timestamp_sources = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT timestamp_source, COUNT(*) FROM photos GROUP BY timestamp_source"
                ).fetchall()
            }
            timestamp_confidence = {
                str(row[0]): row[1]
                for row in connection.execute(
                    "SELECT timestamp_confidence, COUNT(*) FROM photos GROUP BY timestamp_confidence"
                ).fetchall()
            }
            run = connection.execute(
                "SELECT * FROM index_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        latest_run = dict(run) if run else None
        if latest_run:
            latest_run.pop("config_json", None)
            latest_run["elapsed_seconds"] = _duration_seconds(
                latest_run.get("started_at"), latest_run.get("completed_at")
            )
        report = {
            "database": counts,
            "vectors": vectors.counts(),
            "timestamp_sources": timestamp_sources,
            "timestamp_confidence": timestamp_confidence,
            "latest_index_run": latest_run,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        vectors.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
