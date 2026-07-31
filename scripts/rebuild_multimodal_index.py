"""Rebuild or resume the local SQLite + Chroma multimodal index.

Run from the project root, preferably in the existing ``torchtest`` environment::

    python scripts/rebuild_multimodal_index.py --force --organize-events

The command prints one final JSON object so timings and counts can be copied into
the experiment log without parsing progress messages.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_ingestion import scan_photos_directory  # noqa: E402
from event_service import EventService  # noqa: E402
from model_pipeline import get_vision_pipeline  # noqa: E402
from retrieval_engine import MultimodalRetriever  # noqa: E402


def _database_counts(storage) -> dict[str, int]:
    with storage.transaction(write=False) as connection:
        return {
            "photos": connection.execute("SELECT COUNT(*) FROM photos").fetchone()[0],
            "captions": connection.execute(
                "SELECT COUNT(DISTINCT photo_id) FROM captions WHERE status='ok'"
            ).fetchone()[0],
            "scene_graphs": connection.execute(
                "SELECT COUNT(DISTINCT photo_id) FROM scene_graphs WHERE status='ok'"
            ).fetchone()[0],
            "events": connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "failures": connection.execute("SELECT COUNT(*) FROM index_failures").fetchone()[0],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute CLIP and BLIP for every photo")
    parser.add_argument(
        "--organize-events",
        action="store_true",
        help="run deterministic event segmentation after indexing",
    )
    args = parser.parse_args(argv)

    paths = scan_photos_directory()
    pipeline = get_vision_pipeline()
    started = time.perf_counter()
    exit_code = 0
    try:
        pipeline.index_paths(paths, force=args.force, show_progress=True)
        index_seconds = time.perf_counter() - started
        organization = None
        event_seconds = 0.0
        if args.organize_events:
            event_started = time.perf_counter()
            retriever = MultimodalRetriever(
                storage=pipeline.storage,
                vector_store=pipeline.vector_store,
                clip_backend=pipeline.clip_manager,
            )
            organization = EventService(
                storage=pipeline.storage,
                vector_store=pipeline.vector_store,
                retriever=retriever,
            ).organize(persist=True)
            event_seconds = time.perf_counter() - event_started

        payload = {
            "index_run": asdict(pipeline.last_index_summary),
            "elapsed_seconds": round(index_seconds, 3),
            "event_elapsed_seconds": round(event_seconds, 3),
            "database": _database_counts(pipeline.storage),
            "vectors": pipeline.vector_store.counts(),
            "events": (
                {
                    "count": len(organization.events),
                    "assigned_photos": sum(len(event.photo_ids) for event in organization.events),
                    "unassigned_photos": len(organization.unassigned_photo_ids),
                }
                if organization is not None
                else None
            ),
        }
        print("FINAL_REBUILD_REPORT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if pipeline.last_index_summary.failed_count:
            exit_code = 2
    finally:
        pipeline.vector_store.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
