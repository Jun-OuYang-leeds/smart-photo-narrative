"""Replace stored CLIP zero-shot semantic tags with the non-semantic index marker.

The 72 preset CLIP tags were removed from the pipeline.  Their rows in the
``tags`` table, however, also carried the CLIP completion / model-version
identity marker in their ``source`` column, which the incremental indexer reads
to decide whether a photo's image vector is current
(``indexing_service._completion_markers``).  Deleting the rows outright would
therefore invalidate every photo's fast-path marker and force a full CLIP
recomputation, and would drop the v1->v2->v1 stale-vector protection.

This migration keeps each photo's existing ``clip:<model>:<version>`` source but
replaces its semantic tags with a single ``CLIP_INDEX_MARKER_TAG`` sentinel, so
the identity semantics are preserved while no visual label remains.
``PhotoStorage.replace_tags`` refreshes the FTS5 document for each photo it
touches, and the sentinel is excluded from the FTS ``tags`` column, so no
migrated photo keeps searchable tag text.

Read-only by default; pass ``--apply`` to write (a database backup is taken
first).  The Chroma vector store is not modified by this migration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APP_DB_PATH, CLIP_INDEX_MARKER_TAG  # noqa: E402
from storage import PhotoStorage  # noqa: E402


CLIP_TAG_FAMILY = "clip:"


def _survey(database: Path) -> dict[str, object]:
    """Read-only picture of what the migration would change."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        total_rows = connection.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        sentinel_rows = connection.execute(
            "SELECT COUNT(*) FROM tags WHERE tag = ?", (CLIP_INDEX_MARKER_TAG,)
        ).fetchone()[0]
        clip_rows = connection.execute(
            "SELECT COUNT(*) FROM tags WHERE substr(source, 1, ?) = ?",
            (len(CLIP_TAG_FAMILY), CLIP_TAG_FAMILY),
        ).fetchone()[0]
        non_clip_rows = total_rows - clip_rows
        sources = [
            dict(row)
            for row in connection.execute(
                """
                SELECT source, COUNT(*) AS rows, COUNT(DISTINCT photo_id) AS photos
                FROM tags
                WHERE substr(source, 1, ?) = ?
                GROUP BY source
                """,
                (len(CLIP_TAG_FAMILY), CLIP_TAG_FAMILY),
            )
        ]
        # One photo must end up with exactly one source, otherwise the identity
        # marker would be ambiguous after migration.
        multi_source = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT photo_id FROM tags
                WHERE substr(source, 1, ?) = ?
                GROUP BY photo_id HAVING COUNT(DISTINCT source) > 1
            )
            """,
            (len(CLIP_TAG_FAMILY), CLIP_TAG_FAMILY),
        ).fetchone()[0]
        targets = [
            dict(row)
            for row in connection.execute(
                """
                SELECT photo_id, MIN(source) AS source, COUNT(*) AS rows
                FROM tags
                WHERE substr(source, 1, ?) = ?
                GROUP BY photo_id
                ORDER BY photo_id
                """,
                (len(CLIP_TAG_FAMILY), CLIP_TAG_FAMILY),
            )
        ]
        photos_total = connection.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    finally:
        connection.close()
    return {
        "database": str(database),
        "photos_total": photos_total,
        "tag_rows_total": total_rows,
        "clip_family_rows": clip_rows,
        "non_clip_rows": non_clip_rows,
        "sentinel_rows_existing": sentinel_rows,
        "clip_sources": sources,
        "photos_with_multiple_clip_sources": multi_source,
        "photos_to_migrate": len(targets),
        "_targets": targets,
    }


def _verify(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        total_rows = connection.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        sentinel_rows = connection.execute(
            "SELECT COUNT(*) FROM tags WHERE tag = ?", (CLIP_INDEX_MARKER_TAG,)
        ).fetchone()[0]
        semantic_rows = total_rows - sentinel_rows
        photos_with_sentinel = connection.execute(
            "SELECT COUNT(DISTINCT photo_id) FROM tags WHERE tag = ?",
            (CLIP_INDEX_MARKER_TAG,),
        ).fetchone()[0]
        # The sentinel must never be searchable text.
        fts_with_tag_text = connection.execute(
            "SELECT COUNT(*) FROM photo_fts WHERE tags IS NOT NULL AND trim(tags) != ''"
        ).fetchone()[0]
        return {
            "tag_rows_total": total_rows,
            "sentinel_rows": sentinel_rows,
            "remaining_semantic_rows": semantic_rows,
            "photos_with_sentinel": photos_with_sentinel,
            "fts_docs_with_nonempty_tags": fts_with_tag_text,
        }
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=APP_DB_PATH)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "backups",
        help="Parent directory for the pre-migration database copy.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the migration. Without this flag the command is read-only.",
    )
    args = parser.parse_args(argv)

    database = args.database.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    survey = _survey(database)
    targets = survey.pop("_targets")
    report: dict[str, object] = {"dry_run": not args.apply, "survey": survey}

    if survey["photos_with_multiple_clip_sources"]:
        report["blocked"] = (
            "Some photos carry more than one clip: tag source; the identity marker "
            "would be ambiguous. Re-run indexing before migrating."
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    if args.apply:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = args.backup_dir / f"tags_removal_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / database.name
        shutil.copy2(database, backup_path)
        report["backup"] = str(backup_path)

        storage = PhotoStorage(database)
        migrated = 0
        for target in targets:
            # Preserve each photo's own clip:<model>:<version> source so the
            # incremental fast path still recognises the vector as current.
            storage.replace_tags(
                str(target["photo_id"]),
                [(CLIP_INDEX_MARKER_TAG, None)],
                source=str(target["source"]),
                clear_source_prefix=CLIP_TAG_FAMILY,
            )
            migrated += 1
        report["migrated_photos"] = migrated
        report["verification"] = _verify(database)

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
