from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from scene_graph_io import sha256_file
from scene_graph_recovery import (
    RECOVERY_PROVENANCE,
    recover_confirmed_legacy_personal_results,
    write_recovered_scene_graphs,
)
from storage import PhotoStorage


class SceneGraphRecoveryTests(unittest.TestCase):
    def make_photo(self, path: Path, color: str) -> str:
        Image.new("RGB", (8, 8), color).save(path, format="JPEG")
        return sha256_file(path)

    def write_rows(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_confirmed_recovery_labels_weaker_provenance_and_skips_failed_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "photos"
            album.mkdir()
            storage = PhotoStorage(root / "app.db")
            first = album / "first.jpg"
            second = album / "second.jpg"
            first_digest = self.make_photo(first, "red")
            second_digest = self.make_photo(second, "blue")
            first_id = storage.upsert_photo(relative_path=first.name, content_sha256=first_digest)
            storage.upsert_photo(relative_path=second.name, content_sha256=second_digest)
            generated_at = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
            source = root / "legacy.jsonl"
            self.write_rows(
                source,
                [
                    {
                        "image_path": first.name,
                        "scene_graph_source": "remote",
                        "scene_graph": [{"subject": "cup", "predicate": "on", "object": "table"}],
                        "generated_at": generated_at,
                    },
                    {
                        "image_path": second.name,
                        "scene_graph_source": "failed",
                        "scene_graph": [],
                        "remote_error": "truncated JSON",
                        "generated_at": generated_at,
                    },
                ],
            )

            with self.assertRaises(ValueError):
                recover_confirmed_legacy_personal_results(
                    source,
                    album,
                    storage,
                    dataset_id="personal-recovered-test",
                    confirmed_generated_from_current_album=False,
                )

            report = recover_confirmed_legacy_personal_results(
                source,
                album,
                storage,
                dataset_id="personal-recovered-test",
                confirmed_generated_from_current_album=True,
            )
            self.assertEqual(report.blocking_issues, 0)
            self.assertEqual(report.counts["total_rows"], 2)
            self.assertEqual(report.counts["recovered"], 1)
            self.assertEqual(report.counts["failed"], 1)
            row = report.indexable_records[0]
            self.assertEqual(row["local_sqlite_photo_id"], first_id)
            self.assertEqual(row["provenance"], RECOVERY_PROVENANCE)
            self.assertFalse(row["generation_sha256_verified"])
            self.assertEqual(row["sha256_origin"], "current_local_file_at_recovery")
            self.assertEqual(row["output_sha256"], report.source_file_sha256)

            output = root / "indexable.jsonl"
            audit = root / "report.json"
            write_recovered_scene_graphs(report, output, audit)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(json.loads(audit.read_text(encoding="utf-8"))["blocking_issues"], 0)

    def test_batch_subdir_scopes_coverage_to_an_incremental_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "photos"
            album.mkdir()
            batch = album / "pic2"
            batch.mkdir()
            storage = PhotoStorage(root / "app.db")
            # An already-indexed photo living directly in the album root.
            existing = album / "existing.jpg"
            existing_digest = self.make_photo(existing, "red")
            storage.upsert_photo(relative_path=existing.name, content_sha256=existing_digest)
            # A new incremental-batch photo living in photos/pic2/.
            batch_photo = batch / "batch.jpg"
            batch_digest = self.make_photo(batch_photo, "blue")
            batch_id = storage.upsert_photo(
                relative_path="pic2/batch.jpg", content_sha256=batch_digest
            )
            generated_at = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
            source = root / "legacy.jsonl"
            self.write_rows(
                source,
                [
                    {
                        "image_path": "pic2/batch.jpg",
                        "scene_graph_source": "remote",
                        "scene_graph": [{"subject": "cup", "predicate": "on", "object": "table"}],
                        "generated_at": generated_at,
                    }
                ],
            )

            # Without scoping, the pre-existing album photo counts as uncovered.
            unscoped = recover_confirmed_legacy_personal_results(
                source,
                album,
                storage,
                dataset_id="personal-pic2-test",
                confirmed_generated_from_current_album=True,
            )
            self.assertEqual(unscoped.counts["album_uncovered"], 1)
            self.assertGreater(unscoped.blocking_issues, 0)

            # Scoping to the batch subdir ignores the rest of the album.
            scoped = recover_confirmed_legacy_personal_results(
                source,
                album,
                storage,
                dataset_id="personal-pic2-test",
                confirmed_generated_from_current_album=True,
                batch_relative_root="pic2",
            )
            self.assertEqual(scoped.blocking_issues, 0)
            self.assertEqual(scoped.counts["album_uncovered"], 0)
            self.assertEqual(scoped.counts["album_image_files"], 1)
            self.assertEqual(scoped.counts["recovered"], 1)
            self.assertEqual(scoped.batch_relative_root, "pic2")
            record = scoped.indexable_records[0]
            self.assertEqual(record["relative_path"], "pic2/batch.jpg")
            self.assertEqual(record["local_sqlite_photo_id"], batch_id)

    def test_modified_photo_is_blocking_and_no_indexable_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            album = root / "photos"
            album.mkdir()
            storage = PhotoStorage(root / "app.db")
            image = album / "changed.jpg"
            digest = self.make_photo(image, "green")
            storage.upsert_photo(relative_path=image.name, content_sha256=digest)
            generated_at = datetime.now(timezone.utc) - timedelta(days=1)
            os.utime(image, None)
            source = root / "legacy.jsonl"
            self.write_rows(
                source,
                [
                    {
                        "image_path": image.name,
                        "scene_graph_source": "remote",
                        "scene_graph": [{"subject": "tree", "predicate": "near", "object": "road"}],
                        "generated_at": generated_at.isoformat(),
                    }
                ],
            )

            report = recover_confirmed_legacy_personal_results(
                source,
                album,
                storage,
                dataset_id="personal-recovered-test",
                confirmed_generated_from_current_album=True,
            )
            self.assertEqual(report.counts["modified_after_generation"], 1)
            self.assertGreater(report.blocking_issues, 0)
            with self.assertRaises(ValueError):
                write_recovered_scene_graphs(report, root / "indexable.jsonl", root / "report.json")
            self.assertFalse((root / "indexable.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
