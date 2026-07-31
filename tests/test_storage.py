from __future__ import annotations

import sqlite3
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from storage import PhotoStorage, SCHEMA_VERSION, StorageConflictError


class PhotoStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "narrative.sqlite3"
        self.storage = PhotoStorage(self.database_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_photo(
        self,
        name: str,
        digest_character: str,
        *,
        captured_at_sort: int | None = None,
        date_local: str | None = None,
        location: str | None = None,
    ) -> str:
        return self.storage.upsert_photo(
            relative_path=name,
            content_sha256=digest_character * 64,
            file_size_bytes=1234,
            file_mtime_ns=5678,
            image_width=640,
            image_height=480,
            captured_at_sort=captured_at_sort,
            date_local=date_local,
            timestamp_source="exif",
            timestamp_confidence=0.95,
            location=location,
        )

    def test_schema_pragmas_tables_and_migration_are_idempotent(self) -> None:
        expected_tables = {
            "schema_version",
            "photos",
            "captions",
            "scene_graphs",
            "scene_graph_triples",
            "tags",
            "photo_moods",
            "events",
            "event_photos",
            "stories",
            "index_runs",
            "index_failures",
            "photo_fts",
        }

        self.assertEqual(self.storage.migrate(), SCHEMA_VERSION)
        self.assertEqual(self.storage.migrate(), SCHEMA_VERSION)
        with self.storage.transaction(write=False) as connection:
            table_names = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
            schema_rows = connection.execute("SELECT * FROM schema_version").fetchall()
            foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            photo_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)")
            }
            graph_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(scene_graphs)")
            }

        self.assertTrue(expected_tables.issubset(table_names))
        self.assertEqual([row["version"] for row in schema_rows], [SCHEMA_VERSION])
        self.assertEqual(foreign_keys, 1)
        self.assertEqual(journal_mode.lower(), "wal")
        self.assertTrue(
            {
                "photo_id",
                "relative_path",
                "content_sha256",
                "file_size_bytes",
                "file_mtime_ns",
                "image_width",
                "image_height",
                "captured_at",
                "captured_at_sort",
                "date_local",
                "timestamp_source",
                "timestamp_confidence",
                "gps_latitude",
                "gps_longitude",
                "location",
            }.issubset(photo_columns)
        )
        self.assertNotIn("remote_raw", graph_columns)
        self.assertFalse(any("raw" in column.casefold() for column in graph_columns))

    def test_photo_uuid_survives_hash_based_rename(self) -> None:
        first_id = self.add_photo("album/original.jpg", "a")
        uuid.UUID(first_id)

        renamed_id = self.storage.upsert_photo(
            relative_path="album/renamed.jpg",
            content_sha256="a" * 64,
            file_size_bytes=1234,
            file_mtime_ns=9999,
            image_width=640,
            image_height=480,
            timestamp_source="exif",
            timestamp_confidence=0.95,
        )

        self.assertEqual(renamed_id, first_id)
        self.assertIsNone(self.storage.get_photo_by_path("album/original.jpg"))
        by_path = self.storage.get_photo_by_path("album\\renamed.jpg")
        by_hash = self.storage.get_photo_by_hash("A" * 64)
        self.assertEqual(by_path["photo_id"], first_id)
        self.assertEqual(by_hash["photo_id"], first_id)
        self.assertEqual(by_path["file_mtime_ns"], 9999)

    def test_conflicting_path_and_hash_are_rejected_atomically(self) -> None:
        first_id = self.add_photo("first.jpg", "a")
        second_id = self.add_photo("second.jpg", "b")

        with self.assertRaises(StorageConflictError):
            self.storage.upsert_photo(
                relative_path="second.jpg",
                content_sha256="a" * 64,
                timestamp_confidence=0.0,
            )

        self.assertEqual(self.storage.get_photo(first_id)["relative_path"], "first.jpg")
        self.assertEqual(self.storage.get_photo(second_id)["content_sha256"], "b" * 64)

    def test_fts5_indexes_all_text_modalities_and_matches_multiple_words(self) -> None:
        first_id = self.add_photo("cambridge.jpg", "a", location="Cambridge riverside")
        second_id = self.add_photo("street.jpg", "b", location="London")
        self.storage.upsert_caption(
            first_id,
            "a red bicycle beside an old stone bridge",
            model_name="mock-blip",
        )
        self.storage.upsert_caption(second_id, "a red bicycle on a street", model_name="mock-blip")
        self.storage.upsert_scene_graph(
            first_id,
            model_name="mock-qwen",
            scene_graph_text="woman rides bicycle; bicycle near river",
            triples=[("woman", "rides", "bicycle"), ("bicycle", "near", "river")],
        )
        self.storage.replace_tags(first_id, [("holiday", 0.9), ("outdoor", 0.8)])

        caption_hits = self.storage.search_bm25("red bridge")
        graph_hits = self.storage.search_bm25("woman river")
        metadata_hits = self.storage.search_bm25("holiday Cambridge")
        caption_only_graph_hits = self.storage.search_bm25(
            "woman river", fields=("caption",)
        )
        graph_only_hits = self.storage.search_bm25(
            "woman river", fields=("scene_graph_text",)
        )

        self.assertEqual([hit.photo_id for hit in caption_hits], [first_id])
        self.assertEqual([hit.photo_id for hit in graph_hits], [first_id])
        self.assertEqual([hit.photo_id for hit in metadata_hits], [first_id])
        self.assertEqual(caption_only_graph_hits, [])
        self.assertEqual([hit.photo_id for hit in graph_only_hits], [first_id])
        self.assertIn("old stone bridge", caption_hits[0].caption)
        with self.assertRaisesRegex(ValueError, "Unsupported FTS field"):
            self.storage.search_bm25("bridge", fields=("not_a_column",))
        self.assertEqual(self.storage.rebuild_fts(), 2)
        self.assertEqual([hit.photo_id for hit in self.storage.search_bm25("red bridge")], [first_id])

    def test_photo_detail_returns_primary_caption_ordered_graph_and_unique_tags(self) -> None:
        photo_id = self.storage.upsert_photo(
            relative_path="detail.jpg",
            content_sha256="e" * 64,
            image_width=1920,
            image_height=1080,
            captured_at="2026-07-01T12:34:00",
            timestamp_source="exif_original",
            timestamp_confidence=1.0,
            location="Leeds, United Kingdom",
        )
        self.storage.upsert_caption(
            photo_id,
            "primary BLIP caption",
            model_name="blip-primary",
            is_primary=True,
            generated_at="2026-07-01T13:00:00+00:00",
        )
        self.storage.upsert_caption(
            photo_id,
            "newer but non-primary caption",
            model_name="blip-secondary",
            is_primary=False,
            generated_at="2026-07-02T13:00:00+00:00",
        )
        self.storage.upsert_scene_graph(
            photo_id,
            scene_graph_text="person hold camera ; camera near face",
            triples=[
                ("person", "hold", "camera"),
                ("camera", "near", "face"),
            ],
            model_name="qwen-primary",
            is_primary=True,
            generated_at="2026-07-01T13:00:00+00:00",
        )
        self.storage.replace_tags(photo_id, ["Travel", "outdoor"], source="source-one")
        self.storage.replace_tags(photo_id, ["travel"], source="source-two")

        detail = self.storage.get_photo_detail(photo_id)

        self.assertEqual(detail["photo"]["relative_path"], "detail.jpg")
        self.assertEqual(detail["photo"]["image_width"], 1920)
        self.assertEqual(detail["caption"]["caption"], "primary BLIP caption")
        self.assertEqual(detail["scene_graph"]["model_name"], "qwen-primary")
        self.assertEqual(
            [(row["subject"], row["relation"], row["object"]) for row in detail["triples"]],
            [("person", "hold", "camera"), ("camera", "near", "face")],
        )
        self.assertEqual(detail["tags"], ["outdoor", "Travel"])

    def test_photo_detail_falls_back_and_handles_missing_derivatives(self) -> None:
        fallback_id = self.add_photo("fallback.jpg", "e")
        self.storage.upsert_caption(
            fallback_id,
            "older fallback",
            model_name="blip-old",
            is_primary=False,
            generated_at="2026-07-01T00:00:00+00:00",
        )
        self.storage.upsert_caption(
            fallback_id,
            "newest successful fallback",
            model_name="blip-new",
            is_primary=False,
            generated_at="2026-07-02T00:00:00+00:00",
        )
        self.storage.upsert_scene_graph(
            fallback_id,
            scene_graph_text="tree beside road",
            triples=[("tree", "beside", "road")],
            model_name="qwen-fallback",
            is_primary=False,
            generated_at="2026-07-02T00:00:00+00:00",
        )
        fallback = self.storage.get_photo_detail(fallback_id)
        self.assertEqual(fallback["caption"]["caption"], "newest successful fallback")
        self.assertEqual(fallback["triples"][0]["relation"], "beside")

        empty_id = self.add_photo("empty-detail.jpg", "f")
        empty = self.storage.get_photo_detail(empty_id)
        self.assertIsNone(empty["caption"])
        self.assertIsNone(empty["scene_graph"])
        self.assertEqual(empty["triples"], [])
        self.assertEqual(empty["tags"], [])
        self.assertIsNone(self.storage.get_photo_detail("missing-photo-id"))

    def test_photo_moods_are_atomic_validated_and_returned_in_detail(self) -> None:
        first_id = self.add_photo("mood-one.jpg", "1")
        second_id = self.add_photo("mood-two.jpg", "2")

        saved = self.storage.save_photo_moods(
            {first_id: "excited", second_id: "calm"},
            annotation_set="hot_air_balloon_mood_v1",
        )
        self.assertEqual(saved[first_id].mood_label, "excited")
        self.assertTrue(saved[first_id].confirmed)
        self.assertEqual(saved[first_id].subject_role, "photographer")
        self.assertEqual(self.storage.get_photo_detail(first_id)["mood"]["source"], "manual")

        self.storage.save_photo_moods({first_id: "happy", second_id: None})
        self.assertEqual(self.storage.get_photo_mood(first_id).mood_label, "happy")
        self.assertIsNone(self.storage.get_photo_mood(second_id))

        with self.assertRaisesRegex(ValueError, "Unsupported mood"):
            self.storage.save_photo_moods({first_id: "surprised", second_id: "sad"})
        self.assertEqual(self.storage.get_photo_mood(first_id).mood_label, "happy")
        self.assertIsNone(self.storage.get_photo_mood(second_id))

        self.assertEqual(self.storage.delete_photo_moods([first_id, second_id]), 1)
        self.assertEqual(self.storage.get_photo_moods([first_id, second_id]), {})

    def test_mood_writes_do_not_change_fts_documents(self) -> None:
        photo_id = self.add_photo("mood-search.jpg", "3")
        self.storage.upsert_caption(photo_id, "hot air balloons over a field")
        before = self.storage.search_bm25("balloons field")
        self.storage.save_photo_moods({photo_id: "excited"})
        after = self.storage.search_bm25("balloons field")
        self.assertEqual(before, after)
        self.assertEqual(self.storage.search_bm25("excited"), [])

    def test_date_metadata_prefilter_limits_bm25_candidates_before_ranking(self) -> None:
        early_id = self.add_photo(
            "early.jpg", "a", captured_at_sort=100, date_local="2025-01-01"
        )
        target_id = self.add_photo(
            "target.jpg", "b", captured_at_sort=200, date_local="2025-01-02"
        )
        late_id = self.add_photo(
            "late.jpg", "c", captured_at_sort=300, date_local="2025-01-03"
        )
        for photo_id in (early_id, target_id, late_id):
            self.storage.upsert_caption(photo_id, "sunset over the sea", model_name="mock-blip")

        eligible = self.storage.metadata_eligible_ids(
            captured_from_sort=150,
            captured_to_sort=250,
            date_from="2025-01-02",
            date_to="2025-01-02",
            min_timestamp_confidence=0.9,
        )
        hits = self.storage.search_bm25("sunset sea", eligible_photo_ids=eligible)

        self.assertEqual(eligible, [target_id])
        self.assertEqual([hit.photo_id for hit in hits], [target_id])
        self.assertEqual(self.storage.search_bm25("sunset", eligible_photo_ids=[]), [])

    def test_tag_location_gps_and_event_metadata_filters(self) -> None:
        first_id = self.storage.upsert_photo(
            relative_path="first.jpg",
            content_sha256="a" * 64,
            captured_at_sort=100,
            date_local="2025-01-01",
            timestamp_source="exif",
            timestamp_confidence=1.0,
            gps_latitude=52.205,
            gps_longitude=0.119,
            location="Cambridge, United Kingdom",
        )
        second_id = self.add_photo("second.jpg", "b", location="London")
        self.storage.replace_tags(first_id, ["travel", "river"])
        self.storage.replace_tags(second_id, ["travel"])
        event_id = self.storage.upsert_event(method="time_clip_v1", title="Cambridge day")
        self.storage.replace_event_photos(event_id, [first_id])

        eligible = self.storage.metadata_eligible_ids(
            tags_all=["travel", "river"],
            location_contains="bridge",
            require_gps=True,
            event_id=event_id,
        )
        self.assertEqual(eligible, [first_id])

    def test_transaction_rollback_foreign_keys_and_child_upserts_are_idempotent(self) -> None:
        photo_id = self.add_photo("one.jpg", "a")

        with self.assertRaisesRegex(RuntimeError, "force rollback"):
            with self.storage.transaction() as connection:
                connection.execute(
                    "INSERT INTO tags(photo_id, tag, source, created_at) VALUES (?, ?, ?, ?)",
                    (photo_id, "temporary", "test", "now"),
                )
                raise RuntimeError("force rollback")

        with self.storage.transaction(write=False) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM tags WHERE tag = 'temporary'").fetchone()[0],
                0,
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.storage.upsert_caption(
                str(uuid.uuid4()), "missing parent", model_name="mock-blip"
            )

        first_caption_id = self.storage.upsert_caption(
            photo_id, "first caption", model_name="mock-blip"
        )
        second_caption_id = self.storage.upsert_caption(
            photo_id, "updated caption", model_name="mock-blip"
        )
        first_graph_id = self.storage.upsert_scene_graph(
            photo_id,
            model_name="mock-qwen",
            scene_graph_text="person holds cup",
            triples=[("person", "holds", "cup")],
        )
        second_graph_id = self.storage.upsert_scene_graph(
            photo_id,
            model_name="mock-qwen",
            scene_graph_text="person holds mug",
            triples=[("person", "holds", "mug")],
        )

        with self.storage.transaction(write=False) as connection:
            caption_count = connection.execute(
                "SELECT COUNT(*) FROM captions WHERE photo_id = ?", (photo_id,)
            ).fetchone()[0]
            graph_count = connection.execute(
                "SELECT COUNT(*) FROM scene_graphs WHERE photo_id = ?", (photo_id,)
            ).fetchone()[0]
            triples = connection.execute(
                "SELECT subject, relation, object FROM scene_graph_triples"
            ).fetchall()
            fts_count = connection.execute(
                "SELECT COUNT(*) FROM photo_fts WHERE photo_id = ?", (photo_id,)
            ).fetchone()[0]

        self.assertEqual(first_caption_id, second_caption_id)
        self.assertEqual(first_graph_id, second_graph_id)
        self.assertEqual(caption_count, 1)
        self.assertEqual(graph_count, 1)
        self.assertEqual([tuple(row) for row in triples], [("person", "holds", "mug")])
        self.assertEqual(fts_count, 1)

    def test_one_storage_instance_is_safe_for_concurrent_workers(self) -> None:
        def insert(index: int) -> str:
            return self.storage.upsert_photo(
                relative_path=f"batch/{index}.jpg",
                content_sha256=f"{index:064x}",
                file_size_bytes=index,
                timestamp_source="filesystem",
                timestamp_confidence=0.2,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            photo_ids = list(executor.map(insert, range(1, 33)))

        self.assertEqual(len(set(photo_ids)), 32)
        self.assertEqual(len(self.storage.metadata_eligible_ids()), 32)


if __name__ == "__main__":
    unittest.main()
