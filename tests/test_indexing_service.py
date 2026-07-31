from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from config import CLIP_INDEX_MARKER_TAG
from indexing_service import IndexingService
from storage import PhotoStorage


def dominant_colour(image: Image.Image) -> str:
    red, green, blue = image.getpixel((0, 0))[:3]
    if red < 10 and green < 10 and blue < 10:
        return "black"
    return max(((red, "red"), (green, "green"), (blue, "blue")))[1]


class FakeClipBackend:
    model_name = "fake-clip"
    model_version = "1"

    def __init__(self, *, oom_above: int | None = None) -> None:
        self.oom_above = oom_above
        self.encode_batch_sizes: list[int] = []
        self.encoded_colours: list[str] = []

    def encode_images_batch(self, images):
        self.encode_batch_sizes.append(len(images))
        if self.oom_above is not None and len(images) > self.oom_above:
            raise RuntimeError("CUDA out of memory while allocating a tensor")
        colours = [dominant_colour(image) for image in images]
        self.encoded_colours.extend(colours)
        lookup = {
            "red": [1.0, 0.0, 0.0],
            "green": [0.0, 1.0, 0.0],
            "blue": [0.0, 0.0, 1.0],
            "black": [0.0, 0.0, 0.0],
        }
        return [lookup[colour] for colour in colours]


class FakeCaptionBackend:
    model_name = "fake-blip"
    model_version = "1"

    def __init__(self, *, fail_colour: str | None = None) -> None:
        self.fail_colour = fail_colour
        self.batch_sizes: list[int] = []
        self.generated_colours: list[str] = []

    def generate_captions_batch(self, images):
        self.batch_sizes.append(len(images))
        colours = [dominant_colour(image) for image in images]
        if self.fail_colour is not None and self.fail_colour in colours:
            raise RuntimeError(f"caption rejected {self.fail_colour} image")
        self.generated_colours.extend(colours)
        return [f"a {colour} test photo" for colour in colours]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.upsert_batch_sizes: list[int] = []
        self.deleted_ids: list[str] = []

    def all_image_ids(self):
        return list(self.records)

    def upsert_images(self, records):
        self.upsert_batch_sizes.append(len(records))
        for record in records:
            self.records[str(record["photo_id"])] = dict(record)

    def delete_images(self, photo_ids):
        for photo_id in photo_ids:
            self.deleted_ids.append(photo_id)
            self.records.pop(photo_id, None)


class NoIdProbeVectorStore:
    """Minimal protocol implementation: completion falls back to SQLite tags."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.upsert_batch_sizes: list[int] = []

    def upsert_images(self, records):
        self.upsert_batch_sizes.append(len(records))
        for record in records:
            self.records[str(record["photo_id"])] = dict(record)


class IndexingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.photo_root = self.root / "photos"
        self.photo_root.mkdir()
        self.storage = PhotoStorage(self.root / "canonical.sqlite3")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def make_image(path: Path, colour: tuple[int, int, int]) -> None:
        Image.new("RGB", (12, 10), colour).save(path)

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def metadata_extractor(path: Path, base_dir: Path):
        confidence_by_stem = {"a_red": "high", "c_blue": "medium"}
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
        except Exception as error:
            return {
                "image_width": 0,
                "image_height": 0,
                "timestamp_confidence": "low",
                "error": f"Image loading failed: {error}",
            }
        day = 1 + sum(path.stem.encode("utf-8")) % 20
        return {
            # A deliberately wrong path proves the service associates metadata
            # with the input Path rather than zipping an independent list.
            "file_path": "wrong.jpg",
            "image_width": width,
            "image_height": height,
            "datetime_original": f"2026-01-{day:02d}T10:00:00",
            "captured_at_sort": day * 100,
            "date_local": f"2026-01-{day:02d}",
            "timestamp_source": "exif_original",
            "timestamp_confidence": confidence_by_stem.get(path.stem, "low"),
            "location": f"location-{path.stem}",
            "error": None,
        }

    def make_service(
        self,
        *,
        clip: FakeClipBackend | None = None,
        caption: FakeCaptionBackend | None = None,
        vector_store=None,
        clip_batch_size: int = 8,
        caption_batch_size: int = 4,
    ):
        clip = clip or FakeClipBackend()
        caption = caption or FakeCaptionBackend()
        vector_store = vector_store or FakeVectorStore()
        service = IndexingService(
            self.storage,
            clip,
            caption,
            vector_store,
            metadata_extractor=self.metadata_extractor,
            clip_batch_size=clip_batch_size,
            caption_batch_size=caption_batch_size,
        )
        return service, clip, caption, vector_store

    def test_corrupt_middle_image_does_not_shift_photo_metadata_or_outputs(self) -> None:
        red = self.photo_root / "a_red.png"
        broken = self.photo_root / "b_broken.png"
        blue = self.photo_root / "c_blue.png"
        self.make_image(red, (255, 0, 0))
        broken.write_bytes(b"not an image")
        self.make_image(blue, (0, 0, 255))
        service, _, _, vectors = self.make_service()

        summary = service.index_photos([red, broken, blue], self.photo_root)

        self.assertEqual(summary.discovered_count, 3)
        self.assertEqual(summary.prepared_count, 2)
        self.assertEqual(summary.indexed_count, 2)
        self.assertEqual(summary.failed_count, 1)
        self.assertEqual(summary.status, "completed_with_errors")
        self.assertEqual(len(vectors.records), 2)

        records_by_path = {record["relative_path"]: record for record in vectors.records.values()}
        self.assertEqual(set(records_by_path), {"a_red.png", "c_blue.png"})
        self.assertEqual(records_by_path["a_red.png"]["embedding"], [1.0, 0.0, 0.0])
        self.assertEqual(records_by_path["c_blue.png"]["embedding"], [0.0, 0.0, 1.0])
        self.assertEqual(records_by_path["a_red.png"]["timestamp_confidence"], 1.0)
        self.assertEqual(records_by_path["c_blue.png"]["timestamp_confidence"], 0.6)

        red_row = self.storage.get_photo_by_path("a_red.png")
        blue_row = self.storage.get_photo_by_path("c_blue.png")
        broken_row = self.storage.get_photo_by_path("b_broken.png")
        self.assertEqual(red_row["location"], "location-a_red")
        self.assertEqual(blue_row["location"], "location-c_blue")
        self.assertEqual(broken_row["content_sha256"], self.sha256(broken))
        self.assertEqual(
            self.storage.get_caption(
                red_row["photo_id"], model_name="fake-blip", model_version="1"
            )["caption"],
            "a red test photo",
        )
        self.assertEqual(
            self.storage.get_caption(
                blue_row["photo_id"], model_name="fake-blip", model_version="1"
            )["caption"],
            "a blue test photo",
        )
        failures = self.storage.list_index_failures(summary.run_id)
        self.assertEqual([(row["relative_path"], row["stage"]) for row in failures], [("b_broken.png", "metadata")])

    def test_incremental_run_skips_completed_work_and_force_recomputes(self) -> None:
        red = self.photo_root / "red.png"
        blue = self.photo_root / "blue.png"
        self.make_image(red, (255, 0, 0))
        self.make_image(blue, (0, 0, 255))
        service, clip, caption, vectors = self.make_service()

        first = service.index_photos([red, blue], self.photo_root)
        clip_calls_after_first = len(clip.encode_batch_sizes)
        caption_calls_after_first = len(caption.batch_sizes)
        vector_calls_after_first = len(vectors.upsert_batch_sizes)
        second = service.index_photos([red, blue], self.photo_root)

        self.assertEqual(first.indexed_count, 2)
        self.assertEqual(second.indexed_count, 0)
        self.assertEqual(second.skipped_count, 2)
        self.assertEqual(len(clip.encode_batch_sizes), clip_calls_after_first)
        self.assertEqual(len(caption.batch_sizes), caption_calls_after_first)
        self.assertEqual(len(vectors.upsert_batch_sizes), vector_calls_after_first)

        forced = service.index_photos([red, blue], self.photo_root, force=True)
        self.assertEqual(forced.indexed_count, 2)
        self.assertEqual(forced.skipped_count, 0)
        self.assertGreater(len(clip.encode_batch_sizes), clip_calls_after_first)
        self.assertGreater(len(caption.batch_sizes), caption_calls_after_first)

        run_row = self.storage.get_index_run(forced.run_id)
        self.assertEqual(run_row["status"], "completed")
        self.assertEqual(run_row["indexed_count"], 2)
        self.assertIn('"final_clip_batch_size"', run_row["notes"])

    def test_clip_writes_only_a_non_semantic_marker_that_stays_out_of_fts(self) -> None:
        red = self.photo_root / "red.png"
        self.make_image(red, (255, 0, 0))
        service, _, _, _ = self.make_service()
        service.index_photos([red], self.photo_root)

        with self.storage.transaction(write=False) as connection:
            row = connection.execute("SELECT photo_id FROM photos").fetchone()
            tags = [
                dict(value)
                for value in connection.execute(
                    "SELECT tag, source FROM tags WHERE photo_id = ?", (row["photo_id"],)
                )
            ]
            fts_tags = connection.execute(
                "SELECT tags FROM photo_fts WHERE photo_id = ?", (row["photo_id"],)
            ).fetchone()["tags"]

        # Exactly one non-semantic marker carries the CLIP identity; the 72
        # preset zero-shot labels are gone and nothing tag-like is searchable.
        self.assertEqual([value["tag"] for value in tags], [CLIP_INDEX_MARKER_TAG])
        self.assertTrue(tags[0]["source"].startswith("clip:"))
        self.assertEqual((fts_tags or "").strip(), "")
        # The marker still authorises the incremental fast path.
        self.assertEqual(service.index_photos([red], self.photo_root).skipped_count, 1)

    def test_matching_file_fingerprint_skips_hash_and_metadata_but_repairs_stages(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        service, clip, caption, vectors = self.make_service()
        service.index_photos([photo], self.photo_root)
        row = self.storage.get_photo_by_path("red.png")
        stored_caption = self.storage.get_caption(
            row["photo_id"], model_name="fake-blip", model_version="1"
        )

        # Simulate an incomplete derived index while the source file itself is
        # unchanged.  The repair must not reread bytes for SHA-256 or EXIF.
        vectors.records.clear()
        self.storage.upsert_caption(
            row["photo_id"],
            stored_caption["caption"],
            model_name="fake-blip",
            model_version="1",
            language=stored_caption["language"],
            status="stale",
        )

        def unexpected_hash(*_args, **_kwargs):
            raise AssertionError("unchanged fast path must not compute SHA-256")

        def unexpected_metadata(*_args, **_kwargs):
            raise AssertionError("unchanged fast path must not extract metadata")

        service._sha256_file = unexpected_hash
        service.metadata_extractor = unexpected_metadata
        repaired = service.index_photos([photo], self.photo_root)

        self.assertEqual(repaired.clip_indexed_count, 1)
        self.assertEqual(repaired.caption_indexed_count, 1)
        self.assertEqual(clip.encode_batch_sizes, [1, 1])
        self.assertEqual(caption.batch_sizes, [1, 1])
        self.assertIn(row["photo_id"], vectors.records)

        # A vector without its SQLite tags is also incomplete and must rerun only
        # CLIP, still using the same no-hash/no-EXIF fast path.
        self.storage.replace_tags(row["photo_id"], [], source=service._clip_tag_source)
        repaired_tags = service.index_photos([photo], self.photo_root)
        self.assertEqual(repaired_tags.clip_indexed_count, 1)
        self.assertEqual(repaired_tags.caption_indexed_count, 0)
        self.assertEqual(clip.encode_batch_sizes, [1, 1, 1])
        self.assertEqual(caption.batch_sizes, [1, 1])

    def test_force_bypasses_fingerprint_fast_path(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        service, clip, caption, _ = self.make_service()
        service.index_photos([photo], self.photo_root)

        hash_calls: list[Path] = []
        metadata_calls: list[Path] = []
        original_hash = service._sha256_file

        def tracked_hash(path: Path, *args, **kwargs):
            hash_calls.append(path)
            return original_hash(path, *args, **kwargs)

        def tracked_metadata(path: Path, base_dir: Path):
            metadata_calls.append(path)
            return self.metadata_extractor(path, base_dir)

        service._sha256_file = tracked_hash
        service.metadata_extractor = tracked_metadata
        forced = service.index_photos([photo], self.photo_root, force=True)

        self.assertEqual(forced.indexed_count, 1)
        self.assertEqual(hash_calls, [photo])
        self.assertEqual(metadata_calls, [photo])
        self.assertEqual(clip.encode_batch_sizes, [1, 1])
        self.assertEqual(caption.batch_sizes, [1, 1])

    def test_transient_metadata_failure_does_not_poison_fingerprint_fast_path(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        service, clip, caption, _ = self.make_service()
        metadata_calls: list[Path] = []

        def transient_metadata(path: Path, base_dir: Path):
            metadata_calls.append(path)
            if len(metadata_calls) == 1:
                return {"error": "temporary EXIF reader failure"}
            return self.metadata_extractor(path, base_dir)

        service.metadata_extractor = transient_metadata
        first = service.index_photos([photo], self.photo_root)
        failed_row = self.storage.get_photo_by_path("red.png")

        self.assertEqual(first.failed_count, 1)
        self.assertEqual(first.prepared_count, 0)
        self.assertIsNone(failed_row["file_size_bytes"])
        self.assertIsNone(failed_row["file_mtime_ns"])
        self.assertEqual(clip.encode_batch_sizes, [])
        self.assertEqual(caption.batch_sizes, [])

        hash_calls: list[Path] = []
        original_hash = service._sha256_file

        def tracked_hash(path: Path, *args, **kwargs):
            hash_calls.append(path)
            return original_hash(path, *args, **kwargs)

        service._sha256_file = tracked_hash
        repaired = service.index_photos([photo], self.photo_root)
        repaired_row = self.storage.get_photo_by_path("red.png")

        self.assertEqual(repaired.status, "completed")
        self.assertEqual(repaired.indexed_count, 1)
        self.assertEqual(metadata_calls, [photo, photo])
        self.assertEqual(hash_calls, [photo])
        self.assertEqual(repaired_row["file_size_bytes"], photo.stat().st_size)
        self.assertEqual(repaired_row["file_mtime_ns"], photo.stat().st_mtime_ns)
        self.assertEqual(clip.encode_batch_sizes, [1])
        self.assertEqual(caption.batch_sizes, [1])

    def test_failed_force_caption_refresh_is_retried_incrementally(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        service, clip, caption, _ = self.make_service()
        service.index_photos([photo], self.photo_root)
        row = self.storage.get_photo_by_path("red.png")

        caption.fail_colour = "red"
        forced = service.index_photos([photo], self.photo_root, force=True)
        stale_caption = self.storage.get_caption(
            row["photo_id"],
            model_name="fake-blip",
            model_version="1",
            primary_only=False,
        )

        self.assertEqual(forced.status, "completed_with_errors")
        self.assertEqual(forced.failed_count, 1)
        self.assertIsNone(stale_caption)
        self.assertEqual(clip.encode_batch_sizes, [1, 1])
        self.assertEqual(caption.batch_sizes, [1, 1])

        caption.fail_colour = None
        repaired = service.index_photos([photo], self.photo_root)

        self.assertEqual(repaired.status, "completed")
        self.assertEqual(repaired.clip_indexed_count, 0)
        self.assertEqual(repaired.caption_indexed_count, 1)
        self.assertEqual(clip.encode_batch_sizes, [1, 1])
        self.assertEqual(caption.batch_sizes, [1, 1, 1])

    def test_mtime_only_change_rehashes_metadata_and_models(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        service, clip, caption, vectors = self.make_service()
        service.index_photos([photo], self.photo_root)
        first_row = self.storage.get_photo_by_path("red.png")

        hash_calls: list[Path] = []
        metadata_calls: list[Path] = []
        original_hash = service._sha256_file

        def tracked_hash(path: Path, *args, **kwargs):
            hash_calls.append(path)
            return original_hash(path, *args, **kwargs)

        def tracked_metadata(path: Path, base_dir: Path):
            metadata_calls.append(path)
            return self.metadata_extractor(path, base_dir)

        service._sha256_file = tracked_hash
        service.metadata_extractor = tracked_metadata
        stat = photo.stat()
        os.utime(
            photo,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000),
        )
        changed = service.index_photos([photo], self.photo_root)
        second_row = self.storage.get_photo_by_path("red.png")

        self.assertEqual(changed.indexed_count, 1)
        self.assertEqual(first_row["photo_id"], second_row["photo_id"])
        self.assertEqual(hash_calls, [photo])
        self.assertEqual(metadata_calls, [photo])
        self.assertEqual(clip.encode_batch_sizes, [1, 1])
        self.assertEqual(caption.batch_sizes, [1, 1])
        self.assertEqual(vectors.deleted_ids, [first_row["photo_id"]])

    def test_size_only_change_rehashes_metadata_and_models(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        service, clip, caption, _ = self.make_service()
        service.index_photos([photo], self.photo_root)

        hash_calls: list[Path] = []
        metadata_calls: list[Path] = []
        original_hash = service._sha256_file

        def tracked_hash(path: Path, *args, **kwargs):
            hash_calls.append(path)
            return original_hash(path, *args, **kwargs)

        def tracked_metadata(path: Path, base_dir: Path):
            metadata_calls.append(path)
            return self.metadata_extractor(path, base_dir)

        service._sha256_file = tracked_hash
        service.metadata_extractor = tracked_metadata
        stat = photo.stat()
        with photo.open("ab") as handle:
            handle.write(b"trailing-test-bytes")
        os.utime(photo, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(photo.stat().st_size, stat.st_size)
        self.assertEqual(photo.stat().st_mtime_ns, stat.st_mtime_ns)

        changed = service.index_photos([photo], self.photo_root)

        self.assertEqual(changed.indexed_count, 1)
        self.assertEqual(hash_calls, [photo])
        self.assertEqual(metadata_calls, [photo])
        self.assertEqual(clip.encode_batch_sizes, [1, 1])
        self.assertEqual(caption.batch_sizes, [1, 1])

    def test_ctime_mismatch_rehashes_but_reuses_unchanged_model_outputs(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        service, clip, caption, _ = self.make_service()
        service.index_photos([photo], self.photo_root)
        original_row = self.storage.get_photo_by_path("red.png")

        with self.storage.transaction() as connection:
            connection.execute(
                "UPDATE photos SET file_ctime_ns = ? WHERE photo_id = ?",
                (int(original_row["file_ctime_ns"]) + 1, original_row["photo_id"]),
            )

        hash_calls: list[Path] = []
        metadata_calls: list[Path] = []
        original_hash = service._sha256_file

        def tracked_hash(path: Path, *args, **kwargs):
            hash_calls.append(path)
            return original_hash(path, *args, **kwargs)

        def tracked_metadata(path: Path, base_dir: Path):
            metadata_calls.append(path)
            return self.metadata_extractor(path, base_dir)

        service._sha256_file = tracked_hash
        service.metadata_extractor = tracked_metadata
        refreshed = service.index_photos([photo], self.photo_root)

        self.assertEqual(refreshed.skipped_count, 1)
        self.assertEqual(hash_calls, [photo])
        self.assertEqual(metadata_calls, [photo])
        self.assertEqual(clip.encode_batch_sizes, [1])
        self.assertEqual(caption.batch_sizes, [1])

    def test_model_version_change_only_recomputes_affected_stage(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        vectors = FakeVectorStore()
        service, clip, caption_v1, _ = self.make_service(vector_store=vectors)
        service.index_photos([photo], self.photo_root)

        caption_v2 = FakeCaptionBackend()
        upgraded = IndexingService(
            self.storage,
            clip,
            caption_v2,
            vectors,
            metadata_extractor=self.metadata_extractor,
            clip_batch_size=8,
            caption_batch_size=4,
            caption_model_name="fake-blip",
            caption_model_version="2",
        )
        summary = upgraded.index_photos([photo], self.photo_root)

        self.assertEqual(summary.clip_indexed_count, 0)
        self.assertEqual(summary.caption_indexed_count, 1)
        self.assertEqual(clip.encode_batch_sizes, [1])
        self.assertEqual(caption_v1.batch_sizes, [1])
        self.assertEqual(caption_v2.batch_sizes, [1])

    def test_clip_version_round_trip_never_reuses_stale_vector_identity(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        vectors = FakeVectorStore()
        service_v1, clip_v1, caption, _ = self.make_service(vector_store=vectors)
        service_v1.index_photos([photo], self.photo_root)
        row = self.storage.get_photo_by_path("red.png")

        clip_v2 = FakeClipBackend()
        service_v2 = IndexingService(
            self.storage,
            clip_v2,
            caption,
            vectors,
            metadata_extractor=self.metadata_extractor,
            clip_batch_size=8,
            caption_batch_size=4,
            clip_model_name="fake-clip",
            clip_model_version="2",
        )
        upgraded = service_v2.index_photos([photo], self.photo_root)
        self.assertEqual(upgraded.clip_indexed_count, 1)
        self.assertEqual(upgraded.caption_indexed_count, 0)
        self.assertEqual(
            self.storage.list_tag_sources(row["photo_id"], source_prefix="clip:"),
            {"clip:fake-clip:2"},
        )

        downgraded = service_v1.index_photos([photo], self.photo_root)
        self.assertEqual(downgraded.clip_indexed_count, 1)
        self.assertEqual(downgraded.caption_indexed_count, 0)
        self.assertEqual(clip_v1.encode_batch_sizes, [1, 1])
        self.assertEqual(clip_v2.encode_batch_sizes, [1])
        self.assertEqual(caption.batch_sizes, [1])
        self.assertEqual(
            self.storage.list_tag_sources(row["photo_id"], source_prefix="clip:"),
            {"clip:fake-clip:1"},
        )

    def test_clip_version_tag_commit_failure_cannot_authorize_new_vector(self) -> None:
        photo = self.photo_root / "red.png"
        self.make_image(photo, (255, 0, 0))
        vectors = FakeVectorStore()
        service_v1, clip_v1, caption, _ = self.make_service(vector_store=vectors)
        service_v1.index_photos([photo], self.photo_root)
        row = self.storage.get_photo_by_path("red.png")

        clip_v2 = FakeClipBackend()
        service_v2 = IndexingService(
            self.storage,
            clip_v2,
            caption,
            vectors,
            metadata_extractor=self.metadata_extractor,
            clip_batch_size=8,
            caption_batch_size=4,
            clip_model_name="fake-clip",
            clip_model_version="2",
        )
        original_replace_tags = self.storage.replace_tags

        def fail_nonempty_tag_commit(photo_id, tags, **kwargs):
            selected = list(tags)
            if selected:
                raise RuntimeError("simulated tag commit failure after vector upsert")
            return original_replace_tags(photo_id, selected, **kwargs)

        self.storage.replace_tags = fail_nonempty_tag_commit
        failed_upgrade = service_v2.index_photos([photo], self.photo_root)
        self.storage.replace_tags = original_replace_tags

        self.assertEqual(failed_upgrade.status, "completed_with_errors")
        self.assertEqual(clip_v2.encode_batch_sizes, [1])
        self.assertEqual(
            self.storage.list_tag_sources(row["photo_id"], source_prefix="clip:"),
            set(),
        )

        returned_to_v1 = service_v1.index_photos([photo], self.photo_root)
        self.assertEqual(returned_to_v1.clip_indexed_count, 1)
        self.assertEqual(returned_to_v1.caption_indexed_count, 0)
        self.assertEqual(clip_v1.encode_batch_sizes, [1, 1])
        self.assertEqual(
            self.storage.list_tag_sources(row["photo_id"], source_prefix="clip:"),
            {"clip:fake-clip:1"},
        )

    def test_rename_rehashes_and_preserves_stable_uuid(self) -> None:
        original = self.photo_root / "before.png"
        renamed = self.photo_root / "after.png"
        self.make_image(original, (255, 0, 0))
        service, clip, caption, _ = self.make_service()
        service.index_photos([original], self.photo_root)
        first_row = self.storage.get_photo_by_path("before.png")
        original.rename(renamed)

        hash_calls: list[Path] = []
        original_hash = service._sha256_file

        def tracked_hash(path: Path, *args, **kwargs):
            hash_calls.append(path)
            return original_hash(path, *args, **kwargs)

        service._sha256_file = tracked_hash
        renamed_run = service.index_photos([renamed], self.photo_root)
        renamed_row = self.storage.get_photo_by_path("after.png")

        self.assertEqual(hash_calls, [renamed])
        self.assertEqual(first_row["photo_id"], renamed_row["photo_id"])
        self.assertIsNone(self.storage.get_photo_by_path("before.png"))
        self.assertEqual(renamed_run.indexed_count, 0)
        self.assertEqual(renamed_run.skipped_count, 1)
        self.assertEqual(clip.encode_batch_sizes, [1])
        self.assertEqual(caption.batch_sizes, [1])

    def test_incremental_skip_works_with_minimal_vector_store_protocol(self) -> None:
        red = self.photo_root / "red.png"
        self.make_image(red, (255, 0, 0))
        minimal_vectors = NoIdProbeVectorStore()
        service, clip, caption, _ = self.make_service(vector_store=minimal_vectors)

        service.index_photos([red], self.photo_root)
        second = service.index_photos([red], self.photo_root)

        self.assertEqual(second.skipped_count, 1)
        self.assertEqual(clip.encode_batch_sizes, [1])
        self.assertEqual(caption.batch_sizes, [1])
        self.assertEqual(minimal_vectors.upsert_batch_sizes, [1])

    def test_changed_file_reuses_uuid_but_recomputes_both_models(self) -> None:
        photo = self.photo_root / "changing.png"
        self.make_image(photo, (255, 0, 0))
        service, clip, caption, vectors = self.make_service()
        first = service.index_photos([photo], self.photo_root)
        first_row = self.storage.get_photo_by_path("changing.png")
        first_hash = first_row["content_sha256"]

        self.make_image(photo, (0, 0, 255))
        second = service.index_photos([photo], self.photo_root)
        second_row = self.storage.get_photo_by_path("changing.png")
        vector_record = vectors.records[second_row["photo_id"]]
        caption_row = self.storage.get_caption(
            second_row["photo_id"], model_name="fake-blip", model_version="1"
        )

        self.assertEqual(first.status, "completed")
        self.assertEqual(second.indexed_count, 1)
        self.assertEqual(first_row["photo_id"], second_row["photo_id"])
        self.assertNotEqual(first_hash, second_row["content_sha256"])
        self.assertEqual(vector_record["content_sha256"], second_row["content_sha256"])
        self.assertEqual(vector_record["embedding"], [0.0, 0.0, 1.0])
        self.assertEqual(caption_row["caption"], "a blue test photo")
        self.assertEqual(len(clip.encode_batch_sizes), 2)
        self.assertEqual(len(caption.batch_sizes), 2)
        self.assertEqual(vectors.deleted_ids, [second_row["photo_id"]])

    def test_cuda_oom_halves_clip_batch_and_retries_without_losing_photos(self) -> None:
        paths = []
        for index in range(8):
            path = self.photo_root / f"photo_{index}.png"
            # Give every file distinct bytes while keeping a stable dominant
            # colour; the canonical store intentionally deduplicates exact
            # content hashes.
            image = Image.new(
                "RGB",
                (12 + index, 10),
                (255, 0, 0) if index % 2 == 0 else (0, 0, 255),
            )
            image.save(path)
            image.close()
            paths.append(path)
        clip = FakeClipBackend(oom_above=2)
        service, clip, _, vectors = self.make_service(clip=clip)

        summary = service.index_photos(paths, self.photo_root)

        self.assertEqual(clip.encode_batch_sizes[:3], [8, 4, 2])
        self.assertEqual(summary.final_clip_batch_size, 2)
        self.assertEqual(summary.indexed_count, 8)
        self.assertEqual(summary.failed_count, 0)
        self.assertEqual(len(vectors.records), 8)
        self.assertEqual(vectors.upsert_batch_sizes, [2, 2, 2, 2])

    def test_failed_caption_after_content_change_remains_pending_for_next_run(self) -> None:
        photo = self.photo_root / "changing.png"
        self.make_image(photo, (255, 0, 0))
        caption = FakeCaptionBackend()
        service, clip, caption, _ = self.make_service(caption=caption)
        service.index_photos([photo], self.photo_root)

        self.make_image(photo, (0, 0, 255))
        caption.fail_colour = "blue"
        failed = service.index_photos([photo], self.photo_root)
        row = self.storage.get_photo_by_path("changing.png")
        self.assertEqual(failed.status, "completed_with_errors")
        self.assertIsNone(
            self.storage.get_caption(
                row["photo_id"], model_name="fake-blip", model_version="1"
            )
        )

        caption.fail_colour = None
        recovered = service.index_photos([photo], self.photo_root)
        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.clip_skipped_count, 1)
        self.assertEqual(recovered.caption_indexed_count, 1)
        self.assertEqual(len(clip.encode_batch_sizes), 2)
        self.assertEqual(
            self.storage.get_caption(
                row["photo_id"], model_name="fake-blip", model_version="1"
            )["caption"],
            "a blue test photo",
        )

    def test_single_photo_backend_failure_is_recorded_and_other_photos_continue(self) -> None:
        white = self.photo_root / "a_white.png"
        black = self.photo_root / "b_black.png"
        blue = self.photo_root / "c_blue.png"
        self.make_image(white, (255, 255, 255))
        self.make_image(black, (0, 0, 0))
        self.make_image(blue, (0, 0, 255))
        caption = FakeCaptionBackend(fail_colour="black")
        service, _, caption, vectors = self.make_service(caption=caption)

        summary = service.index_photos([white, black, blue], self.photo_root)

        self.assertEqual(summary.status, "completed_with_errors")
        self.assertEqual(summary.clip_indexed_count, 3)
        self.assertEqual(summary.caption_indexed_count, 2)
        self.assertEqual(summary.indexed_count, 2)
        self.assertEqual(summary.failed_count, 1)
        self.assertEqual(len(vectors.records), 3)
        self.assertEqual(caption.batch_sizes, [3, 1, 1, 1])

        black_row = self.storage.get_photo_by_path("b_black.png")
        blue_row = self.storage.get_photo_by_path("c_blue.png")
        self.assertIsNone(
            self.storage.get_caption(
                black_row["photo_id"], model_name="fake-blip", model_version="1"
            )
        )
        self.assertEqual(
            self.storage.get_caption(
                blue_row["photo_id"], model_name="fake-blip", model_version="1"
            )["caption"],
            "a blue test photo",
        )
        failures = self.storage.list_index_failures(summary.run_id)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["photo_id"], black_row["photo_id"])
        self.assertEqual(failures[0]["stage"], "caption_inference")


if __name__ == "__main__":
    unittest.main()
