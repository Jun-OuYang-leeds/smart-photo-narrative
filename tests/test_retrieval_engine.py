from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from retrieval_engine import MultimodalRetriever, SearchFilters
from storage import PhotoStorage
from vector_store import VectorHit


class FakeClip:
    def encode_text(self, _text):
        return np.array([1.0, 0.0])

    def encode_images_batch(self, _images):
        return np.array([[1.0, 0.0]])


class FakeVectors:
    def __init__(self):
        self.last_eligible = None

    def query_images(self, _embedding, *, top_k, eligible_photo_ids=None):
        self.last_eligible = list(eligible_photo_ids or [])
        order = ["p_clip", "p_other"]
        return [VectorHit(pid, index / 10, 1-index/10, pid, {"photo_id": pid})
                for index, pid in enumerate(order) if pid in self.last_eligible][:top_k]

    def query_scene_graph(self, _embedding, *, top_k, eligible_photo_ids=None):
        eligible = set(eligible_photo_ids or [])
        return [VectorHit("p_sg", 0.05, 0.95, "p_sg:000", {"photo_id": "p_sg", "triple_text": "person use laptop"})] if "p_sg" in eligible else []


class RetrievalEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = PhotoStorage(Path(self.tmp.name) / "app.db")
        rows = [
            ("p_clip", "clip.jpg", "a"*64, "2026-01-01T09:00:00", 1, "a beach"),
            ("p_caption", "caption.jpg", "b"*64, "2026-01-02T09:00:00", 2, "rare telescope object"),
            ("p_sg", "sg.jpg", "c"*64, "2026-01-02T10:00:00", 3, "a room"),
            ("p_other", "other.jpg", "d"*64, "2026-01-03T09:00:00", 4, "something else"),
        ]
        for pid, path, digest, captured, sort_value, caption in rows:
            self.storage.upsert_photo(
                photo_id=pid.replace("p_", "00000000-0000-4000-8000-") if False else None,
                relative_path=path, content_sha256=digest, captured_at=captured,
                captured_at_sort=sort_value, date_local=captured[:10], timestamp_source="exif_original",
                timestamp_confidence=1.0, image_width=10, image_height=10,
            )
            actual = self.storage.get_photo_by_path(path)["photo_id"]
            # Tests use predictable IDs by rewriting only inside the private fixture DB.
            with self.storage.transaction() as connection:
                connection.execute("UPDATE photos SET photo_id=? WHERE photo_id=?", (pid, actual))
            self.storage.upsert_caption(pid, caption)
        self.storage.upsert_scene_graph(
            "p_sg", scene_graph_text="person use laptop", triples=[("person", "use", "laptop")],
            model_name="qwen", status="ok",
        )
        self.vectors = FakeVectors()
        self.engine = MultimodalRetriever(self.storage, self.vectors, FakeClip())

    def tearDown(self):
        self.tmp.cleanup()

    def test_caption_is_independent_recall_not_clip_rerank(self):
        response = self.engine.search_standard("rare telescope", top_k=4, enabled_channels=("clip", "caption"))
        ids = [item.photo_id for item in response.results]
        self.assertIn("p_caption", ids)
        target = next(item for item in response.results if item.photo_id == "p_caption")
        self.assertEqual(target.matched_modalities, ["caption"])

    def test_scene_graph_channel_has_per_result_triple(self):
        response = self.engine.search_standard("person using laptop", top_k=4, enabled_channels=("scene_graph",))
        target = response.results[0]
        self.assertEqual(target.photo_id, "p_sg")
        self.assertEqual(target.matched_scene_graph_triples, ["person use laptop"])

    def test_caption_channel_does_not_leak_scene_graph_fts_text(self):
        caption_only = self.engine.search_standard(
            "person laptop", top_k=4, enabled_channels=("caption",)
        )
        self.assertEqual(caption_only.results, [])
        graph_only = self.engine.search_standard(
            "person laptop", top_k=4, enabled_channels=("scene_graph",)
        )
        self.assertEqual(graph_only.results[0].photo_id, "p_sg")

    def test_date_filter_is_passed_before_dense_recall(self):
        filters = SearchFilters(start_date="2026-01-02", end_date="2026-01-02")
        response = self.engine.search_standard("room", top_k=4, filters=filters, enabled_channels=("clip",))
        self.assertEqual(set(self.vectors.last_eligible), {"p_caption", "p_sg"})
        self.assertTrue(all(item.datetime.startswith("2026-01-02") for item in response.results))

    def test_temporal_pair_respects_order(self):
        response = self.engine.search("rare telescope before person using laptop within 2 hours", top_k=3)
        self.assertTrue(response.temporal_pairs)
        pair = response.temporal_pairs[0]
        self.assertEqual(pair.before.photo_id, "p_caption")
        self.assertEqual(pair.after.photo_id, "p_sg")

    def test_temporal_neighbor_before_uses_real_chronology(self):
        response = self.engine.search("what happened before person using laptop within 2 hours", top_k=3)
        self.assertEqual(response.query.query_type, "temporal_neighbor")
        self.assertTrue(response.temporal_pairs)
        pair = response.temporal_pairs[0]
        self.assertEqual(pair.before.photo_id, "p_caption")
        self.assertEqual(pair.after.photo_id, "p_sg")

    def test_temporal_neighbor_after_uses_real_chronology(self):
        response = self.engine.search("what happened after rare telescope within 2 hours", top_k=3)
        self.assertEqual(response.query.query_type, "temporal_neighbor")
        self.assertTrue(response.temporal_pairs)
        pair = response.temporal_pairs[0]
        self.assertEqual(pair.before.photo_id, "p_caption")
        self.assertEqual(pair.after.photo_id, "p_sg")


if __name__ == "__main__":
    unittest.main()
