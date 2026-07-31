from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from vector_store import ChromaVectorStore


class VectorStoreTests(unittest.TestCase):
    def test_independent_image_and_triple_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(Path(tmp))
            vector_a = np.zeros(512, dtype=np.float32)
            vector_a[0] = 1
            vector_b = np.zeros(512, dtype=np.float32)
            vector_b[1] = 1
            records = [
                {"photo_id": "p1", "relative_path": "a.jpg", "content_sha256": "a" * 64,
                 "captured_at_sort": 1, "date_local": "2026-01-01", "timestamp_confidence": 1.0,
                 "embedding": vector_a},
                {"photo_id": "p2", "relative_path": "b.jpg", "content_sha256": "b" * 64,
                 "captured_at_sort": 2, "date_local": "2026-01-02", "timestamp_confidence": 1.0,
                 "embedding": vector_b},
            ]
            store.upsert_images(records)
            store.replace_scene_graph("p2", ["person use laptop"], [vector_a], content_sha256="b" * 64)

            self.assertEqual(store.query_images(vector_a, top_k=1)[0].photo_id, "p1")
            self.assertEqual(store.query_scene_graph(vector_a, top_k=1)[0].photo_id, "p2")
            self.assertEqual(store.query_images(vector_a, top_k=2, eligible_photo_ids=["p2"])[0].photo_id, "p2")
            store.close()

    def test_scene_replacement_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChromaVectorStore(Path(tmp))
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1
            store.replace_scene_graph("p1", ["a on b"], [vector], content_sha256="a" * 64)
            store.replace_scene_graph("p1", ["a beside b"], [vector], content_sha256="a" * 64)
            self.assertEqual(store.counts()["scene_graph_triples"], 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
