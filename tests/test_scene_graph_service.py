from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scene_graph_io import sha256_file
from scene_graph_service import apply_indexable_scene_graphs
from storage import PhotoStorage


class FakeClip:
    def encode_texts_batch(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectors:
    def __init__(self):
        self.records = []

    def replace_scene_graph(self, photo_id, texts, embeddings, *, content_sha256):
        self.records.append((photo_id, list(texts), list(embeddings), content_sha256))


class SceneGraphServiceTests(unittest.TestCase):
    def test_hash_verified_apply_uses_sqlite_photo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "renamed.jpg"
            Image.new("RGB", (8, 8), "red").save(image)
            digest = sha256_file(image)
            storage = PhotoStorage(root / "app.db")
            photo_id = storage.upsert_photo(
                relative_path=image.name, content_sha256=digest, image_width=8, image_height=8,
                timestamp_confidence=0.0,
            )
            payload = {
                "status": "success", "source": "remote", "sha256": digest,
                "dataset_id": "personal", "photo_id": "cloud-id",
                "model_id": "qwen", "prompt_version": "v1",
                "source_ref": "legacy-recovered:abc:1", "output_sha256": "e" * 64,
                "scene_graph": [{"subject": "cup", "predicate": "be on", "object": "table"}],
                "scene_graph_flat_text": "cup be on table",
            }
            path = root / "indexable.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            vectors = FakeVectors()
            summary = apply_indexable_scene_graphs(
                path, storage=storage, vector_store=vectors, clip_backend=FakeClip(), photos_root=root,
            )
            self.assertEqual(summary.applied, 1)
            self.assertEqual(vectors.records[0][0], photo_id)
            with storage.transaction(write=False) as connection:
                graph = connection.execute(
                    "SELECT source_ref, output_sha256 FROM scene_graphs WHERE photo_id = ? AND is_primary = 1",
                    (photo_id,),
                ).fetchone()
            self.assertEqual(graph["source_ref"], "legacy-recovered:abc:1")
            self.assertEqual(graph["output_sha256"], "e" * 64)

    def test_checksum_conflict_is_not_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.jpg"
            Image.new("RGB", (8, 8), "red").save(image)
            digest = sha256_file(image)
            storage = PhotoStorage(root / "app.db")
            storage.upsert_photo(relative_path=image.name, content_sha256=digest, image_width=8, image_height=8)
            path = root / "indexable.jsonl"
            path.write_text(json.dumps({
                "status": "success", "source": "remote", "sha256": "f" * 64,
                "scene_graph": [{"subject": "x", "predicate": "on", "object": "y"}],
            }) + "\n", encoding="utf-8")
            summary = apply_indexable_scene_graphs(
                path, storage=storage, vector_store=FakeVectors(), clip_backend=FakeClip(), photos_root=root,
            )
            self.assertEqual(summary.applied, 0)
            self.assertEqual(summary.checksum_conflicts, 1)


if __name__ == "__main__":
    unittest.main()
