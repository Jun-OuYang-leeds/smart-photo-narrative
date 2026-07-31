from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from event_service import EventService
from retrieval_engine import MultimodalRetriever
from storage import PhotoStorage


class FakeVectors:
    def get_image_embeddings(self, ids):
        return {pid: np.array([1.0, 0.0]) for pid in ids}


class EventServiceTests(unittest.TestCase):
    def test_persists_reliable_events_and_leaves_low_confidence_unassigned(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = PhotoStorage(Path(tmp) / "app.db")
            ids = []
            for index, confidence in enumerate((1.0, 1.0, 0.2)):
                pid = storage.upsert_photo(
                    relative_path=f"{index}.jpg", content_sha256=f"{index + 1:064x}",
                    captured_at=f"2026-01-01T{9 + index * 2:02d}:00:00", captured_at_sort=index * 7200,
                    date_local="2026-01-01", timestamp_source="exif_original" if confidence > .5 else "mtime",
                    timestamp_confidence=confidence, image_width=10, image_height=10,
                )
                storage.upsert_caption(pid, f"photo {index}")
                ids.append(pid)
            retriever = MultimodalRetriever(storage, FakeVectors(), None)
            service = EventService(storage, FakeVectors(), retriever)
            result = service.organize(persist=True)
            self.assertEqual(result.unassigned_photo_ids, (ids[2],))
            stored = service.list_events()
            self.assertEqual(sum(len(event.photo_ids) for event in stored), 2)


if __name__ == "__main__":
    unittest.main()
