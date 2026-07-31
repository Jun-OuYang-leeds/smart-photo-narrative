from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from storage import PhotoStorage


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "scripts" / "mood_story_blind_review_app.py"


class MoodStoryBlindReviewAppTests(unittest.TestCase):
    def test_submit_locks_preference_and_scores_without_revealing_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = PhotoStorage(root / "app.db")
            photo_ids = [
                storage.upsert_photo(
                    relative_path=f"photo-{index}.jpg",
                    content_sha256=format(index + 10, "x") * 64,
                    timestamp_source="unknown",
                    timestamp_confidence=0.0,
                )
                for index in range(5)
            ]
            packet_path = root / "packet.json"
            response_path = root / "response.json"
            packet_path.write_text(json.dumps({
                "protocol_version": "mood-blind-review-v1",
                "case_id": "case",
                "photo_ids": photo_ids,
                "stories": {
                    "A": {"title": "Alpha", "content": "First anonymous story."},
                    "B": {"title": "Beta", "content": "Second anonymous story."},
                },
            }), encoding="utf-8")
            values = {
                "MOOD_BLIND_PACKET": str(packet_path),
                "MOOD_BLIND_RESPONSES": str(response_path),
                "MOOD_BLIND_DATABASE": str(root / "app.db"),
                "MOOD_BLIND_PHOTOS_DIR": str(root),
            }
            previous = {key: os.environ.get(key) for key in values}
            os.environ.update(values)
            try:
                app = AppTest.from_file(str(APP)).run(timeout=30)
                self.assertEqual(len(app.radio), 1)
                self.assertEqual(len(app.slider), 6)
                self.assertNotIn("M0", " ".join(item.value for item in app.markdown))
                self.assertNotIn("M1", " ".join(item.value for item in app.markdown))
                app.radio[0].set_value("A")
                app.slider[0].set_value(4)
                app.slider[1].set_value(5)
                app.slider[2].set_value(3)
                app.button[0].click().run(timeout=30)

                saved = json.loads(response_path.read_text(encoding="utf-8"))
                self.assertTrue(saved["locked"])
                self.assertEqual(saved["preference"], "A")
                self.assertEqual(saved["scores"]["A"]["personalization"], 5)

                restored = AppTest.from_file(str(APP)).run(timeout=30)
                self.assertTrue(any("Review locked" in item.value for item in restored.success))
                self.assertEqual(len(restored.radio), 0)
            finally:
                for key, old in previous.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old


if __name__ == "__main__":
    unittest.main()
