from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "scripts" / "qwen_story_blind_review_app.py"


class QwenStoryBlindReviewAppTests(unittest.TestCase):
    def test_save_restore_and_lock_scores_without_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_path = root / "packet.json"
            response_path = root / "responses.json"
            cases = []
            for index, track in enumerate(("faithful", "creative"), 1):
                dimensions = (
                    ["coherence", "informativeness", "evidence_consistency"]
                    if track == "faithful" else
                    ["coherence", "personalization", "trustworthiness"]
                )
                cases.append({
                    "review_index": index, "case_token": f"token-{index}", "track": track,
                    "language": "en", "source_kind": "event", "dimensions": dimensions,
                    "photo_paths": [],
                    "story_a": {"title": "Alpha", "paragraphs": ["Anonymous A."]},
                    "story_b": {"title": "Beta", "paragraphs": ["Anonymous B."]},
                })
            packet_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            values = {
                "QWEN_STORY_BLIND_PACKET": str(packet_path),
                "QWEN_STORY_BLIND_RESPONSES": str(response_path),
                "QWEN_STORY_PHOTO_ROOT": str(root),
            }
            previous = {key: os.environ.get(key) for key in values}
            os.environ.update(values)
            try:
                app = AppTest.from_file(str(APP)).run(timeout=30)
                self.assertEqual(len(app.radio), 2)
                self.assertEqual(len(app.slider), 12)
                visible = " ".join(item.value for item in app.markdown)
                self.assertNotIn("QF0", visible)
                self.assertNotIn("QC2", visible)
                app.radio[0].set_value("A 更好")
                app.radio[1].set_value("平局")
                app.slider[0].set_value(5)
                app.button[0].click().run(timeout=30)
                saved = json.loads(response_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "in_progress")
                restored = AppTest.from_file(str(APP)).run(timeout=30)
                self.assertEqual(restored.radio[0].value, "A 更好")
                restored.button[1].click().run(timeout=30)
                locked = json.loads(response_path.read_text(encoding="utf-8"))
                self.assertEqual(locked["status"], "locked")
                self.assertEqual(locked["responses"][0]["scores"]["A"]["coherence"], 5)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
