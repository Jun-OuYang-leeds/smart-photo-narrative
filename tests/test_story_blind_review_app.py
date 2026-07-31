from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "scripts" / "story_blind_review_app.py"


class StoryBlindReviewAppTests(unittest.TestCase):
    def test_save_restore_choices_and_final_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            packet_path = tmp_path / "packet.json"
            response_path = tmp_path / "responses.json"
            packet = {
                "protocol_version": "story-blind-review-v1",
                "cases": [{
                    "review_index": index + 1,
                    "case_token": f"token-{index:02d}",
                    "language": "zh" if index < 6 else "en",
                    "source_kind": "event",
                    "photo_paths": [],
                    "story_a": {"title": "A", "paragraphs": ["Alpha"]},
                    "story_b": {"title": "B", "paragraphs": ["Beta"]},
                } for index in range(12)],
            }
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            old_packet = os.environ.get("STORY_BLIND_PACKET")
            old_responses = os.environ.get("STORY_BLIND_RESPONSES")
            os.environ["STORY_BLIND_PACKET"] = str(packet_path)
            os.environ["STORY_BLIND_RESPONSES"] = str(response_path)
            try:
                app = AppTest.from_file(str(APP)).run(timeout=30)
                self.assertEqual(len(app.radio), 12)
                app.radio[0].set_value("A 更好")
                app.button[0].click().run(timeout=30)
                saved = json.loads(response_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["responses"][0]["choice"], "A")
                self.assertEqual(saved["status"], "in_progress")

                restored = AppTest.from_file(str(APP)).run(timeout=30)
                self.assertEqual(restored.radio[0].value, "A 更好")
                choices = ["A 更好", "B 更好", "平局"]
                for index, radio in enumerate(restored.radio):
                    radio.set_value(choices[index % 3])
                restored.run(timeout=30)
                restored.button[1].click().run(timeout=30)
                locked = json.loads(response_path.read_text(encoding="utf-8"))
                self.assertEqual(locked["status"], "locked")
                self.assertEqual(len(locked["responses"]), 12)
                self.assertEqual(
                    {item["choice"] for item in locked["responses"]}, {"A", "B", "tie"},
                )
            finally:
                if old_packet is None:
                    os.environ.pop("STORY_BLIND_PACKET", None)
                else:
                    os.environ["STORY_BLIND_PACKET"] = old_packet
                if old_responses is None:
                    os.environ.pop("STORY_BLIND_RESPONSES", None)
                else:
                    os.environ["STORY_BLIND_RESPONSES"] = old_responses


if __name__ == "__main__":
    unittest.main()
