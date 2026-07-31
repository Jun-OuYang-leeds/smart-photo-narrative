from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mood_evaluation import (
    build_blind_files,
    build_or_validate_manifest,
    build_public_summary,
)
from storage import PhotoStorage


class MoodEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.storage = PhotoStorage(root / "app.db")
        self.manifest_path = root / "private" / "manifest.json"
        self.photo_ids = []
        for index in range(5):
            photo_id = self.storage.upsert_photo(
                relative_path=f"balloon-{index}.jpg",
                content_sha256=format(index + 10, "x") * 64,
                captured_at=f"2026-07-07T06:0{index}:00",
                date_local="2026-07-07",
                timestamp_source="exif_original",
                timestamp_confidence=1.0,
            )
            self.photo_ids.append(photo_id)
        self.storage.upsert_event(
            event_id="58db14d5-e7ad-5975-87a0-59456560e09a",
            title="2026-07-07 · Event 1",
            method="fixture",
        )
        self.storage.replace_event_photos(
            "58db14d5-e7ad-5975-87a0-59456560e09a", self.photo_ids,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_manifest_requires_five_moods_and_refuses_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "All five photos"):
            build_or_validate_manifest(self.storage, self.manifest_path)
        labels = ["excited", "happy", "calm", "excited", "neutral"]
        self.storage.save_photo_moods(dict(zip(self.photo_ids, labels)))
        manifest = build_or_validate_manifest(self.storage, self.manifest_path)
        self.assertEqual(len(manifest["records"]), 5)
        self.assertEqual([row["mood_label"] for row in manifest["records"]], labels)
        self.assertEqual(manifest, build_or_validate_manifest(self.storage, self.manifest_path))

        self.storage.save_photo_moods({self.photo_ids[0]: "sad"})
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            build_or_validate_manifest(self.storage, self.manifest_path)

    def test_blind_packet_hides_variant_mapping(self) -> None:
        records = [
            {
                "variant_id": "M0",
                "normalized_output": {"title": "First title"},
                "content": "First story body.",
            },
            {
                "variant_id": "M1",
                "normalized_output": {"title": "Second title"},
                "content": "Second story body.",
            },
        ]
        packet, mapping = build_blind_files(records, self.photo_ids)
        packet_text = json.dumps(packet)
        self.assertNotIn('"M0"', packet_text)
        self.assertNotIn('"M1"', packet_text)
        self.assertEqual(set(mapping["mapping"].values()), {"M0", "M1"})
        self.assertEqual(set(packet["stories"]), {"A", "B"})

    def test_public_summary_reveals_preference_and_score_labels(self) -> None:
        responses = {
            "locked": True,
            "preference": "B",
            "scores": {
                "A": {"coherence": 2, "personalization": 1, "credibility": 4},
                "B": {"coherence": 5, "personalization": 5, "credibility": 4},
            },
        }
        mapping = {"mapping": {"A": "M0", "B": "M1"}}
        results = {
            "results_sha256": "abc123",
            "records": [
                {"variant_id": "M0", "automatic_metrics": {}, "latency_ms": 10},
                {"variant_id": "M1", "automatic_metrics": {}, "latency_ms": 11},
            ],
        }
        manifest = {"records": [{"mood_label": "happy"}] * 5}

        summary = build_public_summary(responses, mapping, results, manifest)

        self.assertEqual(summary["blind_preference"], "M1")
        self.assertEqual(set(summary["blind_scores"]), {"M0", "M1"})
        self.assertEqual(summary["blind_scores"]["M0"]["coherence"], 2)
        self.assertEqual(summary["blind_scores"]["M1"]["personalization"], 5)
        self.assertNotIn("A", summary["blind_scores"])
        self.assertNotIn("B", summary["blind_scores"])


if __name__ == "__main__":
    unittest.main()
