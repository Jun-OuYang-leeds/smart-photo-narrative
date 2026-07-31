from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from story_agent import (
    EvidenceGroup, EvidenceItem, GeneratedStory, StoryContext, StoryParagraph,
    UncertainObservation,
)
from story_evaluation import blind_n0_n3_order, load_story_cases, score_story


class StoryEvaluationTests(unittest.TestCase):
    @staticmethod
    def context():
        evidence = (
            EvidenceItem("P001", "photo-1", "1.jpg", "2026-01-01T10:00:00", "exif", 1.0, None, "a cup", (), ()),
            EvidenceItem("P002", "photo-2", "2.jpg", "2026-01-01T10:01:00", "exif", 1.0, None, "a cup", (), ()),
        )
        group = EvidenceGroup(
            "G001", ("P001", "P002"), ("photo-1", "photo-2"),
            evidence[0].timestamp, evidence[1].timestamp, (),
            (evidence[0].observation_record(),), ("model conflict",), "P001", ("P002",),
        )
        return StoryContext("case", evidence, (group,))

    def test_story_metrics_cover_language_citations_compression_and_conflicts(self):
        context = self.context()
        story = GeneratedStory(
            date="case", title="Event record",
            paragraphs=[StoryParagraph("A cup is visible on a table.", ("P001", "P002"), "G001")],
            photo_count=2, photo_ids=["photo-1", "photo-2"], model="fake", language="en",
            uncertain_observations=[UncertainObservation("model conflict", ("P001", "P002"))],
        )
        metrics = score_story(story, context, latency_ms=12.5)
        self.assertEqual(metrics["citation_valid_rate"], 1.0)
        self.assertEqual(metrics["evidence_group_coverage"], 1.0)
        self.assertEqual(metrics["conflict_handling_rate"], 1.0)
        self.assertEqual(metrics["paragraph_compression_rate"], 0.5)
        self.assertTrue(metrics["language_compliant"])

    def test_case_loader_enforces_12_case_and_language_contract(self):
        cases = []
        sources = ["single"] * 2 + ["event"] * 8 + ["date"] * 2
        for index, source in enumerate(sources):
            cases.append({
                "case_id": f"case-{index}", "source_kind": source,
                "label": "2026-07-12" if index == 2 else f"label-{index}",
                "photo_ids": [f"photo-{index}"], "language": "zh" if index % 2 == 0 else "en",
            })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            loaded = load_story_cases(path)
        self.assertEqual(len(loaded), 12)

    def test_blind_order_is_stable_and_contains_both_variants(self):
        first = blind_n0_n3_order("case-1")
        self.assertEqual(first, blind_n0_n3_order("case-1"))
        self.assertEqual(set(first), {"N0", "N3"})


if __name__ == "__main__":
    unittest.main()
