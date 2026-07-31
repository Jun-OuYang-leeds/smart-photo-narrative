"""Tests for the Qwen3.7 blinded story review (fake client + 12 temp images)."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_story_review as sr  # noqa: E402
from eval_llm import EvalLLMClient  # noqa: E402

PINNED = "qwen3.7-plus-2026-05-26"


class _Resp:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=content), finish_reason="stop")]
        self.model = PINNED
        self.usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)


class FakeOpenAI:
    def __init__(self):
        self._queue = []
        self.calls = []

    def OpenAI(self, **kwargs):
        outer = self

        def create(**kw):
            outer.calls.append(kw)
            return outer._queue.pop(0)
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))

    def queue(self, obj):
        self._queue.append(_Resp(json.dumps(obj)))


def _client():
    return EvalLLMClient(
        api_key="sk-test-1234567890",
        base_url="https://ws-x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )


def _story(text):
    return types.SimpleNamespace(title=text, paragraphs=[text], creative_transitions=[],
                                 opening="", closing="")


def _img_block_count(call_kwargs):
    content = call_kwargs["messages"][1]["content"]
    return sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image_url")


class ReviewComparisonTests(unittest.TestCase):
    def test_two_comparisons_have_distinct_dimensions(self):
        self.assertNotEqual(sr.COMPARISON_SF_SC0["dimensions"], sr.COMPARISON_SC0_SC1["dimensions"])
        self.assertIn("mood_usage", sr.COMPARISON_SC0_SC1["dimensions"])
        self.assertIn("unsupported_claims", sr.COMPARISON_SF_SC0["dimensions"])

    def test_dimensions_are_scored_per_story_and_remapped(self):
        fake = FakeOpenAI()
        fake.queue({
            "scores": {"A": {"coherence": 5}, "B": {"coherence": 3}},
            "claim_audit": {"A": {"supported": ["s"], "unsupported": [], "creative": [], "non_factual": []},
                            "B": {"supported": [], "unsupported": ["u"], "creative": [], "non_factual": []}},
            "diff_note": "A more faithful, B more reflective.",
        })
        stories = {"S-F": _story("faithful text"), "S-C0": _story("creative text")}
        review_input = sr.ReviewInput(
            images=[], metadata={"date": "2025-12-18", "time_range": "12:19-16:25", "location": "Paris"},
            moods={"p0": "calm", "p1": "happy"},
        )
        with patch.dict(sys.modules, {"openai": fake}):
            result = sr.review_pair(_client(), sr.COMPARISON_SF_SC0, stories, review_input, seed=42)
        # scores remapped to REAL keys, not A/B
        self.assertEqual(set(result["scores"].keys()), {"S-F", "S-C0"})
        self.assertEqual(set(result["claim_audit"].keys()), {"S-F", "S-C0"})
        # mapping correctness: S-F's label tells us which score it got
        sf_label = result["anonymization"]["S-F"]
        expected_coherence = 5 if sf_label == "A" else 3
        self.assertEqual(result["scores"]["S-F"]["coherence"], expected_coherence)
        self.assertEqual(result["diff_note"], "A more faithful, B more reflective.")


class ReviewImagesAndPromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.images = [str(d / f"img{i}.jpg") for i in range(12)]
        for p in self.images:
            Path(p).write_bytes(b"\xff\xd8\xff\xe0img")

    def tearDown(self):
        self.tmp.cleanup()

    def test_passes_12_images_metadata_and_moods(self):
        fake = FakeOpenAI()
        fake.queue({"scores": {"A": {}, "B": {}}, "claim_audit": {"A": {}, "B": {}}, "diff_note": ""})
        stories = {"S-C0": _story("c0"), "S-C1": _story("c1")}
        review_input = sr.ReviewInput(
            images=self.images,
            metadata={"date": "2025-12-18", "time_range": "12:19-16:25", "location": "Paris"},
            moods={"p0": "calm"},
        )
        with patch.dict(sys.modules, {"openai": fake}):
            sr.review_pair(_client(), sr.COMPARISON_SC0_SC1, stories, review_input, seed=7)
        # exactly 12 image blocks sent
        self.assertEqual(_img_block_count(fake.calls[0]), 12)
        # metadata + mood appear in the text prompt; real keys do NOT (blinded)
        text_block = next(
            b for b in fake.calls[0]["messages"][1]["content"]
            if isinstance(b, dict) and b.get("type") == "text"
        )["text"]
        self.assertIn("Paris", text_block)
        self.assertIn("calm", text_block)
        # blinded: the real story identities must not appear as labels
        self.assertNotIn("Story S-C0", text_block)
        self.assertNotIn("Story S-C1", text_block)
        self.assertIn("Story A", text_block)

    def test_review_all_runs_both_comparisons(self):
        fake = FakeOpenAI()
        fake.queue({"scores": {"A": {}, "B": {}}, "claim_audit": {"A": {}, "B": {}}, "diff_note": ""})
        fake.queue({"scores": {"A": {}, "B": {}}, "claim_audit": {"A": {}, "B": {}}, "diff_note": ""})
        stories = {"S-F": _story("f"), "S-C0": _story("c0"), "S-C1": _story("c1")}
        review_input = sr.ReviewInput(images=self.images, metadata={"date": "d"}, moods={})
        with patch.dict(sys.modules, {"openai": fake}):
            results = sr.review_all(_client(), stories, review_input, base_seed=100)
        self.assertEqual([r["comparison_id"] for r in results], ["SF_vs_SC0", "SC0_vs_SC1"])
        self.assertEqual(len(fake.calls), 2)


if __name__ == "__main__":
    unittest.main()
