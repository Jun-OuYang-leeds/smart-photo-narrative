"""Tests for synthetic query generation + blinded audit (fake vector store + LLM)."""

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

import eval_query_gen as qg  # noqa: E402
from eval_llm import EvalLLMClient  # noqa: E402

PINNED = "qwen3.7-plus-2026-05-26"
GOOD_QUERY = "a wooden chair beside a tall green plant near a bright window"  # 11 words


class FakeHit(types.SimpleNamespace):
    pass


class FakeVectorStore:
    def __init__(self, target, distractors):
        self.target = target
        self.distractors = list(distractors)

    def get_image_embeddings(self, ids):
        return {self.target: [1.0, 0.0, 0.0]}

    def query_images(self, embedding, *, top_k, eligible_photo_ids=None):
        hits = [FakeHit(photo_id=self.target, distance=0.0)]
        for i, d in enumerate(self.distractors):
            hits.append(FakeHit(photo_id=d, distance=0.1 + 0.01 * i))
        return hits[:top_k]


class _Resp:
    def __init__(self, content):
        self.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ]
        self.model = PINNED
        self.usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)


class FakeOpenAI:
    """Queue JSON responses for ``OpenAI(...).chat.completions.create``."""

    def __init__(self):
        self._queue: list[_Resp] = []

    def OpenAI(self, **kwargs):
        outer = self

        def create(**kw):
            return outer._queue.pop(0) if outer._queue else _Resp('{"query":"x","cues":[]}')

        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )

    def queue(self, obj):
        self._queue.append(_Resp(json.dumps(obj)))


def _client() -> EvalLLMClient:
    return EvalLLMClient(
        api_key="sk-test-1234567890",
        base_url="https://ws-x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )


class DistractorAndPanelTests(unittest.TestCase):
    def test_select_distractors_excludes_target(self):
        vs = FakeVectorStore("t", ["d1", "d2", "d3", "d4"])
        self.assertEqual(qg.select_distractors("t", vs, ["t", "d1", "d2", "d3", "d4"]), ["d1", "d2", "d3", "d4"])

    def test_select_distractors_raises_when_too_few(self):
        vs = FakeVectorStore("t", ["d1"])
        with self.assertRaises(ValueError):
            qg.select_distractors("t", vs, ["t", "d1"], n=4)

    def test_build_panel_contains_target_once(self):
        panel, idx = qg.build_panel("t", ["d1", "d2", "d3", "d4"], shuffle_seed=42)
        self.assertEqual(len(panel), 5)
        self.assertEqual(panel.count("t"), 1)
        self.assertEqual(panel[idx], "t")

    def test_build_panel_deterministic(self):
        p1, i1 = qg.build_panel("t", ["d1", "d2", "d3", "d4"], shuffle_seed=7)
        p2, i2 = qg.build_panel("t", ["d1", "d2", "d3", "d4"], shuffle_seed=7)
        self.assertEqual((p1, i1), (p2, i2))


class ComplianceTests(unittest.TestCase):
    def test_word_count(self):
        self.assertEqual(qg.query_word_count(GOOD_QUERY), 12)
        self.assertEqual(qg.query_word_count("short"), 1)

    def test_clean_query_compliant(self):
        self.assertEqual(qg.rule_violations(GOOD_QUERY), [])
        self.assertTrue(qg.is_compliant(GOOD_QUERY, model_ok=True))

    def test_short_query_violates(self):
        self.assertTrue(qg.rule_violations("tiny cat"))
        self.assertFalse(qg.is_compliant("tiny cat", model_ok=True))

    def test_numeric_and_uuid_violations(self):
        self.assertTrue(qg.rule_violations("photo IMG 2025 in paris eiffel tower sunny day"))
        self.assertTrue(qg.rule_violations("scene abc12345def near the river bank today"))

    def test_model_unok_blocks_compliance(self):
        self.assertFalse(qg.is_compliant(GOOD_QUERY, model_ok=False))


class GenerateAndAuditTests(unittest.TestCase):
    def setUp(self):
        self.target = "t"
        self.distractors = ["d1", "d2", "d3", "d4"]
        self.corpus = ["t", "d1", "d2", "d3", "d4", "d5"]
        self.vs = FakeVectorStore(self.target, self.distractors)
        # encode_image requires real files on disk; create one per panel photo.
        self._tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self._tmp.name)
        for pid in [self.target, *self.distractors]:
            (tmpdir / f"{pid}.jpg").write_bytes(b"\xff\xd8\xff\xe0panel")
        self.path_of = lambda pid: str(tmpdir / f"{pid}.jpg")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, fake: FakeOpenAI):
        with patch.dict(sys.modules, {"openai": fake}):
            return qg.generate_and_audit(
                self.target, self.path_of, self.vs, self.corpus, _client(), base_seed=1000
            )

    def test_success_first_try(self):
        fake = FakeOpenAI()
        fake.queue({"query": GOOD_QUERY, "cues": ["wooden chair", "green plant"]})
        _, tidx = qg.build_panel(self.target, self.distractors, shuffle_seed=1002)
        fake.queue({"chosen_index": tidx, "compliance_ok": True, "issues": ""})
        outcome = self._run(fake)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.attempts, 1)
        self.assertFalse(outcome.repair_used)
        self.assertEqual(outcome.query, GOOD_QUERY)
        self.assertEqual(outcome.chosen_index, tidx)

    def test_repair_then_success(self):
        fake = FakeOpenAI()
        # attempt 0: wrong choice
        _, tidx0 = qg.build_panel(self.target, self.distractors, shuffle_seed=1002)
        fake.queue({"query": GOOD_QUERY, "cues": []})
        fake.queue({"chosen_index": (tidx0 + 1) % 5, "compliance_ok": True, "issues": ""})
        # attempt 1 (repair): correct choice
        _, tidx1 = qg.build_panel(self.target, self.distractors, shuffle_seed=1012)
        fake.queue({"query": GOOD_QUERY, "cues": []})
        fake.queue({"chosen_index": tidx1, "compliance_ok": True, "issues": ""})
        outcome = self._run(fake)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.attempts, 2)
        self.assertTrue(outcome.repair_used)

    def test_second_failure_is_error(self):
        fake = FakeOpenAI()
        for _ in range(2):  # both attempts choose wrong
            _, tidx = qg.build_panel(self.target, self.distractors,
                                     shuffle_seed=1002 if _ == 0 else 1012)
            fake.queue({"query": GOOD_QUERY, "cues": []})
            fake.queue({"chosen_index": (tidx + 1) % 5, "compliance_ok": True, "issues": ""})
        outcome = self._run(fake)
        self.assertEqual(outcome.status, "query_generation_error")
        self.assertEqual(outcome.attempts, 2)
        self.assertIsNotNone(outcome.query)  # last query kept for the record

    def test_noncompliant_query_triggers_repair(self):
        bad = "photo IMG 2025 eiffel tower sunny paris day"  # has a 4-digit run
        fake = FakeOpenAI()
        # attempt 0: chosen == target BUT query breaks a rule -> not compliant
        _, tidx0 = qg.build_panel(self.target, self.distractors, shuffle_seed=1002)
        fake.queue({"query": bad, "cues": []})
        fake.queue({"chosen_index": tidx0, "compliance_ok": True, "issues": ""})
        # attempt 1: clean query, correct choice
        _, tidx1 = qg.build_panel(self.target, self.distractors, shuffle_seed=1012)
        fake.queue({"query": GOOD_QUERY, "cues": []})
        fake.queue({"chosen_index": tidx1, "compliance_ok": True, "issues": ""})
        outcome = self._run(fake)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.attempts, 2)
        self.assertTrue(outcome.repair_used)

    def test_generation_call_failure_then_success(self):
        fake = FakeOpenAI()
        # attempt 0: malformed query payload (no "query" key) -> raises
        fake.queue({"not_query": "oops"})
        _, tidx1 = qg.build_panel(self.target, self.distractors, shuffle_seed=1012)
        fake.queue({"query": GOOD_QUERY, "cues": []})
        fake.queue({"chosen_index": tidx1, "compliance_ok": True, "issues": ""})
        outcome = self._run(fake)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.attempts, 2)


if __name__ == "__main__":
    unittest.main()
