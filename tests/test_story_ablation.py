from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from story_agent import EvidenceGroup, EvidenceItem, StoryContext
from story_evaluation import (
    AtomicJsonlCheckpoint,
    StoryCase,
    StoryExperimentRunner,
    assert_model_digest,
    build_blind_review_files,
    build_variant_prompt,
    deterministic_seed,
    reveal_blind_preferences,
    score_experiment_output,
)
from scripts.audit_story_claims import _classify


FAKE_DIGEST = "f" * 64


def make_context(label: str = "case", *, conflict: bool = False) -> StoryContext:
    evidence = (
        EvidenceItem(
            "P001", "photo-1", "one.jpg", "2026-01-01T10:00:00", "exif", 1.0,
            "Leeds", "a cup on a table", ("cup | on | table",), ("cup", "table"),
        ),
        EvidenceItem(
            "P002", "photo-2", "two.jpg", "2026-01-01T10:01:00", "exif", 1.0,
            "Leeds", "a person near the table", ("person | near | table",), ("person", "table"),
        ),
    )
    group = EvidenceGroup(
        "G001", ("P001", "P002"), ("photo-1", "photo-2"),
        evidence[0].timestamp, evidence[1].timestamp, ("Leeds",),
        tuple(item.observation_record() for item in evidence),
        ("caption and graph conflict",) if conflict else (), "P001", ("P002",),
    )
    return StoryContext(label, evidence, (group,), source_kind="event")


def valid_grouped(language: str = "en", *, uncertain: bool = False) -> str:
    title = "照片记录" if language == "zh" else "Photo record"
    text = "照片显示桌上的杯子和附近的人。" if language == "zh" else "The photos show a cup on a table and a nearby person."
    payload = {
        "title": title,
        "paragraphs": [{"text": text, "evidence_ids": ["P001", "P002"], "group_id": "G001"}],
        "uncertain_observations": ([{
            "text": "模型观察存在冲突。" if language == "zh" else "Model observations conflict.",
            "evidence_ids": ["P001", "P002"],
        }] if uncertain else []),
        "unused_evidence_ids": [],
        "creative_transitions": [],
    }
    return json.dumps(payload, ensure_ascii=False)


class FakeBackend:
    def __init__(self, *, bad_draft: bool = False, bad_repair: bool = False, digest: str = FAKE_DIGEST):
        self.bad_draft = bad_draft
        self.bad_repair = bad_repair
        self.digest = digest
        self.calls: list[dict] = []

    def model_info(self):
        return {"digest": self.digest, "resolved_model": "llama3:latest"}

    def generate(self, system, user, *, seed, json_mode, temperature, num_predict):
        self.calls.append({
            "system": system, "user": user, "seed": seed, "json_mode": json_mode,
            "temperature": temperature, "num_predict": num_predict,
        })
        zh = "Chinese" in system
        if "warm personal photo diary" in system:
            return "标题：一次回忆\n\n我看见桌上的杯子。" if zh else "Title: A memory\n\nI saw a cup on the table."
        if "one paragraph for every input photo" in system:
            title = "逐图记录" if zh else "Per-photo record"
            paragraphs = [
                {"text": "桌上有一个杯子。" if zh else "A cup is on a table.", "evidence_ids": ["P001"]},
                {"text": "桌边可以看到一个人。" if zh else "A person is visible by the table.", "evidence_ids": ["P002"]},
            ]
            return json.dumps({"title": title, "paragraphs": paragraphs}, ensure_ascii=False)
        if "Repair it once" in user:
            if self.bad_repair:
                return "{still invalid"
            return valid_grouped("zh" if zh else "en")
        if self.bad_draft:
            payload = json.loads(valid_grouped("zh" if zh else "en"))
            payload["paragraphs"][0]["text"] = "可能是在工作。" if zh else "The person is possibly working."
            return json.dumps(payload, ensure_ascii=False)
        return valid_grouped("zh" if zh else "en")


class StoryAblationTests(unittest.TestCase):
    def test_prompt_evidence_isolation_and_paired_v3_prompt(self):
        context = make_context()
        n0 = build_variant_prompt("N0", context, "en")
        n1 = build_variant_prompt("N1", context, "en")
        n2 = build_variant_prompt("N2", context, "en")
        n3 = build_variant_prompt("N3", context, "en")
        self.assertNotIn("qwen", (n0[0] + n0[1]).lower())
        self.assertNotIn("evidence_id", n0[1])
        self.assertIn("qwen_scene_graph_model_observations", n1[1])
        self.assertEqual(n2, n3)
        self.assertFalse(n0[2])
        self.assertTrue(n1[2])

    def test_seed_is_stable_and_shared_across_variants(self):
        self.assertEqual(deterministic_seed("case-a"), deterministic_seed("case-a"))
        self.assertNotEqual(deterministic_seed("case-a"), deterministic_seed("case-b"))

    def test_na_metrics_are_none_not_zero(self):
        context = make_context()
        payload = {"title": "Memory", "paragraphs": [{"text": "I saw a cup."}]}
        metrics = score_experiment_output(
            "N0", payload, context, latency_ms=1, status="ok", json_valid=None,
        )
        self.assertIsNone(metrics["json_valid"])
        self.assertIsNone(metrics["citation_valid_rate"])
        self.assertIsNone(metrics["evidence_group_coverage"])
        self.assertIsNone(metrics["fallback"])

        empty = score_experiment_output(
            "N2", None, context, latency_ms=1, status="invalid", json_valid=False, language="en",
        )
        self.assertIsNone(empty["duplicate_text_unit_rate"])
        self.assertIsNone(empty["risk_term_rate"])
        self.assertIsNone(empty["paragraph_compression_rate"])
        self.assertFalse(empty["language_compliant"])

    def test_claim_audit_catches_terms_missing_from_legacy_validator(self):
        status, _ = _classify(
            "The evidence implies they may be focused on their work routine.",
            "person sit desk laptop keyboard monitor", n3_fallback=False,
        )
        self.assertEqual(status, "unsupported")

    def test_digest_mismatch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            assert_model_digest({"digest": "wrong"}, FAKE_DIGEST)

    def test_runner_reuses_n2_draft_and_checkpoint_resume(self):
        case = StoryCase("case-1", "event", "case", ("photo-1", "photo-2"), "en")
        context = make_context()
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results.jsonl"
            runner = StoryExperimentRunner(
                backend, output, context_factory=lambda _: context, expected_digest=FAKE_DIGEST,
            )
            records = runner.run([case])
            self.assertEqual(len(records), 4)
            self.assertEqual(len(backend.calls), 3)
            by_variant = {item["variant_id"]: item for item in records}
            self.assertEqual(by_variant["N2"]["prompt_hash"], by_variant["N3"]["prompt_hash"])
            self.assertEqual(
                by_variant["N2"]["raw_attempts"][0]["output_hash"],
                by_variant["N3"]["raw_attempts"][0]["output_hash"],
            )
            self.assertEqual(by_variant["N3"]["status"], "ok")
            runner.run([case])
            self.assertEqual(len(backend.calls), 3)

    def test_n3_repairs_once_then_uses_grouped_fallback(self):
        case = StoryCase("case-1", "event", "case", ("photo-1", "photo-2"), "en")
        backend = FakeBackend(bad_draft=True, bad_repair=True)
        with tempfile.TemporaryDirectory() as tmp:
            records = StoryExperimentRunner(
                backend, Path(tmp) / "results.jsonl", context_factory=lambda _: make_context(),
                expected_digest=FAKE_DIGEST,
            ).run([case])
        by_variant = {item["variant_id"]: item for item in records}
        self.assertEqual(by_variant["N2"]["status"], "unvalidated")
        self.assertEqual(by_variant["N3"]["status"], "fallback")
        self.assertEqual(len(by_variant["N3"]["raw_attempts"]), 2)
        self.assertTrue(by_variant["N3"]["fallback"])
        self.assertEqual(len(by_variant["N3"]["normalized_output"]["paragraphs"]), 1)

    def test_fake_twelve_by_four_integration_and_blind_mapping(self):
        cases = [
            StoryCase(f"case-{index:02d}", "event", f"label-{index}", ("photo-1", "photo-2"),
                      "zh" if index < 6 else "en")
            for index in range(12)
        ]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "results.jsonl"
            records = StoryExperimentRunner(
                backend, output, context_factory=lambda case: make_context(case.label),
                expected_digest=FAKE_DIGEST,
            ).run(cases)
            self.assertEqual(len(records), 48)
            self.assertEqual(len({(item["case_id"], item["variant_id"]) for item in records}), 48)
            contexts = {case.case_id: make_context(case.label) for case in cases}
            packet, mapping = build_blind_review_files(
                records, contexts, tmp_path / "packet.json", tmp_path / "mapping.json",
            )
            packet_text = json.dumps(packet)
            self.assertNotIn('"N0"', packet_text)
            self.assertNotIn('"N3"', packet_text)
            responses = {
                "status": "locked",
                "responses": [{
                    "case_token": item["case_token"],
                    "choice": ("A", "B", "tie")[index % 3],
                    "reason": "",
                } for index, item in enumerate(packet["cases"])],
            }
            revealed = reveal_blind_preferences(responses, mapping)
            self.assertEqual(len(revealed), 12)
            self.assertIn("tie", {item["preferred_variant"] for item in revealed})

    def test_checkpoint_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = AtomicJsonlCheckpoint(Path(tmp) / "x.jsonl")
            checkpoint.append({"case_id": "c", "variant_id": "N0"})
            with self.assertRaises(ValueError):
                checkpoint.append({"case_id": "c", "variant_id": "N0"})


if __name__ == "__main__":
    unittest.main()
