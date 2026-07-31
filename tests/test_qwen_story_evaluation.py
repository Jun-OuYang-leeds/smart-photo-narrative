from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_story_evaluation import (
    QWEN_DIGEST,
    QwenCreativeRunner,
    QwenFaithfulRunner,
    _creative_prompt,
    _faithful_prompt,
    build_dual_blind_files,
    build_public_summary,
    deterministic_seed,
    load_creative_case_manifest,
    privacy_scan_public_summary,
    reveal_dual_blind,
)
from story_agent import EvidenceGroup, EvidenceItem, StoryContext
from story_evaluation import StoryCase, canonical_hash


def make_context(label: str = "case") -> StoryContext:
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
        tuple(item.observation_record() for item in evidence), (), "P001", ("P002",),
    )
    return StoryContext(
        label, evidence, (group,), source_kind="event", narrator_role="observer",
        use_photographer_mood=False,
    )


def valid_faithful() -> str:
    return json.dumps({
        "title": "Photo record",
        "paragraphs": [{
            "text": "The photos show a cup on a table and a person nearby.",
            "evidence_ids": ["P001", "P002"], "group_id": "G001",
        }],
        "uncertain_observations": [], "unused_evidence_ids": [], "creative_transitions": [],
    })


def valid_creative() -> str:
    return json.dumps({
        "title": "A table remembered",
        "opening": "I remember pausing to look carefully at a quiet collection of ordinary details in front of me.",
        "groups": [{
            "group_id": "G001",
            "factual_text": (
                "I notice a cup resting on the table while a person remains visible nearby, "
                "and the simple arrangement gives the scene a clear visual centre."
            ),
        }],
        "transitions": [],
        "closing": "I keep this restrained view as a small record of what the camera placed before me.",
    })


class FakeQwenBackend:
    def __init__(self, *, bad_faithful: bool = False, bad_creative: bool = False, digest: str = QWEN_DIGEST):
        self.bad_faithful = bad_faithful
        self.bad_creative = bad_creative
        self.digest = digest
        self.calls: list[dict] = []

    def model_info(self):
        return {"digest": self.digest, "resolved_model": "qwen3:4b"}

    def generate(
        self, system, user, *, seed, json_mode, output_schema=None,
        temperature, num_ctx, num_predict,
    ):
        self.calls.append({
            "system": system, "user": user, "seed": seed, "json_mode": json_mode,
            "output_schema": output_schema, "temperature": temperature,
            "num_ctx": num_ctx, "num_predict": num_predict,
        })
        if "warm personal photo diary" in system:
            return "A memory\n\nI saw a cup on the table and kept the scene in mind."
        if "one paragraph for every input photo" in system:
            return json.dumps({
                "title": "Per photo",
                "paragraphs": [
                    {"text": "A cup is on a table.", "evidence_ids": ["P001"]},
                    {"text": "A person is near the table.", "evidence_ids": ["P002"]},
                ],
            })
        if "natural first-person observer memoir" in system:
            return "Observer memory\n\nI saw a cup on the table and noticed a person nearby before the view ended."
        if "coherent first-person personal-photo memoir" in system:
            if self.bad_creative:
                return "{invalid"
            return valid_creative()
        if "Repair it once" in user and self.bad_faithful:
            return "{invalid"
        if self.bad_faithful:
            payload = json.loads(valid_faithful())
            payload["paragraphs"][0]["text"] = "The person is possibly working."
            return json.dumps(payload)
        return valid_faithful()


class QwenStoryEvaluationTests(unittest.TestCase):
    def test_prompts_isolate_legacy_evidence_and_share_paired_drafts(self):
        context = make_context()
        qf0 = _faithful_prompt("QF0", context, "en")
        qf2 = _faithful_prompt("QF2", context, "en")
        qf3 = _faithful_prompt("QF3", context, "en")
        self.assertNotIn("qwen_scene_graph", qf0[1])
        self.assertNotIn("evidence_id", qf0[1])
        self.assertEqual(qf2, qf3)
        qc1 = _creative_prompt(context, "en")
        qc2 = _creative_prompt(context, "en")
        self.assertEqual(qc1, qc2)
        self.assertFalse(context.use_photographer_mood)

    def test_seed_is_fixed_per_track_and_case(self):
        self.assertEqual(deterministic_seed("case", "faithful"), deterministic_seed("case", "faithful"))
        self.assertNotEqual(deterministic_seed("case", "faithful"), deterministic_seed("case", "creative"))

    def test_faithful_runner_shares_draft_and_resumes(self):
        backend = FakeQwenBackend()
        case = StoryCase("case", "event", "case", ("photo-1", "photo-2"), "en")
        with tempfile.TemporaryDirectory() as tmp:
            runner = QwenFaithfulRunner(
                backend, Path(tmp) / "faithful.jsonl", context_factory=lambda _: make_context(),
            )
            records = runner.run([case])
            self.assertEqual(len(records), 4)
            self.assertEqual(len(backend.calls), 3)
            by_variant = {item["variant_id"]: item for item in records}
            self.assertEqual(by_variant["QF2"]["prompt_hash"], by_variant["QF3"]["prompt_hash"])
            self.assertEqual(
                by_variant["QF2"]["raw_attempts"][0]["output_hash"],
                by_variant["QF3"]["raw_attempts"][0]["output_hash"],
            )
            runner.run([case])
            self.assertEqual(len(backend.calls), 3)

    def test_faithful_bad_repair_uses_grouped_fallback(self):
        backend = FakeQwenBackend(bad_faithful=True)
        case = StoryCase("case", "event", "case", ("photo-1", "photo-2"), "en")
        with tempfile.TemporaryDirectory() as tmp:
            records = QwenFaithfulRunner(
                backend, Path(tmp) / "faithful.jsonl", context_factory=lambda _: make_context(),
            ).run([case])
        qf3 = next(item for item in records if item["variant_id"] == "QF3")
        self.assertEqual(qf3["status"], "fallback")
        self.assertTrue(qf3["fallback"])
        self.assertEqual(len(qf3["raw_attempts"]), 2)

    def test_creative_shares_draft_and_never_falls_back(self):
        case = StoryCase("case", "event", "case", ("photo-1", "photo-2"), "en")
        with tempfile.TemporaryDirectory() as tmp:
            good = FakeQwenBackend()
            records = QwenCreativeRunner(
                good, Path(tmp) / "good.jsonl", context_factory=lambda _: make_context(),
            ).run_cases([case])
            by_variant = {item["variant_id"]: item for item in records}
            self.assertEqual(by_variant["QC1"]["prompt_hash"], by_variant["QC2"]["prompt_hash"])
            self.assertEqual(by_variant["QC2"]["status"], "ok")
            self.assertEqual(len(good.calls), 2)

            bad = FakeQwenBackend(bad_creative=True)
            failed = QwenCreativeRunner(
                bad, Path(tmp) / "bad.jsonl", context_factory=lambda _: make_context(),
            ).run_cases([case])
            qc2 = next(item for item in failed if item["variant_id"] == "QC2")
            self.assertEqual(qc2["status"], "error")
            self.assertIsNone(qc2["normalized_output"])
            self.assertFalse(qc2["fallback"])
            self.assertEqual(len(qc2["raw_attempts"]), 2)

    def test_manifest_validation_checks_hash_quotas_and_non_overlap(self):
        used = 0
        def values(prefix, count, kinds):
            nonlocal used
            result = []
            for index in range(count):
                used += 1
                kind = kinds[index]
                result.append({
                    "case_id": f"{prefix}_{index}", "source_kind": kind,
                    "label": "case", "photo_ids": [f"p-{used}"],
                    "language": "zh" if index % 2 == 0 else "en",
                    "event_id": None, "size_bucket": "1" if kind == "single" else "2-3",
                })
            return result
        payload = {
            "protocol_version": "test", "selection_seed": 1, "criteria": {},
            "excluded_photo_count": 0, "excluded_photo_ids": [],
            "main_cases": values("main", 12, ["single"] * 2 + ["event"] * 8 + ["date"] * 2),
            "reserve_cases": values("reserve", 6, ["single"] + ["event"] * 4 + ["date"]),
        }
        payload["freeze_sha256"] = canonical_hash(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            main, reserve, _ = load_creative_case_manifest(path)
        self.assertEqual((len(main), len(reserve)), (12, 6))

    def test_dual_blind_hides_variants_and_reveals_scores(self):
        context = make_context()
        faithful_contexts = {f"f-{index:02d}": context for index in range(12)}
        creative_contexts = {f"c-{index:02d}": context for index in range(12)}
        faithful = []
        creative = []
        for case_id in faithful_contexts:
            for variant in ("QF0", "QF3"):
                faithful.append({
                    "case_id": case_id, "variant_id": variant, "language": "en", "source_kind": "event",
                    "normalized_output": {"title": "Story", "paragraphs": [{"text": "Visible text."}]},
                })
        for case_id in creative_contexts:
            for variant in ("QC0", "QC2"):
                creative.append({
                    "case_id": case_id, "variant_id": variant, "language": "en", "source_kind": "event",
                    "normalized_output": {"title": "Story", "paragraphs": [{"text": "Visible text."}]},
                })
        with tempfile.TemporaryDirectory() as tmp:
            packet, mapping = build_dual_blind_files(
                faithful, creative, faithful_contexts, creative_contexts, list(creative_contexts),
                packet_path=Path(tmp) / "packet.json", mapping_path=Path(tmp) / "mapping.json",
            )
        packet_text = json.dumps(packet)
        self.assertEqual(len(packet["cases"]), 24)
        self.assertNotIn("QF0", packet_text)
        self.assertNotIn("QC2", packet_text)
        responses = {"status": "locked", "responses": []}
        for case in packet["cases"]:
            responses["responses"].append({
                "case_token": case["case_token"], "choice": "A", "reason": "",
                "scores": {side: {dimension: 4 for dimension in case["dimensions"]} for side in ("A", "B")},
            })
        revealed = reveal_dual_blind(responses, mapping)
        self.assertEqual(set(revealed["public"]), {"faithful", "creative"})
        self.assertEqual(len(revealed["private"]), 24)
        for track in ("faithful", "creative"):
            contrast = next(iter(revealed["public"][track]["paired_score_differences"].values()))
            self.assertTrue(all(value["mean_difference"] == 0.0 for value in contrast.values()))

    def test_dual_blind_preserves_incomplete_frozen_creative_set(self):
        context = make_context()
        faithful_contexts = {f"f-{index:02d}": context for index in range(12)}
        creative_contexts = {f"c-{index:02d}": context for index in range(11)}
        faithful = [
            {
                "case_id": case_id, "variant_id": variant, "language": "en", "source_kind": "event",
                "normalized_output": {"title": "Story", "paragraphs": [{"text": "Visible text."}]},
            }
            for case_id in faithful_contexts for variant in ("QF0", "QF3")
        ]
        creative = [
            {
                "case_id": case_id, "variant_id": variant, "language": "en", "source_kind": "event",
                "normalized_output": {"title": "Story", "paragraphs": [{"text": "Visible text."}]},
            }
            for case_id in creative_contexts for variant in ("QC0", "QC2")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            packet, _ = build_dual_blind_files(
                faithful, creative, faithful_contexts, creative_contexts, list(creative_contexts),
                packet_path=Path(tmp) / "packet.json", mapping_path=Path(tmp) / "mapping.json",
            )
        self.assertEqual(packet["status"], "incomplete")
        self.assertEqual(packet["case_count"], 23)
        self.assertEqual(packet["track_counts"], {"faithful": 12, "creative": 11})
        self.assertEqual(packet["target_case_count"], 24)

    def test_public_summary_contains_no_private_identifiers(self):
        records = []
        for variant in ("QF0", "QF1", "QF2", "QF3"):
            records.append({
                "case_id": "case", "variant_id": variant, "status": "ok", "automatic_metrics": {"latency_ms": 1.0},
            })
        creative = []
        for variant in ("QC0", "QC1", "QC2"):
            creative.append({
                "case_id": "creative", "variant_id": variant, "status": "ok", "automatic_metrics": {"latency_ms": 1.0},
            })
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_public_summary(
                records, creative, {"freeze_sha256": "abc"}, ["creative"] * 12,
                output_path=Path(tmp) / "summary.json",
            )
        self.assertFalse(privacy_scan_public_summary(summary))


if __name__ == "__main__":
    unittest.main()
