from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation import (
    DEFAULT_ABLATIONS,
    QrelsValidationError,
    QueryJudgment,
    RetrievalEvaluator,
    latency_summary,
    load_qrels,
    ndcg_at_k,
    ranking_metrics,
    recall_at_k,
    reciprocal_rank,
    temporal_pair_id,
    validate_frozen_qrels,
    write_report,
)


def result(photo_id: str):
    return SimpleNamespace(photo_id=photo_id)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    def search_standard(self, query, *, top_k, filters, enabled_channels):
        channels = tuple(enabled_channels)
        self.calls.append(("standard", query, channels, filters))
        ranked = ["clip-hit", "other"]
        if "caption" in channels:
            ranked.insert(0, "caption-hit")
        if "scene_graph" in channels:
            ranked.insert(0, "sg-hit")
        if filters.start_date:
            ranked.insert(0, "metadata-hit")
        return SimpleNamespace(results=[result(item) for item in ranked[:top_k]], temporal_pairs=[])

    def search(self, query, *, top_k, filters, enabled_channels, use_ollama_parser):
        self.calls.append(("full", query, tuple(enabled_channels), filters, use_ollama_parser))
        if "before" in query:
            pair = SimpleNamespace(before=result("before-id"), after=result("after-id"))
            return SimpleNamespace(results=[], temporal_pairs=[pair])
        return SimpleNamespace(results=[result("full-hit")], temporal_pairs=[])


class EvaluationMetricTests(unittest.TestCase):
    def test_binary_recall_mrr_and_graded_ndcg(self):
        ranked = ["noise", "best", "okay"]
        relevance = {"best": 2, "okay": 1, "missing": 1}
        self.assertEqual(recall_at_k(ranked, relevance, 1), 0)
        self.assertAlmostEqual(recall_at_k(ranked, relevance, 2), 1 / 3)
        self.assertEqual(reciprocal_rank(ranked, relevance), 0.5)
        expected_dcg = (2**2 - 1) / math.log2(3)
        expected_ideal = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(ranked, relevance, 2), expected_dcg / expected_ideal)
        self.assertEqual(set(ranking_metrics(ranked, relevance, (1, 2))), {
            "MRR", "Recall@1", "nDCG@1", "Recall@2", "nDCG@2"
        })

    def test_duplicate_retrieved_ids_do_not_inflate_metrics(self):
        relevance = {"a": 1, "b": 1}
        self.assertEqual(recall_at_k(["a", "a", "b"], relevance, 2), 1.0)
        self.assertEqual(reciprocal_rank(["a", "a"], relevance), 1.0)

    def test_latency_has_interpolated_p50_and_p95(self):
        summary = latency_summary([10, 20, 30, 40])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["P50"], 25)
        self.assertAlmostEqual(summary["P95"], 38.5)


class QrelsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_jsonl_and_json_query_records(self):
        records = [
            {"query_id": "disabled", "query": "todo", "relevance": {}, "enabled": False},
            {
                "query_id": "q1",
                "query": "beach",
                "relevance": ["photo-1", "photo-2"],
                "filters": {"start_date": "2026-01-01", "tags": ["travel"]},
            },
        ]
        jsonl_path = self.root / "qrels.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
        json_path = self.root / "qrels.json"
        json_path.write_text(json.dumps({"queries": records}), encoding="utf-8")

        for path in (jsonl_path, json_path):
            loaded = load_qrels(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].positive_relevance, {"photo-1": 1.0, "photo-2": 1.0})
            self.assertEqual(loaded[0].search_filters(enabled=True).tags, ("travel",))

    def test_loads_frozen_query_metadata_without_breaking_legacy_records(self):
        path = self.root / "frozen.json"
        path.write_text(json.dumps([{
            "query_id": "q1", "query": "pool", "relevance": {"photo": 2},
            "category": "scene", "language": "en", "split": "dev",
            "judgment_source": "agent_pass2",
        }]), encoding="utf-8")
        judgment = load_qrels(path)[0]
        self.assertEqual((judgment.category, judgment.language, judgment.split), ("scene", "en", "dev"))
        self.assertEqual(judgment.judgment_source, "agent_pass2")

    def test_frozen_48_query_quotas_and_metadata_positives_are_validated(self):
        categories = (
            "scene", "object_attribute", "relation_semantic", "relation_exact",
            "caption_lexical", "metadata",
        )
        judgments = []
        metadata = {}
        for category in categories:
            for index in range(8):
                photo_id = f"{category}-{index}"
                metadata[photo_id] = {
                    "date_local": "2026-01-01", "location": "Leeds, England, GB",
                    "timestamp_confidence": 1.0,
                }
                filters = {"start_date": "2026-01-01", "end_date": "2026-01-01"} if category == "metadata" else {}
                judgments.append(QueryJudgment(
                    query_id=f"{category}-{index}", query=f"unique {category} query {index}",
                    relevance={photo_id: 2}, filters=filters, category=category,
                    language="en" if index < 6 else "zh", split="dev" if index < 2 else "test",
                    judgment_source="agent_pass2",
                ))
        summary = validate_frozen_qrels(
            judgments, existing_photo_ids=metadata, photo_metadata=metadata,
        )
        self.assertEqual(summary["queries"], 48)
        self.assertEqual(summary["language_counts"], {"en": 36, "zh": 12})

    def test_refuses_missing_positive_judgments_instead_of_reporting_zero(self):
        path = self.root / "empty.jsonl"
        path.write_text(json.dumps({
            "query_id": "q1", "query": "unlabelled", "relevance": {}, "enabled": True
        }), encoding="utf-8")
        with self.assertRaisesRegex(QrelsValidationError, "no positive human relevance"):
            load_qrels(path)

        path.write_text(json.dumps({
            "query_id": "q1", "query": "template", "relevance": {}, "enabled": False
        }), encoding="utf-8")
        with self.assertRaisesRegex(QrelsValidationError, "No enabled queries"):
            load_qrels(path)

    def test_temporal_pair_requires_ordered_pair_identifier(self):
        path = self.root / "pair.json"
        path.write_text(json.dumps([{
            "query_id": "q1", "query": "a before b", "task_type": "temporal_pair",
            "relevance": {"single-photo-id": 1}
        }]), encoding="utf-8")
        with self.assertRaisesRegex(QrelsValidationError, "temporal_pair qrel IDs"):
            load_qrels(path)

    def test_metadata_dates_are_validated_before_evaluation(self):
        path = self.root / "bad_date.json"
        path.write_text(json.dumps([{
            "query_id": "q1", "query": "trip", "relevance": {"photo": 1},
            "filters": {"start_date": "2026-05-02", "end_date": "2026-05-01"}
        }]), encoding="utf-8")
        with self.assertRaisesRegex(QrelsValidationError, "start_date is after end_date"):
            load_qrels(path)

    def test_report_write_is_valid_json(self):
        path = write_report({"value": "中文", "metric": 0.5}, self.root / "report.json")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["metric"], 0.5)
        self.assertFalse((self.root / "report.json.tmp").exists())


class AblationRunnerTests(unittest.TestCase):
    def test_variants_accumulate_channels_then_metadata_and_full_logic(self):
        retriever = FakeRetriever()
        judgment = QueryJudgment(
            "q1", "beach", {"metadata-hit": 1}, filters={"start_date": "2026-01-01"}
        )
        report = RetrievalEvaluator(retriever).evaluate(
            [judgment], ks=(1,), repeats=1, warmup=0
        )
        self.assertEqual(list(report["variants"]), [spec.variant_id for spec in DEFAULT_ABLATIONS])
        standard_calls = [call for call in retriever.calls if call[0] == "standard"]
        self.assertEqual(standard_calls[0][2], ("clip",))
        self.assertEqual(standard_calls[1][2], ("clip", "caption"))
        self.assertEqual(standard_calls[2][2], ("clip", "scene_graph"))
        self.assertIsNone(standard_calls[2][3].start_date)
        self.assertEqual(standard_calls[3][2], ("clip", "caption", "scene_graph"))
        self.assertIsNone(standard_calls[3][3].start_date)
        self.assertEqual(standard_calls[4][3].start_date, "2026-01-01")
        self.assertTrue(all(call[0] == "standard" for call in retriever.calls))
        self.assertEqual(report["variants"]["A4"]["metrics_macro"]["Recall@1"], 1.0)
        self.assertEqual(report["accuracy_status"], "computed_from_supplied_qrels")

    def test_full_variant_can_score_ordered_temporal_pairs(self):
        retriever = FakeRetriever()
        judgment = QueryJudgment(
            "q-pair",
            "meal before museum",
            {temporal_pair_id("before-id", "after-id"): 2},
            task_type="temporal_pair",
        )
        report = RetrievalEvaluator(retriever).evaluate(
            [judgment], ks=(1,), repeats=2, warmup=0, include_temporal_mode=True
        )
        self.assertTrue(all(
            value["status"] == "not_computed_no_matching_qrels"
            for value in report["variants"].values()
        ))
        temporal = report["optional_modes"]["temporal_pair"]
        self.assertEqual(temporal["metrics_macro"]["Recall@1"], 1.0)
        self.assertEqual(temporal["latency_ms"]["count"], 2)
        self.assertTrue(temporal["all_rankings_deterministic"])


if __name__ == "__main__":
    unittest.main()
