"""End-to-end logic test for the retrieval experiment orchestrator (fakes)."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_experiment as ex  # noqa: E402
from eval_query_gen import QueryGenOutcome  # noqa: E402


def _make_targets():
    # t1..t5 metadata-complete; t6..t8 incomplete; t5 is a query-gen error.
    rows = []
    for i in range(1, 6):
        rows.append({
            "photo_id": f"t{i}", "timestamp_confidence": 0.8,
            "date_local": f"2025-12-1{i}", "location": f"City{i}",
        })
    for i in range(6, 9):
        rows.append({"photo_id": f"t{i}", "timestamp_confidence": 0.2,
                     "date_local": "", "location": ""})
    return rows


def _query_gen_fn(target_id):
    if target_id == "t5":
        return QueryGenOutcome(target_id=target_id, status="query_generation_error")
    return QueryGenOutcome(target_id=target_id, status="ok", query=f"query for {target_id}")


def _rank_fn(variant, query, target_id):
    # R0 always ranks the target 2nd -> Hit@1 miss; others rank it 1st -> hit.
    if variant == "R0":
        return (["distractor", target_id, "x"], 0.1)
    if variant == "R7":
        return ([target_id, "distractor"], 0.2)
    return ([target_id, "distractor", "x"], 0.1)


class MetadataSubsetTests(unittest.TestCase):
    def test_filters_to_complete_and_seeded(self):
        rows = _make_targets()
        sub = ex.select_metadata_subset(rows, count=3, seed=42)
        self.assertEqual(len(sub), 3)
        self.assertTrue(set(sub) <= {f"t{i}" for i in range(1, 6)})  # only complete
        # deterministic
        self.assertEqual(sub, ex.select_metadata_subset(rows, count=3, seed=42))

    def test_completeness_rule(self):
        self.assertTrue(ex.metadata_complete({"timestamp_confidence": 0.6,
                                              "date_local": "d", "location": "l"}))
        self.assertFalse(ex.metadata_complete({"timestamp_confidence": 0.6,
                                               "date_local": "d", "location": ""}))


class RunExperimentTests(unittest.TestCase):
    def setUp(self):
        self.rows = _make_targets()
        self.targets = [r["photo_id"] for r in self.rows]
        self.corpus = list(self.targets) + ["distractor", "x"]
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(
            "CREATE TABLE photos(photo_id TEXT, date_local TEXT, location TEXT);"
        )
        for r in self.rows:
            self.con.execute(
                "INSERT INTO photos VALUES (?,?,?)",
                (r["photo_id"], r["date_local"], r["location"]),
            )
        self.con.commit()

    def tearDown(self):
        self.con.close()

    def _run(self):
        cfg = ex.ExperimentConfig(
            variants=("R0", "R6"), metadata_subset_count=3,
            comparisons=(("R0", "R6"),),
        )
        return ex.run_experiment(
            self.targets, self.corpus, self.con,
            query_gen_fn=_query_gen_fn, rank_fn=_rank_fn,
            target_rows=self.rows, config=cfg, base_seed=100,
        )

    def test_per_variant_hit_counts(self):
        res = self._run()
        # 8 targets, t5 is a gen error -> 7 valid.
        # R6 ranks target 1st -> 7 Hit@1; R0 ranks it 2nd -> 0 Hit@1.
        self.assertEqual(res["summaries"]["R6"]["hit_at_1_valid"]["hits"], 7)
        self.assertEqual(res["summaries"]["R6"]["hit_at_1_valid"]["n"], 7)
        self.assertEqual(res["summaries"]["R0"]["hit_at_1_valid"]["hits"], 0)
        # end-to-end success counts over all 8
        self.assertEqual(res["summaries"]["R6"]["success_at_1_all"]["n"], 8)
        # one generation error recorded
        self.assertEqual(res["summaries"]["R6"]["query_generation_error"]["hits"], 1)

    def test_metadata_subset_and_r7(self):
        res = self._run()
        self.assertEqual(len(res["metadata_subset"]), 3)
        # R6-M (unfiltered) on the 3 complete targets: all hit (valid + rank1)
        self.assertEqual(res["metadata_r6m_summary"]["hit_at_1_valid"]["hits"], 3)
        # R7 also hits (rank1) on the same 3
        self.assertEqual(res["metadata_r7_summary"]["hit_at_1_valid"]["hits"], 3)
        # Fix 2: candidate-pool sizes are reported (spec: don't misattribute
        # candidate reduction to semantic ranking).
        pool = res["metadata_r7_pool"]
        self.assertEqual(pool["pool_before"], len(self.corpus))
        self.assertEqual(pool["n_targets"], 3)
        self.assertEqual(pool["after_median"], 1)  # each target has unique date+location

    def test_comparisons_have_holm_adjustment(self):
        res = self._run()
        pairs = [c["pair"] for c in res["comparisons"]]
        self.assertIn("R0_vs_R6", pairs)
        self.assertIn("R6M_vs_R7", pairs)
        for rec in res["comparisons"]:
            self.assertIn("p_value", rec)
            self.assertIn("holm_adjusted_p", rec)
            self.assertGreaterEqual(rec["holm_adjusted_p"], rec["p_value"] - 1e-12)

    def test_gen_error_target_still_in_subset(self):
        # t5 is a query-gen error but is metadata-complete; the subset may
        # include it, and it must NOT be replaced (counts as a miss, not dropped).
        rows = [r for r in self.rows if r["photo_id"] != "t6"]  # keep t5 complete
        # force the subset to include t5 by lowering count to the 4 complete ids
        sub = ex.select_metadata_subset(
            [r for r in self.rows if r["photo_id"] in {"t1", "t2", "t3", "t4", "t5"}],
            count=5, seed=7,
        )
        self.assertIn("t5", sub)


if __name__ == "__main__":
    unittest.main()
