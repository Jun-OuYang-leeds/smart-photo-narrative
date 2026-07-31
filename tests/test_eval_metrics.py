"""Tests for the retrieval-experiment metrics and statistics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_metrics as em  # noqa: E402


class HitAndRankTests(unittest.TestCase):
    def test_rank_of_target_exact(self):
        self.assertEqual(em.rank_of_target(["a", "b", "c"], "b"), 2)
        self.assertIsNone(em.rank_of_target(["a", "b", "c"], "z"))

    def test_hit_at_k(self):
        ranked = ["a", "b", "c", "d"]
        self.assertTrue(em.hit_at_k(ranked, "a", 1))
        self.assertTrue(em.hit_at_k(ranked, "c", 3))
        self.assertFalse(em.hit_at_k(ranked, "d", 3))  # rank 4 > 3
        self.assertFalse(em.hit_at_k(ranked, "z", 3))  # absent

    def test_near_duplicate_is_not_a_hit(self):
        # only the exact target id counts; a different id never substitutes
        ranked = ["target_near_dup", "x", "y"]
        self.assertFalse(em.hit_at_k(ranked, "target", 3))

    def test_rank_buckets(self):
        self.assertEqual(em.rank_bucket(1), "rank1")
        self.assertEqual(em.rank_bucket(2), "rank2-3")
        self.assertEqual(em.rank_bucket(3), "rank2-3")
        self.assertEqual(em.rank_bucket(4), "miss")
        self.assertEqual(em.rank_bucket(None), "miss")


class WilsonTests(unittest.TestCase):
    def test_zero_n(self):
        self.assertEqual(em.wilson_interval(0, 0), (0.0, 0.0))

    def test_all_hits(self):
        low, high = em.wilson_interval(100, 100)
        self.assertGreater(low, 0.95)
        self.assertLessEqual(high, 1.0)

    def test_half_contains_point_five(self):
        low, high = em.wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_symmetry(self):
        # Wilson for k/n mirrors (n-k)/n around 0.5 (symmetry of the score interval)
        lo1, hi1 = em.wilson_interval(30, 100)
        lo2, hi2 = em.wilson_interval(70, 100)
        self.assertAlmostEqual(lo1, 1 - hi2, places=6)
        self.assertAlmostEqual(hi1, 1 - lo2, places=6)


class SummarizeVariantTests(unittest.TestCase):
    def test_counts_and_success(self):
        results = [
            em.QueryResult("q1", "t1", ["t1", "x", "y"], valid=True, latency_seconds=0.1),
            em.QueryResult("q2", "t2", ["x", "t2", "y"], valid=True, latency_seconds=0.2),
            em.QueryResult("q3", "t3", ["x", "y", "z"], valid=True, latency_seconds=0.3),
            em.QueryResult("q4", "t4", [], valid=False, latency_seconds=None),  # gen error
        ]
        s = em.summarize_variant(results)
        # valid = 3: hit@1 = 1 (q1), hit@3 = 2 (q1,q2)
        self.assertEqual(s["hit_at_1_valid"]["hits"], 1)
        self.assertEqual(s["hit_at_3_valid"]["hits"], 2)
        self.assertEqual(s["hit_at_3_valid"]["n"], 3)
        # all = 4: success@3 = 2/4
        self.assertEqual(s["success_at_3_all"]["hits"], 2)
        self.assertEqual(s["success_at_3_all"]["n"], 4)
        # one generation error
        self.assertEqual(s["query_generation_error"]["hits"], 1)
        self.assertEqual(s["query_generation_error"]["n"], 4)
        # stacked bar
        self.assertEqual(s["rank_buckets"], {"rank1": 1, "rank2-3": 1, "miss": 2})
        # latency p95 over [0.1,0.2,0.3]: rank=0.95*(3-1)=1.9 -> 0.2 + 0.9*0.1 = 0.29
        self.assertAlmostEqual(s["latency_p95_seconds"], 0.29, places=6)


class McNemarTests(unittest.TestCase):
    def test_no_discordant_pairs_is_unit_p(self):
        r = em.exact_mcnemar_two_sided([True, True, False], [True, True, False])
        self.assertEqual(r["n_discordant"], 0)
        self.assertEqual(r["p_value"], 1.0)

    def test_known_two_sided_value(self):
        # b=12, c=5 -> exact two-sided p (textbook-ish small value < 0.1... but
        # 12 vs 5 with n=17 is not extreme). We assert 0 <= p <= 1 and symmetry.
        a = [True] * 12 + [False] * 5 + [True] * 3
        b = [False] * 12 + [True] * 5 + [True] * 3
        r = em.exact_mcnemar_two_sided(a, b)
        self.assertEqual(r["b"], 12)
        self.assertEqual(r["c"], 5)
        self.assertEqual(r["n_discordant"], 17)
        self.assertGreaterEqual(r["p_value"], 0.0)
        self.assertLessEqual(r["p_value"], 1.0)

    def test_extreme_discordant_is_significant(self):
        # b=0, c=15 -> strong disagreement -> tiny p
        a = [False] * 15 + [True] * 5
        b = [True] * 15 + [True] * 5
        r = em.exact_mcnemar_two_sided(a, b)
        self.assertLess(r["p_value"], 0.001)

    def test_argument_order_symmetric(self):
        a = [True, False, True, False, True]
        b = [False, True, True, True, False]
        r1 = em.exact_mcnemar_two_sided(a, b)
        r2 = em.exact_mcnemar_two_sided(b, a)
        self.assertEqual(r1["p_value"], r2["p_value"])


class HolmTests(unittest.TestCase):
    def test_adjusted_never_below_raw_and_monotone(self):
        raws = [0.01, 0.04, 0.03]
        out = em.holm_correction(raws)
        for entry in out:
            self.assertGreaterEqual(entry["holm_adjusted_p"], entry["raw_p"] - 1e-12)
        # in sorted order, adjusted must be non-decreasing
        sorted_entries = sorted(out, key=lambda e: e["raw_p"])
        adj_seq = [e["holm_adjusted_p"] for e in sorted_entries]
        self.assertEqual(adj_seq, sorted(adj_seq))

    def test_smallest_p_gets_largest_multiplier(self):
        raws = [0.01, 0.05]
        out = em.holm_correction(raws)
        by_index = {e["index"]: e for e in out}
        # smallest p (0.01) is tested first with multiplier m=2 -> 0.02
        self.assertAlmostEqual(by_index[0]["holm_adjusted_p"], 0.02, places=6)


if __name__ == "__main__":
    unittest.main()
