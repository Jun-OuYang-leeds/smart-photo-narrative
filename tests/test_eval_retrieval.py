"""Tests for the R0--R7 retrieval variants (fake backends + real in-memory FTS5)."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_retrieval as er  # noqa: E402
from config import BLIP2_MODEL_NAME, BLIP_MODEL_NAME  # noqa: E402


class FakeHit(types.SimpleNamespace):
    pass


class FakeClip:
    def encode_text(self, text):
        return [1.0, 0.0]


class FakeVec:
    def __init__(self, image_ranking, scene_ranking):
        self.image_ranking = list(image_ranking)
        self.scene_ranking = list(scene_ranking)

    def query_images(self, emb, *, top_k, eligible_photo_ids=None):
        return [FakeHit(photo_id=p) for p in self.image_ranking[:top_k]]

    def query_scene_graph(self, emb, *, top_k, eligible_photo_ids=None):
        return [FakeHit(photo_id=p) for p in self.scene_ranking[:top_k]]


class FakeFTS:
    def __init__(self, ranking):
        self.ranking = list(ranking)

    def search(self, which, query, *, limit, eligible_photo_ids=None):
        return [(p, float(i)) for i, p in enumerate(self.ranking)][:limit]


class RrfCombineTests(unittest.TestCase):
    def test_single_channel_preserves_order(self):
        self.assertEqual(er.rrf_combine({"clip": ["a", "b", "c"]}), ["a", "b", "c"])

    def test_two_channels_fuse_by_weight(self):
        ranked = er.rrf_combine(
            {"clip": ["a", "b", "c"], "blip": ["b", "a", "d"]}
        )
        # a is clip-rank-1 and blip-rank-2 -> beats b (clip-2, blip-1) because
        # the heavier clip channel favours a.
        self.assertEqual(ranked[0], "a")
        self.assertEqual(set(ranked), {"a", "b", "c", "d"})

    def test_tie_break_is_deterministic(self):
        # two ids with identical scores break by photo_id ascending
        ranked = er.rrf_combine({"clip": ["z", "a"]}, weights={"clip": 1.0})
        self.assertEqual(ranked, ["z", "a"])  # z rank1 > a rank2
        # same channel, swapped input order
        ranked2 = er.rrf_combine({"clip": ["a", "z"]}, weights={"clip": 1.0})
        self.assertEqual(ranked2, ["a", "z"])


class FtsQueryTests(unittest.TestCase):
    def test_terms_quoted_ored(self):
        # OR (matches production match_all_terms=False); AND would zero out
        # long queries vs short captions.
        self.assertEqual(er._fts_query("red bicycle"), '"red" OR "bicycle"')
        self.assertEqual(er._fts_query("a,b!c"), '"a" OR "b" OR "c"')

    def test_empty_query(self):
        self.assertEqual(er._fts_query("!!!"), '""')


class EvalCaptionFTSTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fts = er.EvalCaptionFTS(db_path=Path(self.tmp.name) / "caps.db")
        self.prod = sqlite3.connect(":memory:")
        self.prod.executescript(
            "CREATE TABLE captions(photo_id TEXT, model_name TEXT, status TEXT, caption TEXT);"
        )
        # BLIP captions
        self.prod.executemany(
            "INSERT INTO captions VALUES (?,?,?,?)",
            [
                ("p1", BLIP_MODEL_NAME, "ok", "a red bicycle on a stone bridge"),
                ("p2", BLIP_MODEL_NAME, "ok", "a calm lake at sunrise with fog"),
                ("p3", BLIP_MODEL_NAME, "ok", ""),  # empty -> skipped
            ],
        )
        # BLIP2 captions (distinct text so the two indexes differ)
        self.prod.executemany(
            "INSERT INTO captions VALUES (?,?,?,?)",
            [
                ("p1", BLIP2_MODEL_NAME, "ok", "red racing bicycle leaning on a railing"),
                ("p2", BLIP2_MODEL_NAME, "ok", "misty lake reflecting the morning sun"),
            ],
        )
        self.prod.commit()

    def tearDown(self):
        self.prod.close()
        self.tmp.cleanup()

    def test_build_separates_models(self):
        counts = self.fts.build(self.prod)
        self.assertEqual(counts[er.BLIP_TABLE], 2)   # p3 empty skipped
        self.assertEqual(counts[er.BLIP2_TABLE], 2)

    def test_search_blip_vs_blip2(self):
        self.fts.build(self.prod)
        blip = self.fts.search("blip", "red bicycle")
        blip2 = self.fts.search("blip2", "red bicycle")
        # both find p1 (the only photo about a bicycle)
        self.assertEqual([p for p, _ in blip], ["p1"])
        self.assertEqual([p for p, _ in blip2], ["p1"])

    def test_search_eligible_filter(self):
        self.fts.build(self.prod)
        # restrict to p2 only
        hits = self.fts.search("blip", "lake", eligible_photo_ids=["p2"])
        self.assertEqual([p for p, _ in hits], ["p2"])
        # empty eligible -> empty
        self.assertEqual(self.fts.search("blip", "lake", eligible_photo_ids=[]), [])


class MetadataFilterTests(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(
            "CREATE TABLE photos(photo_id TEXT, date_local TEXT, location TEXT);"
        )
        self.con.executemany(
            "INSERT INTO photos VALUES (?,?,?)",
            [("p1", "2025-12-18", "Paris"), ("p2", "2025-12-18", "Lyon"),
             ("p3", "2025-12-19", "Paris"), ("p4", "2025-12-18", "")],
        )
        self.con.commit()

    def tearDown(self):
        self.con.close()

    def test_exact_date_and_location(self):
        corpus = ["p1", "p2", "p3", "p4"]
        self.assertEqual(er.metadata_filter_ids(self.con, date_local="2025-12-18",
                                                location="Paris", corpus=corpus), ["p1"])

    def test_empty_location_returns_none(self):
        self.assertEqual(er.metadata_filter_ids(self.con, date_local="2025-12-18",
                                                location="", corpus=["p1"]), [])


class RunVariantTests(unittest.TestCase):
    def setUp(self):
        self.corpus = ["a", "b", "c", "d", "e"]

    def _backend(self, image=None, scene=None, caption=None):
        return er.RetrievalBackend(
            clip_backend=FakeClip(),
            vector_store=FakeVec(image or ["a", "b", "c"], scene or ["c", "b", "a"]),
            caption_fts=FakeFTS(caption or ["b", "a", "d"]),
        )

    def test_r0_clip_only(self):
        res = er.run_variant("R0", "q", self._backend(image=["a", "b", "c"]), self.corpus)
        self.assertEqual(res.ranked_ids, ["a", "b", "c"])
        self.assertEqual(res.channels_used, ("clip",))

    def test_r1_blip_caption_only(self):
        res = er.run_variant("R1", "q", self._backend(caption=["b", "a", "d"]), self.corpus)
        self.assertEqual(res.ranked_ids, ["b", "a", "d"])
        self.assertEqual(res.channels_used, ("blip",))

    def test_r3_scene_only(self):
        res = er.run_variant("R3", "q", self._backend(scene=["c", "b", "a"]), self.corpus)
        self.assertEqual(res.ranked_ids, ["c", "b", "a"])

    def test_r6_combines_three_channels(self):
        res = er.run_variant("R6", "q", self._backend(), self.corpus)
        self.assertEqual(res.channels_used, ("clip", "blip2", "scene"))
        # b is blip2-rank-1 + scene-rank-2 + clip-rank-2 -> edges out a
        # (clip-rank-1 + blip2-rank-2 + scene-rank-3) under the fixed weights.
        self.assertEqual(res.ranked_ids, ["b", "a", "c", "d"])

    def test_r7_records_pool_sizes_and_restricts(self):
        backend = self._backend(image=["a", "b"], scene=["b"], caption=["a"])
        eligible = ["a", "b"]  # metadata-filtered subset of the corpus
        res = er.run_r7("q", backend, self.corpus, eligible_ids=eligible)
        self.assertEqual(res.pool_before, len(self.corpus))
        self.assertEqual(res.pool_after, 2)
        self.assertTrue(set(res.ranked_ids) <= set(eligible))


if __name__ == "__main__":
    unittest.main()
