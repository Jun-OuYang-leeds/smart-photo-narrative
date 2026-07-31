"""Tests for the frozen-corpus + deterministic target-sampling core."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_corpus as ec  # noqa: E402
from config import BLIP2_MODEL_NAME, BLIP_MODEL_NAME  # noqa: E402


def _build_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE photos (photo_id TEXT PRIMARY KEY, relative_path TEXT,
                             timestamp_confidence REAL, date_local TEXT, location TEXT);
        CREATE TABLE tags (photo_id TEXT, source TEXT);
        CREATE TABLE captions (photo_id TEXT, model_name TEXT);
        CREATE TABLE scene_graphs (photo_id TEXT);
        """
    )
    return con


def _add(con, photo_id, rel, *, clip=True, blip=True, blip2=True, sg=True):
    con.execute("INSERT INTO photos(photo_id, relative_path) VALUES (?,?)", (photo_id, rel))
    if clip:
        con.execute("INSERT INTO tags VALUES (?, 'clip:openai/clip-vit-base-patch32:transformers')", (photo_id,))
    if blip:
        con.execute("INSERT INTO captions VALUES (?,?)", (photo_id, BLIP_MODEL_NAME))
    if blip2:
        con.execute("INSERT INTO captions VALUES (?,?)", (photo_id, BLIP2_MODEL_NAME))
    if sg:
        con.execute("INSERT INTO scene_graphs VALUES (?)", (photo_id,))


class CorpusSqlTests(unittest.TestCase):
    def setUp(self):
        self.con = _build_db()
        # p1, p5: flat + all channels -> corpus AND target
        _add(self.con, "p1", "1.jpg")
        _add(self.con, "p5", "5.jpg")
        # p2: pic2 + all channels -> corpus only
        _add(self.con, "p2", "pic2/x.jpg")
        # p3: flat, missing blip2 -> neither
        _add(self.con, "p3", "3.jpg", blip2=False)
        # p4: flat, missing scene_graph -> neither
        _add(self.con, "p4", "4.jpg", sg=False)
        # p6: pic2, missing clip -> neither
        _add(self.con, "p6", "pic2/y.jpg", clip=False)
        self.con.commit()

    def test_corpus_includes_pic2_all_channel(self):
        self.assertEqual(ec.frozen_corpus_photo_ids(self.con), ["p1", "p2", "p5"])

    def test_target_pool_excludes_pic2(self):
        self.assertEqual(ec.valid_target_pool_photo_ids(self.con), ["p1", "p5"])

    def test_channel_required(self):
        # remove p1's blip2 -> p1 drops out of both corpus and target pool
        self.con.execute("DELETE FROM captions WHERE photo_id='p1' AND model_name=?", (BLIP2_MODEL_NAME,))
        self.con.commit()
        self.assertEqual(ec.frozen_corpus_photo_ids(self.con), ["p2", "p5"])
        self.assertEqual(ec.valid_target_pool_photo_ids(self.con), ["p5"])


class MetadataTierTests(unittest.TestCase):
    def test_complete_requires_time_date_location(self):
        self.assertEqual(ec.metadata_tier({"timestamp_confidence": 0.8, "date_local": "2025-12-18", "location": "Paris"}), "complete")
        self.assertEqual(ec.metadata_tier({"timestamp_confidence": 0.4, "date_local": "2025-12-18", "location": "Paris"}), "partial")
        self.assertEqual(ec.metadata_tier({"timestamp_confidence": 0.8, "date_local": "", "location": "Paris"}), "partial")
        self.assertEqual(ec.metadata_tier({"timestamp_confidence": 0.8, "date_local": "2025-12-18", "location": None}), "partial")


class NearDuplicateTests(unittest.TestCase):
    def test_groups_by_time_and_clip(self):
        ids = ["a", "b", "c", "d"]
        # a,b within 5 min and high cosine -> grouped; c within window but low cosine -> singleton
        ts = {"a": 0.0, "b": 240.0, "c": 120.0, "d": 10000.0}
        vecs = {
            "a": [1.0, 0.0], "b": [0.99, 0.01],  # cosine ~0.9999
            "c": [0.0, 1.0], "d": [1.0, 0.0],    # c orthogonal to a,b
        }
        groups = ec.compute_near_duplicate_groups(ids, vecs, ts)
        # find the component containing a
        comp_a = next(s for s in groups.values() if "a" in s)
        self.assertIn("b", comp_a)
        self.assertNotIn("c", comp_a)
        # d is far in time -> singleton
        comp_d = next(s for s in groups.values() if "d" in s)
        self.assertEqual(comp_d, {"d"})

    def test_time_window_breaks_early(self):
        ids = ["a", "b", "c"]
        ts = {"a": 0.0, "b": 700.0, "c": 1400.0}  # each >10min apart
        vecs = {"a": [1.0], "b": [1.0], "c": [1.0]}
        groups = ec.compute_near_duplicate_groups(ids, vecs, ts)
        sizes = sorted(len(s) for s in groups.values())
        self.assertEqual(sizes, [1, 1, 1])  # all singletons

    def test_index_excludes_singletons(self):
        groups = {"r1": {"a", "b"}, "r2": {"c"}}
        pid_to_group, group_size = ec.near_duplicate_index(groups)
        self.assertEqual(pid_to_group, {"a": 0, "b": 0})
        self.assertEqual(group_size, {0: 2})
        self.assertNotIn("c", pid_to_group)


class StratifiedSampleTests(unittest.TestCase):
    def _make_targets(self, n, *, event_of=None, nd_of=None, stratum_of=None):
        return [f"t{i}" for i in range(n)], event_of or {}, nd_of or {}, stratum_of or {}

    def test_deterministic_same_seed(self):
        targets = [f"t{i}" for i in range(50)]
        stratum = {t: ("complete" if i % 2 else "partial") for i, t in enumerate(targets)}
        a = ec.stratified_sample(targets, stratum, {}, {}, count=20, seed=123)
        b = ec.stratified_sample(targets, stratum, {}, {}, count=20, seed=123)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)

    def test_default_seed_from_protocol_name(self):
        targets = [f"t{i}" for i in range(40)]
        stratum = {t: "partial" for t in targets}
        a = ec.stratified_sample(targets, stratum, {}, {}, count=15)
        b = ec.stratified_sample(targets, stratum, {}, {}, count=15, seed=ec.seed_from_name())
        self.assertEqual(a, b)

    def test_event_cap_respected_when_pool_diverse(self):
        # 60 targets across 10 events (6 each), cap 6, count 30 -> cap never
        # binds, diversity is achievable, every selected event stays <= 6.
        targets = [f"t{i}" for i in range(60)]
        stratum = {t: "partial" for t in targets}
        event = {t: f"E{i // 6}" for i, t in enumerate(targets)}
        sel = ec.stratified_sample(targets, stratum, {}, event, count=30,
                                   max_per_event=6, seed=1)
        self.assertEqual(len(sel), 30)
        from collections import Counter
        evt_counts = Counter(event[t] for t in sel)
        self.assertTrue(all(v <= 6 for v in evt_counts.values()))

    def test_caps_are_soft_and_fill_to_count(self):
        # Degenerate pool: 20 targets all in ONE event and ONE near-dup group.
        # The capped pass yields few; the soft fill phase relaxes the caps to
        # reach the hard count target of 20 ("in principle" caps).
        targets = [f"t{i}" for i in range(20)]
        stratum = {t: "partial" for t in targets}
        event = {t: "E1" for t in targets}
        nd = {t: 0 for t in targets}
        sel = ec.stratified_sample(targets, stratum, nd, event, count=20,
                                   max_per_event=6, max_per_near_dup_group=2, seed=1)
        self.assertEqual(len(sel), 20)

    def test_selection_subset_of_targets(self):
        targets = [f"t{i}" for i in range(60)]
        stratum = {t: ("complete" if i < 20 else "partial") for i, t in enumerate(targets)}
        event = {t: f"E{i // 10}" for i, t in enumerate(targets)}  # 10 per event
        nd = {t: i // 3 for i, t in enumerate(targets)}  # groups of 3
        sel = ec.stratified_sample(targets, stratum, nd, event, count=30, seed=42)
        self.assertEqual(len(sel), 30)
        self.assertTrue(set(sel) <= set(targets))
        # event cap
        ev_counts = {}
        for t in sel:
            ev_counts[event[t]] = ev_counts.get(event[t], 0) + 1
        self.assertTrue(all(c <= 6 for c in ev_counts.values()))
        # nd cap
        nd_counts = {}
        for t in sel:
            nd_counts[nd[t]] = nd_counts.get(nd[t], 0) + 1
        self.assertTrue(all(c <= 2 for c in nd_counts.values()))


class SeedTests(unittest.TestCase):
    def test_seed_from_name_is_stable(self):
        self.assertEqual(ec.seed_from_name(), ec.seed_from_name())
        self.assertEqual(ec.seed_from_name("x"), ec.seed_from_name("x"))
        self.assertNotEqual(ec.seed_from_name("x"), ec.seed_from_name("y"))


if __name__ == "__main__":
    unittest.main()
