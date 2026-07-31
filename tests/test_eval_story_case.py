"""Tests for the single-event Story case study (freeze + contexts + generation)."""

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

import eval_story_case as sc  # noqa: E402


# ----------------------------- Fakes -----------------------------


class FakeAggregator:
    def __init__(self):
        self.calls = []  # list of kwargs

    def aggregate_by_event(self, event_id, *, verified_context="", narrator_role="observer",
                        use_photographer_mood=False):
        kw = {"event_id": event_id, "verified_context": verified_context,
              "narrator_role": narrator_role,
              "use_photographer_mood": use_photographer_mood}
        self.calls.append(kw)
        return types.SimpleNamespace(use_photographer_mood=use_photographer_mood,
                                     tag=len(self.calls))


class FakeStoryGen:
    def __init__(self):
        self.calls = []

    def generate_story_with_context(self, context, temperature=0.0, *, mode="faithful",
                                    language="zh", save=True, seed=None,
                                    allow_deterministic_fallback=True, num_predict=None):
        kw = {"context": context, "temperature": temperature, "mode": mode,
              "language": language, "save": save, "seed": seed,
              "allow_deterministic_fallback": allow_deterministic_fallback,
              "num_predict": num_predict}
        self.calls.append(kw)
        return types.SimpleNamespace(title=f"{mode}-{seed}", paragraphs=["p1", "p2"],
                                     creative_transitions=["t"], opening="", closing="",
                                     mode=mode, status="ok")


class FakeMoodStorage:
    def __init__(self, moods):
        self.moods = moods  # {photo_id: SimpleNamespace(mood_label, annotation_set)}

    def get_photo_moods(self, photo_ids):
        return {pid: self.moods[pid] for pid in photo_ids if pid in self.moods}


def _mood(label, ann):
    return types.SimpleNamespace(mood_label=label, annotation_set=ann)


# ----------------------------- Tests -----------------------------


class FreezeEventTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.photos_dir = Path(self.tmp.name)
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(
            "CREATE TABLE photos(photo_id TEXT, relative_path TEXT);"
            "CREATE TABLE event_photos(event_id TEXT, photo_id TEXT, position INTEGER);"
        )
        for i in range(3):
            pid = f"p{i}"
            (self.photos_dir / f"{pid}.jpg").write_bytes(f"img{i}".encode())
            self.con.execute("INSERT INTO photos VALUES (?,?)", (pid, f"{pid}.jpg"))
            self.con.execute("INSERT INTO event_photos VALUES (?,?,?)",
                             ("EVT", pid, i))
        self.con.commit()

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_freeze_orders_and_hashes(self):
        fe = sc.freeze_event(self.con, "EVT", photos_dir=self.photos_dir)
        self.assertEqual(fe.photo_ids, ["p0", "p1", "p2"])
        self.assertEqual(len(fe.photo_hashes), 3)
        self.assertTrue(all(len(h) == 64 for h in fe.photo_hashes.values()))
        self.assertEqual(fe.order_hash, sc.hashlib.sha256(b"p0\np1\np2").hexdigest())

    def test_missing_file_raises(self):
        (self.photos_dir / "p1.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            sc.freeze_event(self.con, "EVT", photos_dir=self.photos_dir)


class MoodValidationTests(unittest.TestCase):
    def test_all_present_ok(self):
        storage = FakeMoodStorage({p: _mood("calm", sc.DEFAULT_MOOD_SET) for p in ["a", "b"]})
        out = sc.validate_event_moods(storage, ["a", "b"])
        self.assertEqual(out, {"a": "calm", "b": "calm"})

    def test_missing_raises(self):
        storage = FakeMoodStorage({"a": _mood("calm", sc.DEFAULT_MOOD_SET)})
        with self.assertRaises(ValueError):
            sc.validate_event_moods(storage, ["a", "b"])

    def test_wrong_annotation_set_raises(self):
        storage = FakeMoodStorage({"a": _mood("calm", "other_set_v1")})
        with self.assertRaises(ValueError):
            sc.validate_event_moods(storage, ["a"])


class BuildContextsTests(unittest.TestCase):
    def test_variant_flags_and_observer_role(self):
        agg = FakeAggregator()
        sc.build_contexts(agg, "EVT")
        # Three calls, one per variant
        self.assertEqual(len(agg.calls), 3)
        by_event_role = {(c["event_id"], c["narrator_role"], c["verified_context"])
                         for c in agg.calls}
        self.assertEqual(by_event_role, {("EVT", "observer", "")})
        # Map back variant key -> use_mood: S-F, S-C0, S-C1
        use_mood_by_call = [c["use_photographer_mood"] for c in agg.calls]
        self.assertEqual(use_mood_by_call, [False, False, True])

    def test_sc0_and_sc1_differ_only_in_mood(self):
        """The core invariant: S-C0 and S-C1 contexts are identical except the
        ``use_photographer_mood`` flag (single experimental variable)."""
        agg = FakeAggregator()
        sc.build_contexts(agg, "EVT")
        # calls[1] = S-C0, calls[2] = S-C1
        c0, c1 = agg.calls[1], agg.calls[2]
        diff = {k: (c0[k], c1[k]) for k in set(c0) | set(c1) if c0[k] != c1[k]}
        self.assertEqual(diff, {"use_photographer_mood": (False, True)})


class GenerateTests(unittest.TestCase):
    def test_generate_variant_fixed_conditions(self):
        gen = FakeStoryGen()
        ctx = types.SimpleNamespace(tag=1)
        sc.generate_variant(gen, ctx, mode="creative", seed=999)
        kw = gen.calls[0]
        self.assertFalse(kw["save"])                          # no production write
        self.assertFalse(kw["allow_deterministic_fallback"])  # no fallback
        self.assertEqual(kw["mode"], "creative")
        self.assertEqual(kw["language"], "en")
        self.assertEqual(kw["seed"], 999)
        self.assertEqual(kw["num_predict"], 2048)
        self.assertAlmostEqual(kw["temperature"], 0.25)

    def test_generate_three_produces_three_with_distinct_seeds(self):
        gen = FakeStoryGen()
        agg = FakeAggregator()
        stories = sc.generate_three(gen, agg, event_id="EVT", base_seed=1000)
        self.assertEqual(set(stories.keys()), {"S-F", "S-C0", "S-C1"})
        keys = ["S-F", "S-C0", "S-C1"]
        for i, key in enumerate(keys):
            self.assertEqual(gen.calls[i]["mode"], "faithful" if key == "S-F" else "creative")
            # deterministic per-variant seed
            self.assertEqual(gen.calls[i]["seed"], 1000 + sc._stable_int(key))
        seeds = [kw["seed"] for kw in gen.calls]
        self.assertEqual(len(seeds), len(set(seeds)))  # all distinct


class AnonymizeTests(unittest.TestCase):
    def test_deterministic_and_consistent(self):
        a = sc.anonymize_pair("S-C0", "S-C1", "story0", "story1", seed=42)
        b = sc.anonymize_pair("S-C0", "S-C1", "story0", "story1", seed=42)
        self.assertEqual(a, b)
        # mapping is internally consistent
        for label, key in a["mapping"].items():
            self.assertEqual(a["texts"][label],
                             "story0" if key == "S-C0" else "story1")
        self.assertEqual(set(a["mapping"].values()), {"S-C0", "S-C1"})

    def test_both_assignments_possible_across_seeds(self):
        assignments = {
            sc.anonymize_pair("A", "B", "x", "y", seed=s)["labels"]["A"]
            for s in range(50)
        }
        self.assertEqual(assignments, {"A", "B"})


class StoryTextTests(unittest.TestCase):
    def test_flattens(self):
        story = types.SimpleNamespace(title="T", paragraphs=["p1", "p2"],
                                      creative_transitions=["tr"], opening="O", closing="C")
        text = sc.story_text(story)
        self.assertIn("T", text)
        self.assertIn("p1", text)
        self.assertIn("tr", text)
        self.assertIn("O", text)
        self.assertIn("C", text)


if __name__ == "__main__":
    unittest.main()
