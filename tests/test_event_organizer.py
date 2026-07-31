from __future__ import annotations

import unittest

import numpy as np

from event_organizer import PhotoEventInput, merge_events, organize_events, split_event


def photo(pid, timestamp, minute, **kwargs):
    return PhotoEventInput(
        photo_id=pid,
        captured_at=timestamp,
        captured_at_sort=minute * 60,
        timestamp_confidence=kwargs.pop("timestamp_confidence", "high"),
        **kwargs,
    )


class EventOrganizerTests(unittest.TestCase):
    def test_large_time_gap_splits(self):
        result = organize_events([
            photo("a", "2026-01-01T09:00:00", 0),
            photo("b", "2026-01-01T11:00:00", 120),
        ])
        self.assertEqual(len(result.events), 2)

    def test_visual_change_plus_gap_splits(self):
        result = organize_events([
            photo("a", "2026-01-01T09:00:00", 0, clip_embedding=[1, 0]),
            photo("b", "2026-01-01T09:40:00", 40, clip_embedding=[0, 1]),
        ])
        self.assertEqual(len(result.events), 2)

    def test_numpy_embeddings_are_supported(self):
        result = organize_events([
            photo("a", "2026-01-01T09:00:00", 0, clip_embedding=np.array([1.0, 0.0])),
            photo("b", "2026-01-01T09:40:00", 40, clip_embedding=np.array([0.0, 1.0])),
        ])
        self.assertEqual(len(result.events), 2)

    def test_location_change_splits(self):
        result = organize_events([
            photo("a", "2026-01-01T09:00:00", 0, latitude=51.5, longitude=-0.1),
            photo("b", "2026-01-01T09:20:00", 20, latitude=51.55, longitude=-0.1),
        ])
        self.assertEqual(len(result.events), 2)

    def test_low_confidence_is_unassigned(self):
        result = organize_events([
            photo("a", "2026-01-01T09:00:00", 0, timestamp_confidence="low"),
        ])
        self.assertEqual(result.unassigned_photo_ids, ("a",))
        self.assertFalse(result.events)

    def test_cross_day_and_unique_membership(self):
        items = [
            photo("a", "2026-01-01T23:59:00", 0),
            photo("b", "2026-01-02T00:01:00", 2),
            photo("c", "2026-01-02T00:05:00", 6),
        ]
        first = organize_events(items)
        second = organize_events(reversed(items))
        self.assertEqual(first, second)
        memberships = [pid for event in first.events for pid in event.photo_ids]
        self.assertEqual(sorted(memberships), ["a", "b", "c"])
        self.assertEqual(len(memberships), len(set(memberships)))

    def test_representatives_are_capped_and_include_endpoints(self):
        items = [photo(str(i), f"2026-01-01T09:{i:02d}:00", i, clip_embedding=[1, i / 20]) for i in range(20)]
        event = organize_events(items).events[0]
        self.assertLessEqual(len(event.representative_ids), 8)
        self.assertIn("0", event.representative_ids)
        self.assertIn("19", event.representative_ids)

    def test_manual_split_and_merge(self):
        event = organize_events([
            photo("a", "2026-01-01T09:00:00", 0),
            photo("b", "2026-01-01T09:01:00", 1),
        ]).events[0]
        left, right = split_event(event, "b")
        merged = merge_events(left, right)
        self.assertEqual(merged.photo_ids, ("a", "b"))
        self.assertEqual(merged.confidence, "manual")


if __name__ == "__main__":
    unittest.main()
