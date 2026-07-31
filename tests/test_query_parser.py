from __future__ import annotations

import unittest

from query_parser import parse_query


class QueryParserTests(unittest.TestCase):
    def test_english_before_pair_with_window(self):
        parsed = parse_query("waiting at the airport before boarding a plane within 2 hours")
        self.assertEqual(parsed.query_type, "temporal_pair")
        self.assertEqual(parsed.temporal.before, "waiting at the airport")
        self.assertEqual(parsed.temporal.after, "boarding a plane")
        self.assertEqual(parsed.temporal.window_minutes, 120)

    def test_english_after_reverses_pair_order(self):
        parsed = parse_query("boarding a plane after waiting at the airport")
        self.assertEqual(parsed.temporal.before, "waiting at the airport")
        self.assertEqual(parsed.temporal.after, "boarding a plane")

    def test_chinese_pair(self):
        parsed = parse_query("在咖啡店喝咖啡之后乘坐火车")
        self.assertEqual(parsed.query_type, "temporal_pair")
        self.assertIn("咖啡", parsed.temporal.before)
        self.assertIn("火车", parsed.temporal.after)

    def test_neighbor_query(self):
        parsed = parse_query("what happened before boarding the train")
        self.assertEqual(parsed.query_type, "temporal_neighbor")
        self.assertEqual(parsed.temporal.direction, "before")

        parsed_after = parse_query("what happened after boarding the train within 2 hours")
        self.assertEqual(parsed_after.query_type, "temporal_neighbor")
        self.assertEqual(parsed_after.temporal.direction, "after")
        self.assertEqual(parsed_after.temporal.anchor, "boarding the train")
        self.assertEqual(parsed_after.temporal.window_minutes, 120)

    def test_relation_detection(self):
        parsed = parse_query("一个人在桌边使用电脑")
        self.assertEqual(parsed.query_type, "relation")

    def test_iso_date_is_extracted(self):
        parsed = parse_query("beach photos 2025-06-01")
        self.assertEqual(parsed.start_date, "2025-06-01")
        self.assertEqual(parsed.end_date, "2025-06-01")


if __name__ == "__main__":
    unittest.main()
