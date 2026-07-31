from __future__ import annotations

import unittest

from app import _story_markdown
from story_agent import GeneratedStory, StoryParagraph


class StoryUiTests(unittest.TestCase):
    def test_markdown_interleaves_facts_transitions_and_evidence(self):
        story = GeneratedStory(
            date="2026-07-21",
            title="Layered story",
            paragraphs=[
                StoryParagraph("Fact one.", ("P001",), "G001"),
                StoryParagraph("Fact two.", ("P002", "P003"), "G002"),
                StoryParagraph("Fact three.", ("P004",), "G003"),
            ],
            photo_count=4,
            photo_ids=["one", "two", "three", "four"],
            model="qwen3:4b",
            mode="creative",
            language="en",
            creative_transitions=["Creative bridge one.", "Creative bridge two."],
        )

        markdown = _story_markdown(story)
        expected_order = [
            "Fact one.",
            "Creative bridge one.",
            "Fact two.",
            "Creative bridge two.",
            "Fact three.",
        ]
        positions = [markdown.index(value) for value in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(markdown.count("创意过渡——非照片事实"), 2)
        self.assertIn("_Evidence: P002, P003_", markdown)

    def test_v5_markdown_is_continuous_and_moves_labels_into_collapsed_audit(self):
        story = GeneratedStory(
            date="2026-07-07",
            title="海边的一段回忆",
            paragraphs=[
                StoryParagraph("我从画面中看到木质甲板和遮阳伞。", ("P001",), "G001"),
                StoryParagraph("我在下一幅画面中看到椅子上的人物。", ("P002",), "G002"),
                StoryParagraph("我从最后的画面中看到远处水面和山丘。", ("P003",), "G003"),
            ],
            photo_count=3,
            photo_ids=["one", "two", "three"],
            model="qwen3:4b",
            mode="creative",
            language="zh",
            opening="我重新翻开这三张照片，让视线沿着画面慢慢向前。",
            creative_transitions=["我继续看向下一张照片。", "我把目光移向更远的景色。"],
            closing="我合上这段记录，留下对这些清楚细节的记忆。",
            event_summary={"date": "2026-07-07", "time_range": "17:48–17:52", "locations": ["Antalya"]},
            narrator_role="observer",
            prompt_version="grounded-story-v5-first-person-memoir",
        )

        markdown = _story_markdown(story)
        main, audit = markdown.split("<details>", 1)
        expected_order = [
            story.opening,
            story.paragraphs[0].text,
            story.creative_transitions[0],
            story.paragraphs[1].text,
            story.creative_transitions[1],
            story.paragraphs[2].text,
            story.closing,
        ]
        positions = [main.index(value) for value in expected_order]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("Evidence", main)
        self.assertNotIn("非照片事实", main)
        self.assertEqual(main.count("2026-07-07"), 1)
        self.assertIn("Evidence P001", audit)
        self.assertIn("non-photo creative narration", audit)


if __name__ == "__main__":
    unittest.main()
