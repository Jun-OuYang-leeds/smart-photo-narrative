import json
import sys
import types
import unittest
from unittest.mock import patch

from config import (
    OLLAMA_MODEL,
    OLLAMA_QUERY_NUM_CTX,
    OLLAMA_QUERY_NUM_PREDICT,
    OLLAMA_STORY_NUM_CTX,
    OLLAMA_STORY_NUM_PREDICT,
    OLLAMA_THINK,
)
from query_parser import parse_query
from story_agent import OllamaGenerator


class QwenOllamaConfigurationTests(unittest.TestCase):
    def test_production_defaults_use_qwen_with_thinking_disabled(self) -> None:
        self.assertEqual(OLLAMA_MODEL, "qwen3:4b")
        self.assertIs(OLLAMA_THINK, False)

    def test_story_request_explicitly_disables_thinking(self) -> None:
        calls = []

        def fake_list():
            return {"models": [{"model": "qwen3:4b"}]}

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {"message": {"content": '{"title":"ok","paragraphs":[]}'}}

        fake_ollama = types.SimpleNamespace(list=fake_list, chat=fake_chat)
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            output = OllamaGenerator().generate("system", "evidence", temperature=0.0, seed=20260707)

        self.assertTrue(output)
        self.assertEqual(calls[0]["model"], "qwen3:4b")
        self.assertIs(calls[0]["think"], False)
        self.assertEqual(calls[0]["options"]["num_ctx"], OLLAMA_STORY_NUM_CTX)
        self.assertEqual(calls[0]["options"]["num_predict"], OLLAMA_STORY_NUM_PREDICT)
        self.assertEqual(calls[0]["options"]["seed"], 20260707)

    def test_optional_query_parser_uses_qwen_without_thinking(self) -> None:
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            payload = {
                "visual_text": "person at a desk",
                "relation_text": None,
                "before": None,
                "after": None,
                "anchor": None,
                "direction": None,
                "window_minutes": 240,
                "start_date": None,
                "end_date": None,
                "location": None,
            }
            return {"message": {"content": json.dumps(payload)}}

        fake_ollama = types.SimpleNamespace(chat=fake_chat)
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            parsed = parse_query("person at a desk", use_ollama=True)

        self.assertEqual(parsed.visual_text, "person at a desk")
        self.assertEqual(calls[0]["model"], "qwen3:4b")
        self.assertIs(calls[0]["think"], False)
        self.assertEqual(calls[0]["options"]["num_ctx"], OLLAMA_QUERY_NUM_CTX)
        self.assertEqual(calls[0]["options"]["num_predict"], OLLAMA_QUERY_NUM_PREDICT)


if __name__ == "__main__":
    unittest.main()
