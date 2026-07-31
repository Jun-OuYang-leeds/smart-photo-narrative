"""Unit tests for the experimental eval LLM layer (Qwen3.7).

Covers the protocol requirements that live entirely in Layer 1:
  * region config validity + manual-only selection (no auto cross-region)
  * placeholder-address detection and missing-key handling
  * the raw key never appears in repr / errors / masked hints
  * region lock blocks mixing a frozen region/model mid-run
  * the fixed model snapshot is never silently swapped to an alias
  * request shape: temperature 0, fixed seed, thinking disabled, json_object
  * a fake-API end-to-end flow: query-gen -> neighbour audit -> repair ->
    Story review, all vision + structured JSON
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_llm import (  # noqa: E402
    EVAL_LLM_DEFAULT_SEED,
    EvalConfigError,
    EvalLLMClient,
    EvalLLMError,
    EvalModelUnavailableError,
    config_fingerprint,
    encode_image,
    freeze_region_lock,
    mask_secret,
    read_region_lock,
    release_region_lock,
    require_region_lock,
)

PINNED_MODEL = "qwen3.7-plus-2026-05-26"
BEIJING_KEY = "sk-beijing-secret-1234567890abcdef"
FRANKFURT_KEY = "sk-frankfurt-secret-9876543210fedcba"
BEIJING_URL = "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
FRANKFURT_URL = "https://ws-test.eu-central-1.maas.aliyuncs.com/compatible-mode/v1"

EVAL_ENV_KEYS = [
    "SMART_PHOTO_EVAL_ACTIVE_REGION",
    "SMART_PHOTO_EVAL_MODEL",
    "SMART_PHOTO_EVAL_TIMEOUT",
    "SMART_PHOTO_EVAL_BEIJING_BASE_URL",
    "SMART_PHOTO_EVAL_BEIJING_API_KEY",
    "SMART_PHOTO_EVAL_FRANKFURT_BASE_URL",
    "SMART_PHOTO_EVAL_FRANKFURT_API_KEY",
]

EVAL_ENV = {
    "SMART_PHOTO_EVAL_ACTIVE_REGION": "beijing",
    "SMART_PHOTO_EVAL_MODEL": PINNED_MODEL,
    "SMART_PHOTO_EVAL_BEIJING_BASE_URL": BEIJING_URL,
    "SMART_PHOTO_EVAL_BEIJING_API_KEY": BEIJING_KEY,
    "SMART_PHOTO_EVAL_FRANKFURT_BASE_URL": FRANKFURT_URL,
    "SMART_PHOTO_EVAL_FRANKFURT_API_KEY": FRANKFURT_KEY,
}


@contextmanager
def eval_env(**overrides):
    """Control the eval env for one block, clearing any leaked real .env values."""
    base = dict(EVAL_ENV)
    base.update(overrides)
    for k in [k for k, v in base.items() if v is None]:
        base.pop(k)
    saved = {k: os.environ.get(k) for k in EVAL_ENV_KEYS}
    try:
        for k in EVAL_ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in base.items():
            os.environ[k] = v
        yield
    finally:
        for k in EVAL_ENV_KEYS:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


# ----------------------------- Fake openai module -----------------------------


class FakeApiError(Exception):
    def __init__(self, message, *, status_code=None, code=None, body=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.body = body
        self.type = "api_error"


class _Resp:
    def __init__(self, content, *, finish_reason="stop", model=PINNED_MODEL):
        msg = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=msg, finish_reason=finish_reason)
        self.choices = [choice]
        self.model = model
        self.usage = types.SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )


class FakeOpenAIModule:
    """Stand-in for the ``openai`` package used via ``from openai import OpenAI``."""

    def __init__(self):
        self.calls: list[dict] = []
        self.construct_args: list[dict] = []
        self._queue: list = []

    def OpenAI(self, **kwargs):
        self.construct_args.append(kwargs)
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                if not outer._queue:
                    return _Resp("{}")
                factory = outer._queue.pop(0)
                return factory(kw)

        class _Chat:
            completions = _Completions()

        client = types.SimpleNamespace(chat=_Chat())
        return client

    def queue_response(self, content, *, model=PINNED_MODEL, finish_reason="stop"):
        self._queue.append(lambda kw: _Resp(content, finish_reason=finish_reason, model=model))

    def queue_raise(self, exc):
        def _raise(kw):
            raise exc
        self._queue.append(_raise)


def _image_block_count(call_kwargs: dict) -> int:
    content = call_kwargs["messages"][1]["content"]
    if isinstance(content, str):
        return 0
    return sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image_url")


# ----------------------------- Tests -----------------------------


class EvalRegionConfigTests(unittest.TestCase):
    def test_default_region_is_beijing(self):
        with eval_env(SMART_PHOTO_EVAL_ACTIVE_REGION=None):
            client = EvalLLMClient()
        self.assertEqual(client.region, "beijing")
        self.assertEqual(client.model, PINNED_MODEL)

    def test_unknown_region_rejected(self):
        with eval_env():
            with self.assertRaises(EvalConfigError):
                EvalLLMClient(region="tokyo")

    def test_placeholder_base_url_not_configured(self):
        # base_url unset -> client falls back to the region template, which
        # still contains the {WorkspaceId} placeholder.
        with eval_env(SMART_PHOTO_EVAL_BEIJING_BASE_URL=None):
            client = EvalLLMClient()
            self.assertTrue(client.base_url_has_placeholder())
            self.assertFalse(client.is_configured())
            self.assertIn("placeholder", client.get_config_error())

    def test_missing_key_not_configured(self):
        with eval_env(SMART_PHOTO_EVAL_BEIJING_API_KEY=None):
            client = EvalLLMClient()
            self.assertFalse(client.is_configured())
            self.assertIn("SMART_PHOTO_EVAL_BEIJING_API_KEY", client.get_config_error())

    def test_region_reads_only_its_own_pair_no_cross_region(self):
        # Frankfurt creds unset, Beijing creds present. A Frankfurt client must
        # NOT borrow Beijing's endpoint/key (no automatic cross-region).
        with eval_env(
            SMART_PHOTO_EVAL_FRANKFURT_BASE_URL=None,
            SMART_PHOTO_EVAL_FRANKFURT_API_KEY=None,
        ):
            beijing = EvalLLMClient(region="beijing")
            frankfurt = EvalLLMClient(region="frankfurt")
            self.assertTrue(beijing.is_configured())
            self.assertFalse(frankfurt.is_configured())
            self.assertTrue(frankfurt.base_url_has_placeholder())


class EvalSecretDisciplineTests(unittest.TestCase):
    def test_key_never_leaked_in_repr_or_mask(self):
        with eval_env():
            client = EvalLLMClient()
            rep = repr(client)
            hint = client.key_hint()
            self.assertNotIn(BEIJING_KEY, rep)
            self.assertNotIn(BEIJING_KEY, hint)
            self.assertIn("...", hint)  # masked, not the raw value

    def test_mask_secret_collapses_short_values(self):
        self.assertEqual(mask_secret(None), "<unset>")
        self.assertEqual(mask_secret("sk"), "***")
        self.assertIn("...", mask_secret("sk-abcdefgh1234567890"))

    def test_config_error_message_contains_no_key(self):
        with eval_env(SMART_PHOTO_EVAL_BEIJING_API_KEY=None):
            client = EvalLLMClient()
            self.assertNotIn(BEIJING_KEY, client.get_config_error() or "")


class EvalRegionLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock_path = Path(self.tmp.name) / "lock.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_freeze_then_mismatched_region_refused(self):
        freeze_region_lock("beijing", PINNED_MODEL, BEIJING_URL, path=self.lock_path)
        record = read_region_lock(self.lock_path)
        self.assertEqual(record["region"], "beijing")
        self.assertNotIn(BEIJING_KEY, json.dumps(record))  # no secret in lock file
        # same triple is allowed
        require_region_lock("beijing", PINNED_MODEL, BEIJING_URL, path=self.lock_path)
        # different region refused
        with self.assertRaises(EvalConfigError):
            require_region_lock(
                "frankfurt", PINNED_MODEL, FRANKFURT_URL, path=self.lock_path
            )
        # different model refused
        with self.assertRaises(EvalConfigError):
            require_region_lock("beijing", "qwen-other", BEIJING_URL, path=self.lock_path)

    def test_release_allows_switching(self):
        freeze_region_lock("beijing", PINNED_MODEL, BEIJING_URL, path=self.lock_path)
        self.assertTrue(release_region_lock(self.lock_path))
        # after unlock, frankfurt is allowed again
        require_region_lock(
            "frankfurt", PINNED_MODEL, FRANKFURT_URL, path=self.lock_path
        )
        self.assertFalse(release_region_lock(self.lock_path))  # already gone

    def test_check_lock_blocks_mixed_client(self):
        freeze_region_lock("beijing", PINNED_MODEL, BEIJING_URL, path=self.lock_path)
        with eval_env():
            with self.assertRaises(EvalConfigError):
                EvalLLMClient(
                    region="frankfurt", check_lock=True, lock_path=self.lock_path
                )

    def test_fingerprint_is_deterministic(self):
        a = config_fingerprint("beijing", PINNED_MODEL, BEIJING_URL)
        b = config_fingerprint("beijing", PINNED_MODEL, BEIJING_URL)
        self.assertEqual(a, b)
        self.assertNotEqual(a, config_fingerprint("frankfurt", PINNED_MODEL, FRANKFURT_URL))


class EvalCallShapeTests(unittest.TestCase):
    def test_complete_uses_temp0_seed_no_thinking_and_json_object(self):
        fake = FakeOpenAIModule()
        fake.queue_response('{"chosen_index": 1}')
        with eval_env(), patch.dict(sys.modules, {"openai": fake}):
            client = EvalLLMClient()
            resp = client.complete("sys", "usr", seed=123, max_tokens=64)
        self.assertEqual(resp.content, '{"chosen_index": 1}')
        kw = fake.calls[0]
        self.assertEqual(kw["model"], PINNED_MODEL)
        self.assertEqual(kw["temperature"], 0.0)
        self.assertEqual(kw["seed"], 123)
        self.assertEqual(kw["max_tokens"], 64)
        self.assertEqual(kw["response_format"], {"type": "json_object"})
        self.assertEqual(kw["extra_body"], {"enable_thinking": False})

    def test_default_seed_is_fixed_when_omitted(self):
        fake = FakeOpenAIModule()
        fake.queue_response("{}")
        with eval_env(), patch.dict(sys.modules, {"openai": fake}):
            EvalLLMClient().complete("sys", "usr")
        self.assertEqual(fake.calls[0]["seed"], EVAL_LLM_DEFAULT_SEED)

    def test_complete_json_parses_object(self):
        fake = FakeOpenAIModule()
        fake.queue_response('{"chosen_index": 4, "reason": "match"}')
        with eval_env(), patch.dict(sys.modules, {"openai": fake}):
            parsed = EvalLLMClient().complete_json("sys", "usr", seed=1)
        self.assertEqual(parsed["chosen_index"], 4)

    def test_model_unavailable_not_silently_swapped(self):
        fake = FakeOpenAIModule()
        fake.queue_raise(
            FakeApiError("model not exist", status_code=400, code="model_not_found")
        )
        with eval_env(), patch.dict(sys.modules, {"openai": fake}):
            client = EvalLLMClient()
            with self.assertRaises(EvalModelUnavailableError):
                client.complete("sys", "usr", seed=1)
        # exactly one attempt, and the pinned model name was never rewritten
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["model"], PINNED_MODEL)
        self.assertEqual(client.model, PINNED_MODEL)
        self.assertEqual(client.last_error_type, "model_unavailable")

    def test_timeout_and_api_errors_classified(self):
        with eval_env():
            client = EvalLLMClient()
            for message, expected in (("Request timed out", "timeout"),
                                      ("rate limited", "api_error")):
                fake = FakeOpenAIModule()
                fake.queue_raise(FakeApiError(message, status_code=429))
                with patch.dict(sys.modules, {"openai": fake}):
                    with self.assertRaises(EvalLLMError):
                        client.complete("sys", "usr", seed=1)
                self.assertEqual(client.last_error_type, expected)


class EvalVisionFlowTests(unittest.TestCase):
    """Fake-API end-to-end: query-gen -> neighbour audit -> repair -> review."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.img_dir = Path(self.tmp.name)
        # one target image for query generation
        self.target = self.img_dir / "target.jpg"
        self.target.write_bytes(b"\xff\xd8\xff\xe0target")
        # 5-image audit panel (target at index 4) and a 12-image review set
        self.panel = [self.img_dir / f"p{i}.jpg" for i in range(5)]
        for p in self.panel:
            p.write_bytes(b"\xff\xd8panel")
        self.review_set = [self.img_dir / f"r{i}.jpg" for i in range(12)]
        for p in self.review_set:
            p.write_bytes(b"\xff\xd8review")

    def tearDown(self):
        self.tmp.cleanup()

    def test_query_generation_audit_and_review_full_flow(self):
        fake = FakeOpenAIModule()
        fake.queue_response('{"query": "a red bicycle leaning on a stone wall"}')
        fake.queue_response('{"chosen_index": 4, "reason": "match"}')
        fake.queue_response(
            '{"coherence": 4, "evidence": 5, "first_person": 4, '
            '"imaginative": 2, "unsupported_claims": 0}'
        )
        with eval_env(), patch.dict(sys.modules, {"openai": fake}):
            client = EvalLLMClient()
            query = client.complete_json("gen-sys", "gen-usr", images=[self.target], seed=1)
            audit = client.complete_json("audit-sys", "audit-usr", images=self.panel, seed=2)
            review = client.complete_json("rev-sys", "rev-usr", images=self.review_set, seed=3)

        self.assertEqual(query["query"], "a red bicycle leaning on a stone wall")
        self.assertEqual(audit["chosen_index"], 4)  # target index
        self.assertEqual(review["coherence"], 4)
        # vision: the right number of image blocks per call
        self.assertEqual(_image_block_count(fake.calls[0]), 1)
        self.assertEqual(_image_block_count(fake.calls[1]), 5)
        self.assertEqual(_image_block_count(fake.calls[2]), 12)

    def test_repair_after_failed_audit(self):
        # First audit picks the wrong index; a second call (one repair) is
        # allowed and returns the target. The client faithfully sequences them.
        fake = FakeOpenAIModule()
        fake.queue_response('{"chosen_index": 2}')   # wrong (target is 4)
        fake.queue_response('{"chosen_index": 4}')   # repair: correct
        target_index = 4
        with eval_env(), patch.dict(sys.modules, {"openai": fake}):
            client = EvalLLMClient()
            first = client.complete_json("s", "u", images=self.panel, seed=1)
            chosen = first["chosen_index"]
            if chosen != target_index:  # one repair permitted by the protocol
                second = client.complete_json("s", "u", images=self.panel, seed=2)
                chosen = second["chosen_index"]
        self.assertEqual(chosen, target_index)
        self.assertEqual(len(fake.calls), 2)

    def test_encode_image_local_and_url_passthrough(self):
        data_url = encode_image(self.target)
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(encode_image("https://x/y.jpg"), "https://x/y.jpg")
        with self.assertRaises(EvalConfigError):
            encode_image(self.img_dir / "missing.jpg")

    def test_encode_image_downscales_large_image(self):
        # A 2000x2000 image must be downscaled to <= EVAL_IMAGE_MAX_DIM.
        import base64
        import io
        from PIL import Image

        big = self.img_dir / "big.jpg"
        Image.new("RGB", (2000, 2000), (255, 0, 0)).save(big, format="JPEG")
        data_url = encode_image(big)
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        raw_b64 = data_url.split(",", 1)[1]
        decoded = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
        self.assertLessEqual(max(decoded.size), 1024)


class EnvExampleTests(unittest.TestCase):
    def test_env_example_uses_placeholders_only(self):
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        for line in example.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "_API_KEY=" in stripped:
                _, _, value = stripped.partition("=")
                value = value.strip().strip('"').strip("'")
                self.assertTrue(
                    value.startswith("sk-replace"),
                    f".env.example API key is not a placeholder: {stripped}",
                )
        # both region pairs are documented
        self.assertIn("SMART_PHOTO_EVAL_BEIJING_API_KEY", example)
        self.assertIn("SMART_PHOTO_EVAL_FRANKFURT_API_KEY", example)


if __name__ == "__main__":
    unittest.main()
