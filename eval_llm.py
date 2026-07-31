"""Experimental evaluation LLM client (Qwen3.7) for the dissertation experiments.

This is a SECOND, independent remote LLM used only by the new retrieval
experiment (R0--R7) and the single-event Story case study. It never replaces the
production Story backend (``qwen3.5-27b``); ``story_agent`` is untouched.

Contract enforced here, all derived from the experiment protocol:

* **Region-paired, manual only.** An active region (``beijing`` or
  ``frankfurt``) reads ONLY its own ``*_BASE_URL`` / ``*_API_KEY`` env vars.
  There is no automatic cross-region failover; the other region's env is never
  consulted for that client.
* **Region lock.** Once the formal experiment starts the operator freezes the
  region+model (``freeze_region_lock``). Any later client constructed in a
  different region or with a different model is refused rather than silently
  mixed into the run.
* **Fixed model snapshot.** The model is pinned. If that exact snapshot is
  unavailable in the active region the call raises
  :class:`EvalModelUnavailableError`; the client never rewrites the model name
  to a rolling alias or a different model.
* **Deterministic, non-thinking structured output.** ``temperature`` 0, a fixed
  ``seed``, and ``enable_thinking=False`` so query generation, the neighbour
  audit, and the Story review are reproducible run-to-run.
* **Secret discipline.** The api key lives only in the git-ignored ``.env``.
  This module never writes the raw key into logs, errors, ``repr``, or result
  objects; only :meth:`EvalLLMClient.key_hint` (a masked fragment) is exposed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from config import (
    DATA_DIR,
    EVAL_LLM_ACTIVE_REGION_ENV,
    EVAL_LLM_DEFAULT_MODEL,
    EVAL_LLM_DEFAULT_REGION,
    EVAL_LLM_DEFAULT_TIMEOUT,
    EVAL_LLM_ENDPOINT_TEMPLATES,
    EVAL_LLM_MODEL_ENV,
    EVAL_LLM_REGION_ENV,
    EVAL_LLM_REGIONS,
    EVAL_LLM_TEMPERATURE,
    EVAL_LLM_THINKING_ENABLED,
    EVAL_LLM_TIMEOUT_ENV,
    EVAL_REGION_LOCK_PATH,
)

# Fixed project seed so even a caller that forgets to pass one stays
# deterministic (the protocol requires a fixed seed for every Qwen3.7 call).
EVAL_LLM_DEFAULT_SEED = 20260727

# Vision requests are downscaled before base64 so multi-image calls (the 5-image
# query audit, the 12-image story review, the 5-image speed test) stay under the
# endpoint's request-size limit. Original files are never modified; the freeze
# SHA-256 hashes the originals, this resize is transport-only.
EVAL_IMAGE_MAX_DIM = 1024
EVAL_IMAGE_JPEG_QUALITY = 85

ImageSource = Union[str, Path]

# Substrings (case-insensitive) in an API error message/body that indicate the
# requested model snapshot is unavailable in the active region. Hitting any of
# these stops the run instead of silently swapping models.
_MODEL_UNAVAILABLE_MARKERS = (
    "model not exist",
    "model not found",
    "model does not exist",
    "algorithm.serviceunavailable",
    "algorithm.modelunavailable",
    "no such model",
    "model_unavailable",
    "model is not available",
    "not supported in this region",
    "snapshot",
)


class EvalLLMError(RuntimeError):
    """Base error for eval-LLM configuration or call failures."""


class EvalConfigError(EvalLLMError):
    """Region / key / endpoint misconfiguration. Surfaced, never fallen back."""


class EvalModelUnavailableError(EvalLLMError):
    """The fixed model snapshot is unavailable in the active region.

    Raised instead of silently swapping to a rolling alias or another model.
    The operator must stop and re-freeze a region where the snapshot exists.
    """


@dataclass
class EvalLLMResponse:
    """Return value of :meth:`EvalLLMClient.complete`.

    ``model_echoed`` is whatever the endpoint echoed back as the served model;
    it is recorded so a silent alias swap can be detected downstream. It never
    contains a secret.
    """

    content: str
    finish_reason: Optional[str]
    model_echoed: Optional[str]
    usage: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    request_seed: Optional[int] = None


def mask_secret(value: Optional[str], keep: int = 4) -> str:
    """Return a non-reversible hint of a secret for diagnostics/logs.

    Short or missing values collapse to a fixed token so the mask itself leaks
    no length information about a real key.
    """
    if not value:
        return "<unset>"
    if len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}...{value[-keep:]}"


def config_fingerprint(region: str, model: str, base_url: str) -> str:
    """SHA-256 of the non-secret config triple, for the region lock record."""
    payload = f"{region}|{model}|{base_url}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ============================ Region lock ============================


def _read_lock(path: Path = EVAL_REGION_LOCK_PATH) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def freeze_region_lock(
    region: str,
    model: str,
    base_url: str,
    *,
    path: Path = EVAL_REGION_LOCK_PATH,
) -> dict:
    """Freeze the active region+model for the formal experiment.

    Writes a small JSON record (region, model, base_url fingerprint -- no
    secrets). After this, :func:`require_region_lock` refuses any client in a
    different region or with a different model.
    """
    if region not in EVAL_LLM_REGIONS:
        raise EvalConfigError(f"Unknown region {region!r}; allowed {EVAL_LLM_REGIONS}")
    record = {
        "region": region,
        "model": model,
        "base_url_fingerprint": config_fingerprint(region, model, base_url),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def read_region_lock(path: Path = EVAL_REGION_LOCK_PATH) -> Optional[dict]:
    """Return the frozen lock record, or ``None`` if not frozen."""
    return _read_lock(path)


def require_region_lock(
    region: str,
    model: str,
    base_url: str,
    *,
    path: Path = EVAL_REGION_LOCK_PATH,
) -> None:
    """Raise :class:`EvalConfigError` if the run would mix a frozen region/model.

    No-op when no lock exists (the experiment has not started yet). When a lock
    exists, the active region, model, and base_url fingerprint must all match.
    """
    record = _read_lock(path)
    if record is None:
        return
    locked_region = record.get("region")
    locked_model = record.get("model")
    locked_fp = record.get("base_url_fingerprint")
    current_fp = config_fingerprint(region, model, base_url)
    if region != locked_region or model != locked_model or current_fp != locked_fp:
        raise EvalConfigError(
            "Region/model lock mismatch: the formal experiment is frozen to "
            f"region={locked_region!r}, model={locked_model!r}. Refusing to mix "
            f"region={region!r}, model={model!r}. Release the lock explicitly to "
            "switch."
        )


def release_region_lock(path: Path = EVAL_REGION_LOCK_PATH) -> bool:
    """Remove the region lock (explicit unlock). Returns whether a lock existed."""
    if path.is_file():
        path.unlink()
        return True
    return False


# ============================ Image encoding ============================


def encode_image(image: ImageSource) -> str:
    """Return an OpenAI-compatible image reference for one local file or URL.

    Local images are downscaled to at most ``EVAL_IMAGE_MAX_DIM`` on the long
    edge and re-encoded as JPEG (quality ``EVAL_IMAGE_JPEG_QUALITY``) before
    base64, so multi-image requests stay under the endpoint body-size limit.
    The original file is never modified. URLs / data: URLs pass through. If PIL
    cannot read the file it falls back to the raw bytes.
    """
    text = str(image)
    if text.startswith(("http://", "https://", "data:")):
        return text
    path = Path(text)
    if not path.is_file():
        raise EvalConfigError(f"Image file not found: {path}")
    try:
        import io
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            if EVAL_IMAGE_MAX_DIM and max(im.size) > EVAL_IMAGE_MAX_DIM:
                im.thumbnail((EVAL_IMAGE_MAX_DIM, EVAL_IMAGE_MAX_DIM))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=EVAL_IMAGE_JPEG_QUALITY)
            data = buf.getvalue()
    except Exception:
        # PIL missing or file not a valid image -> send raw bytes.
        data = path.read_bytes()
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}"


def _build_user_content(
    user_prompt: str, images: Optional[Sequence[ImageSource]]
) -> Any:
    """OpenAI chat ``content`` for the user turn: text plus optional images."""
    if not images:
        return user_prompt
    blocks: list[dict] = [{"type": "text", "text": user_prompt}]
    for image in images:
        blocks.append({"type": "image_url", "image_url": {"url": encode_image(image)}})
    return blocks


# ============================ Client ============================


class EvalLLMClient:
    """Region-paired, vision-capable Qwen3.7 client for the eval experiments."""

    def __init__(
        self,
        *,
        region: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        check_lock: bool = False,
        lock_path: Optional[Path] = None,
    ) -> None:
        self.region = (
            region or os.getenv(EVAL_LLM_ACTIVE_REGION_ENV, EVAL_LLM_DEFAULT_REGION)
        ).strip().lower()
        if self.region not in EVAL_LLM_REGIONS:
            raise EvalConfigError(
                f"Unknown eval region {self.region!r}; allowed {EVAL_LLM_REGIONS}"
            )
        self.model = model or os.getenv(EVAL_LLM_MODEL_ENV, EVAL_LLM_DEFAULT_MODEL)
        self.timeout = float(
            timeout
            if timeout is not None
            else os.getenv(EVAL_LLM_TIMEOUT_ENV, EVAL_LLM_DEFAULT_TIMEOUT)
        )
        pair = EVAL_LLM_REGION_ENV[self.region]
        # Region-paired env-var names: this client ONLY ever reads its own pair.
        self._base_url_env = pair["base_url"]
        self._api_key_env = pair["api_key"]
        # Explicit overrides are intended for tests; production reads the env.
        self._explicit_base_url = base_url
        self._explicit_api_key = api_key
        self._lock_path = lock_path or EVAL_REGION_LOCK_PATH
        self.last_error_type: Optional[str] = None
        if check_lock:
            require_region_lock(
                self.region, self.model, self.base_url, path=self._lock_path
            )

    # ---- region-paired resolution (call time, never cached at import) ----

    @property
    def base_url(self) -> str:
        if self._explicit_base_url is not None:
            return self._explicit_base_url.strip()
        value = os.getenv(self._base_url_env, "").strip()
        if value:
            return value
        # Fall back to the region template so a missing base_url shows up as the
        # {WorkspaceId} placeholder rather than an empty string.
        return EVAL_LLM_ENDPOINT_TEMPLATES[self.region]

    def _api_key(self) -> Optional[str]:
        return self._explicit_api_key or os.getenv(self._api_key_env)

    # ---- configuration diagnostics (no secrets) ----

    def key_hint(self) -> str:
        """Masked hint of the active-region key, for logs/speed-test banners."""
        return mask_secret(self._api_key())

    def base_url_has_placeholder(self) -> bool:
        return "{WorkspaceId}" in self.base_url

    def is_configured(self) -> bool:
        return self.get_config_error() is None

    def get_config_error(self) -> Optional[str]:
        if self.base_url_has_placeholder():
            return (
                f"{self._base_url_env} still contains the {{WorkspaceId}} "
                "placeholder; set it to the region endpoint."
            )
        if not self._api_key():
            return (
                f"{self._api_key_env} is not set; the {self.region} region key "
                "must be paired with its own base_url."
            )
        return None

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"EvalLLMClient(region={self.region!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, key={self.key_hint()!r})"
        )

    # ---- single call ----

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: Optional[Sequence[ImageSource]] = None,
        temperature: float = EVAL_LLM_TEMPERATURE,
        seed: Optional[int] = EVAL_LLM_DEFAULT_SEED,
        max_tokens: Optional[int] = None,
        structured: bool = True,
        response_format: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> EvalLLMResponse:
        """One deterministic, non-thinking completion. Returns content + metadata.

        ``structured=True`` (default) requests ``json_object`` output. Vision
        inputs go in as OpenAI image-url blocks. Raises
        :class:`EvalConfigError` if misconfigured and
        :class:`EvalModelUnavailableError` if the pinned model snapshot is
        unavailable in the region (never silently swapped).
        """
        error = self.get_config_error()
        if error is not None:
            raise EvalConfigError(error)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EvalLLMError(
                f"openai SDK is not installed ({type(exc).__name__}: {exc}); "
                "run: pip install -U openai"
            ) from exc

        client = OpenAI(
            api_key=self._api_key(), base_url=self.base_url, timeout=self.timeout
        )
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_user_content(user_prompt, images)},
            ],
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = int(max_tokens)
        if seed is not None:
            request_kwargs["seed"] = int(seed)
        if response_format is not None:
            request_kwargs["response_format"] = dict(response_format)
        elif structured:
            request_kwargs["response_format"] = {"type": "json_object"}
        merged_extra = {"enable_thinking": bool(EVAL_LLM_THINKING_ENABLED)}
        if extra_body:
            merged_extra.update(dict(extra_body))
        request_kwargs["extra_body"] = merged_extra

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:  # noqa: BLE001 - classify then re-raise
            kind = self._classify_error(exc)
            self.last_error_type = kind
            message = self._safe_error_message(exc)
            if kind == "model_unavailable":
                raise EvalModelUnavailableError(
                    f"Fixed model snapshot {self.model!r} unavailable in region "
                    f"{self.region!r}: {message}. Stopping; not falling back to "
                    "another model or rolling alias."
                ) from exc
            raise EvalLLMError(
                f"Eval LLM call failed ({kind}): {message}"
            ) from exc

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        usage_dict = (
            {k: v for k, v in vars(usage).items() if not k.startswith("_")}
            if usage is not None and hasattr(usage, "__dict__")
            else {}
        )
        self.last_error_type = None
        return EvalLLMResponse(
            content=getattr(choice.message, "content", None) or "",
            finish_reason=getattr(choice, "finish_reason", None),
            model_echoed=getattr(response, "model", None),
            usage=usage_dict,
            elapsed_seconds=0.0,
            request_seed=request_kwargs.get("seed"),
        )

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: Optional[Sequence[ImageSource]] = None,
        seed: Optional[int] = EVAL_LLM_DEFAULT_SEED,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Convenience wrapper: ``complete(structured=True)`` then JSON-parse."""
        result = self.complete(
            system_prompt,
            user_prompt,
            images=images,
            seed=seed,
            max_tokens=max_tokens,
            structured=True,
        )
        try:
            parsed = json.loads(result.content)
        except json.JSONDecodeError as exc:
            raise EvalLLMError(
                f"Eval LLM did not return valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(parsed, dict):
            raise EvalLLMError(
                f"Eval LLM JSON is not an object: {type(parsed).__name__}"
            )
        return parsed

    # ---- error classification (no secrets ever included) ----

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        """Best-effort message from an SDK exception, stripped of any secret."""
        text = " ".join(
            str(getattr(exc, attr, "") or "")
            for attr in ("message", "body", "code", "type")
        ) or str(exc)
        return text[:500]

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        status = getattr(exc, "status_code", None)
        code = str(getattr(exc, "code", "") or "").lower()
        message = EvalLLMClient._safe_error_message(exc).lower()
        if status == 404:
            return "model_unavailable"
        if code in {"model_not_found", "modelfound"}:
            return "model_unavailable"
        if any(marker in message for marker in _MODEL_UNAVAILABLE_MARKERS):
            return "model_unavailable"
        if isinstance(exc, TimeoutError) or "timeout" in message or "timed out" in message:
            return "timeout"
        return "api_error"


__all__ = [
    "DATA_DIR",
    "EVAL_LLM_DEFAULT_SEED",
    "EvalConfigError",
    "EvalLLMClient",
    "EvalLLMError",
    "EvalLLMResponse",
    "EvalModelUnavailableError",
    "config_fingerprint",
    "encode_image",
    "freeze_region_lock",
    "mask_secret",
    "read_region_lock",
    "release_region_lock",
    "require_region_lock",
]
