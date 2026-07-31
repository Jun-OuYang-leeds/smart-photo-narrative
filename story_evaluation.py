"""Frozen and reproducible N0--N3 Story experiment implementation.

This module deliberately stays outside the production Story database.  It
defines the four ablation prompts, deterministic generation settings,
checkpoint records, automatic metrics, and private review artefacts used for
RQ2.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from story_agent import (
    ContextAggregator,
    GeneratedStory,
    GroundingValidator,
    PromptBuilder,
    StoryContext,
    StoryGenerator,
    StoryParagraph,
    UncertainObservation,
)


EXPERIMENT_MODEL = "llama3:latest"
EXPECTED_MODEL_DIGEST = "365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1"
EXPERIMENT_TEMPERATURE = 0.25
EXPERIMENT_NUM_PREDICT = 1200
EXPERIMENT_PROTOCOL = "story-ablation-n0-n3-v1"


@dataclass(frozen=True)
class StoryVariantSpec:
    variant_id: str
    label: str
    prompt_version: str
    evidence_plan: str
    output_format: str
    validation: str


STORY_VARIANTS = (
    StoryVariantSpec(
        "N0", "Legacy Story", "legacy-story-n0-v1",
        "Caption + tags + reliable time/location; no citations", "text", "none",
    ),
    StoryVariantSpec(
        "N1", "Grounded Story v2 baseline", "grounded-story-v2-per-photo",
        "raw per-photo multimodal evidence; one cited paragraph per photo", "json", "citation_only",
    ),
    StoryVariantSpec(
        "N2", "Event-grouped Story v3", "grounded-story-v3-event-unvalidated",
        "grouping + duplicate compression + conflict separation", "json", "observe_only",
    ),
    StoryVariantSpec(
        "N3", "Validated Story v3", PromptBuilder.PROMPT_VERSION,
        "the exact same first prompt and draft as N2", "json", "strict_repair_fallback",
    ),
)
VARIANT_BY_ID = {item.variant_id: item for item in STORY_VARIANTS}


@dataclass(frozen=True)
class StoryCase:
    case_id: str
    source_kind: str
    label: str
    photo_ids: tuple[str, ...]
    language: str
    event_id: str | None = None
    size_bucket: str = ""


def canonical_hash(value: Any) -> str:
    """SHA-256 of stable UTF-8 JSON (or a supplied string)."""
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_seed(case_id: str) -> int:
    """A stable positive 31-bit seed shared by all variants of one case."""
    return int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def load_story_cases(path: str | Path) -> list[StoryCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    records = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Story cases must be a list or an object with a cases list")
    cases = [StoryCase(
        case_id=str(item["case_id"]), source_kind=str(item["source_kind"]),
        label=str(item["label"]), photo_ids=tuple(item["photo_ids"]),
        language=str(item["language"]), event_id=item.get("event_id"),
        size_bucket=str(item.get("size_bucket") or ""),
    ) for item in records]
    if len(cases) != 12 or len({item.case_id for item in cases}) != 12:
        raise ValueError("Frozen Story set must contain 12 unique cases")
    if sum(item.language == "zh" for item in cases) != 6 or sum(item.language == "en" for item in cases) != 6:
        raise ValueError("Frozen Story set must contain six Chinese and six English cases")
    if sum(item.source_kind == "single" for item in cases) != 2:
        raise ValueError("Frozen Story set must contain two single-photo cases")
    if sum(item.source_kind == "event" for item in cases) != 8:
        raise ValueError("Frozen Story set must contain eight Event cases")
    if sum(item.source_kind == "date" for item in cases) != 2:
        raise ValueError("Frozen Story set must contain two multi-event Date cases")
    if not any(item.source_kind == "event" and item.label == "2026-07-12" for item in cases):
        raise ValueError("The 2026-07-12 large Event regression case is required")
    return cases


def _reliable_time(item: Any) -> Optional[str]:
    return item.timestamp if item.timestamp and float(item.timestamp_confidence or 0.0) >= 0.8 else None


def _n0_projection(context: StoryContext) -> dict[str, Any]:
    return {
        "label": context.label,
        "photos": [{
            "caption": item.caption or None,
            "tags": list(item.tags),
            "reliable_time": _reliable_time(item),
            "location": item.location or None,
        } for item in context.evidence],
    }


def _n1_projection(context: StoryContext) -> dict[str, Any]:
    return {
        "label": context.label,
        "photos": [{
            "evidence_id": item.evidence_id,
            "reliable_time": _reliable_time(item),
            "location": item.location or None,
            "blip_caption_model_observation": item.caption or None,
            "qwen_scene_graph_model_observations": list(item.scene_graph_triples),
            "clip_tags_model_observations": list(item.tags),
        } for item in context.evidence],
    }


def context_record(context: StoryContext) -> dict[str, Any]:
    """Complete private context projection used to detect data drift."""
    return {
        "label": context.label,
        "source_kind": context.source_kind,
        "event_id": context.event_id,
        "verified_context": context.verified_context,
        "evidence": [{
            "evidence_id": item.evidence_id,
            "photo_id": item.photo_id,
            "relative_path": item.relative_path,
            "timestamp": item.timestamp,
            "timestamp_source": item.timestamp_source,
            "timestamp_confidence": item.timestamp_confidence,
            "location": item.location,
            "caption": item.caption,
            "scene_graph_triples": list(item.scene_graph_triples),
            "tags": list(item.tags),
        } for item in context.evidence],
        "groups": [group.to_dict() for group in context.groups],
    }


def build_variant_prompt(
    variant_id: str, context: StoryContext, language: str,
) -> tuple[str, str, bool]:
    """Return system prompt, user prompt, and whether JSON mode is enabled."""
    if variant_id == "N0":
        language_name = "Chinese" if language.startswith("zh") else "English"
        system = f"""You write a warm personal photo diary in {language_name}.
Use first-person prose and connect the supplied photo descriptions into a readable memory.
The source contains only captions, tags, and reliable time/location metadata.
Do not output JSON and do not add evidence citations. Start with a short title on its own line."""
        user = "Write one short diary entry from these photos:\n" + json.dumps(
            _n0_projection(context), ensure_ascii=False, indent=2,
        )
        return system, user, False
    if variant_id == "N1":
        language_name = "Chinese" if language.startswith("zh") else "English"
        system = f"""Write a faithful photo narrative in {language_name} from raw per-photo model observations.
Return JSON only. Write exactly one paragraph for every input photo, in input order.
Each paragraph must cite only that photo's evidence_id. Do not merge or compress photos.
Use neutral observable descriptions; the observations may be fallible.
Return exactly: {{"title":"...","paragraphs":[{{"text":"...","evidence_ids":["P001"]}}]}}"""
        user = "Create the per-photo narrative from this evidence:\n" + json.dumps(
            _n1_projection(context), ensure_ascii=False, indent=2,
        )
        return system, user, True
    if variant_id in {"N2", "N3"}:
        system, user = PromptBuilder.build_story_prompt(context, mode="faithful", language=language)
        return system, user, True
    raise ValueError(f"Unknown Story variant: {variant_id}")


def _split_text_units(text: str) -> list[str]:
    return [value.strip() for value in re.split(r"(?<=[.!?。！？])\s*|\n+", text) if value.strip()]


def _claim_count(text: str) -> int:
    return len(_split_text_units(text))


def _normalise_legacy(content: str, label: str) -> Optional[dict[str, Any]]:
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", content.strip(), flags=re.I)
    if not text:
        return None
    blocks = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
    lines = [value.strip() for value in text.splitlines() if value.strip()]
    first = lines[0] if lines else label
    title_match = re.match(r"^(?:#{1,6}\s*|title\s*:\s*|标题\s*[：:]\s*)(.+)$", first, re.I)
    title = title_match.group(1).strip() if title_match else first[:120]
    if title_match or first.startswith("#"):
        body_text = "\n".join(lines[1:]).strip()
    elif len(blocks) > 1:
        body_text = "\n\n".join(blocks[1:])
    else:
        title, body_text = label, text
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n", body_text) if value.strip()]
    if not paragraphs and body_text:
        paragraphs = [body_text]
    return {"title": title or label, "paragraphs": [{"text": value} for value in paragraphs]}


def _parse_json(content: str) -> tuple[Optional[dict[str, Any]], list[str]]:
    try:
        return GroundingValidator.parse_payload(content), []
    except Exception as exc:
        return None, [f"E_JSON_PARSE: {type(exc).__name__}: {exc}"]


def validate_n1(payload: Optional[dict[str, Any]], context: StoryContext) -> list[str]:
    if payload is None:
        return ["E_JSON_PARSE: no JSON object"]
    errors: list[str] = []
    if not str(payload.get("title") or "").strip():
        errors.append("E_TITLE_MISSING: title is required")
    paragraphs = payload.get("paragraphs")
    if not isinstance(paragraphs, list):
        return errors + ["E_JSON_PARAGRAPHS: paragraphs must be a list"]
    expected = [item.evidence_id for item in context.evidence]
    if len(paragraphs) != len(expected):
        errors.append(f"E_PER_PHOTO_COUNT: expected {len(expected)} paragraphs, got {len(paragraphs)}")
    seen: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict) or not str(paragraph.get("text") or "").strip():
            errors.append(f"E_PARAGRAPH_TEXT: paragraph {index + 1} has no text")
            continue
        citations = paragraph.get("evidence_ids")
        if not isinstance(citations, list) or len(citations) != 1:
            errors.append(f"E_PER_PHOTO_CITATION: paragraph {index + 1} must cite exactly one ID")
            continue
        citation = str(citations[0])
        seen.append(citation)
        if index >= len(expected) or citation != expected[index]:
            errors.append(f"E_PER_PHOTO_ORDER: paragraph {index + 1} cites {citation!r}")
    if seen != expected:
        errors.append("E_EVIDENCE_COVERAGE: each photo must be cited once in input order")
    return list(dict.fromkeys(errors))


def _normalised_payload(payload: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if payload is None:
        return None
    result = {
        "title": str(payload.get("title") or "").strip(),
        "paragraphs": [],
    }
    paragraphs = payload.get("paragraphs")
    if isinstance(paragraphs, list):
        for item in paragraphs:
            if not isinstance(item, Mapping):
                continue
            value: dict[str, Any] = {"text": str(item.get("text") or "").strip()}
            if "evidence_ids" in item:
                ids = item.get("evidence_ids")
                value["evidence_ids"] = [str(item_id) for item_id in ids] if isinstance(ids, list) else []
            if "group_id" in item:
                value["group_id"] = str(item.get("group_id") or "")
            result["paragraphs"].append(value)
    for key in ("uncertain_observations", "unused_evidence_ids", "creative_transitions"):
        if key in payload:
            result[key] = payload.get(key)
    return result


def _payload_text(payload: Optional[Mapping[str, Any]]) -> str:
    if not payload:
        return ""
    values = [str(payload.get("title") or "")]
    values.extend(
        str(item.get("text") or "") for item in payload.get("paragraphs", [])
        if isinstance(item, Mapping)
    )
    return "\n".join(value for value in values if value)


def _duplicate_rate(units: Sequence[str]) -> float:
    duplicate_pairs = 0
    pairs = 0
    for left in range(len(units)):
        for right in range(left + 1, len(units)):
            pairs += 1
            if GroundingValidator._paragraph_similarity(units[left], units[right]) > 0.80:
                duplicate_pairs += 1
    return duplicate_pairs / pairs if pairs else 0.0


def _risk_metrics(text: str) -> tuple[int, float]:
    units = _split_text_units(text)
    risky = sum(
        any(GroundingValidator._contains_term(unit, term) for term in GroundingValidator.SPECULATION_TERMS)
        for unit in units
    )
    return risky, risky / len(units) if units else 0.0


def score_experiment_output(
    variant_id: str,
    payload: Optional[Mapping[str, Any]],
    context: StoryContext,
    *,
    latency_ms: float,
    status: str,
    json_valid: Optional[bool],
    language: Optional[str] = None,
) -> dict[str, Any]:
    """Score a variant without turning unavailable metrics into zero."""
    text = _payload_text(payload)
    units = _split_text_units("\n".join(
        str(item.get("text") or "") for item in (payload or {}).get("paragraphs", [])
        if isinstance(item, Mapping)
    ))
    has_text = bool(text.strip() and units)
    ratio = GroundingValidator._language_ratio(text) if has_text else None
    language_ok = None
    if language is not None:
        language_ok = bool(has_text and (
            ratio >= 0.25 if language.startswith("zh") else ratio <= 0.10
        ))
    risk_count, risk_rate = _risk_metrics(text)
    allowed_ids = {item.evidence_id for item in context.evidence}
    allowed_groups = {group.group_id for group in context.groups}
    paragraphs = list((payload or {}).get("paragraphs", []))
    citations = [
        str(value) for paragraph in paragraphs if isinstance(paragraph, Mapping)
        for value in (paragraph.get("evidence_ids") if isinstance(paragraph.get("evidence_ids"), list) else [])
    ]
    cited_groups = {
        str(paragraph.get("group_id")) for paragraph in paragraphs
        if isinstance(paragraph, Mapping) and paragraph.get("group_id")
    }
    metrics: dict[str, Any] = {
        "json_valid": json_valid,
        "citation_valid_rate": None,
        "evidence_id_coverage": None,
        "evidence_group_coverage": None,
        "evidence_id_accounting_rate": None,
        "language_compliant": language_ok,
        "chinese_character_ratio": ratio,
        "duplicate_text_unit_rate": _duplicate_rate(units) if units else None,
        "risk_term_count": risk_count,
        "risk_term_rate": risk_rate if units else None,
        "claim_count": len(units),
        "unsupported_claim_rate": None,
        "conflict_handling_rate": None,
        "paragraph_compression_rate": None,
        "fallback": None,
        "latency_ms": float(latency_ms),
        "status": status,
    }
    if variant_id in {"N1", "N2", "N3"}:
        metrics["citation_valid_rate"] = (
            sum(value in allowed_ids for value in citations) / len(citations) if citations else 0.0
        )
        metrics["evidence_id_coverage"] = (
            len(set(citations) & allowed_ids) / len(allowed_ids) if allowed_ids else 0.0
        )
    if variant_id in {"N2", "N3"}:
        unused = (payload or {}).get("unused_evidence_ids", [])
        unused_ids = {str(value) for value in unused} if isinstance(unused, list) else set()
        metrics["evidence_group_coverage"] = (
            len(cited_groups & allowed_groups) / len(allowed_groups) if allowed_groups else 0.0
        )
        metrics["evidence_id_accounting_rate"] = (
            len((set(citations) | unused_ids) & allowed_ids) / len(allowed_ids) if allowed_ids else 0.0
        )
        uncertain = (payload or {}).get("uncertain_observations", [])
        uncertain_ids = {
            str(value) for item in uncertain if isinstance(item, Mapping)
            for value in (item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else [])
        } if isinstance(uncertain, list) else set()
        conflict_groups = [group for group in context.groups if group.conflicts]
        metrics["conflict_handling_rate"] = (
            sum(bool(set(group.evidence_ids) & uncertain_ids) for group in conflict_groups) / len(conflict_groups)
            if conflict_groups else 1.0
        )
        metrics["paragraph_compression_rate"] = (
            1.0 - len(paragraphs) / max(1, context.photo_count) if paragraphs else None
        )
    if variant_id == "N3":
        metrics["fallback"] = status == "fallback"
    return metrics


def score_story(story: GeneratedStory, context: StoryContext, *, latency_ms: float) -> dict[str, Any]:
    """Backward-compatible N3 metric adapter used by the three-case acceptance."""
    metrics = score_experiment_output(
        "N3", story.to_dict(), context, latency_ms=latency_ms,
        status=story.status, json_valid=story.status != "error" and bool(story.paragraphs),
        language=story.language,
    )
    text = _payload_text(story.to_dict())
    ratio = GroundingValidator._language_ratio(text)
    metrics["language_compliant"] = ratio >= 0.25 if story.language.startswith("zh") else ratio <= 0.10
    final_errors = GroundingValidator.validate(story.to_dict(), context, story.mode, story.language)
    claims = max(1, int(metrics["claim_count"]))
    unsupported = sum("E_UNSUPPORTED_CLAIM" in value for value in final_errors)
    metrics.update({
        "duplicate_paragraph_rate": metrics["duplicate_text_unit_rate"],
        "unsupported_claim_count": unsupported,
        "unsupported_claim_rate": unsupported / claims if metrics["claim_count"] else 0.0,
        "final_validation_errors": final_errors,
        "prompt_hash": story.prompt_hash,
    })
    return metrics


class AtomicJsonlCheckpoint:
    """Rewrite-through-temp JSONL checkpoint; safe against partial last writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        result = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid checkpoint JSON on line {number}: {exc}") from exc
        keys = [(item.get("case_id"), item.get("variant_id")) for item in result]
        if len(keys) != len(set(keys)):
            raise ValueError("Checkpoint contains duplicate case/variant records")
        return result

    def append(self, record: Mapping[str, Any]) -> None:
        records = self.records()
        key = (record.get("case_id"), record.get("variant_id"))
        if any((item.get("case_id"), item.get("variant_id")) == key for item in records):
            raise ValueError(f"Checkpoint already contains {key}")
        records.append(dict(record))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
        )
        temporary.replace(self.path)


class OllamaExperimentBackend:
    """Small testable wrapper around Ollama's fixed formal-experiment call."""

    def __init__(self, model_name: str = EXPERIMENT_MODEL):
        self.model_name = model_name

    def model_info(self) -> dict[str, Any]:
        import ollama

        listing = ollama.list()
        models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
        for item in models:
            value = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: {})()
            name = str(value.get("model") or value.get("name") or "")
            if name == self.model_name or name.startswith(self.model_name + ":"):
                return {
                    "requested_model": self.model_name,
                    "resolved_model": name,
                    "digest": str(value.get("digest") or ""),
                    "size": value.get("size"),
                    "modified_at": str(value.get("modified_at") or ""),
                }
        raise RuntimeError(f"Model {self.model_name!r} is not available in Ollama")

    def generate(
        self, system_prompt: str, user_prompt: str, *, seed: int,
        json_mode: bool, temperature: float, num_predict: int,
    ) -> str:
        import ollama

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "seed": seed,
            },
        }
        if json_mode:
            kwargs["format"] = "json"
        response = ollama.chat(**kwargs)
        message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response")
        return content


def assert_model_digest(model_info: Mapping[str, Any], expected_digest: str = EXPECTED_MODEL_DIGEST) -> None:
    actual = str(model_info.get("digest") or "")
    if actual != expected_digest:
        raise RuntimeError(
            f"Frozen model digest mismatch: expected {expected_digest}, got {actual or '<missing>'}. "
            "The formal experiment will not switch or pull a model automatically."
        )


class StoryExperimentRunner:
    def __init__(
        self,
        backend: Any,
        output_path: str | Path,
        *,
        context_factory: Optional[Callable[[StoryCase], StoryContext]] = None,
        expected_digest: str = EXPECTED_MODEL_DIGEST,
        progress: Optional[Callable[[Mapping[str, Any], bool], None]] = None,
    ):
        self.backend = backend
        self.checkpoint = AtomicJsonlCheckpoint(output_path)
        self.expected_digest = expected_digest
        self.progress = progress
        if context_factory is None:
            aggregator = ContextAggregator()
            context_factory = lambda case: aggregator.aggregate_by_photo_ids(
                case.photo_ids, label=case.label, event_id=case.event_id,
                source_kind=case.source_kind,
            )
        self.context_factory = context_factory

    @staticmethod
    def _case_hash(case: StoryCase) -> str:
        return canonical_hash(asdict(case))

    @staticmethod
    def _identity_matches(
        record: Mapping[str, Any], *, case_hash: str, prompt_hash: str,
        context_hash: str, model_digest: str, seed: int,
    ) -> bool:
        return all((
            record.get("case_hash") == case_hash,
            record.get("prompt_hash") == prompt_hash,
            record.get("context_hash") == context_hash,
            record.get("model_digest") == model_digest,
            record.get("seed") == seed,
        ))

    def _invoke(
        self, system: str, user: str, *, seed: int, json_mode: bool,
    ) -> tuple[str, float]:
        last_error: Optional[Exception] = None
        for attempt in range(2):
            started = time.perf_counter()
            try:
                content = self.backend.generate(
                    system, user, seed=seed, json_mode=json_mode,
                    temperature=EXPERIMENT_TEMPERATURE, num_predict=EXPERIMENT_NUM_PREDICT,
                )
                return content, (time.perf_counter() - started) * 1000.0
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    break
        raise RuntimeError(f"Generation transport failed twice with the frozen prompt/seed: {last_error}") from last_error

    @staticmethod
    def _attempt(kind: str, content: str, latency_ms: float) -> dict[str, Any]:
        return {
            "kind": kind,
            "content": content,
            "output_hash": canonical_hash(content),
            "latency_ms": float(latency_ms),
        }

    @staticmethod
    def _language_metrics(metrics: dict[str, Any], payload: Optional[Mapping[str, Any]], language: str) -> None:
        text = _payload_text(payload)
        if not text.strip() or not (payload or {}).get("paragraphs"):
            metrics["chinese_character_ratio"] = None
            metrics["language_compliant"] = False
            return
        ratio = GroundingValidator._language_ratio(text)
        metrics["chinese_character_ratio"] = ratio
        metrics["language_compliant"] = ratio >= 0.25 if language.startswith("zh") else ratio <= 0.10

    def _base_record(
        self, case: StoryCase, spec: StoryVariantSpec, *, model_info: Mapping[str, Any],
        seed: int, case_hash: str, prompt_hash: str, context_hash: str,
    ) -> dict[str, Any]:
        return {
            "protocol_version": EXPERIMENT_PROTOCOL,
            "case_id": case.case_id,
            "case_hash": case_hash,
            "variant_id": spec.variant_id,
            "language": case.language,
            "source_kind": case.source_kind,
            "model_name": EXPERIMENT_MODEL,
            "model_digest": str(model_info["digest"]),
            "seed": seed,
            "temperature": EXPERIMENT_TEMPERATURE,
            "num_predict": EXPERIMENT_NUM_PREDICT,
            "prompt_version": spec.prompt_version,
            "prompt_hash": prompt_hash,
            "context_hash": context_hash,
        }

    def _generate_independent(
        self, variant_id: str, context: StoryContext, case: StoryCase,
        system: str, user: str, json_mode: bool, base: dict[str, Any],
    ) -> dict[str, Any]:
        raw, latency = self._invoke(system, user, seed=base["seed"], json_mode=json_mode)
        validation_codes: list[str] = []
        if variant_id == "N0":
            payload = _normalise_legacy(raw, case.label)
            status = "ok" if payload and payload.get("paragraphs") else "invalid"
            if status == "invalid":
                validation_codes.append("E_TEXT_EMPTY: legacy output contains no narrative")
            json_valid: Optional[bool] = None
        else:
            parsed, parse_errors = _parse_json(raw)
            payload = _normalised_payload(parsed)
            validation_codes.extend(parse_errors)
            validation_codes.extend(validate_n1(parsed, context) if parsed is not None else [])
            status = "ok" if not validation_codes else "invalid"
            json_valid = parsed is not None
        metrics = score_experiment_output(
            variant_id, payload, context, latency_ms=latency, status=status,
            json_valid=json_valid, language=case.language,
        )
        self._language_metrics(metrics, payload, case.language)
        return {
            **base,
            "raw_attempts": [self._attempt("generation", raw, latency)],
            "normalized_output": payload,
            "output_hash": canonical_hash(payload) if payload is not None else canonical_hash(raw),
            "status": status,
            "validation_codes": validation_codes,
            "fallback": False,
            "latency_ms": latency,
            "automatic_metrics": metrics,
        }

    def _generate_n2(
        self, context: StoryContext, case: StoryCase, system: str, user: str, base: dict[str, Any],
    ) -> dict[str, Any]:
        raw, latency = self._invoke(system, user, seed=base["seed"], json_mode=True)
        parsed, errors = _parse_json(raw)
        if parsed is not None:
            errors.extend(GroundingValidator.validate(parsed, context, "faithful", case.language))
        payload = _normalised_payload(parsed)
        status = "unvalidated" if parsed is not None else "invalid"
        metrics = score_experiment_output(
            "N2", payload, context, latency_ms=latency, status=status,
            json_valid=parsed is not None, language=case.language,
        )
        self._language_metrics(metrics, payload, case.language)
        return {
            **base,
            "raw_attempts": [self._attempt("shared_n2_n3_draft", raw, latency)],
            "normalized_output": payload,
            "output_hash": canonical_hash(payload) if payload is not None else canonical_hash(raw),
            "status": status,
            "validation_codes": errors,
            "fallback": False,
            "latency_ms": latency,
            "automatic_metrics": metrics,
        }

    def _generate_n3_from_n2(
        self, n2: Mapping[str, Any], context: StoryContext, case: StoryCase,
        system: str, base_user: str, base: dict[str, Any],
    ) -> dict[str, Any]:
        first = n2["raw_attempts"][0]
        raw = str(first["content"])
        draft_latency = float(first["latency_ms"])
        attempts = [dict(first)]
        parsed, errors = _parse_json(raw)
        if parsed is not None:
            errors.extend(GroundingValidator.validate(parsed, context, "faithful", case.language))
        payload: Optional[dict[str, Any]] = None
        status = "ok"
        if parsed is not None and not errors:
            payload = _normalised_payload(parsed)
        else:
            repair_user = base_user + "\n\nThe previous output failed strict validation. Repair it once. Errors:\n- " + "\n- ".join(errors)
            repaired_raw, repair_latency = self._invoke(
                system, repair_user, seed=base["seed"], json_mode=True,
            )
            attempts.append(self._attempt("repair", repaired_raw, repair_latency))
            repaired, repaired_errors = _parse_json(repaired_raw)
            if repaired is not None:
                repaired_errors.extend(GroundingValidator.validate(
                    repaired, context, "faithful", case.language,
                ))
            errors = list(dict.fromkeys([*errors, *repaired_errors]))
            if repaired is not None and not repaired_errors:
                payload = _normalised_payload(repaired)
                status = "repaired"
            else:
                prompt_hash = str(base["prompt_hash"])
                fallback_story = StoryGenerator._fallback(context, case.language, errors, prompt_hash)
                payload = fallback_story.to_dict()
                status = "fallback"
        latency = draft_latency + sum(float(item["latency_ms"]) for item in attempts[1:])
        metrics = score_experiment_output(
            "N3", payload, context, latency_ms=latency, status=status,
            json_valid=True, language=case.language,
        )
        self._language_metrics(metrics, payload, case.language)
        final_validation_codes = GroundingValidator.validate(
            dict(payload), context, "faithful", case.language,
        )
        if final_validation_codes:
            raise RuntimeError(
                f"N3 final artifact failed its hard constraints for {case.case_id}: "
                + "; ".join(final_validation_codes)
            )
        return {
            **base,
            "raw_attempts": attempts,
            "normalized_output": payload,
            "output_hash": canonical_hash(payload),
            "status": status,
            "validation_codes": errors,
            "final_validation_codes": final_validation_codes,
            "fallback": status == "fallback",
            "latency_ms": latency,
            "automatic_metrics": metrics,
            "shared_draft_output_hash": str(first["output_hash"]),
        }

    def run(self, cases: Sequence[StoryCase]) -> list[dict[str, Any]]:
        model_info = self.backend.model_info()
        assert_model_digest(model_info, self.expected_digest)
        existing = {
            (item["case_id"], item["variant_id"]): item for item in self.checkpoint.records()
        }
        for record in existing.values():
            if record.get("model_digest") != model_info["digest"]:
                raise RuntimeError("Existing checkpoint uses a different model digest")
        for case in cases:
            context = self.context_factory(case)
            if not context.evidence or not context.groups:
                raise RuntimeError(f"Frozen case {case.case_id} has no usable evidence/groups")
            context_hash = canonical_hash(context_record(context))
            case_hash = self._case_hash(case)
            seed = deterministic_seed(case.case_id)
            for variant_id in ("N0", "N1", "N2", "N3"):
                spec = VARIANT_BY_ID[variant_id]
                system, user, json_mode = build_variant_prompt(variant_id, context, case.language)
                prompt_hash = canonical_hash(system + "\0" + user)
                base = self._base_record(
                    case, spec, model_info=model_info, seed=seed, case_hash=case_hash,
                    prompt_hash=prompt_hash, context_hash=context_hash,
                )
                key = (case.case_id, variant_id)
                old = existing.get(key)
                if old is not None:
                    if not self._identity_matches(
                        old, case_hash=case_hash, prompt_hash=prompt_hash,
                        context_hash=context_hash, model_digest=str(model_info["digest"]), seed=seed,
                    ):
                        raise RuntimeError(f"Checkpoint identity drift for {case.case_id}/{variant_id}; refusing rerun")
                    if self.progress is not None:
                        self.progress(old, True)
                    continue
                if variant_id in {"N0", "N1"}:
                    record = self._generate_independent(
                        variant_id, context, case, system, user, json_mode, base,
                    )
                elif variant_id == "N2":
                    record = self._generate_n2(context, case, system, user, base)
                else:
                    n2 = existing.get((case.case_id, "N2"))
                    if n2 is None:
                        raise RuntimeError("N3 cannot run before its paired N2 draft is checkpointed")
                    if n2["prompt_hash"] != prompt_hash or n2["seed"] != seed:
                        raise RuntimeError("N2/N3 first prompt or seed differs; paired comparison is invalid")
                    record = self._generate_n3_from_n2(n2, context, case, system, user, base)
                self.checkpoint.append(record)
                existing[key] = record
                if self.progress is not None:
                    self.progress(record, False)
        records = self.checkpoint.records()
        expected_keys = {(case.case_id, variant.variant_id) for case in cases for variant in STORY_VARIANTS}
        actual_keys = {(item["case_id"], item["variant_id"]) for item in records}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise RuntimeError(f"Formal checkpoint is not exactly 12x4; missing={missing}, extra={extra}")
        return records


def blind_n0_n3_order(case_id: str) -> tuple[str, str]:
    """Return stable A/B order without exposing which side is N0 or N3."""
    first = int(hashlib.sha256(("blind-v1\0" + case_id).encode("utf-8")).hexdigest(), 16) % 2
    return ("N0", "N3") if first == 0 else ("N3", "N0")


def build_blind_review_files(
    records: Sequence[Mapping[str, Any]], contexts: Mapping[str, StoryContext],
    packet_path: str | Path, mapping_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_key = {(item["case_id"], item["variant_id"]): item for item in records}
    case_ids = sorted({str(item["case_id"]) for item in records})
    packet_cases = []
    mappings = []
    for position, case_id in enumerate(case_ids, 1):
        if (case_id, "N0") not in by_key or (case_id, "N3") not in by_key:
            raise ValueError(f"Missing N0/N3 output for {case_id}")
        context = contexts[case_id]
        order = blind_n0_n3_order(case_id)
        stories = {}
        for side, variant in zip(("A", "B"), order):
            payload = by_key[(case_id, variant)].get("normalized_output") or {}
            stories[side] = {
                "title": str(payload.get("title") or ""),
                "paragraphs": [
                    str(item.get("text") or "") for item in payload.get("paragraphs", [])
                    if isinstance(item, Mapping)
                ],
            }
        packet_cases.append({
            "review_index": position,
            "case_token": canonical_hash("review-case\0" + case_id)[:16],
            "language": by_key[(case_id, "N3")]["language"],
            "source_kind": by_key[(case_id, "N3")]["source_kind"],
            "photo_paths": [item.relative_path for item in context.evidence],
            "story_a": stories["A"],
            "story_b": stories["B"],
        })
        mappings.append({
            "case_id": case_id,
            "case_token": packet_cases[-1]["case_token"],
            "A": order[0],
            "B": order[1],
        })
    packet = {"protocol_version": "story-blind-review-v1", "locked": True, "cases": packet_cases}
    mapping = {"protocol_version": "story-blind-mapping-v1", "mappings": mappings}
    for path, value in ((Path(packet_path), packet), (Path(mapping_path), mapping)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return packet, mapping


def reveal_blind_preferences(
    responses: Mapping[str, Any], mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if responses.get("status") != "locked":
        raise ValueError("Blind responses must be finalized and locked before reveal")
    lookup = {item["case_token"]: item for item in mapping.get("mappings", [])}
    result = []
    for item in responses.get("responses", []):
        token = str(item.get("case_token") or "")
        choice = str(item.get("choice") or "")
        if token not in lookup or choice not in {"A", "B", "tie"}:
            raise ValueError("Blind response contains an unknown case or choice")
        selected = "tie" if choice == "tie" else lookup[token][choice]
        result.append({
            "case_id": lookup[token]["case_id"],
            "preferred_variant": selected,
            "reason": str(item.get("reason") or ""),
        })
    if len(result) != len(lookup) or len({item["case_id"] for item in result}) != len(lookup):
        raise ValueError("Blind response set is incomplete or duplicated")
    return result


def build_claim_audit_files(
    records: Sequence[Mapping[str, Any]], packet_path: str | Path, mapping_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    packet_items: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for record in records:
        payload = record.get("normalized_output") or {}
        paragraphs = payload.get("paragraphs", []) if isinstance(payload, Mapping) else []
        for paragraph_index, paragraph in enumerate(paragraphs, 1):
            if not isinstance(paragraph, Mapping):
                continue
            evidence_ids = paragraph.get("evidence_ids", [])
            for unit_index, claim in enumerate(_split_text_units(str(paragraph.get("text") or "")), 1):
                key = f"{record['case_id']}\0{record['variant_id']}\0{paragraph_index}\0{unit_index}"
                audit_id = canonical_hash("claim-audit-v1\0" + key)[:20]
                packet_items.append({
                    "audit_id": audit_id,
                    "claim": claim,
                    "evidence_ids": list(evidence_ids) if isinstance(evidence_ids, list) else [],
                    "audited_evidence_ids": [],
                    "status": None,
                    "reason": "",
                    "source": "agent_evidence_audit",
                })
                mappings.append({
                    "audit_id": audit_id,
                    "case_id": record["case_id"],
                    "variant_id": record["variant_id"],
                    "paragraph_index": paragraph_index,
                })
    random.Random(20260720).shuffle(packet_items)
    packet = {"protocol_version": "story-claim-audit-v1", "allowed_statuses": [
        "supported", "unsupported", "uncertain", "non_factual",
    ], "claims": packet_items}
    mapping = {"protocol_version": "story-claim-audit-mapping-v1", "mappings": mappings}
    for path, value in ((Path(packet_path), packet), (Path(mapping_path), mapping)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return packet, mapping


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Optional[float]]:
    if not records:
        raise ValueError("At least one Story metric record is required")
    numeric_keys = {
        key for record in records for key, value in record.items()
        if isinstance(value, (int, float, bool)) and key != "claim_count"
    }
    result: dict[str, Optional[float]] = {}
    for key in sorted(numeric_keys):
        values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float, bool))]
        result[key] = sum(values) / len(values) if values else None
    return result


def paired_bootstrap_difference(
    records: Sequence[Mapping[str, Any]], left_variant: str, right_variant: str,
    metric: str, *, samples: int = 10_000, seed: int = 20260720,
) -> dict[str, Optional[float]]:
    values: dict[str, dict[str, float]] = {}
    for record in records:
        value = record.get("automatic_metrics", {}).get(metric)
        if isinstance(value, (int, float, bool)):
            values.setdefault(str(record["case_id"]), {})[str(record["variant_id"])] = float(value)
    pairs = [
        (item[left_variant], item[right_variant]) for item in values.values()
        if left_variant in item and right_variant in item
    ]
    if not pairs:
        return {"n": 0, "mean_difference": None, "ci_low": None, "ci_high": None}
    differences = [right - left for left, right in pairs]
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(samples)
    )
    return {
        "n": len(pairs),
        "mean_difference": sum(differences) / len(differences),
        "ci_low": bootstrap[max(0, math.floor(0.025 * samples))],
        "ci_high": bootstrap[min(samples - 1, math.ceil(0.975 * samples) - 1)],
    }


def exact_two_sided_binomial(wins: int, losses: int) -> Optional[float]:
    n = wins + losses
    if not n:
        return None
    tail = sum(math.comb(n, index) for index in range(0, min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)
