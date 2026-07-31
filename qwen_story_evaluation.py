"""Qwen-only dual-track Story experiment for the final MSc evaluation.

The historical Llama N0--N3 experiment is intentionally left untouched.  This
module writes only private evaluation artefacts and never persists generated
stories to the production SQLite ``stories`` table.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from config import (
    APP_DB_PATH,
    OLLAMA_STORY_NUM_CTX,
    OLLAMA_STORY_NUM_PREDICT,
    PHOTOS_DIR,
    STORY_NARRATIVE_MAX_PARAGRAPHS,
)
from story_agent import (
    ContextAggregator,
    GroundingValidator,
    PhotographerMoodPromptBuilder,
    PhotographerMoodValidator,
    PromptBuilder,
    StoryContext,
    StoryGenerator,
)
from story_evaluation import (
    AtomicJsonlCheckpoint,
    StoryCase,
    _n0_projection,
    _normalise_legacy,
    _normalised_payload,
    _parse_json,
    _payload_text,
    _split_text_units,
    aggregate_metrics,
    canonical_hash,
    context_record,
    exact_two_sided_binomial,
    paired_bootstrap_difference,
    score_experiment_output,
    validate_n1,
)


ROOT = Path(__file__).resolve().parent
PRIVATE_DIR = ROOT / "evaluation" / "private"
QWEN_MODEL = "qwen3:4b"
QWEN_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
TEMPERATURE = 0.25
NUM_CTX = 8192
NUM_PREDICT = 1200
CASE_SELECTION_SEED = 20260722
BOOTSTRAP_SEED = 20260722
FAITHFUL_PROTOCOL = "story-qwen-faithful-qf0-qf3-v1"
CREATIVE_PROTOCOL = "story-qwen-creative-qc0-qc2-v1"
BLIND_PROTOCOL = "story-qwen-dual-blind-v1"

OLD_CASES_PATH = PRIVATE_DIR / "story_cases_v1.json"
CREATIVE_CASES_PATH = PRIVATE_DIR / "story_qwen_creative_cases_v1.json"
FAITHFUL_RESULTS_PATH = PRIVATE_DIR / "story_qwen_faithful_v1.jsonl"
CREATIVE_RESULTS_PATH = PRIVATE_DIR / "story_qwen_creative_v1.jsonl"
BLIND_PACKET_PATH = PRIVATE_DIR / "story_qwen_dual_blind_packet_v1.json"
BLIND_MAPPING_PATH = PRIVATE_DIR / "story_qwen_dual_blind_mapping_v1.json"
BLIND_RESPONSES_PATH = PRIVATE_DIR / "story_qwen_dual_blind_responses_v1.json"
CLAIM_AUDIT_PATH = PRIVATE_DIR / "story_qwen_claim_audit_v1.json"
PUBLIC_SUMMARY_PATH = ROOT / "evaluation" / "story_qwen_dual_v1_summary.json"

DEVELOPMENT_DATES = {"2026-07-07", "2026-07-10", "2026-07-11", "2026-07-12"}
MOOD_EVENT_ID = "58db14d5-e7ad-5975-87a0-59456560e09a"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    prompt_version: str
    output_format: str
    validation: str


FAITHFUL_VARIANTS = (
    VariantSpec("QF0", "qwen-faithful-legacy-qf0-v1", "text", "none"),
    VariantSpec("QF1", "qwen-faithful-per-photo-qf1-v1", "json", "citation_only"),
    VariantSpec("QF2", "qwen-faithful-grouped-qf2-v1", "json", "observe_only"),
    VariantSpec("QF3", "qwen-faithful-validated-qf3-v1", "json", "repair_fallback"),
)
CREATIVE_VARIANTS = (
    VariantSpec("QC0", "qwen-creative-legacy-qc0-v1", "text", "none"),
    VariantSpec("QC1", PhotographerMoodPromptBuilder.PROMPT_VERSION + "-unvalidated", "json_schema", "observe_only"),
    VariantSpec("QC2", PhotographerMoodPromptBuilder.PROMPT_VERSION, "json_schema", "repair_error"),
)
FAITHFUL_BY_ID = {item.variant_id: item for item in FAITHFUL_VARIANTS}
CREATIVE_BY_ID = {item.variant_id: item for item in CREATIVE_VARIANTS}


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def deterministic_seed(case_id: str, track: str) -> int:
    value = hashlib.sha256(f"story-qwen-v1\0{track}\0{case_id}".encode("utf-8")).hexdigest()
    return int(value[:8], 16) & 0x7FFFFFFF


def assert_qwen_digest(model_info: Mapping[str, Any]) -> None:
    actual = str(model_info.get("digest") or "")
    if actual != QWEN_DIGEST:
        raise RuntimeError(
            f"Frozen Qwen digest mismatch: expected {QWEN_DIGEST}, got {actual or '<missing>'}. "
            "The experiment will not pull or switch a model automatically."
        )


class QwenOllamaBackend:
    """Fixed Ollama adapter with explicit context, seed and thinking settings."""

    def __init__(self, model_name: str = QWEN_MODEL):
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
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        seed: int,
        json_mode: bool,
        output_schema: Optional[Mapping[str, Any]] = None,
        temperature: float = TEMPERATURE,
        num_predict: int = NUM_PREDICT,
        num_ctx: int = NUM_CTX,
    ) -> str:
        import ollama

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "seed": int(seed),
            },
            format=dict(output_schema) if output_schema is not None else ("json" if json_mode else None),
            think=False,
        )
        message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response")
        return content


def _walk_photo_ids(value: Any, result: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "photo_id" and isinstance(child, str):
                result.add(child)
            elif key == "photo_ids" and isinstance(child, list):
                result.update(str(item) for item in child)
            _walk_photo_ids(child, result)
    elif isinstance(value, list):
        for child in value:
            _walk_photo_ids(child, result)


def _rank(value: str, namespace: str) -> str:
    return hashlib.sha256(f"{CASE_SELECTION_SEED}\0{namespace}\0{value}".encode("utf-8")).hexdigest()


def _case_payload(case: StoryCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "source_kind": case.source_kind,
        "label": case.label,
        "photo_ids": list(case.photo_ids),
        "language": case.language,
        "event_id": case.event_id,
        "size_bucket": case.size_bucket,
        "narrator_role": "observer",
        "verified_context": "",
        "use_photographer_mood": False,
    }


def _verify_case_quotas(cases: Sequence[StoryCase], *, reserve: bool) -> None:
    expected_count = 6 if reserve else 12
    if len(cases) != expected_count or len({case.case_id for case in cases}) != expected_count:
        raise ValueError(f"Expected {expected_count} unique {'reserve' if reserve else 'main'} Creative cases")
    expected = {"single": 1, "event": 4, "date": 1} if reserve else {"single": 2, "event": 8, "date": 2}
    actual = Counter(case.source_kind for case in cases)
    if any(actual[key] != value for key, value in expected.items()):
        raise ValueError(f"Creative source-kind quota mismatch: expected={expected}, actual={dict(actual)}")
    language_target = expected_count // 2
    if Counter(case.language for case in cases) != Counter({"zh": language_target, "en": language_target}):
        raise ValueError("Creative language quota must be balanced")


def load_creative_case_manifest(path: str | Path = CREATIVE_CASES_PATH) -> tuple[list[StoryCase], list[StoryCase], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_hash = str(payload.get("freeze_sha256") or "")
    unhashed = dict(payload)
    unhashed.pop("freeze_sha256", None)
    if stored_hash != canonical_hash(unhashed):
        raise ValueError("Creative case manifest SHA-256 does not match its content")

    def parse(items: Sequence[Mapping[str, Any]]) -> list[StoryCase]:
        return [StoryCase(
            case_id=str(item["case_id"]), source_kind=str(item["source_kind"]),
            label=str(item["label"]), photo_ids=tuple(str(value) for value in item["photo_ids"]),
            language=str(item["language"]), event_id=item.get("event_id"),
            size_bucket=str(item.get("size_bucket") or ""),
        ) for item in items]

    main = parse(payload.get("main_cases", []))
    reserve = parse(payload.get("reserve_cases", []))
    _verify_case_quotas(main, reserve=False)
    _verify_case_quotas(reserve, reserve=True)
    all_ids = [photo_id for case in [*main, *reserve] for photo_id in case.photo_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Creative cases must not share photos")
    return main, reserve, payload


def freeze_creative_cases(
    *,
    db_path: str | Path = APP_DB_PATH,
    photo_root: str | Path = PHOTOS_DIR,
    old_cases_path: str | Path = OLD_CASES_PATH,
    output_path: str | Path = CREATIVE_CASES_PATH,
) -> dict[str, Any]:
    """Deterministically freeze unseen Creative main and reserve cases."""
    target = Path(output_path)
    if target.exists():
        return load_creative_case_manifest(target)[2]

    old_payload = json.loads(Path(old_cases_path).read_text(encoding="utf-8"))
    excluded = {
        str(photo_id)
        for case in old_payload.get("cases", [])
        for photo_id in case.get("photo_ids", [])
    }
    connection = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        for row in connection.execute("SELECT grounding_json FROM stories"):
            try:
                grounding = json.loads(row["grounding_json"] or "{}")
            except Exception:
                continue
            _walk_photo_ids(grounding, excluded)
        excluded.update(
            str(row[0]) for row in connection.execute(
                "SELECT photo_id FROM event_photos WHERE event_id = ?", (MOOD_EVENT_ID,),
            )
        )
        excluded.update(
            str(row[0]) for row in connection.execute(
                "SELECT photo_id FROM photos WHERE date_local IN (%s)" % ",".join("?" * len(DEVELOPMENT_DATES)),
                tuple(sorted(DEVELOPMENT_DATES)),
            )
        )

        ready_rows = connection.execute("""
            SELECT p.photo_id, p.relative_path, p.date_local, p.timestamp_confidence
            FROM photos p
            WHERE p.captured_at IS NOT NULL AND p.timestamp_confidence >= 0.6
              AND EXISTS (
                  SELECT 1 FROM captions c
                  WHERE c.photo_id=p.photo_id AND c.status='ok' AND c.is_primary=1
              )
              AND EXISTS (
                  SELECT 1 FROM scene_graphs sg
                  WHERE sg.photo_id=p.photo_id AND sg.status='ok' AND sg.is_primary=1
              )
        """).fetchall()
        photo_root_path = Path(photo_root)
        ready = {
            str(row["photo_id"]): dict(row) for row in ready_rows
            if str(row["photo_id"]) not in excluded
            and (photo_root_path / str(row["relative_path"])).is_file()
        }

        event_candidates: dict[str, list[dict[str, Any]]] = {"2-3": [], "4-7": [], "8-14": []}
        event_rows = connection.execute("""
            SELECT e.event_id, ep.photo_id, ep.position
            FROM events e JOIN event_photos ep ON ep.event_id=e.event_id
            ORDER BY e.event_id, ep.position
        """).fetchall()
        by_event: dict[str, list[str]] = {}
        for row in event_rows:
            by_event.setdefault(str(row["event_id"]), []).append(str(row["photo_id"]))
        for event_id, photo_ids in by_event.items():
            count = len(photo_ids)
            bucket = "2-3" if 2 <= count <= 3 else "4-7" if count <= 7 else "8-14" if count <= 14 else ""
            if bucket and all(photo_id in ready for photo_id in photo_ids):
                event_candidates[bucket].append({
                    "source_kind": "event", "source_id": event_id,
                    "photo_ids": tuple(photo_ids), "event_id": event_id, "size_bucket": bucket,
                })
        for bucket in event_candidates:
            event_candidates[bucket].sort(key=lambda item: _rank(item["source_id"], "event-" + bucket))

        date_candidates: list[dict[str, Any]] = []
        date_rows = connection.execute("""
            SELECT p.photo_id, p.date_local, p.captured_at_sort,
                   COUNT(ep.event_id) OVER (PARTITION BY p.date_local) AS event_links
            FROM photos p LEFT JOIN event_photos ep ON ep.photo_id=p.photo_id
            WHERE p.date_local IS NOT NULL
            ORDER BY p.date_local, p.captured_at_sort, p.photo_id
        """).fetchall()
        by_date: dict[str, list[str]] = {}
        for row in date_rows:
            date = str(row["date_local"])
            if date not in DEVELOPMENT_DATES:
                by_date.setdefault(date, []).append(str(row["photo_id"]))
        for date, raw_ids in by_date.items():
            photo_ids = tuple(dict.fromkeys(raw_ids))
            if not 6 <= len(photo_ids) <= 14 or not all(photo_id in ready for photo_id in photo_ids):
                continue
            event_count = connection.execute(
                "SELECT COUNT(DISTINCT ep.event_id) FROM event_photos ep "
                "JOIN photos p ON p.photo_id=ep.photo_id WHERE p.date_local=?", (date,),
            ).fetchone()[0]
            if int(event_count) >= 2:
                date_candidates.append({
                    "source_kind": "date", "source_id": date,
                    "photo_ids": photo_ids, "event_id": None, "size_bucket": "multi-event",
                })
        date_candidates.sort(key=lambda item: _rank(item["source_id"], "date"))

        totals = {"2-3": 5, "4-7": 4, "8-14": 3}

        def pick_events(used: set[str]) -> Optional[dict[str, list[dict[str, Any]]]]:
            picked: dict[str, list[dict[str, Any]]] = {}
            current = set(used)
            for bucket in ("8-14", "4-7", "2-3"):
                values = []
                for candidate in event_candidates[bucket]:
                    ids = set(candidate["photo_ids"])
                    if not ids & current:
                        values.append(candidate)
                        current.update(ids)
                    if len(values) == totals[bucket]:
                        break
                if len(values) != totals[bucket]:
                    return None
                picked[bucket] = values
            return picked

        selected_dates: Optional[tuple[dict[str, Any], ...]] = None
        selected_events: Optional[dict[str, list[dict[str, Any]]]] = None
        for combination in itertools.combinations(date_candidates, 3):
            ids = [set(item["photo_ids"]) for item in combination]
            if any(ids[left] & ids[right] for left in range(3) for right in range(left + 1, 3)):
                continue
            events = pick_events(set().union(*ids))
            if events is not None:
                selected_dates, selected_events = combination, events
                break
        if selected_dates is None or selected_events is None:
            raise RuntimeError("No deterministic non-overlapping Creative case allocation satisfies the frozen quotas")

        used_ids = {
            photo_id for item in selected_dates for photo_id in item["photo_ids"]
        } | {
            photo_id for values in selected_events.values() for item in values for photo_id in item["photo_ids"]
        }
        single_candidates = sorted(
            (photo_id for photo_id in ready if photo_id not in used_ids),
            key=lambda photo_id: _rank(photo_id, "single"),
        )
        if len(single_candidates) < 3:
            raise RuntimeError("Fewer than three eligible non-overlapping single-photo cases remain")
        singles = [{
            "source_kind": "single", "source_id": photo_id, "photo_ids": (photo_id,),
            "event_id": None, "size_bucket": "1",
        } for photo_id in single_candidates[:3]]

        event_main = [
            *selected_events["2-3"][:3], *selected_events["4-7"][:3], *selected_events["8-14"][:2],
        ]
        event_reserve = [
            *selected_events["2-3"][3:], *selected_events["4-7"][3:], *selected_events["8-14"][2:],
        ]
        main_raw = [*singles[:2], *event_main, *selected_dates[:2]]
        reserve_raw = [singles[2], *event_reserve, selected_dates[2]]

        def make_cases(values: Sequence[Mapping[str, Any]], prefix: str) -> list[StoryCase]:
            result = []
            for index, item in enumerate(values, 1):
                result.append(StoryCase(
                    case_id=f"{prefix}_{index:02d}", source_kind=str(item["source_kind"]),
                    label=f"Creative evaluation case {index:02d}",
                    photo_ids=tuple(str(value) for value in item["photo_ids"]),
                    language="zh" if index % 2 else "en", event_id=item.get("event_id"),
                    size_bucket=str(item["size_bucket"]),
                ))
            return result

        main_cases = make_cases(main_raw, "creative_main")
        reserve_cases = make_cases(reserve_raw, "creative_reserve")
        _verify_case_quotas(main_cases, reserve=False)
        _verify_case_quotas(reserve_cases, reserve=True)
        payload: dict[str, Any] = {
            "protocol_version": "story-qwen-creative-cases-v1",
            "selection_seed": CASE_SELECTION_SEED,
            "criteria": {
                "narrator_role": "observer", "verified_context": "", "use_photographer_mood": False,
                "requires_primary_caption": True, "requires_primary_scene_graph": True,
                "minimum_timestamp_confidence": 0.6, "maximum_case_photos": 14,
                "development_dates_excluded": sorted(DEVELOPMENT_DATES),
            },
            "excluded_photo_count": len(excluded),
            "excluded_photo_ids": sorted(excluded),
            "main_cases": [_case_payload(case) for case in main_cases],
            "reserve_cases": [_case_payload(case) for case in reserve_cases],
        }
        payload["freeze_sha256"] = canonical_hash(payload)
        atomic_write_json(target, payload)
        return payload
    finally:
        connection.close()


def load_faithful_cases(path: str | Path = OLD_CASES_PATH) -> list[StoryCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [StoryCase(
        case_id=str(item["case_id"]), source_kind=str(item["source_kind"]), label=str(item["label"]),
        photo_ids=tuple(str(value) for value in item["photo_ids"]), language=str(item["language"]),
        event_id=item.get("event_id"), size_bucket=str(item.get("size_bucket") or ""),
    ) for item in payload.get("cases", [])]
    if len(cases) != 12 or Counter(case.language for case in cases) != Counter({"zh": 6, "en": 6}):
        raise ValueError("Historical frozen Faithful set must remain 12 cases with 6 zh and 6 en")
    return cases


def default_context_factory(case: StoryCase, *, creative: bool) -> StoryContext:
    context = ContextAggregator().aggregate_by_photo_ids(
        case.photo_ids, label=case.label, event_id=case.event_id, source_kind=case.source_kind,
        verified_context="", narrator_role="observer", use_photographer_mood=False,
    )
    if creative and len(context.groups) > STORY_NARRATIVE_MAX_PARAGRAPHS:
        context = replace(
            context,
            groups=StoryGenerator._reduce_groups(context.groups, STORY_NARRATIVE_MAX_PARAGRAPHS),
        )
    return context


class _RunnerBase:
    def __init__(
        self,
        backend: Any,
        output_path: str | Path,
        *,
        context_factory: Optional[Callable[[StoryCase], StoryContext]] = None,
        progress: Optional[Callable[[Mapping[str, Any], bool], None]] = None,
    ):
        self.backend = backend
        self.checkpoint = AtomicJsonlCheckpoint(output_path)
        self.context_factory = context_factory
        self.progress = progress

    def _invoke(
        self,
        system: str,
        user: str,
        *,
        seed: int,
        json_mode: bool,
        output_schema: Optional[Mapping[str, Any]] = None,
    ) -> tuple[str, float]:
        last_error: Optional[Exception] = None
        for attempt in range(2):
            started = time.perf_counter()
            try:
                content = self.backend.generate(
                    system, user, seed=seed, json_mode=json_mode, output_schema=output_schema,
                    temperature=TEMPERATURE, num_ctx=NUM_CTX, num_predict=NUM_PREDICT,
                )
                return content, (time.perf_counter() - started) * 1000.0
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    break
        raise RuntimeError(
            f"Generation transport failed twice with the same frozen prompt and seed: {last_error}"
        ) from last_error

    @staticmethod
    def _attempt(kind: str, content: str, latency_ms: float) -> dict[str, Any]:
        return {
            "kind": kind, "content": content, "output_hash": canonical_hash(content),
            "latency_ms": float(latency_ms),
        }

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

    @staticmethod
    def _base_record(
        protocol: str,
        case: StoryCase,
        spec: VariantSpec,
        *,
        model_info: Mapping[str, Any],
        seed: int,
        case_hash: str,
        prompt_hash: str,
        context_hash: str,
    ) -> dict[str, Any]:
        return {
            "protocol_version": protocol,
            "case_id": case.case_id,
            "case_hash": case_hash,
            "variant_id": spec.variant_id,
            "language": case.language,
            "source_kind": case.source_kind,
            "size_bucket": case.size_bucket,
            "model_name": QWEN_MODEL,
            "model_digest": str(model_info["digest"]),
            "seed": seed,
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "think": False,
            "prompt_version": spec.prompt_version,
            "prompt_hash": prompt_hash,
            "context_hash": context_hash,
        }

    def _existing(self, model_info: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        existing = {
            (str(item["case_id"]), str(item["variant_id"])): item
            for item in self.checkpoint.records()
        }
        for record in existing.values():
            if record.get("model_digest") != model_info["digest"]:
                raise RuntimeError("Existing checkpoint uses a different model digest")
        return existing

    def _reuse_or_none(
        self,
        existing: Mapping[tuple[str, str], dict[str, Any]],
        key: tuple[str, str],
        *,
        case_hash: str,
        prompt_hash: str,
        context_hash: str,
        model_digest: str,
        seed: int,
    ) -> Optional[dict[str, Any]]:
        record = existing.get(key)
        if record is None:
            return None
        if not self._identity_matches(
            record, case_hash=case_hash, prompt_hash=prompt_hash, context_hash=context_hash,
            model_digest=model_digest, seed=seed,
        ):
            raise RuntimeError(f"Checkpoint identity drift for {key[0]}/{key[1]}; refusing rerun")
        if self.progress is not None:
            self.progress(record, True)
        return record


def _faithful_prompt(variant_id: str, context: StoryContext, language: str) -> tuple[str, str, bool]:
    language_name = "Chinese" if language.startswith("zh") else "English"
    if variant_id == "QF0":
        system = f"""You write a warm personal photo diary in {language_name}.
Use first-person prose and connect the supplied photo descriptions into a readable memory.
The source contains only captions, tags, and reliable time/location metadata.
Do not output JSON and do not add evidence citations. Start with a short title on its own line."""
        user = "Write one short diary entry from these photos:\n" + json.dumps(
            _n0_projection(context), ensure_ascii=False, indent=2,
        )
        return system, user, False
    if variant_id == "QF1":
        photos = [{
            "evidence_id": item.evidence_id,
            "reliable_time": item.timestamp if item.timestamp and item.timestamp_confidence >= 0.8 else None,
            "location": item.location or None,
            "blip_caption_model_observation": item.caption or None,
            "qwen_scene_graph_model_observations": list(item.scene_graph_triples),
            "clip_tags_model_observations": list(item.tags),
        } for item in context.evidence]
        system = f"""Write a faithful photo narrative in {language_name} from raw per-photo model observations.
Return JSON only. Write exactly one paragraph for every input photo, in input order.
Each paragraph must cite only that photo's evidence_id. Do not merge or compress photos.
Use neutral observable descriptions; the observations may be fallible.
Return exactly: {{"title":"...","paragraphs":[{{"text":"...","evidence_ids":["P001"]}}]}}"""
        user = "Create the per-photo narrative from this evidence:\n" + json.dumps(
            {"label": context.label, "photos": photos}, ensure_ascii=False, indent=2,
        )
        return system, user, True
    if variant_id in {"QF2", "QF3"}:
        system, user = PromptBuilder.build_story_prompt(context, mode="faithful", language=language)
        return system, user, True
    raise ValueError(f"Unknown Faithful variant: {variant_id}")


def _common_language_metrics(payload: Optional[Mapping[str, Any]], language: str) -> dict[str, Any]:
    text = _payload_text(payload)
    if not text.strip():
        return {"chinese_character_ratio": None, "language_compliant": False}
    ratio = GroundingValidator._language_ratio(text)
    return {
        "chinese_character_ratio": ratio,
        "language_compliant": ratio >= 0.25 if language.startswith("zh") else ratio <= 0.10,
    }


class QwenFaithfulRunner(_RunnerBase):
    def __init__(self, backend: Any, output_path: str | Path = FAITHFUL_RESULTS_PATH, **kwargs: Any):
        super().__init__(backend, output_path, **kwargs)
        if self.context_factory is None:
            self.context_factory = lambda case: default_context_factory(case, creative=False)

    def _independent(
        self, case: StoryCase, context: StoryContext, base: dict[str, Any],
        system: str, user: str, json_mode: bool,
    ) -> dict[str, Any]:
        raw, latency = self._invoke(system, user, seed=base["seed"], json_mode=json_mode)
        codes: list[str] = []
        if base["variant_id"] == "QF0":
            payload = _normalise_legacy(raw, case.label)
            status = "ok" if payload and payload.get("paragraphs") else "invalid"
            json_valid: Optional[bool] = None
            if status == "invalid":
                codes.append("E_TEXT_EMPTY: legacy output contains no narrative")
            metric_variant = "N0"
        else:
            parsed, parse_errors = _parse_json(raw)
            payload = _normalised_payload(parsed)
            codes.extend(parse_errors)
            codes.extend(validate_n1(parsed, context) if parsed is not None else [])
            status = "ok" if not codes else "invalid"
            json_valid = parsed is not None
            metric_variant = "N1"
        metrics = score_experiment_output(
            metric_variant, payload, context, latency_ms=latency, status=status,
            json_valid=json_valid, language=case.language,
        )
        metrics.update(_common_language_metrics(payload, case.language))
        return {
            **base, "raw_attempts": [self._attempt("generation", raw, latency)],
            "normalized_output": payload,
            "output_hash": canonical_hash(payload) if payload is not None else canonical_hash(raw),
            "status": status, "validation_codes": list(dict.fromkeys(codes)),
            "fallback": False, "latency_ms": latency, "automatic_metrics": metrics,
        }

    def _qf2(
        self, case: StoryCase, context: StoryContext, base: dict[str, Any],
        system: str, user: str,
    ) -> dict[str, Any]:
        raw, latency = self._invoke(system, user, seed=base["seed"], json_mode=True)
        parsed, codes = _parse_json(raw)
        if parsed is not None:
            codes.extend(GroundingValidator.validate(parsed, context, "faithful", case.language))
        payload = _normalised_payload(parsed)
        status = "unvalidated" if parsed is not None else "invalid"
        metrics = score_experiment_output(
            "N2", payload, context, latency_ms=latency, status=status,
            json_valid=parsed is not None, language=case.language,
        )
        metrics.update(_common_language_metrics(payload, case.language))
        return {
            **base, "raw_attempts": [self._attempt("shared_qf2_qf3_draft", raw, latency)],
            "normalized_output": payload,
            "output_hash": canonical_hash(payload) if payload is not None else canonical_hash(raw),
            "status": status, "validation_codes": list(dict.fromkeys(codes)),
            "fallback": False, "latency_ms": latency, "automatic_metrics": metrics,
        }

    def _qf3(
        self, qf2: Mapping[str, Any], case: StoryCase, context: StoryContext,
        base: dict[str, Any], system: str, base_user: str,
    ) -> dict[str, Any]:
        first = dict(qf2["raw_attempts"][0])
        raw = str(first["content"])
        attempts = [first]
        parsed, codes = _parse_json(raw)
        if parsed is not None:
            codes.extend(GroundingValidator.validate(parsed, context, "faithful", case.language))
        payload: Optional[dict[str, Any]] = None
        status = "ok"
        if parsed is not None and not codes:
            payload = _normalised_payload(parsed)
        else:
            repair_user = (
                base_user
                + "\n\nThe previous output failed strict validation. Repair it once and return corrected JSON only."
                + "\nPrevious rejected output:\n<rejected_output>\n"
                + raw
                + "\n</rejected_output>\nValidation errors:\n- "
                + "\n- ".join(codes)
            )
            repaired_raw, repair_latency = self._invoke(
                system, repair_user, seed=base["seed"], json_mode=True,
            )
            attempts.append(self._attempt("repair", repaired_raw, repair_latency))
            repaired, repair_codes = _parse_json(repaired_raw)
            if repaired is not None:
                repair_codes.extend(GroundingValidator.validate(
                    repaired, context, "faithful", case.language,
                ))
            codes = list(dict.fromkeys([*codes, *repair_codes]))
            if repaired is not None and not repair_codes:
                payload = _normalised_payload(repaired)
                status = "repaired"
            else:
                fallback = StoryGenerator._fallback(context, case.language, codes, str(base["prompt_hash"]))
                payload = fallback.to_dict()
                status = "fallback"
        latency = sum(float(item["latency_ms"]) for item in attempts)
        final_codes = GroundingValidator.validate(dict(payload), context, "faithful", case.language)
        if final_codes:
            raise RuntimeError(
                f"QF3 final artifact failed hard constraints for {case.case_id}: " + "; ".join(final_codes)
            )
        metrics = score_experiment_output(
            "N3", payload, context, latency_ms=latency, status=status,
            json_valid=True, language=case.language,
        )
        metrics.update(_common_language_metrics(payload, case.language))
        return {
            **base, "raw_attempts": attempts, "normalized_output": payload,
            "output_hash": canonical_hash(payload), "status": status,
            "validation_codes": codes, "final_validation_codes": final_codes,
            "fallback": status == "fallback", "latency_ms": latency,
            "automatic_metrics": metrics, "shared_draft_output_hash": str(first["output_hash"]),
        }

    def run(self, cases: Sequence[StoryCase]) -> list[dict[str, Any]]:
        model_info = self.backend.model_info()
        assert_qwen_digest(model_info)
        existing = self._existing(model_info)
        for case in cases:
            context = self.context_factory(case)
            if not context.evidence or not context.groups:
                raise RuntimeError(f"Frozen Faithful case {case.case_id} has no usable evidence")
            context_hash = canonical_hash(context_record(context))
            case_hash = canonical_hash(asdict(case))
            seed = deterministic_seed(case.case_id, "faithful")
            for variant_id in ("QF0", "QF1", "QF2", "QF3"):
                spec = FAITHFUL_BY_ID[variant_id]
                system, user, json_mode = _faithful_prompt(variant_id, context, case.language)
                prompt_hash = canonical_hash({"system": system, "user": user, "json_mode": json_mode})
                base = self._base_record(
                    FAITHFUL_PROTOCOL, case, spec, model_info=model_info, seed=seed,
                    case_hash=case_hash, prompt_hash=prompt_hash, context_hash=context_hash,
                )
                key = (case.case_id, variant_id)
                if self._reuse_or_none(
                    existing, key, case_hash=case_hash, prompt_hash=prompt_hash,
                    context_hash=context_hash, model_digest=str(model_info["digest"]), seed=seed,
                ) is not None:
                    continue
                if variant_id in {"QF0", "QF1"}:
                    record = self._independent(case, context, base, system, user, json_mode)
                elif variant_id == "QF2":
                    record = self._qf2(case, context, base, system, user)
                else:
                    qf2 = existing.get((case.case_id, "QF2"))
                    if qf2 is None or qf2["prompt_hash"] != prompt_hash or qf2["seed"] != seed:
                        raise RuntimeError("QF3 cannot run without the exact paired QF2 draft")
                    record = self._qf3(qf2, case, context, base, system, user)
                self.checkpoint.append(record)
                existing[key] = record
                if self.progress is not None:
                    self.progress(record, False)
        records = self.checkpoint.records()
        expected = {(case.case_id, variant.variant_id) for case in cases for variant in FAITHFUL_VARIANTS}
        actual = {(str(item["case_id"]), str(item["variant_id"])) for item in records}
        if actual != expected:
            raise RuntimeError(f"Faithful checkpoint is not exactly 12x4; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        return records


def _creative_legacy_prompt(context: StoryContext, language: str) -> tuple[str, str]:
    language_name = "Chinese" if language.startswith("zh") else "English"
    length = "220–360 Chinese characters" if language.startswith("zh") else "140–230 English words"
    system = f"""Write a natural first-person observer memoir in {language_name}.
Use only the supplied BLIP captions, CLIP tags and reliable time/location metadata.
The narrator is not confirmed as a photographed person: use observer memory such as I saw or I noticed.
Do not claim that the narrator performed a photographed action. Do not infer identity, relationship, purpose or cause.
Do not use evidence IDs or JSON. Start with a short title on its own line, then write flowing prose with an opening and restrained ending.
Target {length}; avoid caption-list phrasing and generic poetic clichés."""
    user = "Create the observer memoir from this legacy evidence:\n" + json.dumps(
        _n0_projection(context), ensure_ascii=False, indent=2,
    )
    return system, user


def _creative_prompt(
    context: StoryContext, language: str,
) -> tuple[str, str, Mapping[str, Any]]:
    system, user = PhotographerMoodPromptBuilder.build_story_prompt(
        context, mode="creative", language=language,
    )
    schema = PhotographerMoodPromptBuilder.output_schema(context, language=language)
    return system, user, schema


def _parse_creative(raw: str, context: StoryContext, language: str) -> tuple[Optional[dict[str, Any]], list[str]]:
    parsed, codes = _parse_json(raw)
    if parsed is None:
        return None, codes
    payload = PhotographerMoodPromptBuilder.normalise_payload(parsed, context)
    payload["mood_reflections"] = []
    codes.extend(PhotographerMoodValidator.validate(payload, context, language))
    return payload, list(dict.fromkeys(codes))


def _creative_full_text(payload: Optional[Mapping[str, Any]]) -> str:
    if not payload:
        return ""
    values = [str(payload.get("title") or ""), str(payload.get("opening") or "")]
    transitions = payload.get("creative_transitions", [])
    for index, paragraph in enumerate(payload.get("paragraphs", [])):
        if isinstance(paragraph, Mapping):
            values.append(str(paragraph.get("text") or ""))
        if isinstance(transitions, list) and index < len(transitions):
            values.append(str(transitions[index] or ""))
    values.append(str(payload.get("closing") or ""))
    return "\n".join(value.strip() for value in values if value and value.strip())


def _creative_metrics(
    variant_id: str,
    payload: Optional[Mapping[str, Any]],
    context: StoryContext,
    language: str,
    *,
    latency_ms: float,
    status: str,
    json_valid: Optional[bool],
    validation_codes: Sequence[str],
) -> dict[str, Any]:
    metric_variant = "N0" if variant_id == "QC0" else "N2"
    metrics = score_experiment_output(
        metric_variant, payload, context, latency_ms=latency_ms, status=status,
        json_valid=json_valid, language=language,
    )
    text = _creative_full_text(payload) if variant_id != "QC0" else _payload_text(payload)
    units = _split_text_units(text)
    normalized = [re.sub(r"\W+", "", unit.casefold()) for unit in units if re.sub(r"\W+", "", unit)]
    ratio = GroundingValidator._language_ratio(text) if text.strip() else None
    if language.startswith("zh"):
        first_person = "我" in text
        prose_length = len(re.findall(r"[\u4e00-\u9fff]", text))
        language_ok = bool(text.strip() and ratio is not None and ratio >= 0.25)
    else:
        first_person = re.search(r"\b(?:I|my|me|we|our)\b", text, re.I) is not None
        prose_length = len(re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", text))
        language_ok = bool(text.strip() and ratio is not None and ratio <= 0.10)
    observer_errors = (
        "E_FIRST_PERSON_VOICE", "E_CONFIRMED_SUBJECT_NO_PERSON", "E_OBSERVER_ROLE", "E_UNSUPPORTED_CLAIM",
    )
    metrics.update({
        "chinese_character_ratio": ratio,
        "language_compliant": language_ok,
        "duplicate_text_unit_rate": (
            1.0 - len(set(normalized)) / len(normalized) if normalized else None
        ),
        "prose_length": prose_length,
        "observer_first_person_compliant": bool(first_person and not any(
            marker in code for code in validation_codes for marker in observer_errors
        )),
        "metadata_repetition_count": sum("E_METADATA_IN_NARRATIVE" in code for code in validation_codes),
        "cliche_count": sum("E_CREATIVE_CLICHE" in code for code in validation_codes),
        "structure_complete": bool(payload) and not any(
            marker in code for code in validation_codes
            for marker in ("E_JSON", "E_MEMOIR_GROUP", "E_MEMOIR_OPENING", "E_MEMOIR_CLOSING", "E_MEMOIR_TRANSITION")
        ),
        "repair": status == "repaired",
        "fallback": False,
        "error": status == "error",
    })
    return metrics


class QwenCreativeRunner(_RunnerBase):
    def __init__(self, backend: Any, output_path: str | Path = CREATIVE_RESULTS_PATH, **kwargs: Any):
        super().__init__(backend, output_path, **kwargs)
        if self.context_factory is None:
            self.context_factory = lambda case: default_context_factory(case, creative=True)

    def _qc0(
        self, case: StoryCase, context: StoryContext, base: dict[str, Any], system: str, user: str,
    ) -> dict[str, Any]:
        raw, latency = self._invoke(system, user, seed=base["seed"], json_mode=False)
        payload = _normalise_legacy(raw, case.label)
        status = "ok" if payload and payload.get("paragraphs") else "invalid"
        codes = [] if status == "ok" else ["E_TEXT_EMPTY: legacy output contains no narrative"]
        metrics = _creative_metrics(
            "QC0", payload, context, case.language, latency_ms=latency, status=status,
            json_valid=None, validation_codes=codes,
        )
        return {
            **base, "raw_attempts": [self._attempt("generation", raw, latency)],
            "normalized_output": payload,
            "output_hash": canonical_hash(payload) if payload is not None else canonical_hash(raw),
            "status": status, "validation_codes": codes, "fallback": False,
            "latency_ms": latency, "automatic_metrics": metrics,
        }

    def _qc1(
        self, case: StoryCase, context: StoryContext, base: dict[str, Any], system: str,
        user: str, schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw, latency = self._invoke(
            system, user, seed=base["seed"], json_mode=True, output_schema=schema,
        )
        payload, codes = _parse_creative(raw, context, case.language)
        status = "unvalidated" if payload is not None else "invalid"
        metrics = _creative_metrics(
            "QC1", payload, context, case.language, latency_ms=latency, status=status,
            json_valid=payload is not None, validation_codes=codes,
        )
        return {
            **base, "raw_attempts": [self._attempt("shared_qc1_qc2_draft", raw, latency)],
            "normalized_output": payload,
            "output_hash": canonical_hash(payload) if payload is not None else canonical_hash(raw),
            "status": status, "validation_codes": codes, "fallback": False,
            "latency_ms": latency, "automatic_metrics": metrics,
        }

    def _qc2(
        self, qc1: Mapping[str, Any], case: StoryCase, context: StoryContext, base: dict[str, Any],
        system: str, base_user: str, schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        first = dict(qc1["raw_attempts"][0])
        raw = str(first["content"])
        attempts = [first]
        payload, initial_codes = _parse_creative(raw, context, case.language)
        history_codes = list(initial_codes)
        final_codes = list(initial_codes)
        status = "ok"
        rejected_payload = payload
        if payload is None or initial_codes:
            repair_user = (
                base_user
                + "\n\nThe previous output failed strict validation. Repair it once and return corrected JSON only."
                + "\nPrevious rejected output:\n<rejected_output>\n"
                + raw
                + "\n</rejected_output>\nValidation errors:\n- "
                + "\n- ".join(initial_codes)
                + "\nRepair guidance:\n"
                + StoryGenerator._creative_repair_guidance(initial_codes, context, case.language)
            )
            repaired_raw, repair_latency = self._invoke(
                system, repair_user, seed=base["seed"], json_mode=True, output_schema=schema,
            )
            attempts.append(self._attempt("repair", repaired_raw, repair_latency))
            repaired_payload, repair_codes = _parse_creative(repaired_raw, context, case.language)
            history_codes = list(dict.fromkeys([*initial_codes, *repair_codes]))
            final_codes = list(repair_codes)
            if repaired_payload is not None and not repair_codes:
                payload = repaired_payload
                status = "repaired"
            else:
                rejected_payload = repaired_payload or rejected_payload
                payload = None
                status = "error"
        latency = sum(float(item["latency_ms"]) for item in attempts)
        metrics = _creative_metrics(
            "QC2", payload, context, case.language, latency_ms=latency, status=status,
            json_valid=payload is not None, validation_codes=final_codes,
        )
        return {
            **base, "raw_attempts": attempts, "normalized_output": payload,
            "rejected_normalized_output": rejected_payload if status == "error" else None,
            "output_hash": canonical_hash(payload) if payload is not None else canonical_hash(attempts[-1]["content"]),
            "status": status, "validation_codes": history_codes,
            "final_validation_codes": final_codes,
            "fallback": False, "latency_ms": latency, "automatic_metrics": metrics,
            "shared_draft_output_hash": str(first["output_hash"]),
        }

    def run_cases(self, cases: Sequence[StoryCase]) -> list[dict[str, Any]]:
        model_info = self.backend.model_info()
        assert_qwen_digest(model_info)
        existing = self._existing(model_info)
        for case in cases:
            context = self.context_factory(case)
            if context.narrator_role != "observer" or context.verified_context or context.use_photographer_mood:
                raise RuntimeError("Formal Creative context must be observer, empty verified_context and Mood disabled")
            if not context.evidence or not context.groups:
                raise RuntimeError(f"Frozen Creative case {case.case_id} has no usable evidence")
            context_hash = canonical_hash(context_record(context))
            case_hash = canonical_hash(asdict(case))
            seed = deterministic_seed(case.case_id, "creative")
            for variant_id in ("QC0", "QC1", "QC2"):
                spec = CREATIVE_BY_ID[variant_id]
                if variant_id == "QC0":
                    system, user = _creative_legacy_prompt(context, case.language)
                    schema = None
                else:
                    system, user, schema = _creative_prompt(context, case.language)
                prompt_hash = canonical_hash({"system": system, "user": user, "schema": schema})
                base = self._base_record(
                    CREATIVE_PROTOCOL, case, spec, model_info=model_info, seed=seed,
                    case_hash=case_hash, prompt_hash=prompt_hash, context_hash=context_hash,
                )
                key = (case.case_id, variant_id)
                if self._reuse_or_none(
                    existing, key, case_hash=case_hash, prompt_hash=prompt_hash,
                    context_hash=context_hash, model_digest=str(model_info["digest"]), seed=seed,
                ) is not None:
                    continue
                if variant_id == "QC0":
                    record = self._qc0(case, context, base, system, user)
                elif variant_id == "QC1":
                    record = self._qc1(case, context, base, system, user, schema or {})
                else:
                    qc1 = existing.get((case.case_id, "QC1"))
                    if qc1 is None or qc1["prompt_hash"] != prompt_hash or qc1["seed"] != seed:
                        raise RuntimeError("QC2 cannot run without the exact paired QC1 draft")
                    record = self._qc2(qc1, case, context, base, system, user, schema or {})
                self.checkpoint.append(record)
                existing[key] = record
                if self.progress is not None:
                    self.progress(record, False)
        return self.checkpoint.records()

    @staticmethod
    def valid_blind_pair(records: Sequence[Mapping[str, Any]], case_id: str) -> bool:
        by_key = {(str(item["case_id"]), str(item["variant_id"])): item for item in records}
        qc0 = by_key.get((case_id, "QC0"))
        qc2 = by_key.get((case_id, "QC2"))
        return bool(
            qc0 and qc0.get("status") == "ok" and qc0.get("normalized_output")
            and qc2 and qc2.get("status") in {"ok", "repaired"} and qc2.get("normalized_output")
        )

    def run_with_reserves(
        self, main_cases: Sequence[StoryCase], reserve_cases: Sequence[StoryCase], *, target_pairs: int = 12,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        records = self.run_cases(main_cases)
        selected = [case.case_id for case in main_cases if self.valid_blind_pair(records, case.case_id)]
        for case in reserve_cases:
            if len(selected) >= target_pairs:
                break
            records = self.run_cases([case])
            if self.valid_blind_pair(records, case.case_id):
                selected.append(case.case_id)
        if len(selected) < target_pairs:
            raise RuntimeError(
                f"Only {len(selected)} valid Creative blind pairs remain after all frozen reserves; "
                "the experiment is incomplete and will not select new cases post hoc"
            )
        return records, selected[:target_pairs]


def _visible_story(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("normalized_output") or {}
    blocks: list[str] = []
    opening = str(payload.get("opening") or "").strip()
    if opening:
        blocks.append(opening)
    paragraphs = payload.get("paragraphs", [])
    transitions = payload.get("creative_transitions", [])
    for index, paragraph in enumerate(paragraphs if isinstance(paragraphs, list) else []):
        if isinstance(paragraph, Mapping) and str(paragraph.get("text") or "").strip():
            blocks.append(str(paragraph["text"]).strip())
        if isinstance(transitions, list) and index < len(transitions) and str(transitions[index] or "").strip():
            blocks.append(str(transitions[index]).strip())
    closing = str(payload.get("closing") or "").strip()
    if closing:
        blocks.append(closing)
    if not blocks:
        blocks = [
            str(item.get("text") or "").strip()
            for item in paragraphs if isinstance(item, Mapping) and str(item.get("text") or "").strip()
        ] if isinstance(paragraphs, list) else []
    return {"title": str(payload.get("title") or "").strip(), "paragraphs": blocks}


def _blind_order(track: str, case_id: str, left: str, right: str) -> tuple[str, str]:
    parity = int(hashlib.sha256(f"{BLIND_PROTOCOL}\0{track}\0{case_id}".encode("utf-8")).hexdigest(), 16) % 2
    return (left, right) if parity == 0 else (right, left)


def build_dual_blind_files(
    faithful_records: Sequence[Mapping[str, Any]],
    creative_records: Sequence[Mapping[str, Any]],
    faithful_contexts: Mapping[str, StoryContext],
    creative_contexts: Mapping[str, StoryContext],
    creative_pair_ids: Sequence[str],
    *,
    packet_path: str | Path = BLIND_PACKET_PATH,
    mapping_path: str | Path = BLIND_MAPPING_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    faithful_by = {(str(item["case_id"]), str(item["variant_id"])): item for item in faithful_records}
    creative_by = {(str(item["case_id"]), str(item["variant_id"])): item for item in creative_records}
    faithful_ids = sorted(faithful_contexts)
    if len(faithful_ids) != 12 or not 1 <= len(creative_pair_ids) <= 12:
        raise ValueError("Dual blind review requires 12 Faithful and 1--12 frozen Creative pairs")

    entries: list[tuple[str, str, Mapping[str, StoryContext], Mapping[tuple[str, str], Mapping[str, Any]], str, str]] = []
    entries.extend(("faithful", case_id, faithful_contexts, faithful_by, "QF0", "QF3") for case_id in faithful_ids)
    entries.extend(("creative", case_id, creative_contexts, creative_by, "QC0", "QC2") for case_id in creative_pair_ids)
    random.Random(CASE_SELECTION_SEED).shuffle(entries)

    packet_cases: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for review_index, (track, case_id, contexts, records, left, right) in enumerate(entries, 1):
        if (case_id, left) not in records or (case_id, right) not in records:
            raise ValueError(f"Missing blind pair records for {track}/{case_id}")
        if track == "creative" and not all(
            records[(case_id, variant)].get("normalized_output") for variant in (left, right)
        ):
            raise ValueError("Creative blind packet cannot contain an empty or rejected story")
        order = _blind_order(track, case_id, left, right)
        context = contexts[case_id]
        token = canonical_hash(f"{BLIND_PROTOCOL}\0{track}\0{case_id}")[:20]
        dimensions = (
            ["coherence", "informativeness", "evidence_consistency"]
            if track == "faithful" else
            ["coherence", "personalization", "trustworthiness"]
        )
        packet_cases.append({
            "review_index": review_index,
            "case_token": token,
            "track": track,
            "language": records[(case_id, left)]["language"],
            "source_kind": records[(case_id, left)]["source_kind"],
            "dimensions": dimensions,
            "photo_paths": [item.relative_path for item in context.evidence],
            "story_a": _visible_story(records[(case_id, order[0])]),
            "story_b": _visible_story(records[(case_id, order[1])]),
        })
        mappings.append({
            "case_token": token, "case_id": case_id, "track": track,
            "A": order[0], "B": order[1],
        })
    achieved_count = len(packet_cases)
    packet = {
        "protocol_version": BLIND_PROTOCOL,
        "status": "complete" if len(creative_pair_ids) == 12 else "incomplete",
        "case_count": achieved_count,
        "target_case_count": 24,
        "track_counts": {"faithful": 12, "creative": len(creative_pair_ids)},
        "target_track_counts": {"faithful": 12, "creative": 12},
        "incomplete_reason": (
            None if len(creative_pair_ids) == 12 else
            "Frozen reserves were exhausted before 12 readable Creative QC0/QC2 pairs were available."
        ),
        "cases": packet_cases,
    }
    mapping = {"protocol_version": BLIND_PROTOCOL + "-mapping", "mappings": mappings}
    atomic_write_json(packet_path, packet)
    atomic_write_json(mapping_path, mapping)
    return packet, mapping


def reveal_dual_blind(
    responses: Mapping[str, Any], mapping: Mapping[str, Any],
) -> dict[str, Any]:
    if responses.get("status") != "locked":
        raise ValueError("All available blind responses must be complete and locked before reveal")
    lookup = {str(item["case_token"]): item for item in mapping.get("mappings", [])}
    response_items = list(responses.get("responses", []))
    if len(response_items) != len(lookup) or len({str(item.get("case_token")) for item in response_items}) != len(lookup):
        raise ValueError("Blind response set is incomplete or duplicated")
    tracks: dict[str, dict[str, Any]] = {}
    revealed_private: list[dict[str, Any]] = []
    for item in response_items:
        token = str(item.get("case_token") or "")
        choice = str(item.get("choice") or "")
        if token not in lookup or choice not in {"A", "B", "tie"}:
            raise ValueError("Blind response contains an unknown case or choice")
        link = lookup[token]
        track = str(link["track"])
        selected = "tie" if choice == "tie" else str(link[choice])
        track_value = tracks.setdefault(
            track, {"wins": Counter(), "ties": 0, "scores": {}, "paired_differences": {}},
        )
        if selected == "tie":
            track_value["ties"] += 1
        else:
            track_value["wins"][selected] += 1
        anonymous_scores = item.get("scores", {})
        scores_by_variant = {
            str(link[side]): dict(anonymous_scores.get(side, {}))
            for side in ("A", "B")
        }
        for variant, values in scores_by_variant.items():
            for dimension, score in values.items():
                track_value["scores"].setdefault(variant, {}).setdefault(dimension, []).append(float(score))
        ordered_variants = sorted(scores_by_variant)
        if len(ordered_variants) == 2:
            baseline, intervention = ordered_variants
            contrast = f"{intervention}_minus_{baseline}"
            shared_dimensions = set(scores_by_variant[baseline]) & set(scores_by_variant[intervention])
            for dimension in shared_dimensions:
                difference = (
                    float(scores_by_variant[intervention][dimension])
                    - float(scores_by_variant[baseline][dimension])
                )
                track_value["paired_differences"].setdefault(contrast, {}).setdefault(
                    dimension, [],
                ).append(difference)
        revealed_private.append({
            "case_id": link["case_id"], "track": track, "preferred_variant": selected,
            "scores": scores_by_variant, "reason": str(item.get("reason") or ""),
        })
    public: dict[str, Any] = {}
    for track, value in tracks.items():
        variants = sorted(value["scores"])
        wins = {variant: int(value["wins"].get(variant, 0)) for variant in variants}
        nonzero = [wins[variant] for variant in variants]
        public[track] = {
            "wins": wins,
            "ties": int(value["ties"]),
            "mean_scores": {
                variant: {
                    dimension: sum(scores) / len(scores)
                    for dimension, scores in dimensions.items()
                }
                for variant, dimensions in value["scores"].items()
            },
            "paired_score_differences": {
                contrast: {
                    dimension: _bootstrap_paired_values(differences)
                    for dimension, differences in dimensions.items()
                }
                for contrast, dimensions in value["paired_differences"].items()
            },
            "exact_binomial_two_sided": exact_two_sided_binomial(nonzero[0], nonzero[1]) if len(nonzero) == 2 else None,
        }
    return {"public": public, "private": revealed_private}


def _bootstrap_paired_values(values: Sequence[float], *, iterations: int = 10_000) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"n": 0, "mean_difference": None, "ci_low": None, "ci_high": None}
    rng = random.Random(BOOTSTRAP_SEED)
    sample_size = len(numeric)
    bootstrap_means = sorted(
        sum(rng.choice(numeric) for _ in range(sample_size)) / sample_size
        for _ in range(iterations)
    )
    return {
        "n": sample_size,
        "mean_difference": sum(numeric) / sample_size,
        "ci_low": bootstrap_means[int(0.025 * iterations)],
        "ci_high": bootstrap_means[min(iterations - 1, int(0.975 * iterations))],
    }


def build_claim_audit_packet(
    records: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, StoryContext],
    *,
    output_path: str | Path = CLAIM_AUDIT_PATH,
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    for record in records:
        payload = record.get("normalized_output")
        if not isinstance(payload, Mapping):
            continue
        context = contexts[str(record["case_id"])]
        fields: list[tuple[str, str, list[str]]] = []
        for name in ("title", "opening"):
            text = str(payload.get(name) or "").strip()
            if text:
                fields.append((name, text, []))
        for index, paragraph in enumerate(payload.get("paragraphs", []), 1):
            if isinstance(paragraph, Mapping):
                evidence_ids = [str(value) for value in paragraph.get("evidence_ids", [])]
                fields.append((f"paragraph_{index}", str(paragraph.get("text") or ""), evidence_ids))
        for index, transition in enumerate(payload.get("creative_transitions", []), 1):
            fields.append((f"transition_{index}", str(transition or ""), []))
        closing = str(payload.get("closing") or "").strip()
        if closing:
            fields.append(("closing", closing, []))
        evidence_lookup = {item.evidence_id: item for item in context.evidence}
        for field, text, evidence_ids in fields:
            audited_ids = evidence_ids or list(evidence_lookup)
            for unit_index, claim in enumerate(_split_text_units(text), 1):
                key = f"{record['case_id']}\0{record['variant_id']}\0{field}\0{unit_index}"
                claims.append({
                    "audit_id": canonical_hash("qwen-claim-audit-v1\0" + key)[:24],
                    "case_id": record["case_id"],
                    "variant_id": record["variant_id"],
                    "field": field,
                    "claim": claim,
                    "evidence_ids": evidence_ids,
                    "audited_evidence_ids": audited_ids,
                    "evidence": [{
                        "evidence_id": evidence_id,
                        "caption": evidence_lookup[evidence_id].caption,
                        "scene_graph_triples": list(evidence_lookup[evidence_id].scene_graph_triples),
                        "tags": list(evidence_lookup[evidence_id].tags),
                        "timestamp": evidence_lookup[evidence_id].timestamp,
                        "timestamp_confidence": evidence_lookup[evidence_id].timestamp_confidence,
                        "location": evidence_lookup[evidence_id].location,
                    } for evidence_id in audited_ids if evidence_id in evidence_lookup],
                    "status": None,
                    "reason": "",
                    "source": "agent_evidence_audit",
                })
    random.Random(BOOTSTRAP_SEED).shuffle(claims)
    packet = {
        "protocol_version": "story-qwen-claim-audit-v1",
        "allowed_statuses": ["supported", "unsupported", "uncertain", "non_factual"],
        "claims": claims,
    }
    atomic_write_json(output_path, packet)
    return packet


def validate_claim_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = set(payload.get("allowed_statuses", []))
    claims = list(payload.get("claims", []))
    if not claims:
        raise ValueError("Claim audit contains no claims")
    if any(item.get("status") not in allowed or not str(item.get("reason") or "").strip() for item in claims):
        raise ValueError("Every claim must have an allowed status and a non-empty audit reason")
    return _summarize_claims(claims)


def _summarize_claims(claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Counter[str]] = {}
    for item in claims:
        by_variant.setdefault(str(item["variant_id"]), Counter())[str(item["status"])] += 1
    return {
        variant: {
            **dict(counts),
            "factual_units": counts["supported"] + counts["unsupported"] + counts["uncertain"],
            "unsupported_rate": (
                counts["unsupported"] / (counts["supported"] + counts["unsupported"] + counts["uncertain"])
                if counts["supported"] + counts["unsupported"] + counts["uncertain"] else None
            ),
        }
        for variant, counts in by_variant.items()
    }


def _variant_summary(records: Sequence[Mapping[str, Any]], variants: Sequence[VariantSpec]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in variants:
        selected = [item for item in records if item.get("variant_id") == variant.variant_id]
        result[variant.variant_id] = {
            "count": len(selected),
            "prompt_version": variant.prompt_version,
            "status_counts": dict(Counter(str(item.get("status")) for item in selected)),
            "automatic_metric_means": aggregate_metrics([
                item.get("automatic_metrics", {}) for item in selected
            ]) if selected else {},
        }
    return result


def build_public_summary(
    faithful_records: Sequence[Mapping[str, Any]],
    creative_records: Sequence[Mapping[str, Any]],
    creative_manifest: Mapping[str, Any],
    creative_pair_ids: Sequence[str],
    *,
    claim_audit: Optional[Mapping[str, Any]] = None,
    blind_reveal: Optional[Mapping[str, Any]] = None,
    output_path: str | Path = PUBLIC_SUMMARY_PATH,
) -> dict[str, Any]:
    audit_summary = validate_claim_audit(claim_audit) if claim_audit is not None else None
    creative_main_ids = {
        str(item["case_id"]) for item in creative_manifest.get("main_cases", [])
        if isinstance(item, Mapping) and item.get("case_id")
    }
    creative_primary_records = [
        item for item in creative_records if str(item.get("case_id")) in creative_main_ids
    ]
    creative_reserve_records = [
        item for item in creative_records if str(item.get("case_id")) not in creative_main_ids
    ]
    if claim_audit is not None:
        claims = list(claim_audit.get("claims", []))
        claim_audit_public = {
            "faithful_primary": _summarize_claims([
                item for item in claims if str(item.get("variant_id", "")).startswith("QF")
            ]),
            "creative_primary": _summarize_claims([
                item for item in claims if str(item.get("case_id")) in creative_main_ids
            ]),
            "creative_all_generated": {
                key: value for key, value in audit_summary.items() if key.startswith("QC")
            },
        }
    else:
        claim_audit_public = None
    summary = {
        "protocol_version": "story-qwen-dual-public-summary-v1",
        "model_name": QWEN_MODEL,
        "model_digest": QWEN_DIGEST,
        "generation": {
            "temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "think": False,
        },
        "faithful": {
            "primary_case_count": 12,
            "record_count": len(faithful_records),
            "variants": _variant_summary(faithful_records, FAITHFUL_VARIANTS),
            "paired_bootstrap": {
                "QF1_to_QF2_duplicate_rate": paired_bootstrap_difference(
                    faithful_records, "QF1", "QF2", "duplicate_text_unit_rate", seed=BOOTSTRAP_SEED,
                ),
                "QF2_to_QF3_language_compliance": paired_bootstrap_difference(
                    faithful_records, "QF2", "QF3", "language_compliant", seed=BOOTSTRAP_SEED,
                ),
            },
        },
        "creative": {
            "primary_case_count": 12,
            "reserve_case_count": 6,
            "generated_record_count": len(creative_records),
            "primary_record_count": len(creative_primary_records),
            "reserve_generated_record_count": len(creative_reserve_records),
            "quality_blind_pair_count": len(creative_pair_ids),
            "quality_blind_pair_target": 12,
            "quality_blind_pair_status": "complete" if len(creative_pair_ids) == 12 else "incomplete",
            "primary_quotas": {"single": 2, "event": 8, "date": 2, "zh": 6, "en": 6},
            "variants": _variant_summary(creative_primary_records, CREATIVE_VARIANTS),
            "reserve_variants": _variant_summary(creative_reserve_records, CREATIVE_VARIANTS),
            "paired_bootstrap": {
                "QC0_to_QC1_duplicate_rate": paired_bootstrap_difference(
                    creative_primary_records, "QC0", "QC1", "duplicate_text_unit_rate", seed=BOOTSTRAP_SEED,
                ),
                "QC1_to_QC2_language_compliance": paired_bootstrap_difference(
                    creative_primary_records, "QC1", "QC2", "language_compliant", seed=BOOTSTRAP_SEED,
                ),
                "QC1_to_QC2_structure_complete": paired_bootstrap_difference(
                    creative_primary_records, "QC1", "QC2", "structure_complete", seed=BOOTSTRAP_SEED,
                ),
            },
        },
        "claim_audit": claim_audit_public,
        "blind_review": blind_reveal.get("public") if blind_reveal is not None else {"status": "pending"},
        "private_hashes": {
            "faithful_results_sha256": canonical_hash(list(faithful_records)),
            "creative_results_sha256": canonical_hash(list(creative_records)),
            "creative_case_manifest_sha256": creative_manifest.get("freeze_sha256"),
        },
        "private_content_committed": False,
        "interpretation": (
            "Qwen-only personal-album case study. Historical Llama outputs are not part of this analysis; "
            "single-user blind preferences do not represent a population-level user study."
        ),
    }
    atomic_write_json(output_path, summary)
    return summary


def privacy_scan_public_summary(payload: Mapping[str, Any]) -> list[str]:
    forbidden_keys = {
        "photo_id", "photo_ids", "relative_path", "photo_paths", "event_id", "case_id",
        "title", "content", "paragraphs", "raw_attempts", "normalized_output", "mapping",
    }
    errors: list[str] = []

    def walk(value: Any, path: str = "$") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in forbidden_keys:
                    errors.append(f"forbidden key {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload)
    raw = json.dumps(payload, ensure_ascii=False)
    if re.search(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", raw):
        errors.append("UUID found in public summary")
    if re.search(r"(?i)(?:[A-Z]:\\|\.jpe?g\b|\.png\b|/photos/)", raw):
        errors.append("local path or image filename found in public summary")
    return errors


__all__ = [
    "BLIND_MAPPING_PATH", "BLIND_PACKET_PATH", "BLIND_RESPONSES_PATH", "CLAIM_AUDIT_PATH",
    "CREATIVE_CASES_PATH", "CREATIVE_RESULTS_PATH", "FAITHFUL_RESULTS_PATH", "PUBLIC_SUMMARY_PATH",
    "QWEN_DIGEST", "QWEN_MODEL", "QwenCreativeRunner", "QwenFaithfulRunner", "QwenOllamaBackend",
    "assert_qwen_digest", "build_claim_audit_packet", "build_dual_blind_files", "build_public_summary",
    "default_context_factory", "freeze_creative_cases", "load_creative_case_manifest",
    "load_faithful_cases", "privacy_scan_public_summary", "reveal_dual_blind", "validate_claim_audit",
]
