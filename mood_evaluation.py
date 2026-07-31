"""Private, reproducible M0/M1 photographer-mood case-study utilities.

M0 and M1 share the same Creative v6 system prompt, photos, model settings and
seed.  The only input difference is whether the frozen manual photographer
mood records are exposed in the user context.  Full stories and identifiers
remain under ``evaluation/private``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import (
    APP_DB_PATH,
    MOOD_EXPERIMENT_ANNOTATION_SET,
    MOOD_EXPERIMENT_EVENT_ID,
    MOOD_EXPERIMENT_SEED,
    OLLAMA_MODEL,
    OLLAMA_STORY_NUM_CTX,
    OLLAMA_STORY_NUM_PREDICT,
    OLLAMA_THINK,
)
from storage import PhotoStorage
from story_agent import (
    GroundingValidator,
    PhotographerMoodPromptBuilder,
    StoryGenerator,
)


ROOT = Path(__file__).resolve().parent
PRIVATE_DIR = ROOT / "evaluation" / "private"
MANIFEST_PATH = PRIVATE_DIR / "mood_m0_m1_v3_manifest.json"
RESULTS_PATH = PRIVATE_DIR / "mood_m0_m1_v3_results.json"
BLIND_PACKET_PATH = PRIVATE_DIR / "mood_m0_m1_v3_blind_packet.json"
BLIND_MAPPING_PATH = PRIVATE_DIR / "mood_m0_m1_v3_blind_mapping.json"
BLIND_RESPONSES_PATH = PRIVATE_DIR / "mood_m0_m1_v3_blind_responses.json"
PUBLIC_SUMMARY_PATH = ROOT / "evaluation" / "mood_m0_m1_v3_summary.json"
CASE_ID = "hot_air_balloon_mood_case_v3"
TEMPERATURE = 0.25


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _ordered_event_photo_ids(storage: PhotoStorage) -> list[str]:
    ids = storage.event_photo_ids(MOOD_EXPERIMENT_EVENT_ID)
    if len(ids) != 5:
        raise RuntimeError(
            f"Frozen mood event must contain exactly 5 photos; found {len(ids)}"
        )
    return ids


def build_or_validate_manifest(
    storage: PhotoStorage,
    path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    photo_ids = _ordered_event_photo_ids(storage)
    moods = storage.get_photo_moods(photo_ids)
    missing = [photo_id for photo_id in photo_ids if photo_id not in moods]
    if missing:
        raise RuntimeError(
            f"All five photos require a manual mood before freezing; missing {len(missing)}"
        )
    records = []
    for photo_id in photo_ids:
        photo = storage.get_photo(photo_id)
        mood = moods[photo_id]
        if mood.annotation_set != MOOD_EXPERIMENT_ANNOTATION_SET:
            raise RuntimeError(
                f"Photo {photo_id} uses annotation_set={mood.annotation_set!r}, expected "
                f"{MOOD_EXPERIMENT_ANNOTATION_SET!r}"
            )
        records.append({
            "photo_id": photo_id,
            "content_sha256": str(photo["content_sha256"]),
            "mood_label": mood.mood_label,
            "source": mood.source,
            "subject_role": mood.subject_role,
            "confirmed": mood.confirmed,
        })
    frozen = {
        "protocol_version": "mood-m0-m1-v3",
        "case_id": CASE_ID,
        "event_id": MOOD_EXPERIMENT_EVENT_ID,
        "annotation_set": MOOD_EXPERIMENT_ANNOTATION_SET,
        "records": records,
    }
    frozen["freeze_sha256"] = sha256_json(frozen)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != frozen:
            raise RuntimeError("Frozen mood manifest differs from current database; refusing to overwrite")
        return existing
    atomic_write_json(path, frozen)
    return frozen


def ollama_model_digest(model: str = OLLAMA_MODEL) -> str:
    import ollama

    response = ollama.list()
    items = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
    for item in items:
        name = getattr(item, "model", None) or (item.get("model") if isinstance(item, dict) else None)
        if name == model or str(name).startswith(model + ":"):
            digest = getattr(item, "digest", None) or (item.get("digest") if isinstance(item, dict) else None)
            if digest:
                return str(digest)
    raise RuntimeError(f"Ollama model {model!r} is unavailable or has no digest")


def _context_hash(context: Any) -> str:
    payload = {
        "label": context.label,
        "event_id": context.event_id,
        "source_kind": context.source_kind,
        "narrator_role": context.narrator_role,
        "use_photographer_mood": context.use_photographer_mood,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "photo_id": item.photo_id,
                "timestamp": item.timestamp,
                "location": item.location,
                "caption": item.caption,
                "scene_graph_triples": list(item.scene_graph_triples),
                "tags": list(item.tags),
                "photographer_mood": item.photographer_mood if context.use_photographer_mood else None,
            }
            for item in context.evidence
        ],
        "groups": [group.to_dict() for group in context.groups],
    }
    return sha256_json(payload)


def _repetition_rate(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return 0.0
    pairs = 0
    repeated = 0
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            pairs += 1
            if GroundingValidator._paragraph_similarity(texts[left], texts[right]) > 0.80:
                repeated += 1
    return repeated / pairs if pairs else 0.0


def automatic_metrics(story: Any, provided_moods: Sequence[str]) -> dict[str, Any]:
    claim_labels = {
        str(label)
        for claim in story.mood_claims
        for label in claim.get("mood_labels", [])
    }
    unique_provided = set(provided_moods)
    texts = [str(block.get("text") or "") for block in story.narrative_blocks]
    return {
        "status": story.status,
        "repair_count": max(0, len(story.raw_attempts) - 1),
        "repetition_rate": _repetition_rate(texts),
        "mood_coverage": (
            len(claim_labels & unique_provided) / len(unique_provided)
            if unique_provided else None
        ),
        "unsupported_mood_claim_count": sum(
            1 for code in story.validation_codes if str(code).startswith("E_UNSUPPORTED_MOOD")
        ),
        "mood_claim_count": len(story.mood_claims),
    }


def build_blind_files(records: Sequence[Mapping[str, Any]], photo_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_variant = {str(record["variant_id"]): record for record in records}
    order = ("M0", "M1")
    if int(hashlib.sha256((CASE_ID + "\0blind").encode("utf-8")).hexdigest(), 16) % 2:
        order = tuple(reversed(order))
    labels = {"A": order[0], "B": order[1]}
    packet = {
        "protocol_version": "mood-blind-review-v3",
        "case_id": CASE_ID,
        "locked": False,
        "photo_ids": list(photo_ids),
        "stories": {
            display: {
                "title": by_variant[variant]["normalized_output"]["title"],
                "content": by_variant[variant]["content"],
            }
            for display, variant in labels.items()
        },
    }
    mapping = {
        "protocol_version": "mood-blind-mapping-v3",
        "case_id": CASE_ID,
        "mapping": labels,
        "packet_sha256": sha256_json(packet),
    }
    return packet, mapping


def run_m0_m1(
    *,
    database_path: Path = APP_DB_PATH,
    results_path: Path = RESULTS_PATH,
    generator: StoryGenerator | None = None,
) -> dict[str, Any]:
    if results_path.exists():
        return json.loads(results_path.read_text(encoding="utf-8"))
    storage = PhotoStorage(database_path)
    manifest = build_or_validate_manifest(storage)
    digest = ollama_model_digest(OLLAMA_MODEL)
    story_generator = generator or StoryGenerator(storage=storage)
    photo_ids = [str(record["photo_id"]) for record in manifest["records"]]
    provided_moods = [str(record["mood_label"]) for record in manifest["records"]]

    contexts = {
        variant: story_generator.aggregator.aggregate_by_photo_ids(
            photo_ids,
            label="2026-07-07 · Event 1",
            event_id=MOOD_EXPERIMENT_EVENT_ID,
            source_kind="event",
            narrator_role="observer",
            use_photographer_mood=variant == "M1",
        )
        for variant in ("M0", "M1")
    }
    prompt_parts = {
        variant: PhotographerMoodPromptBuilder.build_story_prompt(context, mode="creative", language="zh")
        for variant, context in contexts.items()
    }
    system_hashes = {
        hashlib.sha256(parts[0].encode("utf-8")).hexdigest() for parts in prompt_parts.values()
    }
    if len(system_hashes) != 1:
        raise RuntimeError("M0 and M1 system prompts differ; experiment isolation failed")

    records: list[dict[str, Any]] = []
    for variant in ("M0", "M1"):
        started = time.perf_counter()
        story = story_generator.generate_story_with_context(
            contexts[variant],
            temperature=TEMPERATURE,
            mode="creative",
            language="zh",
            save=False,
            seed=MOOD_EXPERIMENT_SEED,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        output = story.to_dict()
        records.append({
            "case_id": CASE_ID,
            "variant_id": variant,
            "model_name": OLLAMA_MODEL,
            "model_digest": digest,
            "seed": MOOD_EXPERIMENT_SEED,
            "temperature": TEMPERATURE,
            "think": OLLAMA_THINK,
            "num_ctx": OLLAMA_STORY_NUM_CTX,
            "num_predict": OLLAMA_STORY_NUM_PREDICT,
            "prompt_version": PhotographerMoodPromptBuilder.PROMPT_VERSION,
            "system_prompt_hash": next(iter(system_hashes)),
            "prompt_hash": story.prompt_hash,
            "context_hash": _context_hash(contexts[variant]),
            "output_hash": sha256_json(output),
            "raw_attempts": list(story.raw_attempts),
            "normalized_output": output,
            "content": story.content,
            "status": story.status,
            "validation_codes": list(story.validation_codes),
            "latency_ms": latency_ms,
            "automatic_metrics": automatic_metrics(
                story, provided_moods if variant == "M1" else []
            ),
        })
    result = {
        "protocol_version": "mood-m0-m1-v3",
        "case_id": CASE_ID,
        "manifest_sha256": manifest["freeze_sha256"],
        "records": records,
    }
    result["results_sha256"] = sha256_json(result)
    atomic_write_json(results_path, result)
    packet, mapping = build_blind_files(records, photo_ids)
    atomic_write_json(BLIND_PACKET_PATH, packet)
    atomic_write_json(BLIND_MAPPING_PATH, mapping)
    return result


def build_public_summary(
    responses: dict[str, Any],
    mapping: dict[str, Any],
    results: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not responses.get("locked"):
        raise RuntimeError("Blind review is not complete and locked")
    choice = str(responses["preference"])
    preferred_variant = "tie" if choice == "Tie" else mapping["mapping"][choice]
    records = results["records"]
    anonymous_scores = responses.get("scores", {})
    scores_by_variant = {
        variant_id: anonymous_scores[display_id]
        for display_id, variant_id in mapping["mapping"].items()
        if display_id in anonymous_scores
    }
    summary = {
        "protocol_version": "mood-m0-m1-public-summary-v3",
        "case_count": 1,
        "photo_count": 5,
        "mood_label_distribution": {
            label: sum(1 for record in manifest["records"] if record["mood_label"] == label)
            for label in sorted({record["mood_label"] for record in manifest["records"]})
        },
        "automatic_metrics": {
            record["variant_id"]: record["automatic_metrics"] for record in records
        },
        "latency_ms": {record["variant_id"]: record["latency_ms"] for record in records},
        "blind_preference": preferred_variant,
        "blind_scores": scores_by_variant,
        "private_results_sha256": results["results_sha256"],
        "private_content_committed": False,
        "interpretation": "Single-case exploratory comparison; no statistical or general performance claim.",
    }
    return summary


def reveal_and_publish(
    responses_path: Path = BLIND_RESPONSES_PATH,
    mapping_path: Path = BLIND_MAPPING_PATH,
    results_path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    summary = build_public_summary(responses, mapping, results, manifest)
    atomic_write_json(PUBLIC_SUMMARY_PATH, summary)
    return summary


__all__ = [
    "BLIND_MAPPING_PATH", "BLIND_PACKET_PATH", "BLIND_RESPONSES_PATH",
    "CASE_ID", "MANIFEST_PATH", "PUBLIC_SUMMARY_PATH", "RESULTS_PATH",
    "automatic_metrics", "build_blind_files", "build_or_validate_manifest",
    "build_public_summary",
    "reveal_and_publish", "run_m0_m1", "sha256_json",
]
