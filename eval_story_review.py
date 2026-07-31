"""Qwen3.7 blinded review of the three Event-1 stories.

Two structured comparisons (NOT a "best overall" ranking), each over the SAME
event so the judge sees the 12 photos, the reliable metadata, and the
photographer's human-confirmed Mood labels:

* **S-F vs S-C0** -- coherence, evidence consistency, first-person voice,
  imaginative content, unsupported claims.
* **S-C0 vs S-C1** -- coherence, personalization, Mood usage, Mood credibility,
  whether it invents a cause for an emotion.

The two stories are anonymized to A/B and deterministically shuffled, so the
judge never knows which is Faithful / Creative / Mood-on. Each dimension is
scored 1-5 per story; claims are audited into supported / unsupported / creative
/ non_factual; a short diff note is required.

Because there is only one event, the paper treats the output as a qualitative
case study (no significance test). The reviewer runs on the EVAL config
(Qwen3.7, temperature 0, fixed seed, non-thinking) -- never on the production
qwen3.5-27b that generated the stories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from eval_llm import EvalLLMClient
from eval_story_case import anonymize_pair, story_text

# The two comparisons. ``dimensions`` are scored 1-5 for each of A and B.
COMPARISON_SF_SC0 = {
    "id": "SF_vs_SC0",
    "keys": ("S-F", "S-C0"),
    "dimensions": (
        "coherence",                 # narrative holds together
        "evidence_consistency",      # only describes what the photos support
        "first_person",              # natural first-person observer voice
        "imaginative_content",       # reflective / atmospheric layer
        "unsupported_claims",        # lower is better (1=few, 5=many)
    ),
}
COMPARISON_SC0_SC1 = {
    "id": "SC0_vs_SC1",
    "keys": ("S-C0", "S-C1"),
    "dimensions": (
        "coherence",                 # narrative holds together
        "personalization",           # feels like a specific person's memory
        "mood_usage",                # Mood woven in naturally (not stapled on)
        "mood_credibility",          # Mood fits the scene as a self-recall
        "invents_emotion_cause",     # lower is better (1=none, 5=fabricates causes)
    ),
}
COMPARISONS = (COMPARISON_SF_SC0, COMPARISON_SC0_SC1)

REVIEW_SYSTEM = (
    "You are a strict, BLINDED reviewer of personal photo narratives. You see "
    "12 photos from one event (in capture order), the event's reliable metadata "
    "(date, time range, location), and the photographer's self-reported Mood "
    "labels. Mood is the photographer's own recollection at the moment they "
    "pressed the shutter -- it is NOT the emotion of anyone in the frame, and a "
    "story must never invent a reason for a mood it cannot see. You are given "
    "two stories labeled A and B (their real identities are hidden). Score each "
    "story 1-5 on every dimension. Then audit each story's claims into "
    "supported / unsupported / creative / non_factual lists. Return JSON: "
    '{"scores": {"A": {dim: int}, "B": {dim: int}}, '
    '"claim_audit": {"A": {"supported": [...], "unsupported": [...], '
    '"creative": [...], "non_factual": [...]}, "B": {...}}, '
    '"diff_note": "<=40 words on the key difference"}.'
)


@dataclass
class ReviewInput:
    images: Sequence[str]                 # 12 event photo paths, capture order
    metadata: Mapping[str, Any]           # reliable: date, time_range, location
    moods: Mapping[str, str]              # photo_id -> mood label


def _user_prompt(comparison: dict, anon_texts: Mapping[str, str],
                 metadata: Mapping[str, Any], moods: Mapping[str, str]) -> str:
    dims = ", ".join(comparison["dimensions"])
    mood_lines = "\n".join(f"  - {pid}: {label}" for pid, label in moods.items()) or "  (none)"
    return (
        f"Event metadata: date={metadata.get('date','')}, "
        f"time_range={metadata.get('time_range','')}, "
        f"location={metadata.get('location','')}.\n"
        f"Photographer Mood labels:\n{mood_lines}\n\n"
        f"Dimensions to score (1-5 each, for BOTH A and B): {dims}.\n\n"
        f"Story A:\n{anon_texts['A']}\n\nStory B:\n{anon_texts['B']}\n\n"
        "Score both stories on every dimension, audit each story's claims, and "
        "give a short diff note. Return JSON only."
    )


def review_pair(
    client: EvalLLMClient,
    comparison: dict,
    stories_by_key: Mapping[str, Any],
    review_input: ReviewInput,
    *,
    seed: int,
) -> dict:
    """Run one blinded A/B comparison. Returns scores/audits mapped back to the
    real story keys, plus the anonymization mapping and the raw judge payload."""
    key_a, key_b = comparison["keys"]
    anon = anonymize_pair(key_a, key_b, story_text(stories_by_key[key_a]),
                           story_text(stories_by_key[key_b]), seed=seed)
    user = _user_prompt(comparison, anon["texts"], review_input.metadata, review_input.moods)
    payload = client.complete_json(
        REVIEW_SYSTEM, user, images=list(review_input.images), seed=seed,
    )
    # Remap the judge's A/B labels back to the real story keys.
    mapping = anon["mapping"]  # {label: real_key}
    scores = {mapping[label]: dict(vals) for label, vals in payload.get("scores", {}).items()}
    audit = {mapping[label]: dict(vals) for label, vals in payload.get("claim_audit", {}).items()}
    return {
        "comparison_id": comparison["id"],
        "keys": comparison["keys"],
        "dimensions": list(comparison["dimensions"]),
        "scores": scores,
        "claim_audit": audit,
        "diff_note": payload.get("diff_note", ""),
        "anonymization": anon["labels"],
        "raw": payload,
    }


def review_all(
    client: EvalLLMClient,
    stories_by_key: Mapping[str, Any],
    review_input: ReviewInput,
    *,
    base_seed: int,
) -> list[dict]:
    """Run both structured comparisons (S-F vs S-C0, then S-C0 vs S-C1)."""
    out = []
    for i, comparison in enumerate(COMPARISONS):
        out.append(review_pair(client, comparison, stories_by_key, review_input,
                                seed=base_seed + i))
    return out


__all__ = [
    "COMPARISONS",
    "COMPARISON_SC0_SC1",
    "COMPARISON_SF_SC0",
    "REVIEW_SYSTEM",
    "ReviewInput",
    "review_all",
    "review_pair",
]
