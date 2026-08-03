"""Synthetic known-item query generation + blinded audit for the retrieval test.

Protocol (per target photo):

1. **Generate** one English query from the target image only (Qwen3.7, temp 0,
   fixed seed). The query is 5--20 words with >=2 observable distinctive cues
   and must not leak filenames / UUIDs / timestamps / places / captions /
   triples / ranking, nor assert identity / emotion / intent / relations.
2. **Select distractors**: the 4 CLIP-most-similar photos from the 1,599-photo
   corpus (the panel is the target + these 4).
3. **Audit**: shuffle the 5 images anonymously and ask Qwen3.7 to pick the best
   match for the query by display index, and to judge query compliance.
4. **One repair**: if the audit did not pick the target OR the query broke a
   rule, exactly one repair attempt is allowed (regenerate + re-audit). A second
   failure is recorded as ``query_generation_error``; the target is NOT replaced.

The benchmark is honestly reported in the paper as a synthetic known-item set
generated and self-audited by the fixed vision LLM -- never as human queries.

The module is decoupled from Chroma and the real LLM: ``vector_store`` only needs
``get_image_embeddings`` + ``query_images`` (duck-typed), and the LLM is the
``EvalLLMClient`` from Layer 1, so the whole flow is unit-testable with fakes.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

from eval_llm import EvalLLMClient

QUERY_MIN_WORDS = 5
QUERY_MAX_WORDS = 20
DISTRACTOR_COUNT = 4
MAX_ATTEMPTS = 2  # one initial try + one repair

# Four-or-more consecutive digits look like a year / timestamp / file number.
_NUMERIC_TOKEN_RE = re.compile(r"\d{4,}")
# A run of 8+ hex chars looks like a UUID / content hash prefix.
_UUIDISH_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
_FILE_EXT_RE = re.compile(r"\.(jpe?g|png|bmp|webp|tiff?|heic)$", re.IGNORECASE)

QUERY_GEN_SYSTEM = (
    "You are a strict known-item retrieval benchmark author. You write exactly "
    "ONE English search query for a single target photo. The query must let a "
    "human or vision model pick THIS exact photo out of very similar images. "
    "Rules: 5-20 English words; at least two observable, distinctive visual "
    "cues; describe ONLY what is visibly in the frame; do NOT mention filenames, "
    "IDs, UUIDs, timestamps, dates, places, captions, scene-graph triples, or "
    "retrieval ranking; do NOT assert identity, emotion, intent, purpose, or "
    "relationships you cannot see. Return JSON {\"query\": str, \"cues\": [str, ...]}."
)
AUDIT_SYSTEM = (
    "You are a strict blinded auditor. You see 5 photos at display indices 0-4 "
    "and ONE query. Choose the SINGLE photo that best matches the query by its "
    "display index. Then judge whether the query follows the rules (5-20 English "
    "words, >=2 observable cues, no filenames/IDs/timestamps/places/captions/"
    "triples, no unsupported identity/emotion/intent/relations). Return JSON "
    "{\"chosen_index\": int, \"compliance_ok\": bool, \"issues\": str}."
)


class VectorProtocol(Protocol):
    """Duck-typed slice of ChromaVectorStore used for distractor selection."""

    def get_image_embeddings(self, photo_ids: Sequence[str]) -> dict: ...
    def query_images(
        self, embedding: Sequence[float], *, top_k: int, eligible_photo_ids=None
    ) -> list: ...


@dataclass
class QueryGenOutcome:
    """Result of generate-and-audit for one target."""

    target_id: str
    status: str  # "ok" | "query_generation_error"
    query: Optional[str] = None
    cues: list[str] = field(default_factory=list)
    panel: list[str] = field(default_factory=list)            # display order
    target_display_index: Optional[int] = None
    chosen_index: Optional[int] = None
    compliance_ok: bool = False
    attempts: int = 0
    repair_used: bool = False
    issues: str = ""


# ============================ Compliance (deterministic) ============================


def query_word_count(query: str) -> int:
    return len([w for w in query.split() if any(ch.isalnum() for ch in w)])


def rule_violations(query: str) -> list[str]:
    """Deterministic, conservative compliance checks (a safety net over the
    model's own ``compliance_ok``). Returns a list of human-readable violations."""
    issues: list[str] = []
    n = query_word_count(query)
    if n < QUERY_MIN_WORDS or n > QUERY_MAX_WORDS:
        issues.append(f"word count {n} outside [{QUERY_MIN_WORDS},{QUERY_MAX_WORDS}]")
    if _NUMERIC_TOKEN_RE.search(query):
        issues.append("contains a 4+ digit run (looks like a year/timestamp/id)")
    if _UUIDISH_RE.search(query):
        issues.append("contains a UUID/hash-like token")
    if _FILE_EXT_RE.search(query.strip()):
        issues.append("contains a file extension")
    return issues


def is_compliant(query: str, model_ok: bool) -> bool:
    """Compliant only when the model says so AND no deterministic rule breaks."""
    return bool(model_ok) and not rule_violations(query)


# ============================ Distractor selection + panel ============================


def select_distractors(
    target_id: str,
    vector_store: VectorProtocol,
    corpus_ids: Sequence[str],
    *,
    n: int = DISTRACTOR_COUNT,
) -> list[str]:
    """The n CLIP-most-similar corpus photos to the target, excluding the target."""
    vecs = vector_store.get_image_embeddings([target_id])
    if target_id not in vecs:
        raise ValueError(f"no CLIP vector for target {target_id!r}")
    hits = vector_store.query_images(
        vecs[target_id], top_k=n + 1, eligible_photo_ids=corpus_ids
    )
    distractors = [h.photo_id for h in hits if h.photo_id != target_id][:n]
    if len(distractors) < n:
        raise ValueError(
            f"only {len(distractors)} distractors available for {target_id!r} "
            f"(need {n})"
        )
    return distractors


def build_panel(
    target_id: str, distractors: Sequence[str], shuffle_seed: int
) -> tuple[list[str], int]:
    """Shuffle [target, *distractors] deterministically.

    Returns (panel_ids_in_display_order, target_display_index).
    """
    panel = [target_id, *distractors]
    rng = random.Random(shuffle_seed)
    rng.shuffle(panel)
    return panel, panel.index(target_id)


# ============================ LLM steps ============================


def generate_query(
    client: EvalLLMClient, target_image: str, *, seed: int
) -> dict:
    """One query-generation call. Returns {query, cues}."""
    payload = client.complete_json(
        QUERY_GEN_SYSTEM,
        "Write the query for this target photo.",
        images=[target_image],
        seed=seed,
    )
    if "query" not in payload or not isinstance(payload["query"], str):
        raise ValueError(f"query generation returned no query string: {payload!r}")
    return {"query": payload["query"].strip(), "cues": list(payload.get("cues") or [])}


def audit_panel(
    client: EvalLLMClient,
    panel_images: Sequence[str],
    query: str,
    *,
    seed: int,
) -> dict:
    """One blinded audit call. Returns {chosen_index, compliance_ok, issues}."""
    user = (
        f"Query: \"{query}\"\n\nPick the single photo (display index 0-"
        f"{len(panel_images) - 1}) that best matches this query, and audit the "
        "query's compliance."
    )
    payload = client.complete_json(
        AUDIT_SYSTEM, user, images=list(panel_images), seed=seed
    )
    if "chosen_index" not in payload:
        raise ValueError(f"audit returned no chosen_index: {payload!r}")
    return {
        "chosen_index": int(payload["chosen_index"]),
        "compliance_ok": bool(payload.get("compliance_ok", True)),
        "issues": str(payload.get("issues") or ""),
    }


# ============================ Orchestrator ============================


def generate_and_audit(
    target_id: str,
    image_path_of: Callable[[str], str],
    vector_store: VectorProtocol,
    corpus_ids: Sequence[str],
    client: EvalLLMClient,
    *,
    base_seed: int,
) -> QueryGenOutcome:
    """Full generate -> select -> audit -> (one repair) flow for one target.

    ``image_path_of(photo_id)`` resolves every panel image (target + 4 CLIP
    distractors) to a real file path for the blinded audit. This is the single
    production entry point and is fully testable: tests inject a fake vector
    store, a fake ``EvalLLMClient`` (via the openai module patch), and a trivial
    ``image_path_of``.
    """
    target_image = image_path_of(target_id)
    distractors = select_distractors(target_id, vector_store, corpus_ids)
    outcome = QueryGenOutcome(target_id=target_id, status="query_generation_error")

    for attempt in range(MAX_ATTEMPTS):
        outcome.attempts = attempt + 1
        outcome.repair_used = attempt > 0
        gen_seed = base_seed + attempt * 10
        audit_seed = base_seed + attempt * 10 + 1
        shuffle_seed = base_seed + attempt * 10 + 2

        try:
            generated = generate_query(client, target_image, seed=gen_seed)
        except Exception:  # noqa: BLE001 - LLM/parse failure counts as an attempt
            outcome.issues = "query generation call failed"
            continue
        panel, target_idx = build_panel(target_id, distractors, shuffle_seed)
        panel_images = [image_path_of(pid) for pid in panel]
        try:
            audited = audit_panel(client, panel_images, generated["query"], seed=audit_seed)
        except Exception:  # noqa: BLE001
            outcome.query, outcome.cues = generated["query"], generated["cues"]
            outcome.panel, outcome.target_display_index = panel, target_idx
            outcome.issues = "audit call failed"
            continue

        chosen = audited["chosen_index"]
        compliant = is_compliant(generated["query"], audited["compliance_ok"])
        outcome.query, outcome.cues = generated["query"], generated["cues"]
        outcome.panel, outcome.target_display_index = panel, target_idx
        outcome.chosen_index = chosen
        outcome.compliance_ok = compliant
        outcome.issues = audited.get("issues", "")
        if chosen == target_idx and compliant:
            outcome.status = "ok"
            return outcome
        outcome.issues = (
            f"attempt {attempt + 1}: chosen={chosen} target={target_idx} "
            f"compliant={compliant}"
        )
    return outcome


__all__ = [
    "AUDIT_SYSTEM",
    "DISTRACTOR_COUNT",
    "MAX_ATTEMPTS",
    "QUERY_GEN_SYSTEM",
    "QUERY_MAX_WORDS",
    "QUERY_MIN_WORDS",
    "QueryGenOutcome",
    "audit_panel",
    "build_panel",
    "generate_and_audit",
    "generate_query",
    "is_compliant",
    "query_word_count",
    "rule_violations",
    "select_distractors",
]
