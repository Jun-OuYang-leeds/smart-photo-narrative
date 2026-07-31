"""Single-event Story case study (Event 1) for the dissertation.

A qualitative deep study of the CURRENT system on one fixed event that
previously suffered prompt leakage under an old local-Qwen fallback. Because
there is only one event, the paper reports this as a qualitative case study --
no significance test, no general-population claim.

Fixed case:

* event_id ``9fb2f2af-ebb4-5f2b-8e7b-d380b898f043`` ("2025-12-18 · Event 1")
* 12 photos, 12:19--16:25, English output, narrator_role = observer,
  verified_context = empty. All 12 have BLIP2 + Scene Graph.

Three stories, identical evidence and fixed generation conditions, differing in
exactly one variable each:

    S-F   Faithful, photographer Mood FORCED OFF
    S-C0  Creative observer, Mood OFF
    S-C1  Creative observer, Mood ON (human-confirmed set)

Generation is over the PRODUCTION remote backend (qwen3.5-27b, temperature 0.25,
max_tokens 2048, fixed seed), with save=False (the production Stories table is
never written) and no fallback / no re-sampling. The S-C0 vs S-C1 contrast
isolates Mood as the single experimental variable; Faithful never sees Mood.

The Story review (Qwen3.7, eval config) lives in :mod:`eval_story_review`.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from config import PHOTOS_DIR, STORY_DEFAULT_TEMPERATURE

EVENT_ID = "9fb2f2af-ebb4-5f2b-8e7b-d380b898f043"
DEFAULT_MOOD_SET = "paris_20251218_mood_v1"
STORY_LANGUAGE = "en"
STORY_MAX_TOKENS = 2048
STORY_TEMPERATURE = STORY_DEFAULT_TEMPERATURE  # 0.25

# (key, mode, use_photographer_mood). S-C0 vs S-C1 differ ONLY in use_mood.
VARIANTS: tuple[tuple[str, str, bool], ...] = (
    ("S-F", "faithful", False),
    ("S-C0", "creative", False),
    ("S-C1", "creative", True),
)


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


# ============================ Freeze ============================


@dataclass
class FrozenEvent:
    event_id: str
    photo_ids: list[str]                       # ordered by event position
    photo_hashes: dict[str, str]               # photo_id -> sha256 of the image file
    order_hash: str                            # sha256 of the ordered id list


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_event(
    conn: sqlite3.Connection,
    event_id: str = EVENT_ID,
    *,
    photos_dir: Path = PHOTOS_DIR,
) -> FrozenEvent:
    """Freeze the event's ordered photo list and per-photo SHA-256.

    Read-only: queries event_photos + photos and hashes the image files; it
    never mutates the database. ``photos_dir`` is injected for tests.
    """
    rows = conn.execute(
        "SELECT ep.photo_id, p.relative_path "
        "FROM event_photos ep JOIN photos p ON p.photo_id = ep.photo_id "
        "WHERE ep.event_id = ? ORDER BY ep.position, ep.photo_id",
        (event_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"event {event_id!r} has no photos")
    photo_ids = [str(r[0]) for r in rows]
    photo_hashes: dict[str, str] = {}
    for photo_id, relative_path in rows:
        path = photos_dir / str(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"photo file missing for {photo_id}: {path}")
        photo_hashes[photo_id] = _sha256_file(path)
    order_hash = hashlib.sha256("\n".join(photo_ids).encode("utf-8")).hexdigest()
    return FrozenEvent(event_id=event_id, photo_ids=photo_ids,
                       photo_hashes=photo_hashes, order_hash=order_hash)


# ============================ Mood validation ============================


class _MoodStorage(Protocol):
    def get_photo_moods(self, photo_ids: Sequence[str]) -> dict: ...


def validate_event_moods(
    storage: _MoodStorage,
    photo_ids: Sequence[str],
    mood_set: str = DEFAULT_MOOD_SET,
) -> dict[str, str]:
    """Every event photo must carry a photographer mood in ``mood_set``.

    Mirrors the existing mood-experiment contract: a photo with no mood, or a
    mood from a different annotation_set, is a hard error (the human must finish
    labeling first). Returns ``{photo_id: mood_label}``.
    """
    moods = storage.get_photo_moods(list(photo_ids))
    out: dict[str, str] = {}
    for photo_id in photo_ids:
        mood = moods.get(photo_id)
        if mood is None:
            raise ValueError(
                f"photo {photo_id} has no photographer mood; label all event "
                f"photos in annotation_set {mood_set!r} first"
            )
        if getattr(mood, "annotation_set", None) != mood_set:
            raise ValueError(
                f"photo {photo_id} mood annotation_set="
                f"{getattr(mood, 'annotation_set', None)!r}, expected {mood_set!r}"
            )
        out[photo_id] = getattr(mood, "mood_label", None) or ""
    return out


# ============================ Contexts ============================


class _Aggregator(Protocol):
    def aggregate_by_event(
        self, event_id: str, *, verified_context: str = "",
        narrator_role: str = "observer", use_photographer_mood: bool = False,
    ) -> Any: ...


def build_contexts(
    aggregator: _Aggregator,
    event_id: str = EVENT_ID,
    *,
    narrator_role: str = "observer",
) -> dict[str, Any]:
    """Build the three variant contexts. S-C0 and S-C1 differ ONLY in
    ``use_photographer_mood``; Faithful is forced off Mood regardless."""
    contexts: dict[str, Any] = {}
    for key, _mode, use_mood in VARIANTS:
        contexts[key] = aggregator.aggregate_by_event(
            event_id, verified_context="", narrator_role=narrator_role,
            use_photographer_mood=use_mood,
        )
    return contexts


# ============================ Generation ============================


class _StoryGenerator(Protocol):
    def generate_story_with_context(
        self, context: Any, temperature: float = ..., *,
        mode: str = ..., language: str = ..., save: bool = ...,
        seed: Optional[int] = ..., allow_deterministic_fallback: bool = ...,
        num_predict: Optional[int] = ...,
    ) -> Any: ...


def generate_variant(
    story_generator: _StoryGenerator,
    context: Any,
    mode: str,
    *,
    seed: int,
) -> Any:
    """Generate one story: save=False (no production write), no fallback, no
    re-sampling, fixed conditions. At most one repair is handled inside the
    generator's 2-attempt loop; a second failure surfaces as status='error'."""
    return story_generator.generate_story_with_context(
        context,
        temperature=STORY_TEMPERATURE,
        mode=mode,
        language=STORY_LANGUAGE,
        save=False,
        seed=seed,
        allow_deterministic_fallback=False,
        num_predict=STORY_MAX_TOKENS,
    )


def generate_three(
    story_generator: _StoryGenerator,
    aggregator: _Aggregator,
    *,
    event_id: str = EVENT_ID,
    base_seed: int,
) -> dict[str, Any]:
    """Generate S-F, S-C0, S-C1 under identical, fixed conditions.

    The seed is derived deterministically per variant from ``base_seed`` so the
    run is reproducible. Mood validation (that the human set is complete) is the
    caller's responsibility -- do it before calling this.
    """
    contexts = build_contexts(aggregator, event_id)
    stories: dict[str, Any] = {}
    for key, mode, _use_mood in VARIANTS:
        stories[key] = generate_variant(
            story_generator, contexts[key], mode, seed=base_seed + _stable_int(key),
        )
    return stories


# ============================ Anonymization (for review) ============================


def anonymize_pair(
    key_a: str, key_b: str, text_a: str, text_b: str, *, seed: int
) -> dict:
    """Deterministically assign A/B labels to two stories for blinded review.

    Returns ``{"labels": {key: "A"|"B"}, "texts": {"A": ..., "B": ...},
    "mapping": {"A": key, "B": key}}``. The same seed always yields the same
    assignment, so the review is reproducible while the judge sees only A/B.
    """
    import random
    rng = random.Random(seed)
    flip = rng.random() < 0.5
    label_a, label_b = ("A", "B") if not flip else ("B", "A")
    return {
        "labels": {key_a: label_a, key_b: label_b},
        "texts": {label_a: text_a, label_b: text_b},
        "mapping": {label_a: key_a, label_b: key_b},
    }


def story_text(story: Any) -> str:
    """Best-effort flat text of a GeneratedStory for the reviewer.

    Includes the deterministic ``mood_reflections`` layer (the verified
    photographer-mood sentences the app appends after generation) so the
    reviewer actually sees the Mood difference between S-C0 and S-C1.
    """
    parts = []
    title = getattr(story, "title", None)
    if title:
        parts.append(str(title))
    for para in getattr(story, "paragraphs", []) or []:
        parts.append(getattr(para, "text", str(para)))
    for tr in getattr(story, "creative_transitions", []) or []:
        parts.append(str(tr))
    opening = getattr(story, "opening", None)
    closing = getattr(story, "closing", None)
    if opening:
        parts.append(str(opening))
    if closing:
        parts.append(str(closing))
    for refl in getattr(story, "mood_reflections", []) or []:
        text = refl.get("text") if isinstance(refl, dict) else str(refl)
        if text:
            parts.append(str(text))
    return "\n\n".join(parts).strip()


__all__ = [
    "DEFAULT_MOOD_SET",
    "EVENT_ID",
    "FrozenEvent",
    "STORY_LANGUAGE",
    "STORY_MAX_TOKENS",
    "STORY_TEMPERATURE",
    "VARIANTS",
    "anonymize_pair",
    "build_contexts",
    "freeze_event",
    "generate_three",
    "generate_variant",
    "story_text",
    "validate_event_moods",
]
