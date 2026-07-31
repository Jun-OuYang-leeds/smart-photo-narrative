"""Event-level, evidence-grounded Story v3 for personal photo collections."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from config import (
    APP_DB_PATH,
    OLLAMA_MODEL,
    OLLAMA_STORY_NUM_CTX,
    OLLAMA_STORY_NUM_PREDICT,
    OLLAMA_THINK,
    REMOTE_LLM_BACKENDS,
    REMOTE_LLM_BASE_URL,
    REMOTE_LLM_DEFAULT_BACKEND,
    REMOTE_LLM_MODEL,
    REMOTE_LLM_NUM_PREDICT,
    REMOTE_LLM_TIMEOUT,
    STORY_DEFAULT_TEMPERATURE,
    STORY_FAITHFUL_SPECULATION_BAN,
    STORY_GROUP_CLIP_THRESHOLD,
    STORY_GROUP_JACCARD_THRESHOLD,
    STORY_GROUP_MAX_GAP_SECONDS,
    STORY_MAX_EVIDENCE_GROUPS,
    STORY_MAX_PHOTOS,
    STORY_MAX_WORDS,
    STORY_NARRATIVE_MAX_PARAGRAPHS,
    STORY_NEAR_DUPLICATE_CLIP_THRESHOLD,
)
from retrieval_engine import MultimodalRetriever, get_multimodal_retriever
from storage import PhotoStorage


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.I)
_GENERIC_TOKENS = {
    "a", "an", "and", "at", "in", "is", "of", "on", "the", "to", "with",
    "image", "photo", "picture", "person", "people", "man", "woman", "someone",
    "show", "shows", "standing", "sitting", "holding", "有", "一", "个", "人", "照片", "图像",
}

MOOD_LABEL_DISPLAY: dict[str, dict[str, str]] = {
    "neutral": {"zh": "中性", "en": "neutral"},
    "calm": {"zh": "平静", "en": "calm"},
    "happy": {"zh": "开心", "en": "happy"},
    "excited": {"zh": "兴奋", "en": "excited"},
    "tense": {"zh": "紧张", "en": "tense"},
    "sad": {"zh": "难过", "en": "sad"},
}


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text or "") if token.casefold() not in _GENERIC_TOKENS}


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(left: Any, right: Any) -> float:
    if left is None or right is None:
        return -1.0
    try:
        a = [float(value) for value in left]
        b = [float(value) for value in right]
        if len(a) != len(b) or not a:
            return -1.0
        denominator = math.sqrt(sum(value * value for value in a)) * math.sqrt(sum(value * value for value in b))
        return sum(x * y for x, y in zip(a, b)) / denominator if denominator else -1.0
    except (TypeError, ValueError):
        return -1.0


def _timestamp_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    photo_id: str
    relative_path: str
    timestamp: Optional[str]
    timestamp_source: str
    timestamp_confidence: float
    location: Optional[str]
    caption: str
    scene_graph_triples: tuple[str, ...]
    tags: tuple[str, ...]
    user_note: str = ""  # legacy, unverified; personal facts must use verified_context
    photographer_mood: Optional[str] = None
    mood_source: Optional[str] = None
    mood_confirmed: bool = False
    mood_annotation_set: Optional[str] = None

    def evidence_text(self) -> str:
        return " ".join(filter(None, [
            self.caption, " ".join(self.scene_graph_triples), self.location or "",
        ]))

    def observation_record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "blip_caption_model_observation": self.caption or None,
            "qwen_scene_graph_model_observations": list(self.scene_graph_triples),
        }

    def mood_record(self) -> dict[str, Any] | None:
        if not self.photographer_mood or not self.mood_confirmed:
            return None
        return {
            "evidence_id": self.evidence_id,
            "photo_id": self.photo_id,
            "mood_label": self.photographer_mood,
            "source": self.mood_source,
            "subject_role": "photographer",
            "confirmed": True,
            "annotation_set": self.mood_annotation_set,
        }


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    evidence_ids: tuple[str, ...]
    photo_ids: tuple[str, ...]
    start_at: Optional[str]
    end_at: Optional[str]
    locations: tuple[str, ...]
    model_observations: tuple[Mapping[str, Any], ...]
    conflicts: tuple[str, ...]
    representative_evidence_id: str
    near_duplicate_evidence_ids: tuple[str, ...] = ()
    source_event_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "evidence_ids": list(self.evidence_ids),
            "photo_ids": list(self.photo_ids),
            "start_at": self.start_at,
            "end_at": self.end_at,
            "locations": list(self.locations),
            "model_observations": [dict(value) for value in self.model_observations],
            "conflicts": list(self.conflicts),
            "representative_evidence_id": self.representative_evidence_id,
            "near_duplicate_evidence_ids": list(self.near_duplicate_evidence_ids),
            "source_event_id": self.source_event_id,
        }


_PERSON_OBSERVATION_TERMS = (
    "person", "people", "man", "woman", "boy", "girl", "child", "adult",
    "人物", "人", "男人", "女人", "男性", "女性", "男孩", "女孩", "儿童",
)


def _group_has_person_observation(group: EvidenceGroup) -> bool:
    """Return whether fallible model observations contain visible-person evidence."""
    text = " ".join(
        str(value)
        for observation in group.model_observations
        for value in observation.values()
        if value
    ).casefold()
    return any(term.casefold() in text for term in _PERSON_OBSERVATION_TERMS)


_PERSON_OBSERVATION_TERMS = (
    "person", "people", "man", "woman", "boy", "girl", "child", "adult",
    "人物", "人", "男人", "女人", "男性", "女性", "男孩", "女孩", "儿童",
)


def _group_has_person_observation(group: EvidenceGroup) -> bool:
    """Return whether fallible model observations contain visible-person evidence."""
    text = " ".join(
        str(value)
        for observation in group.model_observations
        for value in observation.values()
        if value
    ).casefold()
    return any(term.casefold() in text for term in _PERSON_OBSERVATION_TERMS)


@dataclass(frozen=True)
class StoryContext:
    label: str
    evidence: tuple[EvidenceItem, ...]
    groups: tuple[EvidenceGroup, ...] = ()
    event_id: Optional[str] = None
    source_kind: str = "basket"
    verified_context: str = ""
    narrator_role: str = "observer"
    use_photographer_mood: bool = False

    @property
    def date(self) -> str:
        return self.label

    @property
    def photo_count(self) -> int:
        return len(self.evidence)

    @property
    def locations(self) -> list[str]:
        return list(dict.fromkeys(item.location for item in self.evidence if item.location))

    @property
    def photo_descriptions(self) -> list[dict[str, Any]]:
        return [{
            "id": item.relative_path, "photo_id": item.photo_id,
            "time": item.timestamp[11:16] if item.timestamp and len(item.timestamp) >= 16 else "Unknown",
            "caption": item.caption, "location": item.location,
            "scene_graph": list(item.scene_graph_triples),
        } for item in self.evidence]

    @property
    def photographer_moods(self) -> list[dict[str, Any]]:
        if not self.use_photographer_mood:
            return []
        return [record for item in self.evidence if (record := item.mood_record()) is not None]

    def to_prompt_context(self) -> str:
        payload = {
            "label": self.label,
            "source_kind": self.source_kind,
            "verified_context": self.verified_context or None,
            "narrator_role": self.narrator_role,
            "evidence_groups": [group.to_dict() for group in self.groups],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

@dataclass(frozen=True)
class StoryParagraph:
    text: str
    evidence_ids: tuple[str, ...]
    group_id: str = ""


@dataclass(frozen=True)
class UncertainObservation:
    text: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_ids": list(self.evidence_ids)}


@dataclass
class GeneratedStory:
    date: str
    title: str
    paragraphs: list[StoryParagraph]
    photo_count: int
    photo_ids: list[str]
    model: str
    mode: str = "faithful"
    language: str = "zh"
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    uncertain_observations: list[UncertainObservation] = field(default_factory=list)
    unused_evidence_ids: list[str] = field(default_factory=list)
    creative_transitions: list[str] = field(default_factory=list)
    opening: str = ""
    closing: str = ""
    event_summary: dict[str, Any] = field(default_factory=dict)
    narrator_role: str = "observer"
    prompt_version: str = ""
    validation_codes: list[str] = field(default_factory=list)
    prompt_hash: str = ""
    story_id: Optional[str] = None
    use_photographer_mood: bool = False
    photographer_moods: list[dict[str, Any]] = field(default_factory=list)
    mood_claims: list[dict[str, Any]] = field(default_factory=list)
    mood_reflections: list[dict[str, Any]] = field(default_factory=list)
    raw_attempts: list[str] = field(default_factory=list, repr=False)
    # One entry per generation attempt: {"attempt", "phase", "done_reason", "codes"}.
    # Lets the UI report the concrete per-attempt failure reason instead of a
    # low-information deterministic template.
    attempt_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def content(self) -> str:
        if self.opening or self.closing:
            return "\n\n".join(
                str(block["text"]).strip() for block in self.narrative_blocks if str(block.get("text") or "").strip()
            )
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)

    @property
    def narrative_blocks(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if self.opening:
            blocks.append({"kind": "creative_opening", "text": self.opening})
        for index, paragraph in enumerate(self.paragraphs):
            blocks.append({
                "kind": "grounded_fact",
                "text": paragraph.text,
                "group_id": paragraph.group_id,
                "evidence_ids": list(paragraph.evidence_ids),
            })
            for reflection in self.mood_reflections:
                if str(reflection.get("group_id") or "") == paragraph.group_id:
                    blocks.append({
                        "kind": "verified_photographer_mood",
                        "text": str(reflection.get("text") or ""),
                        "group_id": paragraph.group_id,
                        "evidence_id": reflection.get("evidence_id"),
                        "mood_label": reflection.get("mood_label"),
                        "subject_role": "photographer",
                    })
            if index < len(self.creative_transitions):
                blocks.append({
                    "kind": "creative_transition",
                    "text": self.creative_transitions[index],
                    "after_group_id": paragraph.group_id,
                    "before_group_id": (
                        self.paragraphs[index + 1].group_id if index + 1 < len(self.paragraphs) else None
                    ),
                })
        if self.closing:
            blocks.append({"kind": "creative_closing", "text": self.closing})
        return blocks

    @property
    def citations(self) -> dict[str, list[str]]:
        return {str(index + 1): list(paragraph.evidence_ids) for index, paragraph in enumerate(self.paragraphs)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "paragraphs": [{
                "text": item.text, "evidence_ids": list(item.evidence_ids), "group_id": item.group_id,
            } for item in self.paragraphs],
            "uncertain_observations": [item.to_dict() for item in self.uncertain_observations],
            "unused_evidence_ids": list(self.unused_evidence_ids),
            "creative_transitions": list(self.creative_transitions),
            "opening": self.opening,
            "closing": self.closing,
            "event_summary": dict(self.event_summary),
            "narrator_role": self.narrator_role,
            "prompt_version": self.prompt_version,
            "narrative_blocks": self.narrative_blocks,
            "use_photographer_mood": self.use_photographer_mood,
            "photographer_moods": list(self.photographer_moods),
            "mood_claims": list(self.mood_claims),
            "mood_reflections": list(self.mood_reflections),
        }


class ContextAggregator:
    def __init__(
        self,
        retriever: Optional[MultimodalRetriever] = None,
        storage: Optional[PhotoStorage] = None,
    ):
        self.retriever = retriever or get_multimodal_retriever()
        self.storage = storage or self.retriever.storage

    @staticmethod
    def _conflicts(item: EvidenceItem) -> tuple[str, ...]:
        caption_tokens = _tokens(item.caption)
        graph_text = " ".join(item.scene_graph_triples)
        graph_tokens = _tokens(graph_text)
        if len(caption_tokens) < 2 or len(graph_tokens) < 2:
            return ()
        # Low lexical overlap is not itself a contradiction: BLIP and Qwen can
        # describe different, compatible parts of one image. Only explicit,
        # auditable mutually-exclusive concepts are separated from the body.
        conflict_pairs = (
            ({"dish", "rack"}, {"cage"}),
            ({"toilet"}, {"spice"}),
            ({"dog"}, {"cat"}),
            ({"indoor"}, {"outdoor"}),
            ({"day"}, {"night"}),
        )
        explicit = any(
            (left <= caption_tokens and right <= graph_tokens)
            or (right <= caption_tokens and left <= graph_tokens)
            for left, right in conflict_pairs
        )
        if explicit:
            text = (
                f"{item.evidence_id}: BLIP and Qwen observations conflict or have no reliable lexical agreement; "
                f"BLIP={item.caption!r}; Qwen={graph_text!r}."
            )
            return (text,)
        return ()

    def _embeddings(self, photo_ids: Sequence[str]) -> dict[str, Any]:
        store = getattr(self.retriever, "vector_store", None)
        getter = getattr(store, "get_image_embeddings", None)
        if getter is None:
            return {}
        try:
            return dict(getter(photo_ids))
        except Exception:
            return {}

    @staticmethod
    def _adjacent_match(previous: EvidenceItem, current: EvidenceItem, embeddings: Mapping[str, Any]) -> tuple[bool, float]:
        left_time, right_time = _timestamp_seconds(previous.timestamp), _timestamp_seconds(current.timestamp)
        close_in_time = (
            left_time is not None and right_time is not None
            and abs(right_time - left_time) <= STORY_GROUP_MAX_GAP_SECONDS
        )
        clip = _cosine(embeddings.get(previous.photo_id), embeddings.get(current.photo_id))
        lexical = _jaccard(previous.evidence_text(), current.evidence_text())
        return close_in_time and (
            clip >= STORY_GROUP_CLIP_THRESHOLD or lexical >= STORY_GROUP_JACCARD_THRESHOLD
        ), clip

    def _make_group(
        self,
        items: Sequence[EvidenceItem],
        embeddings: Mapping[str, Any],
        *,
        source_event_id: Optional[str] = None,
    ) -> EvidenceGroup:
        observations: list[Mapping[str, Any]] = []
        duplicates: list[str] = []
        conflicts: list[str] = []
        previous: Optional[EvidenceItem] = None
        for item in items:
            near_duplicate = False
            if previous is not None:
                near_duplicate = _cosine(
                    embeddings.get(previous.photo_id), embeddings.get(item.photo_id)
                ) >= STORY_NEAR_DUPLICATE_CLIP_THRESHOLD
            if near_duplicate:
                duplicates.append(item.evidence_id)
            else:
                observations.append(item.observation_record())
            conflicts.extend(self._conflicts(item))
            previous = item
        timestamps = [item.timestamp for item in items if item.timestamp]
        return EvidenceGroup(
            group_id="",
            evidence_ids=tuple(item.evidence_id for item in items),
            photo_ids=tuple(item.photo_id for item in items),
            start_at=min(timestamps) if timestamps else None,
            end_at=max(timestamps) if timestamps else None,
            locations=tuple(dict.fromkeys(item.location for item in items if item.location)),
            model_observations=tuple(observations),
            conflicts=tuple(dict.fromkeys(conflicts)),
            representative_evidence_id=items[0].evidence_id,
            near_duplicate_evidence_ids=tuple(duplicates),
            source_event_id=source_event_id,
        )

    @staticmethod
    def _merge_groups(left: EvidenceGroup, right: EvidenceGroup) -> EvidenceGroup:
        return EvidenceGroup(
            group_id="",
            evidence_ids=left.evidence_ids + right.evidence_ids,
            photo_ids=left.photo_ids + right.photo_ids,
            start_at=left.start_at or right.start_at,
            end_at=right.end_at or left.end_at,
            locations=tuple(dict.fromkeys(left.locations + right.locations)),
            model_observations=left.model_observations + right.model_observations,
            conflicts=tuple(dict.fromkeys(left.conflicts + right.conflicts)),
            representative_evidence_id=left.representative_evidence_id,
            near_duplicate_evidence_ids=left.near_duplicate_evidence_ids + right.near_duplicate_evidence_ids,
            source_event_id=left.source_event_id if left.source_event_id == right.source_event_id else None,
        )

    def _build_groups(
        self,
        evidence: Sequence[EvidenceItem],
        *,
        source_kind: str,
    ) -> tuple[EvidenceGroup, ...]:
        if not evidence:
            return ()
        by_photo = {item.photo_id: item for item in evidence}
        embeddings = self._embeddings(list(by_photo))
        if source_kind == "date":
            raw_partitions = self.storage.event_partitions_for_photos(list(by_photo))
        else:
            raw_partitions = [{"event_id": None, "photo_ids": list(by_photo)}]

        groups: list[EvidenceGroup] = []
        for partition in raw_partitions:
            items = [by_photo[photo_id] for photo_id in partition["photo_ids"] if photo_id in by_photo]
            if not items:
                continue
            current = [items[0]]
            for item in items[1:]:
                matches, _ = self._adjacent_match(current[-1], item, embeddings)
                if matches:
                    current.append(item)
                else:
                    groups.append(self._make_group(current, embeddings, source_event_id=partition.get("event_id")))
                    current = [item]
            groups.append(self._make_group(current, embeddings, source_event_id=partition.get("event_id")))

        while len(groups) > STORY_MAX_EVIDENCE_GROUPS:
            scores: list[tuple[float, int]] = []
            for index in range(len(groups) - 1):
                left_item = by_photo[groups[index].photo_ids[-1]]
                right_item = by_photo[groups[index + 1].photo_ids[0]]
                clip = _cosine(embeddings.get(left_item.photo_id), embeddings.get(right_item.photo_id))
                lexical = _jaccard(left_item.evidence_text(), right_item.evidence_text())
                left_time, right_time = _timestamp_seconds(left_item.timestamp), _timestamp_seconds(right_item.timestamp)
                time_score = 0.0 if left_time is None or right_time is None else 1.0 / (1.0 + abs(right_time - left_time))
                scores.append((max(clip, lexical) + time_score, index))
            _, merge_at = max(scores, key=lambda value: (value[0], -value[1]))
            groups[merge_at:merge_at + 2] = [self._merge_groups(groups[merge_at], groups[merge_at + 1])]

        return tuple(EvidenceGroup(
            group_id=f"G{index:03d}", evidence_ids=group.evidence_ids, photo_ids=group.photo_ids,
            start_at=group.start_at, end_at=group.end_at, locations=group.locations,
            model_observations=group.model_observations, conflicts=group.conflicts,
            representative_evidence_id=group.representative_evidence_id,
            near_duplicate_evidence_ids=group.near_duplicate_evidence_ids,
            source_event_id=group.source_event_id,
        ) for index, group in enumerate(groups, 1))

    def aggregate_by_photo_ids(
        self,
        photo_ids: Sequence[str],
        *,
        label: str = "Selected photos",
        event_id: Optional[str] = None,
        user_notes: Optional[dict[str, str]] = None,
        verified_context: str = "",
        source_kind: str = "basket",
        narrator_role: str = "observer",
        use_photographer_mood: bool = False,
    ) -> StoryContext:
        if narrator_role not in {"observer", "confirmed_subject"}:
            raise ValueError("narrator_role must be 'observer' or 'confirmed_subject'")
        unique_ids = list(dict.fromkeys(photo_ids))[:STORY_MAX_PHOTOS]
        docs = self.retriever._documents(unique_ids)
        moods = self.storage.get_photo_moods(unique_ids)
        evidence: list[EvidenceItem] = []
        for index, photo_id in enumerate(unique_ids, 1):
            doc = docs.get(photo_id)
            if not doc:
                continue
            mood = moods.get(photo_id)
            evidence.append(EvidenceItem(
                evidence_id=f"P{index:03d}", photo_id=photo_id, relative_path=doc["relative_path"],
                timestamp=doc["captured_at"], timestamp_source=doc["timestamp_source"],
                timestamp_confidence=float(doc["timestamp_confidence"]), location=doc["location"],
                caption=doc["caption"], scene_graph_triples=tuple(doc.get("triples", ())),
                tags=(),
                user_note=(user_notes or {}).get(photo_id, ""),
                photographer_mood=mood.mood_label if mood is not None else None,
                mood_source=mood.source if mood is not None else None,
                mood_confirmed=mood.confirmed if mood is not None else False,
                mood_annotation_set=mood.annotation_set if mood is not None else None,
            ))
        groups = self._build_groups(evidence, source_kind=source_kind)
        return StoryContext(
            label=label, evidence=tuple(evidence), groups=groups, event_id=event_id,
            source_kind=source_kind, verified_context=verified_context.strip(), narrator_role=narrator_role,
            use_photographer_mood=bool(use_photographer_mood),
        )

    def aggregate_by_date(
        self, date: str, *, verified_context: str = "", narrator_role: str = "observer",
        use_photographer_mood: bool = False,
    ) -> StoryContext:
        ids = self.storage.metadata_eligible_ids(date_from=date, date_to=date)
        return self.aggregate_by_photo_ids(
            ids, label=date, verified_context=verified_context, source_kind="date", narrator_role=narrator_role,
            use_photographer_mood=use_photographer_mood,
        )

    def aggregate_by_event(
        self, event_id: str, *, verified_context: str = "", narrator_role: str = "observer",
        use_photographer_mood: bool = False,
    ) -> StoryContext:
        ids = self.storage.metadata_eligible_ids(event_id=event_id)
        return self.aggregate_by_photo_ids(
            ids, label=event_id, event_id=event_id, verified_context=verified_context,
            source_kind="event", narrator_role=narrator_role,
            use_photographer_mood=use_photographer_mood,
        )


class PromptBuilder:
    PROMPT_VERSION = "grounded-story-v3-event"

    @classmethod
    def build_story_prompt(
        cls,
        context: StoryContext,
        *,
        mode: str = "faithful",
        language: str = "zh",
    ) -> tuple[str, str]:
        """Grouped faithful contract: the model writes only ``factual_text`` per
        group; the application deterministically injects every evidence ID,
        uncertain observation and unused-evidence accounting via
        :meth:`normalise_payload`. The neutral third-person style is soft
        guidance now that the production speculation ban is config-gated."""
        del mode
        if not context.evidence or not context.groups:
            raise ValueError("No photo evidence is available")
        language_name = "Chinese" if language.startswith("zh") else "English"
        group_count = len(context.groups)
        paragraph_rule = (
            "Write one concise factual_text entry for the single evidence group."
            if context.photo_count == 1
            else f"Write exactly {group_count} factual_text entries, one for each evidence group, in the supplied order."
        )
        system = f"""You write event-level, evidence-grounded personal photo narratives in {language_name}.
Return only the JSON required by the supplied schema.

GROUNDING:
- BLIP captions, Qwen triples and CLIP tags are fallible MODEL OBSERVATIONS, never verified truth.
- factual_text may describe only what is directly visible in the observations, reliable EXIF time/location order, and explicitly verified_context.
- Prefer concrete, checkable visual details: objects, scene, setting, count, posture, and the visible action. Keep neutral third-person observer prose.
- This is guidance, not a hard constraint: avoid asserting an unseen activity, a named identity or relationship, an emotion, or a cause unless verified_context supports it. If you are unsure whether a detail was actually observed, omit it rather than guess.
- Exclude every detail listed under conflicts; the application routes conflicts to uncertain observations for you.
- Reliable EXIF time/location may establish order. verified_context is the only source for personal identity, relationship, emotion or purpose.

STRUCTURE:
- {paragraph_rule}
- Do not output evidence IDs, photo IDs, citations, conflicts, uncertain observations or unused evidence; the application injects them deterministically.
- Weave each group's photos into one flowing paragraph instead of one sentence per photo.
- Keep the whole response concise (at most {STORY_MAX_WORDS} words).

Write the title and every factual_text in {language_name}."""
        user = f"""Create the narrative for {context.label}.
Keep groups in the supplied chronological order. Required group IDs: {', '.join(group.group_id for group in context.groups)}.
Evidence plan:
{context.to_prompt_context()}"""
        return system, user

    @classmethod
    def output_schema(cls, context: StoryContext, language: str = "zh") -> dict[str, Any]:
        """Strict JSON schema for the grouped faithful contract.

        The model returns only ``{title, groups[{group_id, factual_text}]}``;
        the application deterministically injects paragraphs, evidence
        citations, uncertain observations and unused evidence via
        :meth:`normalise_payload`. Semantic checks stay in
        :class:`GroundingValidator`. Faithful never emits transitions.
        """
        del language
        group_ids = [group.group_id for group in context.groups]
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "groups": {
                    "type": "array",
                    "minItems": len(group_ids),
                    "maxItems": len(group_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "group_id": {"type": "string", "enum": group_ids},
                            "factual_text": {"type": "string", "minLength": 1},
                        },
                        "required": ["group_id", "factual_text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["title", "groups"],
            "additionalProperties": False,
        }

    @classmethod
    def normalise_payload(cls, candidate: Mapping[str, Any], context: StoryContext) -> dict[str, Any]:
        """Deterministically inject the faithful bookkeeping the model no longer writes.

        Mirrors :meth:`LayeredCreativePromptBuilder.normalise_payload` but always
        emits empty ``creative_transitions`` (faithful never produces them).
        Each evidence group becomes one paragraph that cites every evidence ID
        in that group; conflicts become uncertain observations; nothing is
        unused. Structural problems (bad group objects, duplicates, unknown
        group IDs) are collected in ``_creative_contract_errors`` and surfaced
        by :class:`GroundingValidator`.
        """
        raw_groups = candidate.get("groups", [])
        group_values = raw_groups if isinstance(raw_groups, list) else []
        by_group: dict[str, str] = {}
        contract_errors: list[str] = []
        for item in group_values:
            if not isinstance(item, Mapping):
                contract_errors.append("E_FAITHFUL_GROUP_FORMAT: every group must be an object")
                continue
            group_id = str(item.get("group_id") or "")
            if group_id in by_group:
                contract_errors.append(f"E_FAITHFUL_GROUP_DUPLICATE: duplicate group_id {group_id!r}")
                continue
            by_group[group_id] = str(item.get("factual_text") or "").strip()
        expected_ids = {group.group_id for group in context.groups}
        unknown_ids = sorted(set(by_group) - expected_ids)
        if unknown_ids:
            contract_errors.append(f"E_FAITHFUL_GROUP_UNKNOWN: unknown group IDs {unknown_ids}")
        paragraphs = [
            {
                "text": by_group.get(group.group_id, ""),
                "evidence_ids": list(group.evidence_ids),
                "group_id": group.group_id,
            }
            for group in context.groups
        ]
        uncertain = [
            {"text": conflict, "evidence_ids": list(group.evidence_ids)}
            for group in context.groups
            for conflict in group.conflicts
        ]
        return {
            "title": str(candidate.get("title") or "").strip(),
            "paragraphs": paragraphs,
            "uncertain_observations": uncertain,
            "unused_evidence_ids": [],
            "creative_transitions": [],
            "_creative_contract_errors": contract_errors,
        }



class LayeredCreativePromptBuilder:
    """Production-only layered Creative contract introduced after frozen N0--N3."""

    PROMPT_VERSION = "grounded-story-v4-layered-creative"

    @staticmethod
    def transition_count(context: StoryContext) -> int:
        return 1 if len(context.groups) == 1 else len(context.groups) - 1

    @classmethod
    def output_schema(cls, context: StoryContext) -> dict[str, Any]:
        group_ids = [group.group_id for group in context.groups]
        transition_count = cls.transition_count(context)
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "groups": {
                    "type": "array",
                    "minItems": len(group_ids),
                    "maxItems": len(group_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "group_id": {"type": "string", "enum": group_ids},
                            "factual_text": {"type": "string", "minLength": 8},
                        },
                        "required": ["group_id", "factual_text"],
                        "additionalProperties": False,
                    },
                },
                "creative_transitions": {
                    "type": "array",
                    "minItems": transition_count,
                    "maxItems": transition_count,
                    "items": {"type": "string", "minLength": 8},
                },
            },
            "required": ["title", "groups", "creative_transitions"],
            "additionalProperties": False,
        }

    @classmethod
    def build_story_prompt(
        cls,
        context: StoryContext,
        *,
        mode: str = "creative",
        language: str = "zh",
    ) -> tuple[str, str]:
        del mode
        if not context.evidence or not context.groups:
            raise ValueError("No photo evidence is available")
        language_name = "Chinese" if language.startswith("zh") else "English"
        group_count = len(context.groups)
        transition_count = cls.transition_count(context)
        transition_rule = (
            "Write one creative reflection after the factual paragraph."
            if group_count == 1
            else f"Write exactly {transition_count} creative transitions, one between each adjacent pair of factual groups."
        )
        system = f"""You write a layered personal-photo narrative in {language_name}.
Return only the JSON required by the supplied schema.

FACTUAL LAYER:
- Write exactly {group_count} factual_text entries, one for every supplied group_id.
- factual_text may describe only observable evidence, reliable time/location, and explicitly verified_context.
- Treat BLIP captions, Qwen triples and CLIP tags as fallible model observations.
- Exclude every detail listed under conflicts.
- Never use first person or infer identity, relationship, emotion, intention, purpose, cause, or unseen events.
- Never use unsupported hedges such as possibly, likely, perhaps, probably, might, maybe, appears to or seems to.

CREATIVE LAYER:
- {transition_rule}
- creative_transitions may use first person, atmosphere, reflection, or an imaginative connection.
- A transition is explicitly non-factual: do not introduce a concrete new person, object, place, action, or event as if photographed.
- Do not place group_id, evidence IDs, conflicts, citations, or unused evidence in the output; the application injects them deterministically.

Write the title, every factual_text, and every creative transition in {language_name}. Keep the whole response under {STORY_MAX_WORDS} words."""
        user = f"""Create a layered creative narrative for {context.label}.
Keep groups in the supplied chronological order. Required group IDs: {', '.join(group.group_id for group in context.groups)}.
Evidence plan:
{context.to_prompt_context()}"""
        return system, user

    @classmethod
    def normalise_payload(cls, candidate: Mapping[str, Any], context: StoryContext) -> dict[str, Any]:
        raw_groups = candidate.get("groups", [])
        group_values = raw_groups if isinstance(raw_groups, list) else []
        by_group: dict[str, str] = {}
        contract_errors: list[str] = []
        for item in group_values:
            if not isinstance(item, Mapping):
                contract_errors.append("E_CREATIVE_GROUP_FORMAT: every group must be an object")
                continue
            group_id = str(item.get("group_id") or "")
            if group_id in by_group:
                contract_errors.append(f"E_CREATIVE_GROUP_DUPLICATE: duplicate group_id {group_id!r}")
                continue
            by_group[group_id] = str(item.get("factual_text") or "").strip()

        expected_ids = {group.group_id for group in context.groups}
        unknown_ids = sorted(set(by_group) - expected_ids)
        if unknown_ids:
            contract_errors.append(f"E_CREATIVE_GROUP_UNKNOWN: unknown group IDs {unknown_ids}")

        paragraphs = [
            {
                "text": by_group.get(group.group_id, ""),
                "evidence_ids": list(group.evidence_ids),
                "group_id": group.group_id,
            }
            for group in context.groups
        ]
        uncertain = [
            {"text": conflict, "evidence_ids": list(group.evidence_ids)}
            for group in context.groups
            for conflict in group.conflicts
        ]
        transitions = candidate.get("creative_transitions", [])
        return {
            "title": str(candidate.get("title") or "").strip(),
            "paragraphs": paragraphs,
            "uncertain_observations": uncertain,
            "unused_evidence_ids": [],
            "creative_transitions": transitions if isinstance(transitions, list) else transitions,
            "_creative_contract_errors": contract_errors,
        }


class FirstPersonMemoirPromptBuilder:
    """Production Creative v5; v4 remains available for historical audit."""

    PROMPT_VERSION = "grounded-story-v5-first-person-memoir"

    @staticmethod
    def transition_count(context: StoryContext) -> int:
        return max(0, len(context.groups) - 1)

    @staticmethod
    def _length_target(context: StoryContext, language: str) -> str:
        count = len(context.groups)
        if language.startswith("zh"):
            if count == 1:
                return "120 to 200 Chinese characters"
            if count <= 3:
                return "220 to 360 Chinese characters"
            return "320 to 520 Chinese characters"
        if count == 1:
            return "70 to 130 English words"
        if count <= 3:
            return "140 to 230 English words"
        return "220 to 360 English words"

    @classmethod
    def output_schema(cls, context: StoryContext, language: str = "zh") -> dict[str, Any]:
        del language
        group_ids = [group.group_id for group in context.groups]
        transition_count = cls.transition_count(context)
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 2},
                "opening": {"type": "string", "minLength": 20},
                "groups": {
                    "type": "array",
                    "minItems": len(group_ids),
                    "maxItems": len(group_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "group_id": {"type": "string", "enum": group_ids},
                            "factual_text": {"type": "string", "minLength": 32},
                        },
                        "required": ["group_id", "factual_text"],
                        "additionalProperties": False,
                    },
                },
                "transitions": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": transition_count,
                    "items": {"type": "string", "minLength": 20},
                },
                "closing": {"type": "string", "minLength": 20},
            },
            "required": ["title", "opening", "groups", "transitions", "closing"],
            "additionalProperties": False,
        }

    @staticmethod
    def _prompt_context(context: StoryContext) -> str:
        payload = {
            "label_hint": context.label,
            "narrator_role": context.narrator_role,
            "verified_context": context.verified_context or None,
            "evidence_groups": [{
                "group_id": group.group_id,
                "model_observations": [dict(value) for value in group.model_observations],
                "conflicts_to_exclude": list(group.conflicts),
                "near_duplicate_count": len(group.near_duplicate_evidence_ids),
                "has_visible_person_evidence": _group_has_person_observation(group),
                "narration_instruction": (
                    "An evidenced action of the principal visible person may be narrated as I/me."
                    if context.narrator_role == "confirmed_subject" and _group_has_person_observation(group)
                    else "Do not place the narrator inside this scene; use observer voice or direct scene description."
                ),
            } for group in context.groups],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def build_story_prompt(
        cls,
        context: StoryContext,
        *,
        mode: str = "creative",
        language: str = "zh",
    ) -> tuple[str, str]:
        del mode
        if not context.evidence or not context.groups:
            raise ValueError("No photo evidence is available")
        language_name = "Chinese" if language.startswith("zh") else "English"
        transition_count = cls.transition_count(context)
        if context.narrator_role == "confirmed_subject":
            narrator_rule = (
                "The user has explicitly confirmed that the principal photographed person is the narrator. "
                "For a group marked has_visible_person_evidence=true, you may turn only an explicitly observed action of "
                "that principal person into I/me narration. For a group marked false, do not invent the narrator inside "
                "the scene: use observer voice or direct scene description. This confirmation does not support emotion, "
                "purpose, relationship or causality."
            )
        else:
            narrator_rule = (
                "The narrator's identity inside the photographs is NOT confirmed. Use first-person observer language such "
                "as 'I see' or 'I notice'. Never claim that a photographed person is I/me or that the narrator performed an action."
            )
        system = f"""Write a coherent first-person personal-photo memoir in {language_name}.
Return JSON only and follow the supplied schema exactly.

VOICE AND STRUCTURE:
- Use a natural memoir voice, not a caption list and not ornate literary prose.
- opening introduces one unifying theme; closing returns to that theme and gives a restrained ending.
- Write one factual_text for every evidence group in chronological order. Each group may contain several scenes: weave them into one flowing paragraph, not one sentence per photo.
- Write 0 to {transition_count} short transitions. A transition is a light temporal or atmospheric bridge between adjacent scenes; it MUST NOT assert a physical, spatial or causal connection between two different photos (for example, a cable seen in one photo does not reach an object shown in another photo).
- Maintain a first-person voice across the story as a whole. The opening must explicitly establish the narrator with 我 in Chinese or I/my in English.
- Natural Chinese may omit 我 in later factual_text, transitions and closing once the voice is established. Do not add 我 mechanically to every paragraph. English should keep explicit first-person grammar where it is needed.
- VARY how each factual_text begins; do not start every paragraph with the same phrase or one fixed template. You choose the openings.
- {narrator_rule}

GROUNDING:
- factual_text may describe the visible model observations, reliable time/place order, and explicitly verified_context.
- BLIP, Qwen and CLIP observations are fallible. Exclude every conflicts_to_exclude detail.
- A light personal impression or gentle atmosphere is allowed. But do NOT assert a specific identity, relationship or name, do NOT state what an unseen person is doing (such as working or studying), and do NOT claim a hard cause (because/therefore) unless verified_context provides it.
- Do not invent a new concrete person, object, place, action or event that is not in the observations.
- opening, transitions and closing are reflective non-photo narration; they may connect scenes in time and mood but must not invent a new concrete person, object, place, action or event.
- Every transition you do write must bridge the adjacent evidence groups at that position using their supplied visual anchors. If no honest bridge exists, omit the transition instead of inventing continuity.
- Do not output evidence IDs, photo IDs, citations, conflicts or unused evidence; the application injects them.

STYLE:
- Do not repeat dates, clock times or locations in the prose; the interface displays them once.
- Avoid generic poetic clichés such as time standing still, frozen moments, new chapters, painted scrolls or time stretched by the wind.
- Vary sentence openings and make the result read as one continuous memory.
- Target {cls._length_target(context, language)} in total.
- The title and every text field must be in {language_name}."""
        user = f"""Create the first-person memoir for this evidence plan.
Required group order: {', '.join(group.group_id for group in context.groups)}.
Evidence plan without display metadata:
{cls._prompt_context(context)}"""
        return system, user

    @classmethod
    def normalise_payload(cls, candidate: Mapping[str, Any], context: StoryContext) -> dict[str, Any]:
        raw_groups = candidate.get("groups", [])
        group_values = raw_groups if isinstance(raw_groups, list) else []
        by_group: dict[str, str] = {}
        contract_errors: list[str] = []
        for item in group_values:
            if not isinstance(item, Mapping):
                contract_errors.append("E_MEMOIR_GROUP_FORMAT: every group must be an object")
                continue
            group_id = str(item.get("group_id") or "")
            if group_id in by_group:
                contract_errors.append(f"E_MEMOIR_GROUP_DUPLICATE: duplicate group_id {group_id!r}")
                continue
            by_group[group_id] = str(item.get("factual_text") or "").strip()
        expected_ids = {group.group_id for group in context.groups}
        unknown_ids = sorted(set(by_group) - expected_ids)
        if unknown_ids:
            contract_errors.append(f"E_MEMOIR_GROUP_UNKNOWN: unknown group IDs {unknown_ids}")
        transitions = candidate.get("transitions", [])
        return {
            "title": str(candidate.get("title") or "").strip(),
            "opening": str(candidate.get("opening") or "").strip(),
            "paragraphs": [{
                "text": by_group.get(group.group_id, ""),
                "evidence_ids": list(group.evidence_ids),
                "group_id": group.group_id,
            } for group in context.groups],
            "uncertain_observations": [
                {"text": conflict, "evidence_ids": list(group.evidence_ids)}
                for group in context.groups for conflict in group.conflicts
            ],
            "unused_evidence_ids": [],
            "creative_transitions": transitions,
            "closing": str(candidate.get("closing") or "").strip(),
            "_creative_contract_errors": contract_errors,
        }


class PhotographerMoodPromptBuilder(FirstPersonMemoirPromptBuilder):
    """Creative v6 with optional, user-confirmed photographer mood metadata."""

    PROMPT_VERSION = "grounded-story-v6.2-deterministic-photographer-mood"

    @classmethod
    def output_schema(cls, context: StoryContext, language: str = "zh") -> dict[str, Any]:
        # Mood is verified metadata and is injected by the application. Keep
        # the model-owned output contract identical for M0 and M1.
        return FirstPersonMemoirPromptBuilder.output_schema(context, language=language)

    @staticmethod
    def _prompt_context(context: StoryContext) -> str:
        mood_by_evidence = {
            str(record["evidence_id"]): record for record in context.photographer_moods
        }
        payload = {
            "label_hint": context.label,
            "narrator_role": context.narrator_role,
            "verified_context": context.verified_context or None,
            "photographer_mood_metadata_enabled": context.use_photographer_mood,
            "photographer_mood_metadata": context.photographer_moods,
            "evidence_groups": [{
                "group_id": group.group_id,
                "model_observations": [dict(value) for value in group.model_observations],
                "conflicts_to_exclude": list(group.conflicts),
                "near_duplicate_count": len(group.near_duplicate_evidence_ids),
                "has_visible_person_evidence": _group_has_person_observation(group),
                "photographer_moods": [
                    mood_by_evidence[evidence_id]
                    for evidence_id in group.evidence_ids
                    if evidence_id in mood_by_evidence
                ],
                "narration_instruction": (
                    "An evidenced action of the principal visible person may be narrated as I/me."
                    if context.narrator_role == "confirmed_subject" and _group_has_person_observation(group)
                    else "Do not place the narrator inside this scene; use observer voice or direct scene description."
                ),
            } for group in context.groups],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def build_story_prompt(
        cls,
        context: StoryContext,
        *,
        mode: str = "creative",
        language: str = "zh",
    ) -> tuple[str, str]:
        system, user = super().build_story_prompt(context, mode=mode, language=language)
        mood_rules = """

PHOTOGRAPHER MOOD METADATA (user-confirmed; you MUST reflect it when enabled):
- photographer_mood_metadata records the photographer's own verified state at each capture. It is NOT a BLIP/Qwen/CLIP observation and NEVER describes a photographed person's emotion.
- When this metadata is enabled, you MUST weave each photo's mood into the narrative as a genuine first-person reflection tied to the act of capturing (e.g., "Capturing the cathedral, I felt excited", "I felt calm watching the river"). Spread the per-photo moods across the opening, factual paragraphs/groups, transitions, and closing so every photo's mood is reflected.
- Use ONLY the supplied mood labels (neutral/calm/happy/excited/tense/sad). Do NOT invent a new mood.
- State the mood as the photographer's own feeling at capture. Do NOT explain WHY (no causal/emotional reasons), do NOT attribute any mood to a photographed person, and do NOT make relationship/medical/intent claims.
- Keep each reflection restrained and grounded in the scene; never turn the mood into a story about its cause.
"""
        return system + mood_rules, user

    @classmethod
    def normalise_payload(cls, candidate: Mapping[str, Any], context: StoryContext) -> dict[str, Any]:
        payload = FirstPersonMemoirPromptBuilder.normalise_payload(candidate, context)
        payload["mood_reflections"] = []
        return payload


class GroundingValidator:
    SPECULATION_TERMS = (
        "possibly", "likely", "perhaps", "probably", "might", "maybe", "appears to", "seems to",
        "working", "studying", "getting ready", "intends", "intention", "purpose",
        "because", "therefore", "decided", "felt", "happy", "sad", "wife", "husband", "daughter", "son",
        "可能", "也许", "或许", "大概", "似乎", "应该", "工作", "学习", "准备", "打算", "想要", "为了",
        "因为", "所以", "导致", "决定", "感到", "开心", "难过", "妻子", "丈夫", "女儿", "儿子",
    )
    FIRST_PERSON_PATTERNS = (
        re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.I),
        re.compile(r"我|我们|我的|我们的"),
    )
    # Reduced ban list for the coherent first-person memoir: it permits light
    # feeling and atmosphere but still blocks concrete identity/relationship
    # claims, unseen-action inference, and hard causal claims.
    NARRATIVE_BANNED_TERMS = (
        "wife", "husband", "daughter", "son",
        "because", "therefore",
        "working", "studying", "intends", "intention", "purpose",
        "妻子", "丈夫", "女儿", "儿子",
        "因为", "所以", "导致",
        "工作", "学习", "打算", "想要", "为了",
    )

    @staticmethod
    def parse_payload(content: str) -> dict[str, Any]:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError("story output must be a JSON object")
        return value

    @staticmethod
    def _prose_char_count(text: str) -> int:
        return sum(1 for ch in text if ch.isalpha())

    @staticmethod
    def _language_ratio(text: str) -> float:
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        return cjk / max(1, cjk + latin)

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        if re.fullmatch(r"[A-Za-z ]+", term):
            return re.search(rf"\b{re.escape(term)}\b", text, re.I) is not None
        return term in text

    @classmethod
    def _echo_targets(cls, context: StoryContext) -> set[str]:
        targets = {context.label.casefold()}
        for item in context.evidence:
            targets.add(item.evidence_id.casefold())
            if item.timestamp:
                targets.update({str(item.timestamp).casefold(), str(item.timestamp)[:10].casefold()})
        return targets

    @staticmethod
    def _paragraph_similarity(left: str, right: str) -> float:
        normalized_left = re.sub(r"\W+", "", left.casefold())
        normalized_right = re.sub(r"\W+", "", right.casefold())
        sequence = difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()
        return max(sequence, _jaccard(left, right))

    @staticmethod
    def _conflict_terms(group: EvidenceGroup) -> set[str]:
        if not group.conflicts:
            return set()
        caption: set[str] = set()
        graph: set[str] = set()
        for observation in group.model_observations:
            caption |= _tokens(str(observation.get("blip_caption_model_observation") or ""))
            graph |= _tokens(" ".join(observation.get("qwen_scene_graph_model_observations") or ()))
        return (caption ^ graph) - _GENERIC_TOKENS

    @classmethod
    def validate(
        cls,
        payload: dict[str, Any],
        context: StoryContext,
        mode: str,
        language: str = "en",
        *,
        allow_first_person_facts: bool = False,
        expected_creative_transitions: Optional[int] = None,
        speculation_terms: Optional[Sequence[str]] = None,
        allow_fewer_transitions: bool = False,
    ) -> list[str]:
        errors: list[str] = []
        allowed = {item.evidence_id: item for item in context.evidence}
        groups = {group.group_id: group for group in context.groups}
        paragraphs = payload.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            return ["E_JSON_PARAGRAPHS: paragraphs must be a non-empty list"]
        if len(paragraphs) != len(groups):
            errors.append(f"E_GROUP_PARAGRAPH_COUNT: expected {len(groups)} paragraphs, got {len(paragraphs)}")

        title = str(payload.get("title") or "").strip()
        if not title:
            errors.append("E_TITLE_MISSING: title is required")
        echo_targets = cls._echo_targets(context)
        cited: set[str] = set()
        referenced_groups: set[str] = set()
        paragraph_texts: list[str] = []
        verified = context.verified_context.casefold()

        for index, paragraph in enumerate(paragraphs, 1):
            if not isinstance(paragraph, dict) or not str(paragraph.get("text", "")).strip():
                errors.append(f"E_PARAGRAPH_TEXT: paragraph {index} has no text")
                continue
            text = str(paragraph["text"]).strip()
            paragraph_texts.append(text)
            if cls._prose_char_count(text) == 0:
                errors.append(f"E_PARAGRAPH_PROSE: paragraph {index} has no descriptive words")
            if text.casefold() in echo_targets:
                errors.append(f"E_PARAGRAPH_ECHO: paragraph {index} only echoes metadata")

            group_id = str(paragraph.get("group_id") or "")
            if not group_id and len(context.groups) == 1:
                group_id = context.groups[0].group_id
            group = groups.get(group_id)
            if group is None:
                errors.append(f"E_GROUP_UNKNOWN: paragraph {index} uses unknown group_id {group_id!r}")
            else:
                referenced_groups.add(group_id)

            citations = paragraph.get("evidence_ids")
            if not isinstance(citations, list) or not citations:
                errors.append(f"E_CITATION_MISSING: paragraph {index} has no evidence citations")
                continue
            unknown = [value for value in citations if value not in allowed]
            if unknown:
                errors.append(f"E_CITATION_UNKNOWN: paragraph {index} uses {unknown}")
                continue
            if group is not None and not set(citations) <= set(group.evidence_ids):
                errors.append(f"E_CITATION_GROUP_MISMATCH: paragraph {index} cites outside {group_id}")
            cited.update(str(value) for value in citations)

            if group is not None and group.conflicts:
                lowered_tokens = _tokens(text)
                used_conflicts = lowered_tokens & cls._conflict_terms(group)
                if used_conflicts:
                    errors.append(
                        f"E_CONFLICT_IN_BODY: paragraph {index} uses conflicted detail {sorted(used_conflicts)}"
                    )

            if not allow_first_person_facts and any(pattern.search(text) for pattern in cls.FIRST_PERSON_PATTERNS):
                errors.append(f"E_FIRST_PERSON_IN_FACT: paragraph {index} places first person in factual prose")
            if mode in {"faithful", "creative"}:
                active_terms = cls.SPECULATION_TERMS if speculation_terms is None else speculation_terms
                for term in active_terms:
                    if cls._contains_term(text, term) and not cls._contains_term(verified, term):
                        errors.append(f"E_UNSUPPORTED_CLAIM: paragraph {index} contains {term!r}")

        for left in range(len(paragraph_texts)):
            for right in range(left + 1, len(paragraph_texts)):
                if cls._paragraph_similarity(paragraph_texts[left], paragraph_texts[right]) > 0.80:
                    errors.append(f"E_PARAGRAPH_DUPLICATE: paragraphs {left + 1} and {right + 1} exceed 0.80")

        missing_groups = set(groups) - referenced_groups
        if missing_groups:
            errors.append(f"E_GROUP_COVERAGE: unreferenced groups {sorted(missing_groups)}")

        unused = payload.get("unused_evidence_ids", [])
        if not isinstance(unused, list) or any(value not in allowed for value in unused):
            errors.append("E_UNUSED_IDS: unused_evidence_ids contains unknown IDs or is not a list")
            unused = []
        missing_evidence = set(allowed) - cited
        if missing_evidence != set(unused):
            errors.append(
                f"E_EVIDENCE_ACCOUNTING: expected unused IDs {sorted(missing_evidence)}, got {sorted(set(unused))}"
            )

        uncertain = payload.get("uncertain_observations", [])
        if not isinstance(uncertain, list):
            errors.append("E_UNCERTAIN_FORMAT: uncertain_observations must be a list")
            uncertain = []
        uncertain_ids: set[str] = set()
        for item in uncertain:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                errors.append("E_UNCERTAIN_FORMAT: each uncertain observation needs text and evidence_ids")
                continue
            ids = item.get("evidence_ids")
            if not isinstance(ids, list) or any(value not in allowed for value in ids):
                errors.append("E_UNCERTAIN_CITATION: uncertain observation has invalid evidence IDs")
                continue
            uncertain_ids.update(ids)
        for group in context.groups:
            if group.conflicts and not (set(group.evidence_ids) & uncertain_ids):
                errors.append(f"E_CONFLICT_UNREPORTED: {group.group_id} conflict is absent from uncertain_observations")

        transitions = payload.get("creative_transitions", [])
        if not isinstance(transitions, list) or any(not isinstance(value, str) or not value.strip() for value in transitions):
            errors.append("E_CREATIVE_FORMAT: creative_transitions must be a list of non-empty strings")
            transitions = []
        elif mode == "faithful" and transitions:
            errors.append("E_CREATIVE_IN_FAITHFUL: faithful mode cannot contain creative transitions")
        elif mode == "creative":
            expected_transitions = (
                LayeredCreativePromptBuilder.transition_count(context)
                if expected_creative_transitions is None else expected_creative_transitions
            )
            if allow_fewer_transitions:
                if len(transitions) > expected_transitions:
                    errors.append(
                        f"E_CREATIVE_TRANSITION_COUNT: expected at most {expected_transitions} transitions, got {len(transitions)}"
                    )
            elif len(transitions) != expected_transitions:
                errors.append(
                    f"E_CREATIVE_TRANSITION_COUNT: expected {expected_transitions} transitions, got {len(transitions)}"
                )

        if mode in {"faithful", "creative"}:
            errors.extend(str(value) for value in payload.get("_creative_contract_errors", []) if value)

        language_text = " ".join([title, *paragraph_texts, *(transitions if mode == "creative" else [])])
        ratio = cls._language_ratio(language_text)
        if language.startswith("zh") and ratio < 0.25:
            errors.append(f"E_LANGUAGE_ZH: Chinese character ratio {ratio:.3f} is below 0.25")
        if not language.startswith("zh") and ratio > 0.10:
            errors.append(f"E_LANGUAGE_EN: Chinese character ratio {ratio:.3f} exceeds 0.10")
        return list(dict.fromkeys(errors))


class FirstPersonMemoirValidator:
    CLICHES = (
        "仿佛时间被", "岁月定格", "定格了时光", "展开新的篇章", "翻开新的篇章",
        "时光在这一刻", "轻轻诉说", "画卷般", "时间仿佛静止",
        "time stood still", "frozen in time", "a new chapter", "painted a picture",
        "whispered softly", "like a painting",
    )
    DATE_TIME_PATTERNS = (
        re.compile(r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)"),
        re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)"),
    )
    ZH_OBSERVER_PATTERNS = (
        re.compile(r"我(?:从|在)?(?:照片|画面|镜头|眼前|这些画面)?(?:中|里)?(?:可以|能)?(?:看到|看见|注意到|留意到|观察到|望见)"),
        re.compile(
            r"我(?:从|在)?(?:(?:第[一二三四五六七八九十]+|(?:这|那|下|上|最后)(?:一张|一幅|一组)?)(?:的)?|"
            r"(?:一张|一幅|一组|些))?(?:照片|画面|镜头)(?:中|里)?(?:可以|能)?"
            r"(?:看到|看见|注意到|留意到|观察到|望见)"
        ),
        re.compile(r"映入我眼帘的是"),
        re.compile(r"我的视线(?:落在|转向|停在)"),
        re.compile(r"我(?:把)?(?:目光|视线)(?:落在|投向|移向|转向|停在)"),
        re.compile(r"我(?:望向|看向|看着|注视着)"),
    )
    EN_OBSERVER_PATTERNS = (
        re.compile(r"\bI (?:can )?(?:see|notice|observe|look at)\b", re.I),
        re.compile(r"\bMy attention (?:rests on|moves to|turns to)\b", re.I),
        re.compile(r"\bWhat I see is\b", re.I),
    )
    @staticmethod
    def _field_language_error(name: str, text: str, language: str) -> Optional[str]:
        ratio = GroundingValidator._language_ratio(text)
        if language.startswith("zh") and ratio < 0.25:
            return f"E_FIELD_LANGUAGE_ZH: {name} Chinese character ratio {ratio:.3f} is below 0.25"
        if not language.startswith("zh") and ratio > 0.10:
            return f"E_FIELD_LANGUAGE_EN: {name} Chinese character ratio {ratio:.3f} exceeds 0.10"
        return None

    @classmethod
    def _contains_first_person(cls, text: str, language: str) -> bool:
        pattern = (
            GroundingValidator.FIRST_PERSON_PATTERNS[1]
            if language.startswith("zh") else GroundingValidator.FIRST_PERSON_PATTERNS[0]
        )
        return pattern.search(text) is not None

    @classmethod
    def _observer_only(cls, text: str, language: str) -> tuple[bool, bool]:
        patterns = cls.ZH_OBSERVER_PATTERNS if language.startswith("zh") else cls.EN_OBSERVER_PATTERNS
        found = any(pattern.search(text) for pattern in patterns)
        scrubbed = text
        for pattern in patterns:
            scrubbed = pattern.sub("", scrubbed)
        remaining_first_person = cls._contains_first_person(scrubbed, language)
        return found, not remaining_first_person

    @classmethod
    def _group_has_person(cls, group: EvidenceGroup) -> bool:
        return _group_has_person_observation(group)

    @staticmethod
    def _minimum_length(context: StoryContext, language: str) -> int:
        count = len(context.groups)
        if language.startswith("zh"):
            return 70 if count == 1 else 140 if count <= 3 else 200
        return 45 if count == 1 else 95 if count <= 3 else 130

    @classmethod
    def validate(
        cls,
        payload: dict[str, Any],
        context: StoryContext,
        language: str,
    ) -> list[str]:
        errors = GroundingValidator.validate(
            payload,
            context,
            "creative",
            language,
            allow_first_person_facts=True,
            expected_creative_transitions=FirstPersonMemoirPromptBuilder.transition_count(context),
            speculation_terms=GroundingValidator.NARRATIVE_BANNED_TERMS,
            allow_fewer_transitions=True,
        )
        title = str(payload.get("title") or "").strip()
        opening = str(payload.get("opening") or "").strip()
        closing = str(payload.get("closing") or "").strip()
        paragraphs = payload.get("paragraphs", [])
        transitions = payload.get("creative_transitions", [])
        if not opening:
            errors.append("E_MEMOIR_OPENING: opening is required")
        if not closing:
            errors.append("E_MEMOIR_CLOSING: closing is required")

        named_fields: list[tuple[str, str]] = [("title", title), ("opening", opening)]
        named_fields.extend(
            (f"factual_text[{index}]", str(item.get("text") or "").strip())
            for index, item in enumerate(paragraphs, 1) if isinstance(item, Mapping)
        )
        if isinstance(transitions, list):
            named_fields.extend((f"transition[{index}]", str(value).strip()) for index, value in enumerate(transitions, 1))
        named_fields.append(("closing", closing))
        for name, text in named_fields:
            if text:
                language_error = cls._field_language_error(name, text, language)
                if language_error:
                    errors.append(language_error)

        if opening and not cls._contains_first_person(opening, language):
            errors.append("E_FIRST_PERSON_VOICE: opening must establish first-person narration")
        if not language.startswith("zh") and closing and not cls._contains_first_person(closing, language):
            errors.append("E_FIRST_PERSON_VOICE: closing must use explicit first-person narration in English")
        if not language.startswith("zh") and isinstance(transitions, list):
            for index, text in enumerate(transitions, 1):
                if isinstance(text, str) and text.strip() and not cls._contains_first_person(text, language):
                    errors.append(
                        f"E_FIRST_PERSON_VOICE: transition {index} must use explicit first-person narration in English"
                    )

        groups = {group.group_id: group for group in context.groups}
        for index, paragraph in enumerate(paragraphs, 1):
            if not isinstance(paragraph, Mapping):
                continue
            text = str(paragraph.get("text") or "").strip()
            if not text:
                continue
            group = groups.get(str(paragraph.get("group_id") or ""))
            # First-person voice is established at story level. Natural Chinese
            # frequently omits the subject after the opening, and object-only
            # groups should not be forced into a mechanical "我看到" template.
            # Identity confirmation only authorises narrator action where the
            # evidence group actually contains a visible person.
            if context.narrator_role == "confirmed_subject":
                _, observer_only = cls._observer_only(text, language)
                if not observer_only and group is not None and not cls._group_has_person(group):
                    errors.append(
                        f"E_CONFIRMED_SUBJECT_NO_PERSON: factual_text {index} claims narrator action without person evidence"
                    )

        narrative_fields = [(name, text) for name, text in named_fields if name != "title"]
        for name, text in narrative_fields:
            lowered = text.casefold()
            for pattern in cls.DATE_TIME_PATTERNS:
                if pattern.search(text):
                    errors.append(f"E_METADATA_IN_NARRATIVE: {name} repeats a date or clock time")
                    break
            for location in context.locations:
                if len(location.strip()) >= 2 and location.casefold() in lowered:
                    errors.append(f"E_METADATA_IN_NARRATIVE: {name} repeats location {location!r}")
            for cliché in cls.CLICHES:
                if cliché.casefold() in lowered:
                    errors.append(f"E_CREATIVE_CLICHE: {name} contains {cliché!r}")

        narrative_text = " ".join(text for _, text in narrative_fields)
        length = (
            len(re.findall(r"[\u4e00-\u9fff]", narrative_text))
            if language.startswith("zh")
            else len(re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", narrative_text))
        )
        minimum = cls._minimum_length(context, language)
        if length < minimum:
            unit = "Chinese characters" if language.startswith("zh") else "English words"
            errors.append(f"E_MEMOIR_TOO_SHORT: got {length} {unit}, minimum is {minimum}")

        # Factual paragraphs are already compared by GroundingValidator. An
        # opening and closing intentionally share a theme, and a bridge may echo
        # that theme; comparing every block pair made valid memoirs fail. Only
        # repeated transitions need an additional style check here.
        transition_texts = [str(value).strip() for value in transitions] if isinstance(transitions, list) else []
        for left in range(len(transition_texts)):
            for right in range(left + 1, len(transition_texts)):
                if GroundingValidator._paragraph_similarity(transition_texts[left], transition_texts[right]) > 0.92:
                    errors.append(
                        f"E_MEMOIR_REPETITION: transitions {left + 1} and {right + 1} exceed 0.92"
                    )
        return list(dict.fromkeys(errors))


class PhotographerMoodValidator:
    """Validate that Creative v6 uses only confirmed photographer moods."""

    TERMS: dict[str, tuple[str, ...]] = {
        "neutral": ("neutral", "中性"),
        "calm": ("calm", "peaceful", "平静", "平和"),
        "happy": ("happy", "glad", "开心", "高兴", "愉快"),
        "excited": ("excited", "thrilled", "兴奋", "激动"),
        "tense": ("tense", "nervous", "anxious", "紧张", "焦虑"),
        "sad": ("sad", "unhappy", "难过", "悲伤"),
    }
    EN_MOOD_CUE = re.compile(
        r"\b(?:feel|feels|felt|feeling|mood|am|was|became|remained|seemed)\b", re.I
    )
    ZH_MOOD_CUE = re.compile(r"感到|感觉|觉得|心情|心里|内心|变得|保持|带着|怀着|有些|很|十分")

    @classmethod
    def _labels_in_text(cls, text: str) -> set[str]:
        labels: set[str] = set()
        for label, terms in cls.TERMS.items():
            if any(GroundingValidator._contains_term(text, term) for term in terms):
                labels.add(label)
        return labels

    @classmethod
    def _is_mood_claim(cls, text: str, language: str, *, title: bool = False) -> bool:
        if not cls._labels_in_text(text):
            return False
        if title:
            return True
        if language.startswith("zh"):
            return bool(GroundingValidator.FIRST_PERSON_PATTERNS[1].search(text) and cls.ZH_MOOD_CUE.search(text))
        return bool(GroundingValidator.FIRST_PERSON_PATTERNS[0].search(text) and cls.EN_MOOD_CUE.search(text))

    @classmethod
    def _attributes_mood_to_visible_person(cls, text: str, language: str) -> bool:
        terms = [re.escape(term) for values in cls.TERMS.values() for term in values]
        mood_pattern = "(?:" + "|".join(terms) + ")"
        if language.startswith("zh"):
            return re.search(
                rf"(?:照片中的人|画面中的人|人物|男人|女人|男孩|女孩|他|她|他们|她们)"
                rf"[^。！？]{{0,16}}(?:感到|感觉|显得|看起来|似乎|很|十分|有些)?[^。！？]{{0,6}}{mood_pattern}",
                text,
                re.I,
            ) is not None
        return re.search(
            rf"\b(?:person|people|man|woman|boy|girl|he|she|they)\b"
            rf"[^.!?]{{0,28}}\b(?:felt|feels|looked|seemed|was|were|became)?\b[^.!?]{{0,8}}{mood_pattern}",
            text,
            re.I,
        ) is not None

    @classmethod
    def extract_claims(
        cls, payload: Mapping[str, Any], context: StoryContext, language: str
    ) -> list[dict[str, Any]]:
        blocks: list[tuple[str, str, Optional[str]]] = [
            ("title", str(payload.get("title") or ""), None),
            ("opening", str(payload.get("opening") or ""), None),
        ]
        for item in payload.get("paragraphs", []):
            if isinstance(item, Mapping):
                blocks.append((
                    "grounded_fact",
                    str(item.get("text") or ""),
                    str(item.get("group_id") or "") or None,
                ))
        for value in payload.get("creative_transitions", []):
            blocks.append(("creative_transition", str(value or ""), None))
        for value in payload.get("mood_reflections", []):
            if isinstance(value, Mapping):
                blocks.append(("verified_photographer_mood", str(value.get("text") or ""), None))
        blocks.append(("closing", str(payload.get("closing") or ""), None))

        claims: list[dict[str, Any]] = []
        for kind, text, group_id in blocks:
            labels = cls._labels_in_text(text)
            if labels and cls._is_mood_claim(text, language, title=kind == "title"):
                claims.append({
                    "block_kind": kind,
                    "group_id": group_id,
                    "mood_labels": sorted(labels),
                    "text": text,
                    "subject_role": "photographer",
                })
        return claims

    @classmethod
    def validate(
        cls, payload: dict[str, Any], context: StoryContext, language: str
    ) -> list[str]:
        errors = FirstPersonMemoirValidator.validate(payload, context, language)
        claims = cls.extract_claims(payload, context, language)
        all_allowed = {
            str(record["mood_label"]) for record in context.photographer_moods
        }
        mood_records_by_evidence = {
            str(record["evidence_id"]): record for record in context.photographer_moods
        }
        allowed_by_group: dict[str, set[str]] = {}
        mood_by_evidence = {
            str(record["evidence_id"]): str(record["mood_label"])
            for record in context.photographer_moods
        }
        for group in context.groups:
            allowed_by_group[group.group_id] = {
                mood_by_evidence[evidence_id]
                for evidence_id in group.evidence_ids
                if evidence_id in mood_by_evidence
            }

        if not context.use_photographer_mood:
            if payload.get("mood_reflections"):
                errors.append("E_MOOD_REFLECTION_UNEXPECTED: mood_reflections must be empty when metadata is disabled")
            if claims:
                errors.append("E_UNSUPPORTED_MOOD: photographer mood metadata is disabled")
        else:
            reflections = payload.get("mood_reflections", [])
            if not isinstance(reflections, list):
                reflections = []
            # mood_reflections is OPTIONAL: mood may now be LLM-woven directly
            # into the narrative (opening/transitions/closing). The structured
            # field, when present, is still validated below.
            seen_reflections: set[str] = set()
            for index, reflection in enumerate(reflections, 1):
                if not isinstance(reflection, Mapping):
                    errors.append(f"E_MOOD_REFLECTION_FORMAT: reflection {index} must be an object")
                    continue
                evidence_id = str(reflection.get("evidence_id") or "")
                label = str(reflection.get("mood_label") or "").casefold()
                text = str(reflection.get("text") or "").strip()
                record = mood_records_by_evidence.get(evidence_id)
                if record is None:
                    errors.append(
                        f"E_MOOD_REFLECTION_EVIDENCE: reflection {index} uses unknown mood evidence {evidence_id!r}"
                    )
                elif label != str(record["mood_label"]):
                    errors.append(
                        f"E_MOOD_REFLECTION_LABEL: reflection {index} label {label!r} does not match evidence {evidence_id}"
                    )
                if evidence_id in seen_reflections:
                    errors.append(f"E_MOOD_REFLECTION_DUPLICATE: duplicate evidence {evidence_id!r}")
                seen_reflections.add(evidence_id)
                labels_in_text = cls._labels_in_text(text)
                if label not in labels_in_text or labels_in_text - {label}:
                    errors.append(
                        f"E_MOOD_REFLECTION_TEXT: reflection {index} must express only its exact mood label {label!r}"
                    )
                if not cls._is_mood_claim(text, language):
                    errors.append(
                        f"E_MOOD_REFLECTION_VOICE: reflection {index} must be an explicit first-person mood statement"
                    )
                language_error = FirstPersonMemoirValidator._field_language_error(
                    f"mood_reflection[{index}]", text, language,
                )
                if language_error:
                    errors.append(language_error)
                for term in GroundingValidator.NARRATIVE_BANNED_TERMS:
                    if GroundingValidator._contains_term(text, term):
                        errors.append(
                            f"E_MOOD_REFLECTION_UNSUPPORTED: reflection {index} contains {term!r}"
                        )
            for claim in claims:
                labels = set(claim["mood_labels"])
                group_id = claim.get("group_id")
                allowed = allowed_by_group.get(str(group_id), set()) if group_id else all_allowed
                unsupported = sorted(labels - allowed)
                if unsupported:
                    errors.append(
                        f"E_UNSUPPORTED_MOOD: {claim['block_kind']} uses unconfirmed mood labels {unsupported}"
                    )
                # Mood is now allowed anywhere in the narrative (opening/groups/
                # transitions/closing), not only in a verified block. Only the
                # label must be confirmed (checked above) and causes are still
                # banned by NARRATIVE_BANNED_TERMS.
            if all_allowed and not claims:
                errors.append("E_MOOD_UNUSED: confirmed photographer mood metadata was not used")

        narrative_values = [
            str(payload.get("opening") or ""),
            *(str(item.get("text") or "") for item in payload.get("paragraphs", []) if isinstance(item, Mapping)),
            *(str(value or "") for value in payload.get("creative_transitions", [])),
            *(str(value.get("text") or "") for value in payload.get("mood_reflections", []) if isinstance(value, Mapping)),
            str(payload.get("closing") or ""),
        ]
        for text in narrative_values:
            if cls._labels_in_text(text) and cls._attributes_mood_to_visible_person(text, language):
                errors.append("E_MOOD_SUBJECT: mood must describe the photographer, not a photographed person")
        return list(dict.fromkeys(errors))


class OllamaGenerator:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self._error_message: Optional[str] = None
        # Set after each generate() so the caller can tell an over-length
        # truncation (done_reason == "length") apart from an ordinary parse
        # failure. num_predict truncation yields invalid JSON downstream.
        self.last_done_reason: Optional[str] = None

    def is_available(self) -> bool:
        try:
            import ollama
            models = ollama.list()
            items = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
            names = [
                getattr(item, "model", None) or (item.get("model") if isinstance(item, dict) else "")
                for item in items
            ]
            available = any(name == self.model or str(name).startswith(self.model + ":") for name in names)
            self._error_message = None if available else f"Model {self.model} is not downloaded"
            return available
        except Exception as exc:
            self._error_message = f"Ollama unavailable: {type(exc).__name__}"
            return False

    def get_error_message(self) -> Optional[str]:
        return self._error_message

    def generate(
        self, system_prompt: str, user_prompt: str,
        temperature: float = STORY_DEFAULT_TEMPERATURE,
        output_schema: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        num_predict: Optional[int] = None,
        **_: Any,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(self.get_error_message() or "Ollama unavailable")
        import ollama
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": OLLAMA_STORY_NUM_CTX,
            "num_predict": int(num_predict) if num_predict else OLLAMA_STORY_NUM_PREDICT,
        }
        if seed is not None:
            options["seed"] = int(seed)
        self.last_done_reason = None
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            format=dict(output_schema) if output_schema is not None else "json",
            think=OLLAMA_THINK,
            options=options,
        )
        self.last_done_reason = (
            response.get("done_reason") if isinstance(response, dict) else getattr(response, "done_reason", None)
        )
        return response["message"]["content"]


class DashScopeGenerator:
    """Remote story backend over the DashScope / Bailian OpenAI-compatible API.

    Implements the exact same contract as ``OllamaGenerator`` so ``StoryGenerator``
    can use either via dependency injection without any change to the core
    generate / validate / repair loop. The API key is read ONLY from the
    ``DASHSCOPE_API_KEY`` environment variable; on any failure (missing key,
    network error, SDK missing) it raises a plain ``RuntimeError`` so the caller
    surfaces an auditable ``E_GENERATION_*`` code and never silently falls back
    to the local model.
    """

    def __init__(
        self,
        model: str = REMOTE_LLM_MODEL,
        *,
        base_url: str = REMOTE_LLM_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = REMOTE_LLM_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url
        # Read from the env var at call time so a key rotated after import is
        # still picked up; an explicit api_key is only used by tests.
        self._api_key_env = "DASHSCOPE_API_KEY"
        self._explicit_api_key = api_key
        self.timeout = timeout
        self._error_message: Optional[str] = None
        # Mirror the Ollama attribute so the generation loop can tell a
        # max_tokens truncation (finish_reason == "length") apart from an
        # ordinary JSON parse failure.
        self.last_done_reason: Optional[str] = None

    def _api_key(self) -> Optional[str]:
        return self._explicit_api_key or os.getenv(self._api_key_env)

    def is_available(self) -> bool:
        key = self._api_key()
        if not key:
            self._error_message = f"{self._api_key_env} environment variable is not set"
            return False
        if "{WorkspaceId}" in self.base_url:
            self._error_message = (
                "REMOTE_LLM_BASE_URL still contains the {WorkspaceId} placeholder; "
                "set SMART_PHOTO_REMOTE_BASE_URL or edit config.py"
            )
            return False
        self._error_message = None
        return True

    def get_error_message(self) -> Optional[str]:
        return self._error_message

    @staticmethod
    def _schema_instruction(output_schema: Optional[Mapping[str, Any]]) -> str:
        """Render the dynamic Story schema into the prompt sent to DashScope.

        DashScope ``json_object`` mode guarantees syntactically valid JSON, but
        it does not enforce our application-specific property names.  Ollama
        receives the schema through its ``format`` parameter; the remote backend
        therefore needs the same contract stated explicitly in the prompt.
        """
        if output_schema is None:
            return ""
        schema_json = json.dumps(
            dict(output_schema), ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        )
        return (
            "\n\nREMOTE STRUCTURED OUTPUT CONTRACT:\n"
            "Return exactly one JSON object that matches the following JSON Schema. "
            "Use every property name exactly as written; do not substitute legacy "
            "fields such as paragraphs/text for groups/factual_text. Do not add "
            "properties that the schema does not allow.\n"
            "<json_schema>\n"
            f"{schema_json}\n"
            "</json_schema>"
        )

    def generate(
        self, system_prompt: str, user_prompt: str,
        temperature: float = STORY_DEFAULT_TEMPERATURE,
        output_schema: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        num_predict: Optional[int] = None,
        **_: Any,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(self.get_error_message() or "Remote LLM unavailable")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                f"openai SDK is not installed ({type(exc).__name__}: {exc}); run: pip install -U openai"
            ) from exc
        client = OpenAI(
            api_key=self._api_key(), base_url=self.base_url, timeout=self.timeout,
        )
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + self._schema_instruction(output_schema),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": int(num_predict) if num_predict else REMOTE_LLM_NUM_PREDICT,
            # DashScope's compatible endpoint reliably honours json_object mode;
            # the prompt already pins the schema and our validator parses and
            # re-validates the JSON, so we do not need a json_schema response format.
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            request_kwargs["seed"] = int(seed)
        response = client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        self.last_done_reason = getattr(choice, "finish_reason", None)
        return getattr(choice.message, "content", None) or ""


class StoryGenerator:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        *,
        aggregator: Optional[ContextAggregator] = None,
        generator: Any = None,
        storage: Optional[PhotoStorage] = None,
    ):
        self.aggregator = aggregator or ContextAggregator(storage=storage)
        self.storage = storage or self.aggregator.storage
        self.generator = generator or OllamaGenerator(model)

    @staticmethod
    def _context_uncertainties(context: StoryContext) -> list[UncertainObservation]:
        values: list[UncertainObservation] = []
        for group in context.groups:
            for conflict in group.conflicts:
                values.append(UncertainObservation(conflict, group.evidence_ids))
        return values

    @staticmethod
    def _event_summary(context: StoryContext) -> dict[str, Any]:
        reliable = [
            item for item in context.evidence
            if item.timestamp and item.timestamp_confidence >= 0.6 and _timestamp_seconds(item.timestamp) is not None
        ]
        reliable.sort(key=lambda item: (_timestamp_seconds(item.timestamp) or 0.0, item.evidence_id))
        date_label = ""
        time_range = ""
        if reliable:
            first = str(reliable[0].timestamp)
            last = str(reliable[-1].timestamp)
            first_date, last_date = first[:10], last[:10]
            date_label = first_date if first_date == last_date else f"{first_date} – {last_date}"
            first_time = first[11:16] if len(first) >= 16 else ""
            last_time = last[11:16] if len(last) >= 16 else ""
            if first_time:
                time_range = first_time if first_time == last_time or not last_time else f"{first_time}–{last_time}"
        return {
            "date": date_label,
            "time_range": time_range,
            "locations": context.locations,
            "reliable_time_count": len(reliable),
            "photo_count": context.photo_count,
        }

    @staticmethod
    def _reduce_groups(groups: Sequence[EvidenceGroup], target: int) -> tuple[EvidenceGroup, ...]:
        """Chronologically merge adjacent groups down to at most ``target``.

        Used only by the coherent first-person memoir so heterogeneous
        selections are woven into a few flowing paragraphs instead of one
        paragraph per photo. Merging is purely time-based and never invents a
        connection; it only narrows how many paragraphs the model must write.
        The closest-in-time adjacent pair is merged first.
        """
        reduced = list(groups)
        while len(reduced) > max(1, target):
            best_index = 0
            best_gap: Optional[float] = None
            for index in range(len(reduced) - 1):
                left_end = _timestamp_seconds(reduced[index].end_at)
                right_start = _timestamp_seconds(reduced[index + 1].start_at)
                gap = (
                    abs(right_start - left_end)
                    if left_end is not None and right_start is not None
                    else math.inf
                )
                if best_gap is None or gap < best_gap:
                    best_gap, best_index = gap, index
            merged = ContextAggregator._merge_groups(reduced[best_index], reduced[best_index + 1])
            reduced[best_index:best_index + 2] = [merged]
        return tuple(
            replace(group, group_id=f"G{index:03d}")
            for index, group in enumerate(reduced, 1)
        )

    @staticmethod
    def _fallback(
        context: StoryContext, language: str, warnings: list[str], prompt_hash: str,
    ) -> GeneratedStory:
        paragraphs: list[StoryParagraph] = []
        zh_templates = (
            "开头的照片组按可靠时间保存了画面中可直接观察的场景与物体。",
            "随后的一组照片记录了可见的人物、物体与周围环境，不加入额外解释。",
            "中间的照片组保留了所选画面中能够直接核查的视觉内容。",
            "接下来的图像继续按时间记录可观察的环境与物体信息。",
            "最后一组照片以画面中可见的场景内容结束这段记录。",
        )
        en_templates = (
            "The opening photographs preserve a direct visual record of their observable scene and objects.",
            "The next chronological group documents visible people, objects, and surroundings without interpretation.",
            "The middle set preserves what is directly visible in the selected photographs.",
            "The following images continue the time-ordered record using only observable visual details.",
            "The final group closes the photographic record with its visible scene content.",
        )
        for index, group in enumerate(context.groups):
            if context.photo_count == 1:
                text = (
                    "这张照片提供了能够直接核查的场景与物体记录。"
                    if language.startswith("zh")
                    else "This photo provides a directly checkable record of its visible scene and objects."
                )
            else:
                text = (zh_templates if language.startswith("zh") else en_templates)[index]
            paragraphs.append(StoryParagraph(text, group.evidence_ids, group.group_id))
        title = ("照片记录" if context.photo_count == 1 else "事件照片记录") if language.startswith("zh") else (
            "Photo note" if context.photo_count == 1 else "Event photo record"
        )
        return GeneratedStory(
            date=context.label, title=title, paragraphs=paragraphs, photo_count=context.photo_count,
            photo_ids=[item.photo_id for item in context.evidence], model="deterministic-group-fallback",
            language=language, status="fallback", warnings=list(warnings),
            uncertain_observations=StoryGenerator._context_uncertainties(context),
            validation_codes=list(dict.fromkeys(warnings)), prompt_hash=prompt_hash,
            event_summary=StoryGenerator._event_summary(context), narrator_role=context.narrator_role,
            prompt_version=PromptBuilder.PROMPT_VERSION,
        )

    def _save_story(
        self,
        story: GeneratedStory,
        context: StoryContext,
        prompt_version: str = PromptBuilder.PROMPT_VERSION,
    ) -> None:
        story.story_id = self.storage.add_story(
            content=story.content, title=story.title, model_name=story.model, event_id=context.event_id,
            style=story.mode, language=story.language, prompt_version=prompt_version,
            grounding={
                "prompt_version": prompt_version,
                "prompt_hash": story.prompt_hash,
                "photo_ids": story.photo_ids,
                "evidence_groups": [group.to_dict() for group in context.groups],
                "conflicts": [value for group in context.groups for value in group.conflicts],
                "citations": story.citations,
                "unused_evidence_ids": story.unused_evidence_ids,
                "uncertain_observations": [value.to_dict() for value in story.uncertain_observations],
                "creative_transitions": story.creative_transitions,
                "opening": story.opening,
                "closing": story.closing,
                "event_summary": story.event_summary,
                "narrator_role": story.narrator_role,
                "narrative_blocks": story.narrative_blocks,
                "use_photographer_mood": story.use_photographer_mood,
                "photographer_moods": story.photographer_moods,
                "mood_claims": story.mood_claims,
                "mood_reflections": story.mood_reflections,
                "validation_codes": story.validation_codes,
                "warnings": story.warnings,
                "source_kind": context.source_kind,
                "verified_context": context.verified_context or None,
            },
        )

    @staticmethod
    def _creative_repair_guidance(
        errors: Sequence[str], context: StoryContext, language: str,
    ) -> str:
        """Translate validator codes into concrete one-shot repair instructions."""
        guidance = [
            "Preserve the exact JSON structure and group order.",
            "Keep every factual statement grounded in the supplied observations and exclude conflicts.",
        ]
        if any("E_FIRST_PERSON_VOICE" in error for error in errors):
            guidance.append(
                "Establish first-person voice explicitly in the opening with 我; later Chinese blocks may omit the subject."
                if language.startswith("zh") else
                "Establish first-person voice with explicit I/my narration and keep natural English first-person grammar."
            )
        if any("E_CONFIRMED_SUBJECT_NO_PERSON" in error for error in errors):
            guidance.append(
                "A group with has_visible_person_evidence=false must use observer voice or direct description; "
                "do not claim that the narrator performs an action in that scene."
            )
        if any("REPETITION" in error or "DUPLICATE" in error for error in errors):
            guidance.append(
                "Rewrite the repeated factual paragraph or transition with distinct concrete details; do not copy the opening."
            )
        if any("E_UNSUPPORTED_MOOD" in error for error in errors):
            guidance.append(
                "Use only the supplied photographer_mood_metadata labels. If metadata is disabled, remove all specific photographer mood claims."
            )
        if any("E_MOOD_UNUSED" in error for error in errors):
            guidance.append(
                "Use at least one supplied photographer mood naturally as the narrator's state while taking a photograph."
            )
        if any("E_MOOD_REFLECTION" in error for error in errors):
            guidance.append(
                "Return at least one mood_reflections object using an exact supplied evidence_id/mood_label pair; its text must explicitly say in first person that the photographer felt that label."
            )
        if any("E_MOOD_SUBJECT" in error for error in errors):
            guidance.append(
                "Attribute mood only to the first-person photographer, never to a visible person in the image."
            )
        if context.narrator_role == "confirmed_subject":
            guidance.append(
                "Use I/me for an action only in groups marked has_visible_person_evidence=true and only when the action is observed."
            )
        return "\n".join(f"- {value}" for value in guidance)

    @staticmethod
    def _paragraphs(payload: Mapping[str, Any], context: StoryContext) -> list[StoryParagraph]:
        result: list[StoryParagraph] = []
        for index, item in enumerate(payload["paragraphs"]):
            group_id = str(item.get("group_id") or (context.groups[index].group_id if index < len(context.groups) else ""))
            result.append(StoryParagraph(
                str(item["text"]).strip(), tuple(str(value) for value in item["evidence_ids"]), group_id,
            ))
        return result

    @staticmethod
    def _mood_reflections(payload: Mapping[str, Any], context: StoryContext) -> list[dict[str, Any]]:
        group_by_evidence = {
            evidence_id: group.group_id
            for group in context.groups
            for evidence_id in group.evidence_ids
        }
        return [
            {
                "evidence_id": str(value.get("evidence_id") or ""),
                "mood_label": str(value.get("mood_label") or ""),
                "text": str(value.get("text") or "").strip(),
                "group_id": group_by_evidence.get(str(value.get("evidence_id") or ""), ""),
                "subject_role": "photographer",
                "source": "manual",
                "confirmed": True,
            }
            for value in payload.get("mood_reflections", [])
            if isinstance(value, Mapping)
        ]

    @staticmethod
    def _deterministic_mood_reflections(
        context: StoryContext, language: str,
    ) -> list[dict[str, str]]:
        """Render one verified sentence per distinct mood in evidence order."""

        if not context.use_photographer_mood:
            return []
        zh_phrases = {
            "neutral": "我的心情较为平常（中性）",
            "calm": "我感到平静",
            "happy": "我感到开心",
            "excited": "我感到兴奋",
            "tense": "我感到紧张",
            "sad": "我感到难过",
        }
        zh_prefixes = (
            "按下这张照片的快门时，",
            "拍摄随后这幅画面时，",
            "记录后面的景色时，",
            "继续拍摄这一幕时，",
            "为最后的画面按下快门时，",
            "回到这段拍摄记录时，",
        )
        en_prefixes = (
            "When I pressed the shutter for this photograph, ",
            "While I photographed the next view, ",
            "As I recorded the later scene, ",
            "While I continued photographing the sequence, ",
            "When I took the final photograph, ",
            "As I returned to this photographic record, ",
        )
        seen: set[str] = set()
        reflections: list[dict[str, str]] = []
        for record in context.photographer_moods:
            label = str(record["mood_label"])
            if label in seen:
                continue
            seen.add(label)
            index = len(reflections)
            if language.startswith("zh"):
                text = zh_prefixes[min(index, len(zh_prefixes) - 1)] + zh_phrases[label] + "。"
            else:
                text = en_prefixes[min(index, len(en_prefixes) - 1)] + f"I felt {label}."
            reflections.append({
                "evidence_id": str(record["evidence_id"]),
                "mood_label": label,
                "text": text,
            })
        return reflections

    def generate_story_with_context(
        self,
        context: StoryContext,
        temperature: float = STORY_DEFAULT_TEMPERATURE,
        *,
        mode: str = "faithful",
        language: str = "zh",
        save: bool = True,
        seed: Optional[int] = None,
        allow_deterministic_fallback: bool = True,
        num_predict: Optional[int] = None,
    ) -> GeneratedStory:
        model_name = str(getattr(self.generator, "model", OLLAMA_MODEL))
        effective_num_predict = int(num_predict) if num_predict else OLLAMA_STORY_NUM_PREDICT
        if not context.evidence or not context.groups:
            return GeneratedStory(
                date=context.label, title="", paragraphs=[], photo_count=0, photo_ids=[], model=model_name,
                mode=mode, language=language, status="error", warnings=["E_NO_PHOTOS: No photos"],
            )
        if len(context.groups) > STORY_NARRATIVE_MAX_PARAGRAPHS:
            # Weave a heterogeneous selection into a few flowing paragraphs
            # instead of one paragraph per photo. Applied to both Faithful and
            # Creative so a large event never forces one paragraph per group.
            context = replace(
                context, groups=self._reduce_groups(context.groups, STORY_NARRATIVE_MAX_PARAGRAPHS)
            )
        prompt_builder = PhotographerMoodPromptBuilder if mode == "creative" else PromptBuilder
        system, base_user = prompt_builder.build_story_prompt(context, mode=mode, language=language)
        prompt_hash = hashlib.sha256((system + "\0" + base_user).encode("utf-8")).hexdigest()
        warnings: list[str] = []
        payload: Optional[dict[str, Any]] = None
        successful_attempt = 0
        user = base_user
        raw_attempts: list[str] = []
        attempt_diagnostics: list[dict[str, Any]] = []
        for attempt in range(2):
            content = ""
            try:
                generation_options: dict[str, Any] = {
                    "output_schema": prompt_builder.output_schema(context, language=language),
                }
                content = self.generator.generate(
                    system, user, temperature=temperature, seed=seed,
                    num_predict=effective_num_predict, **generation_options,
                )
                raw_attempts.append(content)
                candidate = GroundingValidator.parse_payload(content)
                candidate = prompt_builder.normalise_payload(candidate, context)
                if mode == "creative":
                    # Mood is LLM-woven into the narrative (opening/transitions/
                    # closing) when metadata is enabled; no deterministic appendix.
                    candidate["mood_reflections"] = []
                    errors = PhotographerMoodValidator.validate(candidate, context, language)
                else:
                    # Faithful: the production speculation ban is config-gated.
                    # When disabled the validator gets an empty term set, so the
                    # 46 speculation/emotion/causal/kinship terms no longer hard-
                    # reject the narrative; grounding is enforced by prompt
                    # guidance plus the deterministically injected citations.
                    speculation_terms = None if STORY_FAITHFUL_SPECULATION_BAN else ()
                    errors = GroundingValidator.validate(
                        candidate, context, mode, language, speculation_terms=speculation_terms,
                    )
                if not errors:
                    payload = candidate
                    successful_attempt = attempt + 1
                    attempt_diagnostics.append({
                        "attempt": attempt + 1, "phase": "ok",
                        "done_reason": getattr(self.generator, "last_done_reason", None), "codes": [],
                    })
                    break
                warnings.extend(errors)
                attempt_diagnostics.append({
                    "attempt": attempt + 1, "phase": "validation",
                    "done_reason": getattr(self.generator, "last_done_reason", None), "codes": list(errors),
                })
                if mode == "creative":
                    user = (
                        base_user
                        + "\n\nThe previous output failed strict validation. Repair it once and return corrected JSON only."
                        + "\nPrevious rejected output:\n<rejected_output>\n"
                        + content
                        + "\n</rejected_output>\nValidation errors:\n- "
                        + "\n- ".join(errors)
                        + "\nRepair guidance:\n"
                        + self._creative_repair_guidance(errors, context, language)
                    )
                else:
                    user = (
                        base_user
                        + "\n\nThe previous output failed strict validation. Repair it once and return corrected JSON only."
                        + "\nPrevious rejected output:\n<rejected_output>\n"
                        + content
                        + "\n</rejected_output>\nValidation errors:\n- "
                        + "\n- ".join(errors)
                        + "\nThe corrected object must use the exact title + groups[{group_id, factual_text}] "
                        "contract supplied in the system JSON Schema."
                    )
            except Exception as exc:
                done_reason = getattr(self.generator, "last_done_reason", None)
                if done_reason == "length" and isinstance(exc, (json.JSONDecodeError, ValueError)):
                    error = (
                        f"E_LENGTH_TRUNCATED_{attempt + 1}: attempt {attempt + 1} exceeded "
                        f"num_predict={effective_num_predict}; output was truncated to invalid JSON"
                    )
                else:
                    error = f"E_GENERATION_{attempt + 1}: {type(exc).__name__}: {exc}"
                warnings.append(error)
                attempt_diagnostics.append({
                    "attempt": attempt + 1, "phase": "generation",
                    "done_reason": done_reason, "codes": [error],
                })
                if mode == "creative" and attempt == 0:
                    user = (
                        base_user
                        + "\n\nThe previous output could not be parsed or validated. Repair it once and return corrected JSON only."
                        + "\nPrevious rejected output:\n<rejected_output>\n"
                        + content
                        + "\n</rejected_output>\nValidation errors:\n- "
                        + error
                    )
        if payload is None:
            if mode == "creative" or not allow_deterministic_fallback:
                # Surface an explicit, auditable failure with per-attempt reasons
                # instead of substituting a low-information deterministic template.
                if mode == "creative":
                    fail_title = "创意故事生成未通过校验" if language.startswith("zh") else "Creative story validation failed"
                else:
                    fail_title = "故事生成未通过校验" if language.startswith("zh") else "Story validation failed"
                return GeneratedStory(
                    date=context.label,
                    title=fail_title,
                    paragraphs=[],
                    photo_count=context.photo_count,
                    photo_ids=[item.photo_id for item in context.evidence],
                    model=model_name,
                    mode=mode,
                    language=language,
                    status="error",
                    warnings=list(warnings),
                    validation_codes=list(dict.fromkeys(warnings)),
                    prompt_hash=prompt_hash,
                    event_summary=self._event_summary(context),
                    narrator_role=context.narrator_role,
                    prompt_version=prompt_builder.PROMPT_VERSION,
                    use_photographer_mood=context.use_photographer_mood,
                    photographer_moods=context.photographer_moods,
                    raw_attempts=raw_attempts,
                    attempt_diagnostics=attempt_diagnostics,
                )
            story = self._fallback(context, language, warnings, prompt_hash)
            story.mode = mode
            story.attempt_diagnostics = attempt_diagnostics
            if save:
                self._save_story(story, context, prompt_builder.PROMPT_VERSION)
            return story

        uncertain = self._context_uncertainties(context)
        known_uncertain = {(item.text, item.evidence_ids) for item in uncertain}
        for item in payload.get("uncertain_observations", []):
            value = UncertainObservation(
                str(item["text"]).strip(), tuple(str(evidence_id) for evidence_id in item.get("evidence_ids", [])),
            )
            if (value.text, value.evidence_ids) not in known_uncertain:
                uncertain.append(value)
        story = GeneratedStory(
            date=context.label, title=str(payload.get("title") or context.label),
            paragraphs=self._paragraphs(payload, context), photo_count=context.photo_count,
            photo_ids=[item.photo_id for item in context.evidence], model=model_name, mode=mode,
            language=language, status="repaired" if successful_attempt == 2 else "ok",
            warnings=warnings, uncertain_observations=uncertain,
            unused_evidence_ids=[str(value) for value in payload.get("unused_evidence_ids", [])],
            creative_transitions=[str(value) for value in payload.get("creative_transitions", [])],
            opening=str(payload.get("opening") or "").strip(), closing=str(payload.get("closing") or "").strip(),
            event_summary=self._event_summary(context), narrator_role=context.narrator_role,
            prompt_version=prompt_builder.PROMPT_VERSION,
            validation_codes=list(dict.fromkeys(warnings)), prompt_hash=prompt_hash,
            use_photographer_mood=context.use_photographer_mood,
            photographer_moods=context.photographer_moods,
            mood_claims=PhotographerMoodValidator.extract_claims(payload, context, language) if mode == "creative" else [],
            mood_reflections=self._mood_reflections(payload, context) if mode == "creative" else [],
            raw_attempts=raw_attempts,
            attempt_diagnostics=attempt_diagnostics,
        )
        if save:
            self._save_story(story, context, prompt_builder.PROMPT_VERSION)
        return story

    def generate_story(
        self,
        date: str,
        temperature: float = STORY_DEFAULT_TEMPERATURE,
        stream: bool = False,
        *,
        mode: str = "faithful",
        language: str = "zh",
        verified_context: str = "",
        narrator_role: str = "observer",
    ) -> GeneratedStory:
        del stream
        return self.generate_story_with_context(
            self.aggregator.aggregate_by_date(
                date, verified_context=verified_context, narrator_role=narrator_role,
            ),
            temperature, mode=mode, language=language,
        )

    def generate_from_photo_ids(
        self,
        photo_ids: Sequence[str],
        *,
        label: str = "Selected photos",
        temperature: float = STORY_DEFAULT_TEMPERATURE,
        mode: str = "faithful",
        language: str = "zh",
        user_notes: Optional[dict[str, str]] = None,
        verified_context: str = "",
        narrator_role: str = "observer",
        source_kind: str = "basket",
        event_id: Optional[str] = None,
        save: bool = True,
        use_photographer_mood: bool = False,
        seed: Optional[int] = None,
    ) -> GeneratedStory:
        context = self.aggregator.aggregate_by_photo_ids(
            photo_ids, label=label, event_id=event_id, user_notes=user_notes,
            verified_context=verified_context, source_kind=source_kind, narrator_role=narrator_role,
            use_photographer_mood=use_photographer_mood,
        )
        return self.generate_story_with_context(
            context, temperature, mode=mode, language=language, save=save, seed=seed,
        )

    def get_context(self, date: str) -> StoryContext:
        return self.aggregator.aggregate_by_date(date)


_story_generators: dict[str, StoryGenerator] = {}


def get_story_generator(
    model: str = OLLAMA_MODEL,
    *,
    backend: str = REMOTE_LLM_DEFAULT_BACKEND,
) -> StoryGenerator:
    """Return a cached ``StoryGenerator`` for the requested backend.

    ``backend="local"`` (default) uses the local Ollama model and is the only
    path the frozen experiments ever need. ``backend="remote"`` injects a
    ``DashScopeGenerator`` pointing at the configured DashScope OpenAI-compatible
    endpoint; remote failures are raised by the generator and never fall back to
    the local model. One instance is cached per backend so switching in the UI
    does not rebuild the aggregator/storage pipeline.
    """
    backend = backend if backend in REMOTE_LLM_BACKENDS else REMOTE_LLM_DEFAULT_BACKEND
    cached = _story_generators.get(backend)
    if cached is not None:
        return cached
    if backend == "remote":
        instance = StoryGenerator(
            REMOTE_LLM_MODEL, generator=DashScopeGenerator(REMOTE_LLM_MODEL),
        )
    else:
        instance = StoryGenerator(model)
    _story_generators[backend] = instance
    return instance


def generate_diary(date: str, temperature: float = STORY_DEFAULT_TEMPERATURE) -> GeneratedStory:
    return get_story_generator().generate_story(date, temperature)
