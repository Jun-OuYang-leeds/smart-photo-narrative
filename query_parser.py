"""Deterministic-first query parsing with an optional local Ollama fallback.

The parser deliberately keeps hard metadata filters separate from visual text.
This makes retrieval reproducible when Ollama is unavailable while still
allowing richer Chinese/English temporal expressions when it is running.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from config import (
    OLLAMA_MODEL,
    OLLAMA_QUERY_NUM_CTX,
    OLLAMA_QUERY_NUM_PREDICT,
    OLLAMA_THINK,
)


@dataclass(frozen=True)
class TemporalQuery:
    before: Optional[str] = None
    after: Optional[str] = None
    anchor: Optional[str] = None
    direction: Optional[str] = None  # before|after
    window_minutes: int = 240

    @property
    def is_pair(self) -> bool:
        return bool(self.before and self.after)

    @property
    def is_neighbor(self) -> bool:
        return bool(self.anchor and self.direction)


@dataclass(frozen=True)
class ParsedQuery:
    original: str
    visual_text: str
    relation_text: Optional[str] = None
    temporal: Optional[TemporalQuery] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    parser_source: str = "rules"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def query_type(self) -> str:
        if self.temporal and self.temporal.is_pair:
            return "temporal_pair"
        if self.temporal and self.temporal.is_neighbor:
            return "temporal_neighbor"
        if self.relation_text:
            return "relation"
        return "standard"


_DURATION_RE = re.compile(
    r"(?:within|in|间隔|之内|以内)\s*(\d+)\s*(minutes?|mins?|分钟|hours?|hrs?|小时|days?|天)",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_RELATION_HINT_RE = re.compile(
    r"\b(using|holding|wearing|riding|sitting on|standing (?:by|near)|next to|beside|under|above|behind|in front of|on top of)\b"
    r"|使用|拿着|手持|佩戴|骑|坐在|站在|旁边|附近|下面|上面|后面|前面|放在|位于",
    re.IGNORECASE,
)


def _duration_minutes(text: str, default: int) -> int:
    match = _DURATION_RE.search(text)
    if not match:
        return default
    value = max(1, int(match.group(1)))
    unit = match.group(2).lower()
    if unit.startswith(("day", "天")):
        return value * 24 * 60
    if unit.startswith(("hour", "hr", "小时")):
        return value * 60
    return value


def _clean_clause(value: str) -> str:
    value = _DURATION_RE.sub("", value)
    return value.strip(" ,，。?？;；")


def parse_with_rules(query: str, default_window_minutes: int = 240) -> ParsedQuery:
    """Parse common temporal/relation forms without any model dependency."""
    original = query.strip()
    window = _duration_minutes(original, default_window_minutes)
    temporal: Optional[TemporalQuery] = None

    # A before B / A 在 B 之前.  The non-greedy groups intentionally retain
    # full event phrases rather than extracting individual keywords.
    match = re.match(r"^(.+?)\s+before\s+(.+?)(?:\s+(?:within|in)\s+\d+\s+\w+)?$", original, re.I)
    if match and _clean_clause(match.group(1)).lower() != "what happened":
        temporal = TemporalQuery(
            before=_clean_clause(match.group(1)),
            after=_clean_clause(match.group(2)),
            window_minutes=window,
        )
    if temporal is None:
        match = re.match(r"^(.+?)\s+after\s+(.+?)(?:\s+(?:within|in)\s+\d+\s+\w+)?$", original, re.I)
        if match and _clean_clause(match.group(1)).lower() != "what happened":
            temporal = TemporalQuery(
                before=_clean_clause(match.group(2)),
                after=_clean_clause(match.group(1)),
                window_minutes=window,
            )

    # Chinese pair forms: A之后B / A以后B / A之前B.
    if temporal is None:
        match = re.match(r"^(.+?)(?:之后|以后|过后)(.+?)$", _DURATION_RE.sub("", original))
        if match:
            temporal = TemporalQuery(
                before=_clean_clause(match.group(1)),
                after=_clean_clause(match.group(2)),
                window_minutes=window,
            )
    if temporal is None:
        match = re.match(r"^(.+?)(?:之前|以前)(.+?)$", _DURATION_RE.sub("", original))
        if match and _clean_clause(match.group(1)) and _clean_clause(match.group(2)):
            temporal = TemporalQuery(
                before=_clean_clause(match.group(1)),
                after=_clean_clause(match.group(2)),
                window_minutes=window,
            )

    # One-anchor neighborhood forms are evaluated only after pair forms.
    if temporal is None:
        match = re.match(r"^(?:what happened\s+)?before\s+(.+)$", original, re.I)
        if match:
            temporal = TemporalQuery(anchor=_clean_clause(match.group(1)), direction="before", window_minutes=window)
    if temporal is None:
        match = re.match(r"^(?:what happened\s+)?after\s+(.+)$", original, re.I)
        if match:
            temporal = TemporalQuery(anchor=_clean_clause(match.group(1)), direction="after", window_minutes=window)
    if temporal is None:
        match = re.match(r"^(.+?)(?:之前|以前)(?:发生了什么|我在做什么|的照片)?[？?]?$", original)
        if match:
            temporal = TemporalQuery(anchor=_clean_clause(match.group(1)), direction="before", window_minutes=window)
    if temporal is None:
        match = re.match(r"^(.+?)(?:之后|以后)(?:发生了什么|我在做什么|的照片)?[？?]?$", original)
        if match:
            temporal = TemporalQuery(anchor=_clean_clause(match.group(1)), direction="after", window_minutes=window)

    dates = _ISO_DATE_RE.findall(original)
    start_date = dates[0] if dates else None
    end_date = dates[1] if len(dates) > 1 else start_date
    relation = original if _RELATION_HINT_RE.search(original) else None

    visual_text = original
    if temporal and temporal.is_pair:
        visual_text = f"{temporal.before} {temporal.after}"
    elif temporal and temporal.is_neighbor:
        visual_text = temporal.anchor or original

    return ParsedQuery(
        original=original,
        visual_text=visual_text,
        relation_text=relation,
        temporal=temporal,
        start_date=start_date,
        end_date=end_date,
    )


class OllamaQueryParser:
    """Optional structured parser; failures always fall back to rules."""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def parse(self, query: str, rule_result: ParsedQuery) -> ParsedQuery:
        if rule_result.temporal is not None:
            return rule_result
        try:
            import ollama

            prompt = (
                "Return JSON only. Parse the photo-search query into: visual_text, relation_text, "
                "before, after, anchor, direction(before|after|null), window_minutes, start_date, "
                "end_date, location. Do not invent absent constraints. Query: " + query
            )
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                think=OLLAMA_THINK,
                options={
                    "temperature": 0,
                    "num_ctx": OLLAMA_QUERY_NUM_CTX,
                    "num_predict": OLLAMA_QUERY_NUM_PREDICT,
                },
            )
            payload: Dict[str, Any] = json.loads(response["message"]["content"])
            temporal = None
            before = payload.get("before")
            after = payload.get("after")
            anchor = payload.get("anchor")
            direction = payload.get("direction")
            if before and after:
                temporal = TemporalQuery(before=str(before), after=str(after), window_minutes=int(payload.get("window_minutes") or 240))
            elif anchor and direction in {"before", "after"}:
                temporal = TemporalQuery(anchor=str(anchor), direction=direction, window_minutes=int(payload.get("window_minutes") or 240))
            return ParsedQuery(
                original=query,
                visual_text=str(payload.get("visual_text") or rule_result.visual_text),
                relation_text=payload.get("relation_text") or rule_result.relation_text,
                temporal=temporal,
                start_date=payload.get("start_date") or rule_result.start_date,
                end_date=payload.get("end_date") or rule_result.end_date,
                location=payload.get("location"),
                parser_source="ollama",
            )
        except Exception as exc:
            return ParsedQuery(
                **{**rule_result.__dict__, "warnings": (f"Ollama parser unavailable: {type(exc).__name__}",)}
            )


def parse_query(query: str, use_ollama: bool = False, model: str = OLLAMA_MODEL) -> ParsedQuery:
    result = parse_with_rules(query)
    if use_ollama:
        return OllamaQueryParser(model).parse(query, result)
    return result
