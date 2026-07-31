"""Deterministic event segmentation for timestamped personal photos."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class PhotoEventInput:
    photo_id: str
    captured_at: Optional[str]
    captured_at_sort: Optional[int]
    timestamp_confidence: str
    clip_embedding: Optional[Sequence[float]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    caption: str = ""
    location: str = ""

    @property
    def date_local(self) -> Optional[str]:
        return self.captured_at[:10] if self.captured_at else None


@dataclass(frozen=True)
class EventCandidate:
    event_id: str
    date_local: str
    start_at: str
    end_at: str
    photo_ids: tuple[str, ...]
    representative_ids: tuple[str, ...]
    boundary_reasons: tuple[str, ...] = field(default_factory=tuple)
    confidence: str = "automatic"


@dataclass(frozen=True)
class EventOrganization:
    events: tuple[EventCandidate, ...]
    unassigned_photo_ids: tuple[str, ...]


def _cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    # NumPy arrays deliberately reject implicit truth-value testing.  Dense
    # embeddings loaded from Chroma are arrays, while unit tests often use
    # lists, so test the length explicitly for both representations.
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return None
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def _distance_km(a: PhotoEventInput, b: PhotoEventInput) -> Optional[float]:
    if None in {a.latitude, a.longitude, b.latitude, b.longitude}:
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, [a.latitude, a.longitude, b.latitude, b.longitude])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _boundary_reason(previous: PhotoEventInput, current: PhotoEventInput) -> Optional[str]:
    if previous.date_local != current.date_local:
        return "date_change"
    gap_minutes = ((current.captured_at_sort or 0) - (previous.captured_at_sort or 0)) / 60
    if gap_minutes > 90:
        return "time_gap_gt_90m"
    similarity = _cosine(previous.clip_embedding, current.clip_embedding)
    if gap_minutes > 30 and similarity is not None and similarity < 0.65:
        return "time_and_visual_change"
    distance = _distance_km(previous, current)
    if gap_minutes > 10 and distance is not None and distance > 2:
        return "location_change_gt_2km"
    return None


def _event_id(date_local: str, photo_ids: Sequence[str], ordinal: int) -> str:
    value = f"{date_local}:{ordinal}:" + "\0".join(photo_ids)
    return str(uuid.uuid5(uuid.UUID("72da4ef8-1ee2-46a9-a33b-3017484ad0e7"), value))


def _representatives(photos: Sequence[PhotoEventInput], limit: int = 8) -> tuple[str, ...]:
    if len(photos) <= limit:
        return tuple(photo.photo_id for photo in photos)
    chosen = [0, len(photos) - 1]
    candidates = set(range(1, len(photos) - 1))
    while candidates and len(chosen) < limit:
        best_idx = None
        best_distance = -1.0
        for idx in sorted(candidates):
            similarities = [
                _cosine(photos[idx].clip_embedding, photos[selected].clip_embedding)
                for selected in chosen
            ]
            valid = [value for value in similarities if value is not None]
            if valid:
                distance = 1.0 - max(valid)
            else:
                distance = min(abs(idx - selected) for selected in chosen) / max(len(photos) - 1, 1)
            if distance > best_distance:
                best_idx, best_distance = idx, distance
        chosen.append(best_idx)
        candidates.remove(best_idx)
    return tuple(photos[index].photo_id for index in sorted(chosen))


def organize_events(photos: Iterable[PhotoEventInput]) -> EventOrganization:
    """Segment reliable photos; low-confidence mtime photos remain explicit."""
    source = list(photos)
    reliable = [
        photo for photo in source
        if photo.timestamp_confidence in {"high", "medium"}
        and photo.captured_at_sort is not None
        and photo.captured_at
    ]
    reliable.sort(key=lambda photo: (photo.captured_at_sort, photo.photo_id))
    reliable_ids = {photo.photo_id for photo in reliable}
    unassigned = tuple(sorted(photo.photo_id for photo in source if photo.photo_id not in reliable_ids))
    if not reliable:
        return EventOrganization(events=(), unassigned_photo_ids=unassigned)

    groups: list[tuple[list[PhotoEventInput], list[str]]] = []
    current = [reliable[0]]
    reasons: list[str] = []
    for photo in reliable[1:]:
        reason = _boundary_reason(current[-1], photo)
        if reason:
            groups.append((current, reasons))
            current = [photo]
            reasons = [reason]
        else:
            current.append(photo)
    groups.append((current, reasons))

    events = []
    day_ordinals: dict[str, int] = {}
    for group, boundary_reasons in groups:
        date_local = group[0].date_local or "unknown"
        day_ordinals[date_local] = day_ordinals.get(date_local, 0) + 1
        photo_ids = tuple(photo.photo_id for photo in group)
        events.append(EventCandidate(
            event_id=_event_id(date_local, photo_ids, day_ordinals[date_local]),
            date_local=date_local,
            start_at=group[0].captured_at or "",
            end_at=group[-1].captured_at or "",
            photo_ids=photo_ids,
            representative_ids=_representatives(group),
            boundary_reasons=tuple(boundary_reasons),
        ))
    return EventOrganization(events=tuple(events), unassigned_photo_ids=unassigned)


def split_event(event: EventCandidate, before_photo_id: str) -> tuple[EventCandidate, EventCandidate]:
    """Split immediately before a member, preserving deterministic manual IDs."""
    index = event.photo_ids.index(before_photo_id)
    if index <= 0:
        raise ValueError("split point must not be the first photo")
    left_ids, right_ids = event.photo_ids[:index], event.photo_ids[index:]
    left = replace(event, event_id=f"{event.event_id}_a", photo_ids=left_ids,
                   representative_ids=tuple(pid for pid in event.representative_ids if pid in left_ids),
                   confidence="manual")
    right = replace(event, event_id=f"{event.event_id}_b", photo_ids=right_ids,
                    representative_ids=tuple(pid for pid in event.representative_ids if pid in right_ids),
                    confidence="manual")
    return left, right


def merge_events(first: EventCandidate, second: EventCandidate) -> EventCandidate:
    if first.date_local != second.date_local:
        raise ValueError("events on different dates cannot be merged")
    ids = first.photo_ids + tuple(pid for pid in second.photo_ids if pid not in first.photo_ids)
    reps = tuple(dict.fromkeys(first.representative_ids + second.representative_ids))[:8]
    return EventCandidate(
        event_id=f"{first.event_id}_merged",
        date_local=first.date_local,
        start_at=min(first.start_at, second.start_at),
        end_at=max(first.end_at, second.end_at),
        photo_ids=ids,
        representative_ids=reps,
        boundary_reasons=("manual_merge",),
        confidence="manual",
    )
