"""Independent multimodal recall, metadata pre-filtering, and weighted RRF."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from config import (
    APP_DB_PATH,
    CLIP_INDEX_MARKER_TAG,
    OLLAMA_MODEL,
    PHOTOS_DIR,
    RETRIEVAL_CANDIDATE_K,
    RRF_K,
    SEARCH_TOP_K,
)
from query_parser import ParsedQuery, parse_query
from query_translation import translate_query_to_english
from storage import BM25Result, PhotoStorage
from vector_store import ChromaVectorStore, VectorHit


# Searched-optimal weights (clip 1.0, caption 0.4, scene_graph 0.2) from
# retrieval_human_ablation.json::grid_b_all_channels.best. The report still
# documents the intuitive baseline (clip 1.0, caption 0.8, scene_graph 0.9) as the
# shipped config; this live override does not change the frozen evaluation tables.
CHANNEL_WEIGHTS = {"clip": 1.0, "caption": 0.4, "scene_graph": 0.2}


@dataclass(frozen=True)
class SearchFilters:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tags: tuple[str, ...] = ()
    location: Optional[str] = None
    min_timestamp_confidence: Optional[float] = None
    event_id: Optional[str] = None


@dataclass
class SearchResult:
    photo_id: str
    id: str  # relative path retained for the existing Streamlit UI
    score: float
    datetime: Optional[str]
    timestamp_source: str
    timestamp_confidence: float
    location: Optional[str]
    tags: list[str]
    caption: str
    scene_graph_text: str
    image_width: int
    image_height: int
    channel_scores: dict[str, float] = field(default_factory=dict)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    matched_modalities: list[str] = field(default_factory=list)
    matched_caption_terms: list[str] = field(default_factory=list)
    matched_scene_graph_triples: list[str] = field(default_factory=list)
    filters_applied: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    @property
    def image_path(self) -> str:
        return str(PHOTOS_DIR / self.id)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["image_path"] = self.image_path
        return payload


@dataclass(frozen=True)
class TemporalPair:
    before: SearchResult
    after: SearchResult
    delta_minutes: float
    score: float


@dataclass
class SearchResponse:
    query: ParsedQuery
    results: list[SearchResult] = field(default_factory=list)
    temporal_pairs: list[TemporalPair] = field(default_factory=list)
    eligible_count: int = 0
    channels_used: tuple[str, ...] = ()


class MultimodalRetriever:
    def __init__(
        self,
        storage: Optional[PhotoStorage] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        clip_backend: Any = None,
    ):
        self.storage = storage or PhotoStorage(APP_DB_PATH)
        self.vector_store = vector_store or ChromaVectorStore()
        self._clip_backend = clip_backend

    @property
    def clip_backend(self):
        if self._clip_backend is None:
            from model_pipeline import CLIPModelManager

            self._clip_backend = CLIPModelManager()
        return self._clip_backend

    def _eligible_ids(self, filters: SearchFilters) -> list[str]:
        return self.storage.metadata_eligible_ids(
            date_from=filters.start_date,
            date_to=filters.end_date,
            min_timestamp_confidence=filters.min_timestamp_confidence,
            tags_any=filters.tags,
            location_contains=filters.location,
            event_id=filters.event_id,
        )

    @staticmethod
    def _filter_summary(filters: SearchFilters) -> dict[str, Any]:
        return {
            key: value for key, value in {
                "start_date": filters.start_date,
                "end_date": filters.end_date,
                "tags": list(filters.tags) if filters.tags else None,
                "location": filters.location,
                "min_timestamp_confidence": filters.min_timestamp_confidence,
                "event_id": filters.event_id,
            }.items() if value is not None
        }

    def _documents(self, photo_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not photo_ids:
            return {}
        photo_ids = list(dict.fromkeys(photo_ids))
        with self.storage.transaction(write=False) as connection:
            connection.execute("CREATE TEMP TABLE requested_photo_ids(photo_id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO requested_photo_ids(photo_id) VALUES (?)",
                [(photo_id,) for photo_id in photo_ids],
            )
            rows = connection.execute(
                """
                SELECT p.*,
                       COALESCE(c.caption, '') AS caption,
                       COALESCE(s.scene_graph_text, '') AS scene_graph_text,
                       COALESCE(group_concat(DISTINCT t.tag), '') AS tags
                FROM photos p
                JOIN requested_photo_ids r ON r.photo_id = p.photo_id
                LEFT JOIN captions c ON c.photo_id = p.photo_id AND c.is_primary = 1 AND c.status = 'ok'
                LEFT JOIN scene_graphs s ON s.photo_id = p.photo_id AND s.is_primary = 1 AND s.status = 'ok'
                LEFT JOIN tags t ON t.photo_id = p.photo_id
                GROUP BY p.photo_id
                """
            ).fetchall()
            triples = connection.execute(
                """
                SELECT sg.photo_id, st.subject, st.relation, st.object
                FROM scene_graphs sg
                JOIN scene_graph_triples st ON st.scene_graph_id = sg.scene_graph_id
                JOIN requested_photo_ids r ON r.photo_id = sg.photo_id
                WHERE sg.is_primary = 1 AND sg.status = 'ok'
                ORDER BY sg.photo_id, st.ordinal
                """
            ).fetchall()
        result = {row["photo_id"]: dict(row) for row in rows}
        for item in result.values():
            item["triples"] = []
        for row in triples:
            result[row["photo_id"]]["triples"].append(
                f"{row['subject']} {row['relation']} {row['object']}"
            )
        return result

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [token.casefold() for token in re.findall(r"[\w-]+", text) if len(token) > 1]

    def _assemble(
        self,
        ranked: Mapping[str, Sequence[Any]],
        query_text: str,
        top_k: int,
        filters: SearchFilters,
    ) -> list[SearchResult]:
        contributions: dict[str, dict[str, Any]] = {}
        channels_used = [channel for channel, hits in ranked.items() if hits]
        for channel, hits in ranked.items():
            weight = CHANNEL_WEIGHTS[channel]
            for rank, hit in enumerate(hits, 1):
                photo_id = hit.photo_id
                info = contributions.setdefault(photo_id, {"raw": 0.0, "ranks": {}, "scores": {}, "triples": []})
                info["raw"] += weight / (RRF_K + rank)
                info["ranks"][channel] = rank
                if isinstance(hit, VectorHit):
                    info["scores"][channel] = hit.score
                    if channel == "scene_graph" and hit.metadata.get("triple_text"):
                        info["triples"].append(str(hit.metadata["triple_text"]))
                elif isinstance(hit, BM25Result):
                    info["scores"][channel] = -hit.bm25_rank

        denominator = sum(CHANNEL_WEIGHTS[channel] / (RRF_K + 1) for channel in channels_used) or 1.0
        ordered = sorted(contributions, key=lambda pid: (-contributions[pid]["raw"], pid))[:top_k]
        docs = self._documents(ordered)
        query_terms = self._terms(query_text)
        filter_summary = self._filter_summary(filters)
        results = []
        for photo_id in ordered:
            doc = docs.get(photo_id)
            if not doc:
                continue
            info = contributions[photo_id]
            caption_lower = doc["caption"].casefold()
            matched_caption = sorted({term for term in query_terms if term in caption_lower})
            triple_matches = list(dict.fromkeys(info["triples"]))
            if not triple_matches:
                triple_matches = [triple for triple in doc["triples"] if any(term in triple.casefold() for term in query_terms)][:3]
            modalities = sorted(info["ranks"], key=lambda channel: info["ranks"][channel])
            explanation_parts = [f"{channel} rank {info['ranks'][channel]}" for channel in modalities]
            result = SearchResult(
                photo_id=photo_id,
                id=doc["relative_path"],
                score=max(0.0, min(1.0, info["raw"] / denominator)),
                datetime=doc["captured_at"],
                timestamp_source=doc["timestamp_source"],
                timestamp_confidence=float(doc["timestamp_confidence"]),
                location=doc["location"],
                tags=[value for value in doc["tags"].split(",") if value],
                caption=doc["caption"],
                scene_graph_text=doc["scene_graph_text"],
                image_width=int(doc["image_width"] or 0),
                image_height=int(doc["image_height"] or 0),
                channel_scores=dict(info["scores"]),
                channel_ranks=dict(info["ranks"]),
                matched_modalities=modalities,
                matched_caption_terms=matched_caption,
                matched_scene_graph_triples=triple_matches,
                filters_applied=filter_summary,
                explanation="; ".join(explanation_parts),
            )
            results.append(result)
        return results

    def search_standard(
        self,
        query: str,
        *,
        top_k: int = SEARCH_TOP_K,
        filters: Optional[SearchFilters] = None,
        enabled_channels: Iterable[str] = ("clip", "caption", "scene_graph"),
        parsed_query: Optional[ParsedQuery] = None,
    ) -> SearchResponse:
        filters = filters or SearchFilters()
        parsed = parsed_query or parse_query(query)
        channels = tuple(channel for channel in enabled_channels if channel in CHANNEL_WEIGHTS)
        eligible = self._eligible_ids(filters)
        candidate_k = max(top_k * 4, min(RETRIEVAL_CANDIDATE_K, max(len(eligible), top_k)))
        translated = translate_query_to_english(parsed.visual_text)
        relation_text = translate_query_to_english(parsed.relation_text or parsed.visual_text)
        ranked: dict[str, Sequence[Any]] = {channel: [] for channel in channels}
        query_embedding = None
        if "clip" in channels or "scene_graph" in channels:
            query_embedding = self.clip_backend.encode_text(translated)
        if "clip" in channels:
            ranked["clip"] = self.vector_store.query_images(
                query_embedding, top_k=candidate_k, eligible_photo_ids=eligible
            )
        if "caption" in channels:
            try:
                ranked["caption"] = self.storage.search_bm25(
                    translated,
                    limit=candidate_k,
                    eligible_photo_ids=eligible,
                    match_all_terms=False,
                    fields=("caption",),
                )
            except ValueError:
                ranked["caption"] = []
        if "scene_graph" in channels:
            if relation_text != translated:
                query_embedding = self.clip_backend.encode_text(relation_text)
            ranked["scene_graph"] = self.vector_store.query_scene_graph(
                query_embedding, top_k=candidate_k, eligible_photo_ids=eligible
            )
        results = self._assemble(ranked, translated, top_k, filters)
        return SearchResponse(parsed, results, eligible_count=len(eligible), channels_used=channels)

    def search(
        self,
        query: str,
        *,
        top_k: int = SEARCH_TOP_K,
        filters: Optional[SearchFilters] = None,
        enabled_channels: Iterable[str] = ("clip", "caption", "scene_graph"),
        use_ollama_parser: bool = False,
    ) -> SearchResponse:
        parsed = parse_query(query, use_ollama=use_ollama_parser, model=OLLAMA_MODEL)
        if parsed.temporal and parsed.temporal.is_pair:
            return self.search_temporal_pair(parsed, top_k=top_k, filters=filters, enabled_channels=enabled_channels)
        if parsed.temporal and parsed.temporal.is_neighbor:
            return self.search_temporal_neighbor(
                parsed, top_k=top_k, filters=filters, enabled_channels=enabled_channels
            )
        return self.search_standard(
            parsed.visual_text,
            top_k=top_k,
            filters=filters,
            enabled_channels=enabled_channels,
            parsed_query=parsed,
        )

    def search_temporal_pair(
        self,
        parsed: ParsedQuery,
        *,
        top_k: int,
        filters: Optional[SearchFilters],
        enabled_channels: Iterable[str],
    ) -> SearchResponse:
        temporal = parsed.temporal
        if not temporal or not temporal.is_pair:
            raise ValueError("A temporal pair query is required")
        base = filters or SearchFilters()
        reliable_filters = SearchFilters(
            start_date=base.start_date,
            end_date=base.end_date,
            tags=base.tags,
            location=base.location,
            min_timestamp_confidence=max(base.min_timestamp_confidence or 0.0, 0.6),
            event_id=base.event_id,
        )
        before = self.search_standard(temporal.before or "", top_k=max(20, top_k * 4), filters=reliable_filters, enabled_channels=enabled_channels)
        after = self.search_standard(temporal.after or "", top_k=max(20, top_k * 4), filters=reliable_filters, enabled_channels=enabled_channels)
        pairs = []
        max_seconds = temporal.window_minutes * 60
        docs = self._documents([item.photo_id for item in before.results + after.results])
        for first in before.results:
            first_doc = docs[first.photo_id]
            first_time = first_doc["captured_at_sort"]
            if first_time is None:
                continue
            for second in after.results:
                second_doc = docs[second.photo_id]
                second_time = second_doc["captured_at_sort"]
                if second_time is None or first_doc["date_local"] != second_doc["date_local"]:
                    continue
                delta = int(second_time) - int(first_time)
                if 0 < delta <= max_seconds:
                    closeness = 1.0 - delta / max_seconds
                    pair_score = ((first.score + second.score) / 2) * (0.85 + 0.15 * closeness)
                    pairs.append(TemporalPair(first, second, delta / 60, pair_score))
        pairs.sort(key=lambda item: (-item.score, item.delta_minutes, item.before.photo_id, item.after.photo_id))
        return SearchResponse(
            query=parsed,
            temporal_pairs=pairs[:top_k],
            eligible_count=before.eligible_count,
            channels_used=before.channels_used,
        )

    def search_temporal_neighbor(
        self,
        parsed: ParsedQuery,
        *,
        top_k: int,
        filters: Optional[SearchFilters],
        enabled_channels: Iterable[str],
    ) -> SearchResponse:
        """Find real chronological neighbours around semantically matched anchors."""

        temporal = parsed.temporal
        if not temporal or not temporal.is_neighbor:
            raise ValueError("A temporal-neighbour query is required")
        base = filters or SearchFilters()
        reliable_filters = SearchFilters(
            start_date=base.start_date,
            end_date=base.end_date,
            tags=base.tags,
            location=base.location,
            min_timestamp_confidence=max(base.min_timestamp_confidence or 0.0, 0.6),
            event_id=base.event_id,
        )
        anchors = self.search_standard(
            temporal.anchor or "",
            top_k=max(20, top_k * 4),
            filters=reliable_filters,
            enabled_channels=enabled_channels,
        )
        eligible = self._eligible_ids(reliable_filters)
        docs = self._documents(eligible)
        max_seconds = temporal.window_minutes * 60
        candidates: list[tuple[float, int, SearchResult, str]] = []
        for anchor in anchors.results:
            anchor_doc = docs.get(anchor.photo_id)
            if not anchor_doc or anchor_doc["captured_at_sort"] is None:
                continue
            anchor_time = int(anchor_doc["captured_at_sort"])
            for candidate_id, candidate_doc in docs.items():
                candidate_time = candidate_doc["captured_at_sort"]
                if (
                    candidate_id == anchor.photo_id
                    or candidate_time is None
                    or candidate_doc["date_local"] != anchor_doc["date_local"]
                ):
                    continue
                signed_delta = int(candidate_time) - anchor_time
                delta = -signed_delta if temporal.direction == "before" else signed_delta
                if not 0 < delta <= max_seconds:
                    continue
                closeness = 1.0 - delta / max_seconds
                pair_score = anchor.score * (0.8 + 0.2 * closeness)
                candidates.append((pair_score, delta, anchor, candidate_id))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2].photo_id, item[3]))
        selected = candidates[:top_k]
        neighbour_ids = list(dict.fromkeys(item[3] for item in selected))
        neighbours = {item.photo_id: item for item in self.get_by_ids(neighbour_ids)}
        pairs: list[TemporalPair] = []
        for pair_score, delta, anchor, neighbour_id in selected:
            neighbour = replace(
                neighbours[neighbour_id],
                score=max(0.0, min(1.0, 1.0 - delta / max_seconds)),
                explanation=f"chronological {temporal.direction} neighbour of {anchor.id}",
            )
            if temporal.direction == "before":
                pairs.append(TemporalPair(neighbour, anchor, delta / 60, pair_score))
            else:
                pairs.append(TemporalPair(anchor, neighbour, delta / 60, pair_score))
        return SearchResponse(
            query=parsed,
            temporal_pairs=pairs,
            eligible_count=len(eligible),
            channels_used=anchors.channels_used,
        )

    def search_by_image(
        self,
        image: Any,
        *,
        top_k: int = SEARCH_TOP_K,
        filters: Optional[SearchFilters] = None,
    ) -> SearchResponse:
        filters = filters or SearchFilters()
        if isinstance(image, (str, Path)):
            from PIL import Image

            with Image.open(image) as source:
                image_value = source.convert("RGB").copy()
        else:
            image_value = image.convert("RGB")
        embedding = self.clip_backend.encode_images_batch([image_value])[0]
        eligible = self._eligible_ids(filters)
        hits = self.vector_store.query_images(embedding, top_k=max(top_k * 4, top_k), eligible_photo_ids=eligible)
        parsed = ParsedQuery(original="[image]", visual_text="[image]")
        results = self._assemble({"clip": hits}, "", top_k, filters)
        return SearchResponse(parsed, results, eligible_count=len(eligible), channels_used=("clip",))

    def get_by_date(self, date: str) -> list[SearchResult]:
        ids = self.storage.metadata_eligible_ids(date_from=date, date_to=date)
        return self.get_by_ids(ids, filters_applied={"date": date})

    def get_by_ids(
        self,
        photo_ids: Sequence[str],
        *,
        filters_applied: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        ids = list(dict.fromkeys(photo_ids))
        docs = self._documents(ids)
        results = []
        for photo_id in ids:
            doc = docs[photo_id]
            results.append(SearchResult(
                photo_id=photo_id, id=doc["relative_path"], score=1.0,
                datetime=doc["captured_at"], timestamp_source=doc["timestamp_source"],
                timestamp_confidence=float(doc["timestamp_confidence"]), location=doc["location"],
                tags=[value for value in doc["tags"].split(",") if value], caption=doc["caption"],
                scene_graph_text=doc["scene_graph_text"], image_width=int(doc["image_width"] or 0),
                image_height=int(doc["image_height"] or 0), filters_applied=filters_applied or {},
            ))
        return results

    def get_available_dates(self) -> list[str]:
        with self.storage.transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT DISTINCT date_local FROM photos WHERE date_local IS NOT NULL ORDER BY date_local DESC"
            ).fetchall()
        return [row["date_local"] for row in rows]

    def get_available_tags(self) -> list[str]:
        # The CLIP index marker is a non-semantic completion/identity flag, not
        # a user-visible tag, so it is never offered as a search filter.
        with self.storage.transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT DISTINCT tag FROM tags WHERE tag != ? ORDER BY tag COLLATE NOCASE",
                (CLIP_INDEX_MARKER_TAG,),
            ).fetchall()
        return [row["tag"] for row in rows]


_retriever: Optional[MultimodalRetriever] = None


def get_multimodal_retriever() -> MultimodalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MultimodalRetriever()
    return _retriever


def reset_multimodal_retriever() -> None:
    global _retriever
    _retriever = None
