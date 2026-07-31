"""Compatibility facade for the v2 multimodal retrieval engine.

Existing callers can keep using ``HybridRetriever`` while all ranking now uses
independent CLIP, FTS5/BM25, and Scene Graph recall followed by weighted RRF.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from query_translation import PHRASES as CHINESE_TO_ENGLISH
from query_translation import translate_query_to_english
from retrieval_engine import (
    MultimodalRetriever,
    SearchFilters,
    SearchResponse,
    SearchResult,
    get_multimodal_retriever,
    reset_multimodal_retriever,
)


class FilterBuilder:
    """Legacy name retained; v2 filters are executed in SQLite before recall."""

    @staticmethod
    def date_range(start_date: Optional[str] = None, end_date: Optional[str] = None) -> SearchFilters:
        return SearchFilters(start_date=start_date, end_date=end_date)


class HybridRetriever:
    def __init__(self, engine: Optional[MultimodalRetriever] = None):
        self.engine = engine or get_multimodal_retriever()
        self.last_response: Optional[SearchResponse] = None

    @staticmethod
    def _filters(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        location_keyword: Optional[str] = None,
        min_timestamp_confidence: Optional[float] = None,
        event_id: Optional[str] = None,
        **_: Any,
    ) -> SearchFilters:
        return SearchFilters(
            start_date=start_date,
            end_date=end_date,
            tags=tuple(tags or ()),
            location=location_keyword,
            min_timestamp_confidence=min_timestamp_confidence,
            event_id=event_id,
        )

    def search_hybrid(
        self,
        query: str,
        top_k: int = 20,
        mode: str = "hybrid",
        vector_weight: float = 0.7,
        caption_keyword: Optional[str] = None,
        use_ollama_parser: bool = False,
        **filter_values: Any,
    ) -> list[SearchResult]:
        del vector_weight  # v2 uses rank fusion, not incomparable score addition.
        channels = {
            "vector": ("clip",),
            "keyword": ("caption",),
            "scene_graph": ("scene_graph",),
            "hybrid": ("clip", "caption", "scene_graph"),
        }.get(mode, ("clip", "caption", "scene_graph"))
        effective_query = " ".join(value for value in [query.strip(), (caption_keyword or "").strip()] if value)
        self.last_response = self.engine.search(
            effective_query,
            top_k=top_k,
            filters=self._filters(**filter_values),
            enabled_channels=channels,
            use_ollama_parser=use_ollama_parser,
        )
        if self.last_response.temporal_pairs:
            # Legacy grids show the chronologically later target once per pair.
            seen = set()
            results = []
            for pair in self.last_response.temporal_pairs:
                if pair.after.photo_id not in seen:
                    pair.after.score = pair.score
                    results.append(pair.after)
                    seen.add(pair.after.photo_id)
            return results[:top_k]
        return self.last_response.results

    def search(self, query: str, top_k: int = 20, **filters: Any) -> list[SearchResult]:
        return self.search_hybrid(query, top_k=top_k, mode="hybrid", **filters)

    def search_by_image(self, image_path: str, top_k: int = 20, **filter_values: Any) -> list[SearchResult]:
        self.last_response = self.engine.search_by_image(
            image_path,
            top_k=top_k,
            filters=self._filters(**filter_values),
        )
        return self.last_response.results

    def get_by_date(self, date: str) -> list[SearchResult]:
        return self.engine.get_by_date(date)

    def get_by_date_range(self, start_date: str, end_date: str) -> list[SearchResult]:
        ids = self.engine.storage.metadata_eligible_ids(date_from=start_date, date_to=end_date)
        docs = self.engine._documents(ids)
        results = []
        for date in sorted({doc["date_local"] for doc in docs.values() if doc["date_local"]}):
            results.extend(self.engine.get_by_date(date))
        return results

    def get_available_dates(self) -> list[str]:
        return self.engine.get_available_dates()

    def get_available_tags(self) -> list[str]:
        return self.engine.get_available_tags()

    def get_available_locations(self) -> list[str]:
        with self.engine.storage.transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT DISTINCT location FROM photos WHERE location IS NOT NULL AND location <> '' ORDER BY location"
            ).fetchall()
        return [row["location"] for row in rows]


_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def reset_hybrid_retriever() -> None:
    global _hybrid_retriever
    _hybrid_retriever = None
    reset_multimodal_retriever()


def search_photos(query: str, top_k: int = 20, **filters: Any) -> list[SearchResult]:
    return get_hybrid_retriever().search(query, top_k=top_k, **filters)


def get_photos_by_date(date: str) -> list[SearchResult]:
    return get_hybrid_retriever().get_by_date(date)
