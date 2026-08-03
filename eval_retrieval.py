"""R0--R7 retrieval variants for the retrieval experiment.

Each variant ranks the frozen 1,596-photo corpus (current DB: 1,599 photos) for one English known-item
query and returns a ranked photo-id list (the metric layer checks whether the
target is in the top-k). Channels and weights are frozen by the protocol:

* CLIP image vector           -- weight 1.0
* Caption BM25 (BLIP / BLIP2) -- weight 0.8
* Scene Graph triple vector   -- weight 0.9
* RRF k = 60, candidate depth = 100

Variants:

    R0  CLIP
    R1  base-BLIP caption BM25
    R2  BLIP2 caption BM25
    R3  Qwen Scene Graph vector
    R4  CLIP + base-BLIP
    R5  CLIP + BLIP2
    R6  CLIP + BLIP2 + Scene Graph
    R7  R6 restricted to the target's exact date + full location (metadata
        subset only); reports pool size before and after the filter so a smaller
        candidate pool is never misreported as a semantic-ranking gain.

R1/R2 use two **read-only experimental FTS5 indexes** (one per caption model) in
a separate SQLite file. The production ``photo_fts`` and the captions Primary
flag are never touched.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

from config import BLIP2_MODEL_NAME, BLIP_MODEL_NAME, DATA_DIR, RRF_K, RETRIEVAL_CANDIDATE_K

WEIGHT_CLIP = 1.0
WEIGHT_CAPTION = 0.8
WEIGHT_SCENE_GRAPH = 0.9
CANDIDATE_DEPTH = RETRIEVAL_CANDIDATE_K  # 100

CHANNEL_WEIGHT = {
    "clip": WEIGHT_CLIP,
    "blip": WEIGHT_CAPTION,
    "blip2": WEIGHT_CAPTION,
    "scene": WEIGHT_SCENE_GRAPH,
}

# Variant -> ordered channels. R7 is R6 + a metadata filter (handled separately).
VARIANTS: dict[str, tuple[str, ...]] = {
    "R0": ("clip",),
    "R1": ("blip",),
    "R2": ("blip2",),
    "R3": ("scene",),
    "R4": ("clip", "blip"),
    "R5": ("clip", "blip2"),
    "R6": ("clip", "blip2", "scene"),
}
R6_CHANNELS = VARIANTS["R6"]

EVAL_CAPTION_FTS_PATH = DATA_DIR / "eval_caption_fts.db"
BLIP_TABLE = "eval_blip_fts"
BLIP2_TABLE = "eval_blip2_fts"


# ============================ Protocols ============================


class _ClipBackend(Protocol):
    def encode_text(self, text: str) -> Sequence[float]: ...


class _VectorStore(Protocol):
    def query_images(self, embedding, *, top_k, eligible_photo_ids=None) -> list: ...
    def query_scene_graph(self, embedding, *, top_k, eligible_photo_ids=None) -> list: ...


# ============================ Reciprocal Rank Fusion ============================


def rrf_combine(
    channel_lists: dict[str, Sequence[str]],
    *,
    weights: dict[str, float] = None,
    k: int = RRF_K,
) -> list[str]:
    """Weighted Reciprocal Rank Fusion over per-channel ranked id lists.

    ``score(id) = sum_c weight_c / (k + rank_c)`` where ``rank_c`` is the 1-based
    rank of ``id`` in channel ``c``. Ties break by photo_id (deterministic).
    """
    weights = weights or CHANNEL_WEIGHT
    scores: dict[str, float] = {}
    for channel, ids in channel_lists.items():
        w = weights.get(channel, 0.0)
        if not w:
            continue
        for rank, pid in enumerate(ids, start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank)
    return sorted(scores, key=lambda pid: (-scores[pid], pid))


# ============================ Read-only caption FTS ============================


def _fts_query(query: str) -> str:
    """Build a safe FTS5 OR query, matching production (``match_all_terms=False``).

    Each token is a quoted phrase; tokens join with OR so a caption need only
    share SOME terms with the query, and BM25 ranks by relevance. Strict AND
    (space-joined) would require every query term to appear in the caption,
    returning 0 candidates for long visual queries vs short generic captions.
    """
    terms = re.findall(r"[A-Za-z0-9]+", query.lower())
    return " OR ".join(f'"{t}"' for t in terms) if terms else '""'


class EvalCaptionFTS:
    """Two read-only FTS5 indexes (BLIP, BLIP2) in a standalone SQLite file.

    Built once from the production ``captions`` table (read-only); the
    production ``photo_fts`` and Primary flag are never modified.
    """

    def __init__(self, db_path: Path = EVAL_CAPTION_FTS_PATH):
        self.db_path = Path(db_path)

    def build(
        self,
        production_conn: sqlite3.Connection,
        *,
        blip_model: str = BLIP_MODEL_NAME,
        blip2_model: str = BLIP2_MODEL_NAME,
    ) -> dict[str, int]:
        """(Re)create the two FTS5 tables from the production captions."""
        if self.db_path.exists():
            self.db_path.unlink()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                f"""
                CREATE VIRTUAL TABLE {BLIP_TABLE} USING fts5(
                    photo_id UNINDEXED, caption,
                    tokenize = 'unicode61 remove_diacritics 2');
                CREATE VIRTUAL TABLE {BLIP2_TABLE} USING fts5(
                    photo_id UNINDEXED, caption,
                    tokenize = 'unicode61 remove_diacritics 2');
                """
            )
            counts = {}
            for model, table in ((blip_model, BLIP_TABLE), (blip2_model, BLIP2_TABLE)):
                rows = production_conn.execute(
                    "SELECT photo_id, caption FROM captions WHERE model_name=? AND status='ok' "
                    "AND caption IS NOT NULL AND TRIM(caption)<>''",
                    (model,),
                ).fetchall()
                conn.executemany(
                    f"INSERT INTO {table}(photo_id, caption) VALUES (?,?)", rows
                )
                counts[table] = len(rows)
            conn.commit()
            return counts
        finally:
            conn.close()

    def search(
        self,
        which: str,
        query: str,
        *,
        limit: int = CANDIDATE_DEPTH,
        eligible_photo_ids: Optional[Sequence[str]] = None,
    ) -> list[tuple[str, float]]:
        """BM25 search of one caption index. Returns [(photo_id, bm25_rank)]."""
        table = {BLIP_MODEL_NAME: BLIP_TABLE, "blip": BLIP_TABLE,
                 BLIP2_MODEL_NAME: BLIP2_TABLE, "blip2": BLIP2_TABLE}.get(which)
        if table is None:
            raise ValueError(f"unknown caption channel {which!r}")
        if limit <= 0 or not query.strip():
            return []
        fts_query = _fts_query(query)
        conn = sqlite3.connect(self.db_path)
        try:
            params: list = [fts_query]
            join = ""
            if eligible_photo_ids is not None:
                eligible = list(dict.fromkeys(eligible_photo_ids))
                if not eligible:
                    return []
                conn.execute("CREATE TEMP TABLE eligible(photo_id TEXT PRIMARY KEY)")
                conn.executemany("INSERT INTO eligible(photo_id) VALUES (?)",
                                 [(pid,) for pid in eligible])
                join = f"JOIN eligible e ON e.photo_id = {table}.photo_id"
            rows = conn.execute(
                f"""
                SELECT {table}.photo_id, bm25({table}) AS rank
                FROM {table} {join}
                WHERE {table} MATCH ?
                ORDER BY rank ASC, {table}.photo_id ASC
                LIMIT ?
                """,
                [*params, int(limit)],
            ).fetchall()
            return [(str(r[0]), float(r[1])) for r in rows]
        finally:
            conn.close()


# ============================ Metadata filter (R7) ============================


def metadata_filter_ids(
    conn: sqlite3.Connection,
    *,
    date_local: str,
    location: str,
    corpus: Sequence[str],
) -> list[str]:
    """Corpus photos with the target's exact local date AND full location.

    "Full location" = a non-empty geocoded ``location`` string; the filter
    requires an exact match on both date and location.
    """
    if not date_local or not location:
        return []
    placeholders = ",".join("?" for _ in corpus)
    rows = conn.execute(
        f"""
        SELECT photo_id FROM photos
        WHERE photo_id IN ({placeholders})
          AND date_local = ? AND location = ?
        ORDER BY photo_id
        """,
        [*corpus, date_local, location],
    ).fetchall()
    return [str(r[0]) for r in rows]


# ============================ Retrieval backend + variants ============================


@dataclass
class VariantResult:
    ranked_ids: list[str]
    channels_used: tuple[str, ...] = ()
    pool_before: Optional[int] = None  # R7 only
    pool_after: Optional[int] = None   # R7 only


@dataclass
class RetrievalBackend:
    """Injectable backends so the variants are unit-testable with fakes."""

    clip_backend: _ClipBackend
    vector_store: _VectorStore
    caption_fts: EvalCaptionFTS
    depth: int = CANDIDATE_DEPTH

    def _channel_list(self, channel: str, query: str, pool: Sequence[str]) -> list[str]:
        if channel == "clip":
            emb = self.clip_backend.encode_text(query)
            hits = self.vector_store.query_images(
                emb, top_k=self.depth, eligible_photo_ids=pool
            )
            return [str(h.photo_id) for h in hits]
        if channel == "scene":
            emb = self.clip_backend.encode_text(query)
            hits = self.vector_store.query_scene_graph(
                emb, top_k=self.depth, eligible_photo_ids=pool
            )
            return [str(h.photo_id) for h in hits]
        if channel in ("blip", "blip2"):
            rows = self.caption_fts.search(
                channel, query, limit=self.depth, eligible_photo_ids=pool
            )
            return [pid for pid, _ in rows]
        raise ValueError(f"unknown channel {channel!r}")


def run_variant(
    variant: str,
    query: str,
    backend: RetrievalBackend,
    corpus: Sequence[str],
    *,
    pool: Optional[Sequence[str]] = None,
) -> VariantResult:
    """Run one R0--R6 variant over ``corpus`` (or a restricted ``pool``).

    ``pool`` restricts every channel to a subset of the corpus (used by R7's
    metadata filter); when given, ``pool_before``/``pool_after`` are recorded.
    """
    channels = VARIANTS[variant]
    search_pool = list(pool) if pool is not None else list(corpus)
    channel_lists = {
        ch: backend._channel_list(ch, query, search_pool) for ch in channels
    }
    ranked = rrf_combine(channel_lists)
    result = VariantResult(ranked_ids=ranked, channels_used=tuple(channels))
    if pool is not None:
        result.pool_before = len(corpus)
        result.pool_after = len(search_pool)
    return result


def run_r7(
    query: str,
    backend: RetrievalBackend,
    corpus: Sequence[str],
    *,
    eligible_ids: Sequence[str],
) -> VariantResult:
    """R7 = R6 ranking restricted to the metadata-filtered candidate pool."""
    result = run_variant("R6", query, backend, corpus, pool=eligible_ids)
    return result


__all__ = [
    "BLIP2_TABLE",
    "BLIP_TABLE",
    "CANDIDATE_DEPTH",
    "CHANNEL_WEIGHT",
    "EVAL_CAPTION_FTS_PATH",
    "EvalCaptionFTS",
    "RetrievalBackend",
    "R6_CHANNELS",
    "VARIANTS",
    "VariantResult",
    "WEIGHT_CAPTION",
    "WEIGHT_CLIP",
    "WEIGHT_SCENE_GRAPH",
    "metadata_filter_ids",
    "rrf_combine",
    "run_r7",
    "run_variant",
]
