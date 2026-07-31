"""Versioned Chroma vector storage for image and scene-graph channels."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from config import (
    CHROMA_PERSIST_DIR,
    CLIP_EMBEDDING_DIM,
    CLIP_MODEL_NAME,
    COLLECTION_NAME,
    INDEX_SCHEMA_VERSION,
    SCENE_GRAPH_COLLECTION_NAME,
)


class VectorStoreCompatibilityError(RuntimeError):
    """Raised instead of recursively retrying an incompatible collection."""


@dataclass(frozen=True)
class VectorHit:
    photo_id: str
    distance: float
    score: float
    vector_id: str
    metadata: Mapping[str, Any]


class ChromaVectorStore:
    def __init__(self, persist_dir: str | Path = CHROMA_PERSIST_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collections: dict[str, Any] = {}

    @property
    def client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    @staticmethod
    def _expected_metadata(kind: str) -> dict[str, Any]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "kind": kind,
            "embedding_model": CLIP_MODEL_NAME,
            "embedding_dim": CLIP_EMBEDDING_DIM,
            "hnsw:space": "cosine",
        }

    def _collection(self, kind: str):
        if kind in self._collections:
            return self._collections[kind]
        name = COLLECTION_NAME if kind == "image" else SCENE_GRAPH_COLLECTION_NAME
        expected = self._expected_metadata(kind)
        try:
            collection = self.client.get_or_create_collection(name=name, metadata=expected)
        except Exception as exc:
            raise VectorStoreCompatibilityError(
                f"Cannot open Chroma collection {name!r}: {exc}. "
                "Use a new versioned index directory or rebuild the derived index."
            ) from exc
        actual = collection.metadata or {}
        mismatches = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if key != "hnsw:space" and actual.get(key) != value
        }
        if mismatches:
            raise VectorStoreCompatibilityError(
                f"Collection {name!r} manifest mismatch: {mismatches}"
            )
        self._collections[kind] = collection
        return collection

    @property
    def image_collection(self):
        return self._collection("image")

    @property
    def scene_collection(self):
        return self._collection("scene_graph")

    def counts(self) -> dict[str, int]:
        return {
            "image": self.image_collection.count(),
            "scene_graph_triples": self.scene_collection.count(),
        }

    def close(self) -> None:
        """Release Windows HNSW file handles (mainly needed by tests/tools)."""
        client = self._client
        self._collections.clear()
        self._client = None
        if client is not None:
            try:
                client._system.stop()  # Chroma currently exposes no public close().
            except Exception:
                pass
            try:
                from chromadb.api.client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
        gc.collect()

    def all_image_ids(self) -> list[str]:
        return list(self.image_collection.get(include=[]).get("ids", []))

    def upsert_images(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        self.image_collection.upsert(
            ids=[str(record["photo_id"]) for record in records],
            embeddings=[np.asarray(record["embedding"], dtype=np.float32).tolist() for record in records],
            metadatas=[{
                "photo_id": str(record["photo_id"]),
                "relative_path": str(record["relative_path"]),
                "content_sha256": str(record["content_sha256"]),
                "captured_at_sort": int(record["captured_at_sort"] or 0),
                "date_local": str(record["date_local"] or ""),
                "timestamp_confidence": float(record.get("timestamp_confidence", 0.0)),
            } for record in records],
        )

    def delete_images(self, photo_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(str(item) for item in photo_ids))
        if ids:
            self.image_collection.delete(ids=ids)

    def replace_scene_graph(
        self,
        photo_id: str,
        triple_texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        *,
        content_sha256: str,
    ) -> None:
        if len(triple_texts) != len(embeddings):
            raise ValueError("triple_texts and embeddings must have the same length")
        # A graph update replaces all previous triple vectors atomically enough
        # for this derived index; SQLite remains the canonical transaction log.
        self.scene_collection.delete(where={"photo_id": photo_id})
        if not triple_texts:
            return
        ids = [f"{photo_id}:{index:03d}" for index in range(len(triple_texts))]
        self.scene_collection.upsert(
            ids=ids,
            embeddings=[np.asarray(value, dtype=np.float32).tolist() for value in embeddings],
            documents=list(triple_texts),
            metadatas=[{
                "photo_id": photo_id,
                "triple_text": text,
                "triple_ordinal": index,
                "content_sha256": content_sha256,
            } for index, text in enumerate(triple_texts)],
        )

    @staticmethod
    def _chunks(items: Sequence[str], size: int = 500):
        for offset in range(0, len(items), size):
            yield items[offset:offset + size]

    def _query(
        self,
        collection,
        embedding: Sequence[float],
        top_k: int,
        eligible_photo_ids: Optional[Iterable[str]],
    ) -> list[VectorHit]:
        if top_k <= 0:
            return []
        eligible = None if eligible_photo_ids is None else list(dict.fromkeys(eligible_photo_ids))
        if eligible == []:
            return []
        where_clauses: list[Optional[dict[str, Any]]]
        if eligible is None:
            where_clauses = [None]
        else:
            where_clauses = [{"photo_id": {"$in": chunk}} for chunk in self._chunks(eligible)]

        best_by_vector: dict[str, VectorHit] = {}
        for where in where_clauses:
            available = collection.count()
            if available == 0:
                continue
            params: dict[str, Any] = {
                "query_embeddings": [np.asarray(embedding, dtype=np.float32).tolist()],
                "n_results": min(top_k, available),
                "include": ["metadatas", "distances", "documents"],
            }
            if where:
                params["where"] = where
            try:
                payload = collection.query(**params)
            except Exception as exc:
                # Chroma raises when a filter chunk has no matching records.
                if "nothing found" in str(exc).lower() or "no results" in str(exc).lower():
                    continue
                raise
            ids = (payload.get("ids") or [[]])[0]
            metadata = (payload.get("metadatas") or [[]])[0]
            distances = (payload.get("distances") or [[]])[0]
            documents = (payload.get("documents") or [[]])[0]
            for index, vector_id in enumerate(ids):
                meta = dict(metadata[index] or {})
                if index < len(documents) and documents[index] is not None:
                    meta.setdefault("document", documents[index])
                distance = float(distances[index])
                hit = VectorHit(
                    photo_id=str(meta.get("photo_id") or vector_id),
                    distance=distance,
                    score=max(-1.0, min(1.0, 1.0 - distance)),
                    vector_id=str(vector_id),
                    metadata=meta,
                )
                previous = best_by_vector.get(hit.vector_id)
                if previous is None or hit.distance < previous.distance:
                    best_by_vector[hit.vector_id] = hit
        return sorted(best_by_vector.values(), key=lambda hit: (hit.distance, hit.vector_id))[:top_k]

    def query_images(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        eligible_photo_ids: Optional[Iterable[str]] = None,
    ) -> list[VectorHit]:
        return self._query(self.image_collection, embedding, top_k, eligible_photo_ids)

    def query_scene_graph(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        eligible_photo_ids: Optional[Iterable[str]] = None,
    ) -> list[VectorHit]:
        hits = self._query(self.scene_collection, embedding, max(top_k * 4, top_k), eligible_photo_ids)
        # Multiple triples from one photo collapse to its best relation match.
        best: dict[str, VectorHit] = {}
        for hit in hits:
            if hit.photo_id not in best or hit.distance < best[hit.photo_id].distance:
                best[hit.photo_id] = hit
        return sorted(best.values(), key=lambda hit: (hit.distance, hit.photo_id))[:top_k]

    def get_image_embeddings(self, photo_ids: Sequence[str]) -> dict[str, np.ndarray]:
        if not photo_ids:
            return {}
        payload = self.image_collection.get(ids=list(photo_ids), include=["embeddings"])
        embeddings = payload.get("embeddings")
        if embeddings is None:
            return {}
        return {
            str(photo_id): np.asarray(embedding, dtype=np.float32)
            for photo_id, embedding in zip(payload.get("ids", []), embeddings)
        }
