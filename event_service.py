"""Bridge deterministic event segmentation to SQLite and Chroma embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import APP_DB_PATH
from event_organizer import EventOrganization, PhotoEventInput, organize_events
from retrieval_engine import MultimodalRetriever, get_multimodal_retriever
from storage import PhotoStorage
from vector_store import ChromaVectorStore


ALGORITHM_VERSION = "time90-visual065-gps2km-v1"
EVENT_METHOD = "automatic_time_clip_gps"


@dataclass(frozen=True)
class StoredEvent:
    event_id: str
    title: str
    date_local: str
    start_at: str
    end_at: str
    photo_ids: tuple[str, ...]
    representative_ids: tuple[str, ...]
    method: str


class EventService:
    def __init__(
        self,
        storage: Optional[PhotoStorage] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        retriever: Optional[MultimodalRetriever] = None,
    ):
        self.storage = storage or PhotoStorage(APP_DB_PATH)
        self.vector_store = vector_store or ChromaVectorStore()
        self.retriever = retriever or get_multimodal_retriever()

    def organize(self, *, persist: bool = True) -> EventOrganization:
        ids = self.storage.metadata_eligible_ids()
        docs = self.retriever._documents(ids)
        embeddings = self.vector_store.get_image_embeddings(ids)
        inputs = []
        for photo_id in ids:
            doc = docs[photo_id]
            inputs.append(PhotoEventInput(
                photo_id=photo_id,
                captured_at=doc["captured_at"],
                captured_at_sort=doc["captured_at_sort"],
                timestamp_confidence=("high" if float(doc["timestamp_confidence"]) >= 0.9
                                      else "medium" if float(doc["timestamp_confidence"]) >= 0.6 else "low"),
                clip_embedding=embeddings.get(photo_id),
                latitude=doc["gps_latitude"],
                longitude=doc["gps_longitude"],
                caption=doc["caption"],
                location=doc["location"] or "",
            ))
        organization = organize_events(inputs)
        if persist:
            with self.storage.transaction() as connection:
                connection.execute("DELETE FROM events WHERE method = ?", (EVENT_METHOD,))
            day_counts: dict[str, int] = {}
            for event in organization.events:
                day_counts[event.date_local] = day_counts.get(event.date_local, 0) + 1
                title = f"{event.date_local} · Event {day_counts[event.date_local]}"
                start_doc = docs[event.photo_ids[0]]
                end_doc = docs[event.photo_ids[-1]]
                self.storage.upsert_event(
                    event_id=event.event_id,
                    title=title,
                    start_at=event.start_at,
                    start_at_sort=start_doc["captured_at_sort"],
                    end_at=event.end_at,
                    end_at_sort=end_doc["captured_at_sort"],
                    date_local=event.date_local,
                    method=EVENT_METHOD,
                    algorithm_version=ALGORITHM_VERSION,
                    parameters={
                        "time_gap_minutes": 90,
                        "visual_gap_minutes": 30,
                        "visual_similarity_threshold": 0.65,
                        "gps_gap_minutes": 10,
                        "gps_distance_km": 2,
                        "boundary_reasons": list(event.boundary_reasons),
                        "representative_ids": list(event.representative_ids),
                    },
                )
                roles = ["representative" if pid in event.representative_ids else "member" for pid in event.photo_ids]
                self.storage.replace_event_photos(event.event_id, event.photo_ids, roles=roles)
        return organization

    def list_events(self) -> list[StoredEvent]:
        with self.storage.transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT e.*, ep.photo_id, ep.role, ep.position
                FROM events e LEFT JOIN event_photos ep ON ep.event_id=e.event_id
                ORDER BY e.start_at_sort, ep.position
                """
            ).fetchall()
        grouped: dict[str, dict] = {}
        for row in rows:
            item = grouped.setdefault(row["event_id"], {
                "event_id": row["event_id"], "title": row["title"], "date_local": row["date_local"] or "",
                "start_at": row["start_at"] or "", "end_at": row["end_at"] or "", "method": row["method"],
                "photo_ids": [], "representative_ids": [],
            })
            if row["photo_id"]:
                item["photo_ids"].append(row["photo_id"])
                if row["role"] == "representative":
                    item["representative_ids"].append(row["photo_id"])
        return [StoredEvent(
            event_id=item["event_id"], title=item["title"], date_local=item["date_local"],
            start_at=item["start_at"], end_at=item["end_at"], photo_ids=tuple(item["photo_ids"]),
            representative_ids=tuple(item["representative_ids"]), method=item["method"],
        ) for item in grouped.values()]

