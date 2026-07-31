"""Restartable, memory-bounded photo indexing orchestration.

The service keeps model inference behind small injectable protocols.  It scans
metadata without retaining pixels, completes the CLIP phase in batches, releases
those image objects, and only then starts the BLIP phase.  SQLite is canonical;
Chroma receives stable UUIDs and content hashes as a derived vector index.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from PIL import Image

from config import CLIP_INDEX_MARKER_TAG
from storage import PhotoStorage


DEFAULT_CLIP_BATCH_SIZE = 8
DEFAULT_CAPTION_BATCH_SIZE = 4
DEFAULT_MIN_BATCH_SIZE = 1

_TIMESTAMP_CONFIDENCE = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.2,
    "missing": 0.2,
}


class ClipBackend(Protocol):
    def encode_images_batch(self, images: Sequence[Image.Image]) -> Any: ...


class CaptionBackend(Protocol):
    def generate_captions_batch(self, images: Sequence[Image.Image]) -> Sequence[str]: ...


class ImageVectorStore(Protocol):
    def upsert_images(self, records: Sequence[Mapping[str, Any]]) -> None: ...


@dataclass(frozen=True)
class PreparedPhoto:
    photo_id: str
    absolute_path: Path
    relative_path: str
    content_sha256: str
    captured_at_sort: int | None
    date_local: str | None
    timestamp_confidence: float
    needs_clip: bool
    needs_caption: bool


@dataclass(frozen=True)
class IndexRunSummary:
    run_id: str
    status: str
    discovered_count: int
    prepared_count: int
    completed_count: int
    indexed_count: int
    skipped_count: int
    failed_count: int
    clip_indexed_count: int
    caption_indexed_count: int
    clip_skipped_count: int
    caption_skipped_count: int
    final_clip_batch_size: int
    final_caption_batch_size: int


@dataclass
class _RunState:
    run_id: str
    discovered_count: int
    prepared: list[PreparedPhoto] = field(default_factory=list)
    clip_complete: set[str] = field(default_factory=set)
    caption_complete: set[str] = field(default_factory=set)
    clip_processed: set[str] = field(default_factory=set)
    caption_processed: set[str] = field(default_factory=set)
    fully_skipped: set[str] = field(default_factory=set)
    invalid_image_ids: set[str] = field(default_factory=set)
    failed_entities: set[str] = field(default_factory=set)
    recorded_failures: set[tuple[str, str]] = field(default_factory=set)


class IndexingService:
    """Coordinate metadata, CLIP, caption, vector, and checkpoint writes."""

    def __init__(
        self,
        storage: PhotoStorage,
        clip_backend: ClipBackend,
        caption_backend: CaptionBackend,
        vector_store: ImageVectorStore,
        *,
        metadata_extractor: Callable[..., Any] | None = None,
        clip_batch_size: int = DEFAULT_CLIP_BATCH_SIZE,
        caption_batch_size: int = DEFAULT_CAPTION_BATCH_SIZE,
        min_batch_size: int = DEFAULT_MIN_BATCH_SIZE,
        clip_model_name: str | None = None,
        clip_model_version: str | None = None,
        caption_model_name: str | None = None,
        caption_model_version: str | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if clip_batch_size < 1 or caption_batch_size < 1 or min_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if min_batch_size > clip_batch_size or min_batch_size > caption_batch_size:
            raise ValueError("min_batch_size cannot exceed a configured batch size")
        self.storage = storage
        self.clip_backend = clip_backend
        self.caption_backend = caption_backend
        self.vector_store = vector_store
        self.metadata_extractor = metadata_extractor or self._default_metadata_extractor
        self.clip_batch_size = int(clip_batch_size)
        self.caption_batch_size = int(caption_batch_size)
        self.min_batch_size = int(min_batch_size)
        self.clip_model_name = clip_model_name or self._backend_name(clip_backend)
        self.clip_model_version = (
            clip_model_version if clip_model_version is not None else self._backend_version(clip_backend)
        )
        self.caption_model_name = caption_model_name or self._backend_name(caption_backend)
        self.caption_model_version = (
            caption_model_version
            if caption_model_version is not None
            else self._backend_version(caption_backend)
        )
        clip_identity = f"{self.clip_model_name}:{self.clip_model_version}".rstrip(":")
        self._clip_tag_family = "clip:"
        self._clip_tag_source = f"clip:{clip_identity}"
        self.progress_callback = progress_callback
        self._run_lock = threading.Lock()

    @staticmethod
    def _backend_name(backend: Any) -> str:
        for attribute in ("model_name", "name"):
            value = getattr(backend, attribute, None)
            if value:
                return str(value)
        return type(backend).__name__

    @staticmethod
    def _backend_version(backend: Any) -> str:
        value = getattr(backend, "model_version", "")
        return str(value) if value is not None else ""

    @staticmethod
    def _default_metadata_extractor(path: Path, base_dir: Path) -> Any:
        # Kept lazy so unit tests and command-line schema tools do not import the
        # geocoder/model stack.
        from data_ingestion import extract_photo_metadata

        return extract_photo_metadata(path, base_dir)

    def _call_metadata_extractor(self, path: Path, base_dir: Path) -> Any:
        try:
            signature = inspect.signature(self.metadata_extractor)
            signature.bind(path, base_dir)
        except (TypeError, ValueError):
            return self.metadata_extractor(path)
        return self.metadata_extractor(path, base_dir)

    @staticmethod
    def _metadata_value(metadata: Any, name: str, default: Any = None) -> Any:
        if isinstance(metadata, Mapping):
            return metadata.get(name, default)
        return getattr(metadata, name, default)

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _relative_path(path: Path, base_dir: Path) -> str:
        try:
            return path.resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"Photo {path} is outside the configured photo root {base_dir}") from error

    @staticmethod
    def _sort_value(captured_at: str | None) -> int | None:
        if not captured_at:
            return None
        try:
            parsed = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).replace(tzinfo=None)
            return int((parsed - datetime(1970, 1, 1)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _confidence(value: Any) -> float:
        if isinstance(value, str):
            return _TIMESTAMP_CONFIDENCE.get(value.strip().casefold(), 0.2)
        if value is None:
            return 0.2
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("timestamp confidence must be in [0, 1]")
        return numeric

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        if value is None:
            return None
        numeric = int(value)
        return numeric if numeric > 0 else None

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        name = type(error).__name__.casefold()
        message = str(error).casefold()
        return (
            "outofmemory" in name
            or "cuda out of memory" in message
            or ("out of memory" in message and ("cuda" in message or "cudnn" in message))
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def _emit(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _known_vector_ids(self) -> tuple[set[str], bool]:
        getter = getattr(self.vector_store, "all_image_ids", None)
        if not callable(getter):
            return set(), False
        try:
            return {str(value) for value in getter()}, True
        except Exception:
            # Recomputing is safer than treating an unavailable derived index as
            # complete.
            return set(), True

    @staticmethod
    def _stored_file_fingerprint_matches(
        row: Mapping[str, Any],
        *,
        relative_path: str,
        file_size_bytes: int,
        file_mtime_ns: int,
        file_ctime_ns: int | None,
    ) -> bool:
        """Return whether a filesystem observation is identical to SQLite.

        This deliberately includes the exact relative path.  A rename therefore
        falls back to SHA-256 reconciliation, which is what preserves the stable
        UUID instead of accidentally creating a second logical photo.
        """

        return (
            str(row.get("relative_path", "")) == relative_path
            and row.get("file_size_bytes") is not None
            and int(row["file_size_bytes"]) == int(file_size_bytes)
            and row.get("file_mtime_ns") is not None
            and int(row["file_mtime_ns"]) == int(file_mtime_ns)
            and row.get("file_ctime_ns") is not None
            and file_ctime_ns is not None
            and int(row["file_ctime_ns"]) == int(file_ctime_ns)
        )

    def _completion_state(
        self,
        photo_id: str,
        *,
        vector_ids: set[str],
        vector_ids_available: bool,
    ) -> tuple[bool, bool]:
        """Check the independent CLIP/tag and caption completion markers."""

        # Exactly one active CLIP tag source is the persistent identity marker
        # for the single image vector stored under this photo_id.  Clearing old
        # family members on every successful CLIP write prevents a v1->v2->v1
        # switch from pairing stale v1 tags with the current v2 vector.
        clip_tag_sources = self.storage.list_tag_sources(
            photo_id,
            source_prefix=self._clip_tag_family,
        )
        tags_exist = clip_tag_sources == {self._clip_tag_source}
        # A store that can enumerate IDs must contain both the derived vector and
        # the SQLite tags.  Minimal protocol implementations cannot expose a
        # vector probe, so their historical tag marker remains the safe fallback.
        vector_exists = photo_id in vector_ids if vector_ids_available else tags_exist
        clip_exists = vector_exists and tags_exist
        caption_exists = self.storage.get_caption(
            photo_id,
            model_name=self.caption_model_name,
            model_version=self.caption_model_version,
            primary_only=False,
        ) is not None
        return clip_exists, caption_exists

    @staticmethod
    def _append_prepared_photo(
        state: _RunState,
        *,
        photo_id: str,
        path: Path,
        relative_path: str,
        content_sha256: str,
        captured_at_sort: Any,
        date_local: str | None,
        timestamp_confidence: float,
        clip_exists: bool,
        caption_exists: bool,
    ) -> None:
        prepared = PreparedPhoto(
            photo_id=photo_id,
            absolute_path=path,
            relative_path=relative_path,
            content_sha256=content_sha256,
            captured_at_sort=(
                None if captured_at_sort is None else int(captured_at_sort)
            ),
            date_local=date_local,
            timestamp_confidence=timestamp_confidence,
            needs_clip=not clip_exists,
            needs_caption=not caption_exists,
        )
        state.prepared.append(prepared)
        if clip_exists:
            state.clip_complete.add(photo_id)
        if caption_exists:
            state.caption_complete.add(photo_id)
        if clip_exists and caption_exists:
            state.fully_skipped.add(photo_id)

    def _record_failure(
        self,
        state: _RunState,
        *,
        stage: str,
        error: BaseException | str,
        photo: PreparedPhoto | None = None,
        relative_path: str | None = None,
        photo_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        selected_id = photo.photo_id if photo is not None else photo_id
        selected_path = photo.relative_path if photo is not None else relative_path
        identity = selected_id or selected_path or "__run__"
        key = (identity, stage)
        if key in state.recorded_failures:
            return
        state.recorded_failures.add(key)
        state.failed_entities.add(identity)
        self.storage.record_index_failure(
            run_id=state.run_id,
            photo_id=selected_id,
            relative_path=selected_path,
            stage=stage,
            error=error,
            retryable=retryable,
        )
        self._emit(
            run_id=state.run_id,
            phase=stage,
            status="error",
            photo_id=selected_id,
            relative_path=selected_path,
            error=str(error),
        )

    def _prepare_photos(
        self,
        paths: Sequence[Path],
        base_dir: Path,
        state: _RunState,
        *,
        force: bool,
    ) -> None:
        vector_ids, vector_ids_available = self._known_vector_ids()
        delete_vectors = getattr(self.vector_store, "delete_images", None)

        for offset, path in enumerate(paths, start=1):
            try:
                relative_path = self._relative_path(path, base_dir)
            except ValueError as error:
                self._record_failure(
                    state, stage="metadata", error=error, relative_path=path.name
                )
                continue

            try:
                stat = path.stat()
            except (OSError, PermissionError) as error:
                self._record_failure(
                    state,
                    stage="stat",
                    error=error,
                    relative_path=relative_path,
                    retryable=True,
                )
                continue

            existing_by_path = self.storage.get_photo_by_path(relative_path)

            # The common incremental case is metadata-only: stat is cheap and a
            # matching path/size/mtime/ctime tuple lets us trust the previously
            # stored digest and EXIF fields.  Missing derived stages are still detected
            # and repaired below without re-reading the source file here.
            if (
                not force
                and existing_by_path is not None
                and self._stored_file_fingerprint_matches(
                    existing_by_path,
                    relative_path=relative_path,
                    file_size_bytes=stat.st_size,
                    file_mtime_ns=stat.st_mtime_ns,
                    file_ctime_ns=getattr(stat, "st_ctime_ns", None),
                )
            ):
                photo_id = str(existing_by_path["photo_id"])
                clip_exists, caption_exists = self._completion_state(
                    photo_id,
                    vector_ids=vector_ids,
                    vector_ids_available=vector_ids_available,
                )
                self._append_prepared_photo(
                    state,
                    photo_id=photo_id,
                    path=path,
                    relative_path=relative_path,
                    content_sha256=str(existing_by_path["content_sha256"]),
                    captured_at_sort=existing_by_path.get("captured_at_sort"),
                    date_local=existing_by_path.get("date_local"),
                    timestamp_confidence=self._confidence(
                        existing_by_path.get("timestamp_confidence")
                    ),
                    clip_exists=clip_exists,
                    caption_exists=caption_exists,
                )
                if offset % self.clip_batch_size == 0:
                    self._checkpoint(state, phase="metadata", batch_size=self.clip_batch_size)
                continue

            try:
                digest = self._sha256_file(path)
            except (OSError, PermissionError) as error:
                self._record_failure(
                    state,
                    stage="hash",
                    error=error,
                    relative_path=relative_path,
                    retryable=True,
                )
                continue

            existing_by_hash = self.storage.get_photo_by_hash(digest)
            content_changed = bool(
                existing_by_path is not None
                and str(existing_by_path["content_sha256"]).casefold() != digest.casefold()
            )
            file_observation_changed = bool(
                existing_by_path is not None
                and (
                    existing_by_path.get("file_size_bytes") is None
                    or int(existing_by_path["file_size_bytes"]) != int(stat.st_size)
                    or existing_by_path.get("file_mtime_ns") is None
                    or int(existing_by_path["file_mtime_ns"]) != int(stat.st_mtime_ns)
                )
            )
            requires_model_refresh = content_changed or file_observation_changed

            metadata: Any = {}
            metadata_error: BaseException | str | None = None
            try:
                metadata = self._call_metadata_extractor(path, base_dir)
                reported_error = self._metadata_value(metadata, "error")
                if reported_error:
                    metadata_error = str(reported_error)
            except Exception as error:
                metadata_error = error

            captured_at = self._metadata_value(
                metadata,
                "datetime_original",
                self._metadata_value(metadata, "captured_at"),
            )
            captured_sort = self._metadata_value(metadata, "captured_at_sort")
            if captured_sort is None:
                captured_sort = self._sort_value(captured_at)
            date_local = self._metadata_value(metadata, "date_local")
            if not date_local and captured_at:
                date_local = str(captured_at)[:10]
            gps = self._metadata_value(metadata, "gps_coords")
            gps_latitude = gps[0] if gps and len(gps) >= 2 else None
            gps_longitude = gps[1] if gps and len(gps) >= 2 else None

            try:
                confidence = self._confidence(
                    self._metadata_value(metadata, "timestamp_confidence", "low")
                )
                photo_id = self.storage.upsert_photo(
                    relative_path=relative_path,
                    content_sha256=digest,
                    # Do not persist a fast-path-complete fingerprint when
                    # metadata extraction failed.  Leaving these observations
                    # NULL makes the next incremental run retry SHA-256 and
                    # metadata even when the source bytes did not change.
                    file_size_bytes=None if metadata_error is not None else stat.st_size,
                    file_mtime_ns=None if metadata_error is not None else stat.st_mtime_ns,
                    file_ctime_ns=(
                        None
                        if metadata_error is not None
                        else getattr(stat, "st_ctime_ns", None)
                    ),
                    image_width=self._positive_int(self._metadata_value(metadata, "image_width")),
                    image_height=self._positive_int(self._metadata_value(metadata, "image_height")),
                    captured_at=captured_at,
                    captured_at_sort=captured_sort,
                    date_local=date_local,
                    timestamp_source=str(
                        self._metadata_value(metadata, "timestamp_source", "missing")
                    ),
                    timestamp_confidence=confidence,
                    gps_latitude=gps_latitude,
                    gps_longitude=gps_longitude,
                    location=self._metadata_value(metadata, "location"),
                )
            except Exception as error:
                self._record_failure(
                    state,
                    stage="metadata_upsert",
                    error=error,
                    relative_path=relative_path,
                )
                continue

            if force or requires_model_refresh:
                # Removing the successful-tag marker prevents a failed rewrite
                # from looking complete on the next incremental run.
                self.storage.replace_tags(
                    photo_id,
                    [],
                    source=self._clip_tag_source,
                    clear_source_prefix=self._clip_tag_family,
                )
                previous_caption = self.storage.get_caption(
                    photo_id,
                    model_name=self.caption_model_name,
                    model_version=self.caption_model_version,
                    primary_only=False,
                )
                if previous_caption is not None:
                    # Keep the old text for auditability but exclude it from FTS
                    # and incremental-completion checks until BLIP succeeds for
                    # the new bytes.
                    self.storage.upsert_caption(
                        photo_id,
                        previous_caption["caption"],
                        model_name=self.caption_model_name,
                        model_version=self.caption_model_version,
                        language=previous_caption["language"],
                        status="stale",
                        is_primary=bool(previous_caption["is_primary"]),
                    )
                if callable(delete_vectors):
                    try:
                        delete_vectors([photo_id])
                        vector_ids.discard(photo_id)
                    except Exception as error:
                        self._record_failure(
                            state,
                            stage="vector_invalidation",
                            error=error,
                            photo_id=photo_id,
                            relative_path=relative_path,
                            retryable=True,
                        )

            if metadata_error is not None:
                self._record_failure(
                    state,
                    stage="metadata",
                    error=metadata_error,
                    photo_id=photo_id,
                    relative_path=relative_path,
                    retryable=True,
                )
                continue

            unchanged = existing_by_path is not None or existing_by_hash is not None
            if force or requires_model_refresh or not unchanged:
                clip_exists = False
                caption_exists = False
            else:
                clip_exists, caption_exists = self._completion_state(
                    photo_id,
                    vector_ids=vector_ids,
                    vector_ids_available=vector_ids_available,
                )

            self._append_prepared_photo(
                state,
                photo_id=photo_id,
                path=path,
                relative_path=relative_path,
                content_sha256=digest,
                captured_at_sort=captured_sort,
                date_local=date_local,
                timestamp_confidence=confidence,
                clip_exists=clip_exists,
                caption_exists=caption_exists,
            )

            if offset % self.clip_batch_size == 0:
                self._checkpoint(state, phase="metadata", batch_size=self.clip_batch_size)

    def _load_images(
        self,
        photos: Sequence[PreparedPhoto],
        state: _RunState,
        *,
        phase: str,
    ) -> list[tuple[PreparedPhoto, Image.Image]]:
        loaded: list[tuple[PreparedPhoto, Image.Image]] = []
        for photo in photos:
            if photo.photo_id in state.invalid_image_ids:
                continue
            try:
                with Image.open(photo.absolute_path) as source:
                    source.load()
                    image = source.convert("RGB").copy()
                loaded.append((photo, image))
            except Exception as error:
                state.invalid_image_ids.add(photo.photo_id)
                self._record_failure(
                    state,
                    stage=f"{phase}_image_load",
                    error=error,
                    photo=photo,
                )
        return loaded

    @staticmethod
    def _close_images(loaded: Sequence[tuple[PreparedPhoto, Image.Image]]) -> None:
        for _, image in loaded:
            image.close()

    def _process_clip_batch(
        self,
        photos: Sequence[PreparedPhoto],
        state: _RunState,
    ) -> list[str]:
        loaded = self._load_images(photos, state, phase="clip")
        if not loaded:
            return []
        try:
            image_objects = [image for _, image in loaded]
            embeddings = self.clip_backend.encode_images_batch(image_objects)
            if len(embeddings) != len(loaded):
                raise ValueError("CLIP returned a different number of embeddings than images")

            records = []
            for index, (photo, _) in enumerate(loaded):
                records.append(
                    {
                        "photo_id": photo.photo_id,
                        "relative_path": photo.relative_path,
                        "content_sha256": photo.content_sha256,
                        "captured_at_sort": photo.captured_at_sort,
                        "date_local": photo.date_local,
                        "timestamp_confidence": photo.timestamp_confidence,
                        "embedding": embeddings[index],
                    }
                )

            # Clear the persistent identity marker *before* touching Chroma.
            # If a version-switch vector upsert succeeds (or partially succeeds)
            # and the subsequent tag transaction fails, no old-model marker may
            # authorize that new vector on a later run.
            for photo, _ in loaded:
                self.storage.replace_tags(
                    photo.photo_id,
                    [],
                    source=self._clip_tag_source,
                    clear_source_prefix=self._clip_tag_family,
                )

            # Vector upsert is idempotent.  The single non-semantic
            # CLIP_INDEX_MARKER_TAG row is committed only after the upsert
            # succeeds and serves as the completion / model-version identity
            # marker (its source carries clip:<model>:<version>). It is NOT a
            # semantic tag and is excluded from the FTS5 tags column.
            self.vector_store.upsert_images(records)
            for photo, _ in loaded:
                self.storage.replace_tags(
                    photo.photo_id,
                    [(CLIP_INDEX_MARKER_TAG, None)],
                    source=self._clip_tag_source,
                    clear_source_prefix=self._clip_tag_family,
                )
            return [photo.photo_id for photo, _ in loaded]
        finally:
            self._close_images(loaded)

    def _process_caption_batch(
        self,
        photos: Sequence[PreparedPhoto],
        state: _RunState,
    ) -> list[str]:
        loaded = self._load_images(photos, state, phase="caption")
        if not loaded:
            return []
        try:
            captions = self.caption_backend.generate_captions_batch(
                [image for _, image in loaded]
            )
            if len(captions) != len(loaded):
                raise ValueError("Caption backend returned a different number of captions than images")
            for (photo, _), caption in zip(loaded, captions):
                if caption is None or not str(caption).strip():
                    raise ValueError("Caption backend returned an empty caption")
                self.storage.upsert_caption(
                    photo.photo_id,
                    str(caption),
                    model_name=self.caption_model_name,
                    model_version=self.caption_model_version,
                    language="en",
                )
            return [photo.photo_id for photo, _ in loaded]
        finally:
            self._close_images(loaded)

    def _run_phase(
        self,
        photos: Sequence[PreparedPhoto],
        *,
        state: _RunState,
        phase: str,
        initial_batch_size: int,
        process_batch: Callable[[Sequence[PreparedPhoto], _RunState], list[str]],
    ) -> int:
        if not photos:
            return initial_batch_size
        batch_size = min(initial_batch_size, len(photos))
        offset = 0
        while offset < len(photos):
            chunk = photos[offset : offset + batch_size]
            try:
                successful_ids = process_batch(chunk, state)
            except Exception as error:
                if self._is_cuda_oom(error):
                    self._clear_cuda_cache()
                    if len(chunk) > 1 and batch_size > self.min_batch_size:
                        batch_size = max(self.min_batch_size, batch_size // 2)
                        self._emit(
                            run_id=state.run_id,
                            phase=phase,
                            status="oom_retry",
                            batch_size=batch_size,
                        )
                        continue
                    self._record_failure(
                        state,
                        stage=f"{phase}_oom",
                        error=error,
                        photo=chunk[0],
                        retryable=True,
                    )
                    offset += 1
                    self._checkpoint(state, phase=phase, batch_size=batch_size)
                    continue

                if len(chunk) > 1:
                    # A backend may reject one image while the rest are valid.
                    # Retry as singletons to preserve exact photo/result mapping.
                    for photo in chunk:
                        try:
                            individual_ids = process_batch([photo], state)
                            self._mark_phase_success(state, phase, individual_ids)
                        except Exception as individual_error:
                            self._record_failure(
                                state,
                                stage=f"{phase}_inference",
                                error=individual_error,
                                photo=photo,
                                retryable=True,
                            )
                        self._checkpoint(state, phase=phase, batch_size=1)
                    offset += len(chunk)
                    continue

                self._record_failure(
                    state,
                    stage=f"{phase}_inference",
                    error=error,
                    photo=chunk[0],
                    retryable=True,
                )
                offset += 1
                self._checkpoint(state, phase=phase, batch_size=batch_size)
                continue

            self._mark_phase_success(state, phase, successful_ids)
            offset += len(chunk)
            self._checkpoint(state, phase=phase, batch_size=batch_size)
        return batch_size

    @staticmethod
    def _mark_phase_success(state: _RunState, phase: str, photo_ids: Iterable[str]) -> None:
        selected = set(photo_ids)
        if phase == "clip":
            state.clip_complete.update(selected)
            state.clip_processed.update(selected)
        else:
            state.caption_complete.update(selected)
            state.caption_processed.update(selected)

    @staticmethod
    def _run_counts(state: _RunState) -> tuple[int, int, int]:
        eligible_ids = {
            photo.photo_id
            for photo in state.prepared
            if photo.photo_id not in state.invalid_image_ids
        }
        complete = eligible_ids & state.clip_complete & state.caption_complete
        newly_indexed = complete - state.fully_skipped
        return len(complete), len(newly_indexed), len(state.fully_skipped)

    def _checkpoint(self, state: _RunState, *, phase: str, batch_size: int) -> None:
        completed_count, indexed_count, skipped_count = self._run_counts(state)
        notes = json.dumps(
            {
                "checkpoint_phase": phase,
                "batch_size": batch_size,
                "prepared_count": len(state.prepared),
                "completed_count": completed_count,
                "clip_processed": len(state.clip_processed),
                "caption_processed": len(state.caption_processed),
            },
            sort_keys=True,
        )
        self.storage.finish_index_run(
            state.run_id,
            status="running",
            indexed_count=indexed_count,
            skipped_count=skipped_count,
            failed_count=len(state.failed_entities),
            notes=notes,
        )
        self._emit(
            run_id=state.run_id,
            phase=phase,
            status="checkpoint",
            indexed_count=indexed_count,
            failed_count=len(state.failed_entities),
        )

    def index_photos(
        self,
        image_paths: Iterable[str | Path],
        base_dir: str | Path,
        *,
        force: bool = False,
    ) -> IndexRunSummary:
        """Index photos incrementally and return an auditable run summary."""

        with self._run_lock:
            root = Path(base_dir)
            unique_paths: dict[str, Path] = {}
            for value in image_paths:
                path = Path(value)
                key = str(path.resolve()).casefold()
                unique_paths.setdefault(key, path)
            paths = sorted(unique_paths.values(), key=lambda item: str(item).casefold())
            run_id = self.storage.start_index_run(
                pipeline_version="staged-clip-caption-v1",
                config={
                    "clip_batch_size": self.clip_batch_size,
                    "caption_batch_size": self.caption_batch_size,
                    "min_batch_size": self.min_batch_size,
                    "force": bool(force),
                    "clip_model": self.clip_model_name,
                    "clip_model_version": self.clip_model_version,
                    "caption_model": self.caption_model_name,
                    "caption_model_version": self.caption_model_version,
                },
                discovered_count=len(paths),
            )
            state = _RunState(run_id=run_id, discovered_count=len(paths))
            final_clip_batch = self.clip_batch_size
            final_caption_batch = self.caption_batch_size

            try:
                self._prepare_photos(paths, root, state, force=force)
                self._checkpoint(state, phase="metadata", batch_size=self.clip_batch_size)

                clip_candidates = [photo for photo in state.prepared if photo.needs_clip]
                final_clip_batch = self._run_phase(
                    clip_candidates,
                    state=state,
                    phase="clip",
                    initial_batch_size=self.clip_batch_size,
                    process_batch=self._process_clip_batch,
                )
                unload_clip = getattr(self.clip_backend, "unload_model", None)
                if callable(unload_clip):
                    unload_clip()
                self._clear_cuda_cache()

                # BLIP starts only after every CLIP batch is checkpointed.  This
                # avoids retaining both phases' image tensors simultaneously.
                caption_candidates = [
                    photo
                    for photo in state.prepared
                    if photo.needs_caption and photo.photo_id not in state.invalid_image_ids
                ]
                final_caption_batch = self._run_phase(
                    caption_candidates,
                    state=state,
                    phase="caption",
                    initial_batch_size=self.caption_batch_size,
                    process_batch=self._process_caption_batch,
                )
                unload_caption = getattr(self.caption_backend, "unload_model", None)
                if callable(unload_caption):
                    unload_caption()
                self._clear_cuda_cache()
            except Exception as error:
                self._record_failure(state, stage="pipeline", error=error, retryable=True)
                completed_count, indexed_count, skipped_count = self._run_counts(state)
                self.storage.finish_index_run(
                    run_id,
                    status="failed",
                    indexed_count=indexed_count,
                    skipped_count=skipped_count,
                    failed_count=len(state.failed_entities),
                    notes=f"Fatal pipeline error: {error}",
                )
                raise

            completed_count, indexed_count, skipped_count = self._run_counts(state)
            status = "completed_with_errors" if state.failed_entities else "completed"
            self.storage.finish_index_run(
                run_id,
                status=status,
                indexed_count=indexed_count,
                skipped_count=skipped_count,
                failed_count=len(state.failed_entities),
                notes=json.dumps(
                    {
                        "prepared_count": len(state.prepared),
                        "completed_count": completed_count,
                        "clip_processed": len(state.clip_processed),
                        "caption_processed": len(state.caption_processed),
                        "final_clip_batch_size": final_clip_batch,
                        "final_caption_batch_size": final_caption_batch,
                    },
                    sort_keys=True,
                ),
            )
            return IndexRunSummary(
                run_id=run_id,
                status=status,
                discovered_count=len(paths),
                prepared_count=len(state.prepared),
                completed_count=completed_count,
                indexed_count=indexed_count,
                skipped_count=skipped_count,
                failed_count=len(state.failed_entities),
                clip_indexed_count=len(state.clip_processed),
                caption_indexed_count=len(state.caption_processed),
                clip_skipped_count=sum(not photo.needs_clip for photo in state.prepared),
                caption_skipped_count=sum(not photo.needs_caption for photo in state.prepared),
                final_clip_batch_size=final_clip_batch,
                final_caption_batch_size=final_caption_batch,
            )

    # Short alias useful for CLI callers.
    index = index_photos


PhotoIndexingService = IndexingService


__all__ = [
    "CaptionBackend",
    "ClipBackend",
    "DEFAULT_CAPTION_BATCH_SIZE",
    "DEFAULT_CLIP_BATCH_SIZE",
    "DEFAULT_MIN_BATCH_SIZE",
    "ImageVectorStore",
    "IndexRunSummary",
    "IndexingService",
    "PhotoIndexingService",
    "PreparedPhoto",
]
