"""Canonical SQLite storage for the Smart Photo Narrative project.

The vector database remains responsible for dense CLIP/scene-graph vectors.  This
module owns stable photo identity, searchable text, metadata filters, events,
stories, and indexing audit data.  It deliberately stores only normalized scene
graphs and triples; raw responses from a remote model belong in an external
sidecar file, never in this database.

All public operations use a short-lived SQLite connection.  That makes a single
``PhotoStorage`` instance safe to share between Streamlit worker threads while
WAL mode allows readers to continue during an indexing write.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

from config import CLIP_INDEX_MARKER_TAG


SCHEMA_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FTS_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
MOOD_LABELS: tuple[str, ...] = ("neutral", "calm", "happy", "excited", "tense", "sad")


class StorageError(RuntimeError):
    """Base error raised by the canonical data layer."""


class StorageConflictError(StorageError):
    """Raised when stable photo identity conflicts with an existing record."""


class StorageCapabilityError(StorageError):
    """Raised when the local SQLite build lacks a required capability."""


@dataclass(frozen=True)
class BM25Result:
    """One FTS5 hit.  Smaller ``bm25_rank`` values rank first."""

    photo_id: str
    bm25_rank: float
    caption: str
    scene_graph_text: str
    tags: str
    location: str


@dataclass(frozen=True)
class SceneGraphTriple:
    subject: str
    relation: str
    object: str
    confidence: float | None = None


@dataclass(frozen=True)
class PhotoMood:
    """One user-confirmed photographer mood attached to a photo."""

    photo_id: str
    mood_label: str
    source: str = "manual"
    subject_role: str = "photographer"
    confirmed: bool = True
    annotation_set: str = "hot_air_balloon_mood_v1"
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "mood_label": self.mood_label,
            "source": self.source,
            "subject_role": self.subject_role,
            "confirmed": self.confirmed,
            "annotation_set": self.annotation_set,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


_MIGRATION_1: tuple[str, ...] = (
    """
    CREATE TABLE photos (
        photo_id TEXT PRIMARY KEY,
        relative_path TEXT NOT NULL COLLATE NOCASE UNIQUE,
        content_sha256 TEXT NOT NULL COLLATE NOCASE UNIQUE,
        file_size_bytes INTEGER,
        file_mtime_ns INTEGER,
        file_ctime_ns INTEGER,
        image_width INTEGER,
        image_height INTEGER,
        captured_at TEXT,
        captured_at_sort INTEGER,
        date_local TEXT,
        timestamp_source TEXT NOT NULL DEFAULT 'unknown',
        timestamp_confidence REAL NOT NULL DEFAULT 0.0
            CHECK (timestamp_confidence >= 0.0 AND timestamp_confidence <= 1.0),
        gps_latitude REAL CHECK (gps_latitude IS NULL OR (gps_latitude >= -90 AND gps_latitude <= 90)),
        gps_longitude REAL CHECK (gps_longitude IS NULL OR (gps_longitude >= -180 AND gps_longitude <= 180)),
        gps_altitude REAL,
        location TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX photos_captured_sort_idx ON photos(captured_at_sort)",
    "CREATE INDEX photos_date_local_idx ON photos(date_local)",
    "CREATE INDEX photos_sha256_idx ON photos(content_sha256)",
    """
    CREATE TABLE captions (
        caption_id INTEGER PRIMARY KEY AUTOINCREMENT,
        photo_id TEXT NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL DEFAULT '',
        caption TEXT NOT NULL,
        language TEXT NOT NULL DEFAULT 'en',
        status TEXT NOT NULL DEFAULT 'ok',
        is_primary INTEGER NOT NULL DEFAULT 1 CHECK (is_primary IN (0, 1)),
        generated_at TEXT NOT NULL,
        UNIQUE(photo_id, model_name, model_version)
    )
    """,
    "CREATE INDEX captions_photo_idx ON captions(photo_id, is_primary)",
    """
    CREATE TABLE scene_graphs (
        scene_graph_id INTEGER PRIMARY KEY AUTOINCREMENT,
        photo_id TEXT NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL DEFAULT '',
        scene_graph_text TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'offline_jsonl',
        source_ref TEXT,
        output_sha256 TEXT,
        status TEXT NOT NULL DEFAULT 'ok',
        is_primary INTEGER NOT NULL DEFAULT 1 CHECK (is_primary IN (0, 1)),
        generated_at TEXT NOT NULL,
        UNIQUE(photo_id, model_name, model_version)
    )
    """,
    "CREATE INDEX scene_graphs_photo_idx ON scene_graphs(photo_id, is_primary)",
    """
    CREATE TABLE scene_graph_triples (
        triple_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_graph_id INTEGER NOT NULL REFERENCES scene_graphs(scene_graph_id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        subject TEXT NOT NULL,
        relation TEXT NOT NULL,
        object TEXT NOT NULL,
        confidence REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
        UNIQUE(scene_graph_id, ordinal)
    )
    """,
    "CREATE INDEX scene_graph_triples_terms_idx ON scene_graph_triples(subject, relation, object)",
    """
    CREATE TABLE tags (
        photo_id TEXT NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
        tag TEXT NOT NULL COLLATE NOCASE,
        source TEXT NOT NULL DEFAULT 'manual',
        score REAL CHECK (score IS NULL OR (score >= 0.0 AND score <= 1.0)),
        created_at TEXT NOT NULL,
        PRIMARY KEY(photo_id, tag, source)
    )
    """,
    "CREATE INDEX tags_tag_idx ON tags(tag COLLATE NOCASE)",
    """
    CREATE TABLE events (
        event_id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        summary TEXT NOT NULL DEFAULT '',
        start_at TEXT,
        start_at_sort INTEGER,
        end_at TEXT,
        end_at_sort INTEGER,
        date_local TEXT,
        method TEXT NOT NULL,
        algorithm_version TEXT NOT NULL DEFAULT '',
        parameters_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX events_time_idx ON events(start_at_sort, end_at_sort)",
    """
    CREATE TABLE event_photos (
        event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        photo_id TEXT NOT NULL REFERENCES photos(photo_id) ON DELETE CASCADE,
        position INTEGER NOT NULL,
        role TEXT NOT NULL DEFAULT 'member',
        similarity_to_previous REAL,
        PRIMARY KEY(event_id, photo_id),
        UNIQUE(event_id, position)
    )
    """,
    "CREATE INDEX event_photos_photo_idx ON event_photos(photo_id)",
    """
    CREATE TABLE stories (
        story_id TEXT PRIMARY KEY,
        event_id TEXT REFERENCES events(event_id) ON DELETE SET NULL,
        title TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        style TEXT NOT NULL DEFAULT 'grounded',
        language TEXT NOT NULL DEFAULT 'zh',
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL DEFAULT '',
        prompt_version TEXT NOT NULL DEFAULT '',
        grounding_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX stories_event_idx ON stories(event_id, created_at)",
    """
    CREATE TABLE index_runs (
        run_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        pipeline_version TEXT NOT NULL DEFAULT '',
        config_json TEXT NOT NULL DEFAULT '{}',
        started_at TEXT NOT NULL,
        completed_at TEXT,
        discovered_count INTEGER NOT NULL DEFAULT 0,
        indexed_count INTEGER NOT NULL DEFAULT 0,
        skipped_count INTEGER NOT NULL DEFAULT 0,
        failed_count INTEGER NOT NULL DEFAULT 0,
        notes TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE index_failures (
        failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT REFERENCES index_runs(run_id) ON DELETE CASCADE,
        photo_id TEXT REFERENCES photos(photo_id) ON DELETE SET NULL,
        relative_path TEXT,
        stage TEXT NOT NULL,
        error_type TEXT NOT NULL,
        message TEXT NOT NULL,
        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX index_failures_run_idx ON index_failures(run_id, stage)",
    """
    CREATE VIRTUAL TABLE photo_fts USING fts5(
        photo_id UNINDEXED,
        caption,
        scene_graph_text,
        tags,
        location,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER photos_delete_fts AFTER DELETE ON photos BEGIN
        DELETE FROM photo_fts WHERE photo_id = OLD.photo_id;
    END
    """,
)


_MIGRATION_2: tuple[str, ...] = (
    """
    CREATE TABLE photo_moods (
        photo_id TEXT PRIMARY KEY REFERENCES photos(photo_id) ON DELETE CASCADE,
        mood_label TEXT NOT NULL
            CHECK (mood_label IN ('neutral', 'calm', 'happy', 'excited', 'tense', 'sad')),
        source TEXT NOT NULL DEFAULT 'manual'
            CHECK (source = 'manual'),
        subject_role TEXT NOT NULL DEFAULT 'photographer'
            CHECK (subject_role = 'photographer'),
        confirmed INTEGER NOT NULL DEFAULT 1
            CHECK (confirmed = 1),
        annotation_set TEXT NOT NULL DEFAULT 'hot_air_balloon_mood_v1',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX photo_moods_annotation_idx ON photo_moods(annotation_set, mood_label)",
)


class PhotoStorage:
    """Thread-safe canonical SQLite data layer.

    ``content_sha256`` is unique and is the identity reconciliation key.  Calling
    :meth:`upsert_photo` with the same hash and a new path is treated as a rename,
    so the existing UUID and every event/story relationship remain stable.
    Exact duplicate files are intentionally represented by one logical photo.
    """

    def __init__(self, database_path: str | Path, *, timeout_seconds: float = 30.0) -> None:
        self.database_path = Path(database_path)
        self.timeout_seconds = float(timeout_seconds)
        self._write_lock = threading.RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout_seconds * 1000))}")
        return connection

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        """Yield one atomic connection and commit or roll back as a unit.

        Storage methods intentionally open their own short transactions.  Use
        this context for custom atomic SQL and do not call another storage method
        from inside it.
        """

        guard = self._write_lock if write else nullcontext()
        with guard:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def migrate(self) -> int:
        """Apply schema migrations exactly once and return the active version."""

        with self._write_lock:
            connection = self._connect()
            try:
                # WAL persists on the database and must be selected outside an
                # active transaction.
                journal_mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
                if journal_mode != "wal":
                    raise StorageCapabilityError(f"SQLite could not enable WAL mode: {journal_mode}")
                connection.execute("PRAGMA synchronous = NORMAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    )
                    """
                )
                row = connection.execute("SELECT COALESCE(MAX(version), 0) AS version FROM schema_version").fetchone()
                current = int(row["version"])
                if current > SCHEMA_VERSION:
                    raise StorageError(
                        f"Database schema {current} is newer than supported version {SCHEMA_VERSION}"
                    )
                # A real v1 album is backed up once before the first v2 write.
                # Fresh test databases move from 0 to 2 atomically and do not
                # need a redundant empty backup.
                if current == 1:
                    backup_path = self.database_path.with_name(
                        f"{self.database_path.stem}.schema_v1_backup{self.database_path.suffix}"
                    )
                    if not backup_path.exists():
                        with sqlite3.connect(backup_path) as backup_connection:
                            connection.backup(backup_connection)

                connection.execute("BEGIN IMMEDIATE")
                if current < 1:
                    try:
                        for statement in _MIGRATION_1:
                            connection.execute(statement)
                    except sqlite3.OperationalError as error:
                        if "fts5" in str(error).lower():
                            raise StorageCapabilityError("This SQLite build does not provide FTS5") from error
                        raise
                    connection.execute(
                        "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                        (1, self._utc_now()),
                    )
                if current < 2:
                    for statement in _MIGRATION_2:
                        connection.execute(statement)
                    # This project stores the active schema version rather than
                    # a full migration history; preserve that v1 convention.
                    connection.execute("DELETE FROM schema_version")
                    connection.execute(
                        "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                        (2, self._utc_now()),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
        return self.schema_version

    @property
    def schema_version(self) -> int:
        with self.transaction(write=False) as connection:
            row = connection.execute("SELECT COALESCE(MAX(version), 0) AS version FROM schema_version").fetchone()
        return int(row["version"])

    @staticmethod
    def _normalize_relative_path(relative_path: str | Path) -> str:
        raw = str(relative_path).strip().replace("\\", "/")
        if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
            raise ValueError("relative_path must be a non-empty path relative to the photo root")
        normalized = PurePosixPath(raw)
        if any(part == ".." for part in normalized.parts):
            raise ValueError("relative_path must not escape the photo root")
        value = normalized.as_posix()
        if value in {"", "."}:
            raise ValueError("relative_path must name a file")
        return value

    @staticmethod
    def _normalize_sha256(content_sha256: str) -> str:
        value = content_sha256.strip().lower()
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("content_sha256 must contain exactly 64 hexadecimal characters")
        return value

    @staticmethod
    def _normalize_timestamp(
        captured_at: str | datetime | None,
        captured_at_sort: int | None,
        date_local: str | None,
    ) -> tuple[str | None, int | None, str | None]:
        if captured_at is None:
            return None, captured_at_sort, date_local

        if isinstance(captured_at, datetime):
            parsed = captured_at
        else:
            raw = captured_at.strip()
            if not raw:
                return None, captured_at_sort, date_local
            normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                try:
                    parsed = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                except ValueError as error:
                    raise ValueError(f"captured_at is not an ISO-8601 or EXIF timestamp: {raw}") from error

        local_date = date_local or parsed.date().isoformat()
        if parsed.tzinfo is None:
            # EXIF commonly has no offset.  Treating the wall clock as UTC makes
            # the sort key deterministic across machines; date_local preserves
            # the actual calendar date shown to the user.
            sortable = parsed.replace(tzinfo=timezone.utc)
            stored = parsed.isoformat(timespec="seconds")
        else:
            sortable = parsed.astimezone(timezone.utc)
            stored = parsed.isoformat(timespec="seconds")
        sort_value = captured_at_sort
        if sort_value is None:
            sort_value = int(sortable.timestamp() * 1_000_000)
        return stored, int(sort_value), local_date

    @staticmethod
    def _validate_photo_values(
        *,
        timestamp_confidence: float,
        gps_latitude: float | None,
        gps_longitude: float | None,
        image_width: int | None,
        image_height: int | None,
    ) -> None:
        if not 0.0 <= timestamp_confidence <= 1.0:
            raise ValueError("timestamp_confidence must be between 0 and 1")
        if gps_latitude is not None and not -90 <= gps_latitude <= 90:
            raise ValueError("gps_latitude must be between -90 and 90")
        if gps_longitude is not None and not -180 <= gps_longitude <= 180:
            raise ValueError("gps_longitude must be between -180 and 180")
        if image_width is not None and image_width <= 0:
            raise ValueError("image_width must be positive")
        if image_height is not None and image_height <= 0:
            raise ValueError("image_height must be positive")

    def upsert_photo(
        self,
        *,
        relative_path: str | Path,
        content_sha256: str,
        file_size_bytes: int | None = None,
        file_mtime_ns: int | None = None,
        file_ctime_ns: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        captured_at: str | datetime | None = None,
        captured_at_sort: int | None = None,
        date_local: str | None = None,
        timestamp_source: str = "unknown",
        timestamp_confidence: float = 0.0,
        gps_latitude: float | None = None,
        gps_longitude: float | None = None,
        gps_altitude: float | None = None,
        location: str | None = None,
        photo_id: str | None = None,
    ) -> str:
        """Insert/update a photo and return its stable UUID.

        Hash identity takes precedence over the path: the same content at a new
        path is a rename.  A caller-supplied ``photo_id`` is accepted only for a
        new row and must be a valid UUID.
        """

        path = self._normalize_relative_path(relative_path)
        digest = self._normalize_sha256(content_sha256)
        confidence = float(timestamp_confidence)
        self._validate_photo_values(
            timestamp_confidence=confidence,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            image_width=image_width,
            image_height=image_height,
        )
        captured_text, sort_value, local_date = self._normalize_timestamp(
            captured_at, captured_at_sort, date_local
        )
        if photo_id is not None:
            photo_id = str(uuid.UUID(photo_id))
        now = self._utc_now()

        with self.transaction() as connection:
            hash_row = connection.execute(
                "SELECT photo_id, relative_path FROM photos WHERE content_sha256 = ? COLLATE NOCASE",
                (digest,),
            ).fetchone()
            path_row = connection.execute(
                "SELECT photo_id, content_sha256 FROM photos WHERE relative_path = ? COLLATE NOCASE",
                (path,),
            ).fetchone()

            if hash_row is not None and path_row is not None and hash_row["photo_id"] != path_row["photo_id"]:
                raise StorageConflictError(
                    f"Path {path!r} and SHA-256 {digest!r} identify different photos"
                )

            existing = hash_row or path_row
            stable_id = existing["photo_id"] if existing is not None else (photo_id or str(uuid.uuid4()))
            if existing is not None and photo_id is not None and photo_id != stable_id:
                raise StorageConflictError("photo_id cannot replace an existing stable UUID")

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO photos(
                        photo_id, relative_path, content_sha256,
                        file_size_bytes, file_mtime_ns, file_ctime_ns,
                        image_width, image_height,
                        captured_at, captured_at_sort, date_local,
                        timestamp_source, timestamp_confidence,
                        gps_latitude, gps_longitude, gps_altitude, location,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id, path, digest,
                        file_size_bytes, file_mtime_ns, file_ctime_ns,
                        image_width, image_height,
                        captured_text, sort_value, local_date,
                        timestamp_source, confidence,
                        gps_latitude, gps_longitude, gps_altitude, location,
                        now, now,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE photos SET
                        relative_path = ?, content_sha256 = ?,
                        file_size_bytes = ?, file_mtime_ns = ?, file_ctime_ns = ?,
                        image_width = ?, image_height = ?,
                        captured_at = ?, captured_at_sort = ?, date_local = ?,
                        timestamp_source = ?, timestamp_confidence = ?,
                        gps_latitude = ?, gps_longitude = ?, gps_altitude = ?, location = ?,
                        updated_at = ?
                    WHERE photo_id = ?
                    """,
                    (
                        path, digest,
                        file_size_bytes, file_mtime_ns, file_ctime_ns,
                        image_width, image_height,
                        captured_text, sort_value, local_date,
                        timestamp_source, confidence,
                        gps_latitude, gps_longitude, gps_altitude, location,
                        now, stable_id,
                    ),
                )
            self._refresh_photo_fts(connection, stable_id)
        return stable_id

    def get_photo(self, photo_id: str) -> dict[str, Any] | None:
        with self.transaction(write=False) as connection:
            row = connection.execute("SELECT * FROM photos WHERE photo_id = ?", (photo_id,)).fetchone()
        return dict(row) if row is not None else None

    def get_photo_by_path(self, relative_path: str | Path) -> dict[str, Any] | None:
        path = self._normalize_relative_path(relative_path)
        with self.transaction(write=False) as connection:
            row = connection.execute(
                "SELECT * FROM photos WHERE relative_path = ? COLLATE NOCASE", (path,)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_photo_by_hash(self, content_sha256: str) -> dict[str, Any] | None:
        digest = self._normalize_sha256(content_sha256)
        with self.transaction(write=False) as connection:
            row = connection.execute(
                "SELECT * FROM photos WHERE content_sha256 = ? COLLATE NOCASE", (digest,)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_caption(
        self,
        photo_id: str,
        *,
        model_name: str | None = None,
        model_version: str = "",
        primary_only: bool = True,
    ) -> dict[str, Any] | None:
        """Return the selected caption without exposing a writable connection."""

        where = ["photo_id = ?", "status = 'ok'"]
        parameters: list[Any] = [photo_id]
        if model_name is not None:
            where.extend(["model_name = ?", "model_version = ?"])
            parameters.extend([model_name, model_version])
        if primary_only:
            where.append("is_primary = 1")
        sql = (
            "SELECT * FROM captions WHERE "
            + " AND ".join(where)
            + " ORDER BY generated_at DESC, caption_id DESC LIMIT 1"
        )
        with self.transaction(write=False) as connection:
            row = connection.execute(sql, parameters).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _photo_mood_from_row(row: sqlite3.Row) -> PhotoMood:
        return PhotoMood(
            photo_id=str(row["photo_id"]),
            mood_label=str(row["mood_label"]),
            source=str(row["source"]),
            subject_role=str(row["subject_role"]),
            confirmed=bool(row["confirmed"]),
            annotation_set=str(row["annotation_set"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get_photo_mood(self, photo_id: str) -> PhotoMood | None:
        """Return a photo's confirmed photographer mood, if annotated."""

        with self.transaction(write=False) as connection:
            row = connection.execute(
                "SELECT * FROM photo_moods WHERE photo_id = ?", (photo_id,)
            ).fetchone()
        return self._photo_mood_from_row(row) if row is not None else None

    def get_photo_moods(self, photo_ids: Sequence[str]) -> dict[str, PhotoMood]:
        """Batch-read moods without changing caller order or touching FTS."""

        unique_ids = list(dict.fromkeys(str(photo_id) for photo_id in photo_ids))
        if not unique_ids:
            return {}
        placeholders = ", ".join("?" for _ in unique_ids)
        with self.transaction(write=False) as connection:
            rows = connection.execute(
                f"SELECT * FROM photo_moods WHERE photo_id IN ({placeholders})", unique_ids
            ).fetchall()
        by_id = {str(row["photo_id"]): self._photo_mood_from_row(row) for row in rows}
        return {photo_id: by_id[photo_id] for photo_id in unique_ids if photo_id in by_id}

    def save_photo_moods(
        self,
        moods: Mapping[str, str | None],
        *,
        annotation_set: str = "hot_air_balloon_mood_v1",
    ) -> dict[str, PhotoMood]:
        """Atomically create, replace, or clear manual photographer moods.

        ``None`` and an empty string delete an annotation.  Every photo and
        every non-empty label is validated before the first write, so a bad
        row cannot partially update an experiment set.
        """

        normalized: dict[str, str | None] = {}
        for raw_photo_id, raw_label in moods.items():
            photo_id = str(raw_photo_id).strip()
            if not photo_id:
                raise ValueError("photo_id must not be empty")
            label = None if raw_label is None else str(raw_label).strip().casefold()
            if not label:
                label = None
            elif label not in MOOD_LABELS:
                raise ValueError(
                    f"Unsupported mood label {raw_label!r}; expected one of {', '.join(MOOD_LABELS)}"
                )
            normalized[photo_id] = label
        if not normalized:
            return {}
        annotation_set = str(annotation_set).strip()
        if not annotation_set:
            raise ValueError("annotation_set must not be empty")

        now = self._utc_now()
        with self.transaction() as connection:
            placeholders = ", ".join("?" for _ in normalized)
            existing = {
                str(row["photo_id"])
                for row in connection.execute(
                    f"SELECT photo_id FROM photos WHERE photo_id IN ({placeholders})",
                    list(normalized),
                )
            }
            missing = sorted(set(normalized) - existing)
            if missing:
                raise KeyError(f"Unknown photo IDs: {missing}")
            for photo_id, label in normalized.items():
                if label is None:
                    connection.execute("DELETE FROM photo_moods WHERE photo_id = ?", (photo_id,))
                    continue
                connection.execute(
                    """
                    INSERT INTO photo_moods(
                        photo_id, mood_label, source, subject_role, confirmed,
                        annotation_set, created_at, updated_at
                    ) VALUES (?, ?, 'manual', 'photographer', 1, ?, ?, ?)
                    ON CONFLICT(photo_id) DO UPDATE SET
                        mood_label = excluded.mood_label,
                        source = 'manual',
                        subject_role = 'photographer',
                        confirmed = 1,
                        annotation_set = excluded.annotation_set,
                        updated_at = excluded.updated_at
                    """,
                    (photo_id, label, annotation_set, now, now),
                )
        return self.get_photo_moods(list(normalized))

    def delete_photo_moods(self, photo_ids: Sequence[str]) -> int:
        """Delete selected mood annotations and return the affected row count."""

        unique_ids = list(dict.fromkeys(str(photo_id) for photo_id in photo_ids))
        if not unique_ids:
            return 0
        placeholders = ", ".join("?" for _ in unique_ids)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"DELETE FROM photo_moods WHERE photo_id IN ({placeholders})", unique_ids
            )
        return int(cursor.rowcount)

    def get_photo_detail(self, photo_id: str) -> dict[str, Any] | None:
        """Return one photo's display details in a single read transaction.

        Primary successful BLIP/Qwen records take precedence.  If an older
        import has no primary marker, the newest successful record is used so
        the detail dialog remains useful without repairing stored data.
        """

        with self.transaction(write=False) as connection:
            photo_row = connection.execute(
                """
                SELECT
                    relative_path, image_width, image_height, captured_at,
                    date_local, timestamp_source, timestamp_confidence, location
                FROM photos
                WHERE photo_id = ?
                """,
                (photo_id,),
            ).fetchone()
            if photo_row is None:
                return None

            caption_row = connection.execute(
                """
                SELECT *
                FROM captions
                WHERE photo_id = ? AND status = 'ok'
                ORDER BY is_primary DESC, generated_at DESC, caption_id DESC
                LIMIT 1
                """,
                (photo_id,),
            ).fetchone()
            graph_row = connection.execute(
                """
                SELECT *
                FROM scene_graphs
                WHERE photo_id = ? AND status = 'ok'
                ORDER BY is_primary DESC, generated_at DESC, scene_graph_id DESC
                LIMIT 1
                """,
                (photo_id,),
            ).fetchone()
            triple_rows = []
            if graph_row is not None:
                triple_rows = connection.execute(
                    """
                    SELECT subject, relation, object, confidence
                    FROM scene_graph_triples
                    WHERE scene_graph_id = ?
                    ORDER BY ordinal
                    """,
                    (graph_row["scene_graph_id"],),
                ).fetchall()
            tag_rows = connection.execute(
                """
                SELECT tag
                FROM tags
                WHERE photo_id = ?
                ORDER BY tag COLLATE NOCASE, tag
                """,
                (photo_id,),
            ).fetchall()
            mood_row = connection.execute(
                "SELECT * FROM photo_moods WHERE photo_id = ?", (photo_id,)
            ).fetchone()

        tags: list[str] = []
        seen_tags: set[str] = set()
        for row in tag_rows:
            tag = str(row["tag"])
            key = tag.casefold()
            if key not in seen_tags:
                seen_tags.add(key)
                tags.append(tag)
        return {
            "photo": dict(photo_row),
            "caption": dict(caption_row) if caption_row is not None else None,
            "scene_graph": dict(graph_row) if graph_row is not None else None,
            "triples": [dict(row) for row in triple_rows],
            "tags": tags,
            "mood": self._photo_mood_from_row(mood_row).to_dict() if mood_row is not None else None,
        }

    def has_tags(self, photo_id: str, *, source: str | None = None) -> bool:
        """Return whether a photo has at least one tag, optionally from one model."""

        sql = "SELECT 1 FROM tags WHERE photo_id = ?"
        parameters: list[Any] = [photo_id]
        if source is not None:
            sql += " AND source = ?"
            parameters.append(source)
        sql += " LIMIT 1"
        with self.transaction(write=False) as connection:
            return connection.execute(sql, parameters).fetchone() is not None

    def list_tag_sources(self, photo_id: str, *, source_prefix: str | None = None) -> set[str]:
        """Return distinct persisted tag sources, optionally within one family."""

        sql = "SELECT DISTINCT source FROM tags WHERE photo_id = ?"
        parameters: list[Any] = [photo_id]
        if source_prefix is not None:
            sql += " AND substr(source, 1, ?) = ?"
            parameters.extend([len(source_prefix), source_prefix])
        with self.transaction(write=False) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return {str(row["source"]) for row in rows}

    def get_index_run(self, run_id: str) -> dict[str, Any] | None:
        with self.transaction(write=False) as connection:
            row = connection.execute("SELECT * FROM index_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_index_failures(self, run_id: str) -> list[dict[str, Any]]:
        with self.transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT * FROM index_failures WHERE run_id = ? ORDER BY failure_id", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_caption(
        self,
        photo_id: str,
        caption: str,
        *,
        model_name: str = "Salesforce/blip-image-captioning-base",
        model_version: str = "",
        language: str = "en",
        status: str = "ok",
        is_primary: bool = True,
        generated_at: str | None = None,
    ) -> int:
        text = caption.strip()
        if not text:
            raise ValueError("caption must not be empty")
        with self.transaction() as connection:
            if is_primary:
                connection.execute("UPDATE captions SET is_primary = 0 WHERE photo_id = ?", (photo_id,))
            connection.execute(
                """
                INSERT INTO captions(
                    photo_id, model_name, model_version, caption, language,
                    status, is_primary, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(photo_id, model_name, model_version) DO UPDATE SET
                    caption = excluded.caption,
                    language = excluded.language,
                    status = excluded.status,
                    is_primary = excluded.is_primary,
                    generated_at = excluded.generated_at
                """,
                (
                    photo_id, model_name, model_version, text, language,
                    status, int(is_primary), generated_at or self._utc_now(),
                ),
            )
            row = connection.execute(
                """
                SELECT caption_id FROM captions
                WHERE photo_id = ? AND model_name = ? AND model_version = ?
                """,
                (photo_id, model_name, model_version),
            ).fetchone()
            self._refresh_photo_fts(connection, photo_id)
        return int(row["caption_id"])

    @staticmethod
    def _coerce_triple(value: SceneGraphTriple | Mapping[str, Any] | Sequence[Any]) -> SceneGraphTriple:
        if isinstance(value, SceneGraphTriple):
            triple = value
        elif isinstance(value, Mapping):
            triple = SceneGraphTriple(
                subject=str(value.get("subject", "")),
                relation=str(value.get("relation", value.get("predicate", ""))),
                object=str(value.get("object", "")),
                confidence=(None if value.get("confidence") is None else float(value["confidence"])),
            )
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) in {3, 4}:
            confidence = None if len(value) == 3 or value[3] is None else float(value[3])
            triple = SceneGraphTriple(str(value[0]), str(value[1]), str(value[2]), confidence)
        else:
            raise ValueError("Each scene-graph triple must be a triple object, mapping, or 3/4-item sequence")
        clean = SceneGraphTriple(
            triple.subject.strip(), triple.relation.strip(), triple.object.strip(), triple.confidence
        )
        if not clean.subject or not clean.relation or not clean.object:
            raise ValueError("Scene-graph subject, relation, and object must be non-empty")
        if clean.confidence is not None and not 0.0 <= clean.confidence <= 1.0:
            raise ValueError("Scene-graph confidence must be between 0 and 1")
        return clean

    def upsert_scene_graph(
        self,
        photo_id: str,
        *,
        scene_graph_text: str,
        triples: Iterable[SceneGraphTriple | Mapping[str, Any] | Sequence[Any]],
        model_name: str,
        model_version: str = "",
        source: str = "offline_jsonl",
        source_ref: str | None = None,
        output_sha256: str | None = None,
        status: str = "ok",
        is_primary: bool = True,
        generated_at: str | None = None,
    ) -> int:
        """Store a normalized graph and triples (never the remote raw response)."""

        text = scene_graph_text.strip()
        normalized_triples = [self._coerce_triple(item) for item in triples]
        if not text and normalized_triples:
            text = " ; ".join(
                f"{item.subject} {item.relation} {item.object}" for item in normalized_triples
            )
        if not text and status == "ok":
            raise ValueError("A successful scene graph must have text or triples")
        if output_sha256 is not None:
            output_sha256 = self._normalize_sha256(output_sha256)

        with self.transaction() as connection:
            if is_primary:
                connection.execute("UPDATE scene_graphs SET is_primary = 0 WHERE photo_id = ?", (photo_id,))
            connection.execute(
                """
                INSERT INTO scene_graphs(
                    photo_id, model_name, model_version, scene_graph_text,
                    source, source_ref, output_sha256, status, is_primary, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(photo_id, model_name, model_version) DO UPDATE SET
                    scene_graph_text = excluded.scene_graph_text,
                    source = excluded.source,
                    source_ref = excluded.source_ref,
                    output_sha256 = excluded.output_sha256,
                    status = excluded.status,
                    is_primary = excluded.is_primary,
                    generated_at = excluded.generated_at
                """,
                (
                    photo_id, model_name, model_version, text,
                    source, source_ref, output_sha256, status, int(is_primary),
                    generated_at or self._utc_now(),
                ),
            )
            row = connection.execute(
                """
                SELECT scene_graph_id FROM scene_graphs
                WHERE photo_id = ? AND model_name = ? AND model_version = ?
                """,
                (photo_id, model_name, model_version),
            ).fetchone()
            graph_id = int(row["scene_graph_id"])
            connection.execute("DELETE FROM scene_graph_triples WHERE scene_graph_id = ?", (graph_id,))
            connection.executemany(
                """
                INSERT INTO scene_graph_triples(
                    scene_graph_id, ordinal, subject, relation, object, confidence
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (graph_id, index, item.subject, item.relation, item.object, item.confidence)
                    for index, item in enumerate(normalized_triples)
                ],
            )
            self._refresh_photo_fts(connection, photo_id)
        return graph_id

    def replace_tags(
        self,
        photo_id: str,
        tags: Iterable[str | tuple[str, float | None]],
        *,
        source: str = "clip_zero_shot",
        clear_source_prefix: str | None = None,
    ) -> None:
        if clear_source_prefix == "":
            raise ValueError("clear_source_prefix must not be empty")
        normalized: dict[str, tuple[str, float | None]] = {}
        for value in tags:
            if isinstance(value, str):
                tag, score = value.strip(), None
            else:
                tag, score = str(value[0]).strip(), value[1]
                score = None if score is None else float(score)
            if not tag:
                continue
            if score is not None and not 0.0 <= score <= 1.0:
                raise ValueError("tag score must be between 0 and 1")
            normalized[tag.casefold()] = (tag, score)
        now = self._utc_now()
        with self.transaction() as connection:
            if clear_source_prefix is None:
                connection.execute(
                    "DELETE FROM tags WHERE photo_id = ? AND source = ?",
                    (photo_id, source),
                )
            else:
                connection.execute(
                    "DELETE FROM tags WHERE photo_id = ? AND substr(source, 1, ?) = ?",
                    (photo_id, len(clear_source_prefix), clear_source_prefix),
                )
            connection.executemany(
                "INSERT INTO tags(photo_id, tag, source, score, created_at) VALUES (?, ?, ?, ?, ?)",
                [(photo_id, tag, source, score, now) for tag, score in normalized.values()],
            )
            self._refresh_photo_fts(connection, photo_id)

    def _refresh_photo_fts(self, connection: sqlite3.Connection, photo_id: str) -> None:
        photo = connection.execute(
            "SELECT location FROM photos WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        connection.execute("DELETE FROM photo_fts WHERE photo_id = ?", (photo_id,))
        if photo is None:
            return
        caption_row = connection.execute(
            """
            SELECT COALESCE(group_concat(caption, ' '), '') AS text
            FROM captions
            WHERE photo_id = ? AND status = 'ok'
              AND (
                is_primary = 1 OR NOT EXISTS(
                    SELECT 1 FROM captions c2
                    WHERE c2.photo_id = captions.photo_id AND c2.status = 'ok' AND c2.is_primary = 1
                )
              )
            """,
            (photo_id,),
        ).fetchone()
        graph_row = connection.execute(
            """
            SELECT COALESCE(group_concat(scene_graph_text, ' '), '') AS text
            FROM scene_graphs
            WHERE photo_id = ? AND status = 'ok'
              AND (
                is_primary = 1 OR NOT EXISTS(
                    SELECT 1 FROM scene_graphs s2
                    WHERE s2.photo_id = scene_graphs.photo_id AND s2.status = 'ok' AND s2.is_primary = 1
                )
              )
            """,
            (photo_id,),
        ).fetchone()
        tag_rows = connection.execute(
            "SELECT tag FROM tags WHERE photo_id = ? AND tag != ? ORDER BY tag COLLATE NOCASE",
            (photo_id, CLIP_INDEX_MARKER_TAG),
        ).fetchall()
        connection.execute(
            """
            INSERT INTO photo_fts(photo_id, caption, scene_graph_text, tags, location)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                photo_id,
                caption_row["text"],
                graph_row["text"],
                " ".join(row["tag"] for row in tag_rows),
                photo["location"] or "",
            ),
        )

    def refresh_photo_fts(self, photo_id: str) -> None:
        """Upsert one aggregate FTS document from normalized source tables."""

        with self.transaction() as connection:
            self._refresh_photo_fts(connection, photo_id)

    def rebuild_fts(self) -> int:
        """Recreate every aggregate FTS document and return its row count."""

        with self.transaction() as connection:
            connection.execute("DELETE FROM photo_fts")
            photo_ids = [row["photo_id"] for row in connection.execute("SELECT photo_id FROM photos")]
            for photo_id in photo_ids:
                self._refresh_photo_fts(connection, photo_id)
            row = connection.execute("SELECT COUNT(*) AS count FROM photo_fts").fetchone()
        return int(row["count"])

    @staticmethod
    def _build_fts_query(query: str, *, match_all_terms: bool) -> str:
        tokens = _FTS_TOKEN_RE.findall(query)
        if not tokens:
            raise ValueError("FTS query must contain at least one letter or number")
        escaped = ['"' + token.replace('"', '""') + '"' for token in tokens]
        return (" AND " if match_all_terms else " OR ").join(escaped)

    def search_bm25(
        self,
        query: str,
        *,
        limit: int = 20,
        eligible_photo_ids: Iterable[str] | None = None,
        match_all_terms: bool = True,
        fields: Iterable[str] | None = None,
    ) -> list[BM25Result]:
        """Search caption/graph/tags/location using FTS5 BM25.

        ``eligible_photo_ids`` is normally produced by
        :meth:`metadata_eligible_ids`, ensuring date/location/tag filtering
        happens before ranking rather than after a small top-k result set.
        ``fields`` provides an explicit FTS5 column filter.  This is essential
        for controlled ablations: the retrieval ``caption`` channel must not
        silently consume Scene Graph text stored in the same FTS document.
        """

        if limit <= 0:
            return []
        fts_query = self._build_fts_query(query, match_all_terms=match_all_terms)
        searchable_fields = ("caption", "scene_graph_text", "tags", "location")
        selected_fields = searchable_fields if fields is None else tuple(
            dict.fromkeys(str(field).strip() for field in fields if str(field).strip())
        )
        invalid_fields = sorted(set(selected_fields).difference(searchable_fields))
        if invalid_fields:
            raise ValueError(f"Unsupported FTS field(s): {', '.join(invalid_fields)}")
        if not selected_fields:
            return []
        if selected_fields != searchable_fields:
            fts_query = f"{{{' '.join(selected_fields)}}} : ({fts_query})"
        eligible = None if eligible_photo_ids is None else list(dict.fromkeys(eligible_photo_ids))
        if eligible == []:
            return []

        with self.transaction(write=False) as connection:
            eligibility_join = ""
            if eligible is not None:
                connection.execute("CREATE TEMP TABLE eligible_photo_ids(photo_id TEXT PRIMARY KEY)")
                connection.executemany(
                    "INSERT INTO eligible_photo_ids(photo_id) VALUES (?)",
                    [(photo_id,) for photo_id in eligible],
                )
                eligibility_join = "JOIN eligible_photo_ids e ON e.photo_id = photo_fts.photo_id"
            rows = connection.execute(
                f"""
                SELECT
                    photo_fts.photo_id,
                    bm25(photo_fts, 0.0, 1.0, 1.15, 0.7, 0.6) AS bm25_rank,
                    photo_fts.caption,
                    photo_fts.scene_graph_text,
                    photo_fts.tags,
                    photo_fts.location
                FROM photo_fts
                {eligibility_join}
                WHERE photo_fts MATCH ?
                ORDER BY bm25_rank ASC, photo_fts.photo_id ASC
                LIMIT ?
                """,
                (fts_query, int(limit)),
            ).fetchall()
        return [
            BM25Result(
                photo_id=row["photo_id"],
                bm25_rank=float(row["bm25_rank"]),
                caption=row["caption"] or "",
                scene_graph_text=row["scene_graph_text"] or "",
                tags=row["tags"] or "",
                location=row["location"] or "",
            )
            for row in rows
        ]

    def metadata_eligible_ids(
        self,
        *,
        captured_from_sort: int | None = None,
        captured_to_sort: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_timestamp_confidence: float | None = None,
        tags_all: Iterable[str] = (),
        tags_any: Iterable[str] = (),
        location_contains: str | None = None,
        require_gps: bool | None = None,
        event_id: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Return stable IDs eligible for dense/sparse retrieval.

        Date and time bounds are inclusive and evaluated in SQL before any
        ranking.  Results are chronological, with unknown timestamps last.
        """

        where: list[str] = []
        parameters: list[Any] = []
        joins = ""
        if captured_from_sort is not None:
            where.append("p.captured_at_sort >= ?")
            parameters.append(int(captured_from_sort))
        if captured_to_sort is not None:
            where.append("p.captured_at_sort <= ?")
            parameters.append(int(captured_to_sort))
        if date_from is not None:
            where.append("p.date_local >= ?")
            parameters.append(date_from)
        if date_to is not None:
            where.append("p.date_local <= ?")
            parameters.append(date_to)
        if min_timestamp_confidence is not None:
            value = float(min_timestamp_confidence)
            if not 0.0 <= value <= 1.0:
                raise ValueError("min_timestamp_confidence must be between 0 and 1")
            where.append("p.timestamp_confidence >= ?")
            parameters.append(value)
        if location_contains:
            where.append("p.location LIKE ? ESCAPE '\\' COLLATE NOCASE")
            escaped = location_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            parameters.append(f"%{escaped}%")
        if require_gps is True:
            where.append("p.gps_latitude IS NOT NULL AND p.gps_longitude IS NOT NULL")
        elif require_gps is False:
            where.append("p.gps_latitude IS NULL OR p.gps_longitude IS NULL")
        if event_id is not None:
            joins += " JOIN event_photos ep ON ep.photo_id = p.photo_id AND ep.event_id = ?"
            parameters.insert(0, event_id)

        normalized_all = [tag.strip() for tag in tags_all if tag.strip()]
        for tag in normalized_all:
            where.append(
                "EXISTS (SELECT 1 FROM tags ta WHERE ta.photo_id = p.photo_id AND ta.tag = ? COLLATE NOCASE)"
            )
            parameters.append(tag)
        normalized_any = [tag.strip() for tag in tags_any if tag.strip()]
        if normalized_any:
            placeholders = ", ".join("?" for _ in normalized_any)
            where.append(
                f"EXISTS (SELECT 1 FROM tags ty WHERE ty.photo_id = p.photo_id "
                f"AND ty.tag COLLATE NOCASE IN ({placeholders}))"
            )
            parameters.extend(normalized_any)

        sql = f"SELECT DISTINCT p.photo_id FROM photos p{joins}"
        if where:
            sql += " WHERE " + " AND ".join(f"({clause})" for clause in where)
        sql += " ORDER BY p.captured_at_sort IS NULL, p.captured_at_sort, p.relative_path COLLATE NOCASE"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must not be negative")
            sql += " LIMIT ?"
            parameters.append(int(limit))

        with self.transaction(write=False) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [row["photo_id"] for row in rows]

    def upsert_event(
        self,
        *,
        event_id: str | None = None,
        title: str = "",
        summary: str = "",
        start_at: str | datetime | None = None,
        start_at_sort: int | None = None,
        end_at: str | datetime | None = None,
        end_at_sort: int | None = None,
        date_local: str | None = None,
        method: str,
        algorithm_version: str = "",
        parameters: Mapping[str, Any] | None = None,
    ) -> str:
        event_id = str(uuid.UUID(event_id)) if event_id else str(uuid.uuid4())
        start_text, start_sort, inferred_date = self._normalize_timestamp(start_at, start_at_sort, date_local)
        end_text, end_sort, _ = self._normalize_timestamp(end_at, end_at_sort, None)
        now = self._utc_now()
        parameters_json = json.dumps(parameters or {}, ensure_ascii=False, sort_keys=True)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO events(
                    event_id, title, summary, start_at, start_at_sort,
                    end_at, end_at_sort, date_local, method,
                    algorithm_version, parameters_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    title = excluded.title,
                    summary = excluded.summary,
                    start_at = excluded.start_at,
                    start_at_sort = excluded.start_at_sort,
                    end_at = excluded.end_at,
                    end_at_sort = excluded.end_at_sort,
                    date_local = excluded.date_local,
                    method = excluded.method,
                    algorithm_version = excluded.algorithm_version,
                    parameters_json = excluded.parameters_json,
                    updated_at = excluded.updated_at
                """,
                (
                    event_id, title, summary, start_text, start_sort,
                    end_text, end_sort, date_local or inferred_date, method,
                    algorithm_version, parameters_json, now, now,
                ),
            )
        return event_id

    def replace_event_photos(
        self,
        event_id: str,
        photo_ids: Sequence[str],
        *,
        roles: Sequence[str] | None = None,
        similarities_to_previous: Sequence[float | None] | None = None,
    ) -> None:
        if len(set(photo_ids)) != len(photo_ids):
            raise ValueError("photo_ids must not contain duplicates")
        roles = roles or ["member"] * len(photo_ids)
        similarities_to_previous = similarities_to_previous or [None] * len(photo_ids)
        if len(roles) != len(photo_ids) or len(similarities_to_previous) != len(photo_ids):
            raise ValueError("roles and similarities must align with photo_ids")
        with self.transaction() as connection:
            connection.execute("DELETE FROM event_photos WHERE event_id = ?", (event_id,))
            connection.executemany(
                """
                INSERT INTO event_photos(event_id, photo_id, position, role, similarity_to_previous)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (event_id, photo_id, position, roles[position], similarities_to_previous[position])
                    for position, photo_id in enumerate(photo_ids)
                ],
            )

    def event_partitions_for_photos(self, photo_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Return saved-event partitions for a selected set of photos.

        This is a read-only Story helper. It preserves event order and photo
        position, and returns unassigned photos as singleton partitions. The
        caller can therefore use stored Events as the first grouping level
        without changing the database schema.
        """
        unique_ids = list(dict.fromkeys(str(photo_id) for photo_id in photo_ids))
        if not unique_ids:
            return []
        placeholders = ", ".join("?" for _ in unique_ids)
        with self.transaction(write=False) as connection:
            rows = connection.execute(
                f"""
                SELECT e.event_id, e.title, e.start_at_sort, ep.photo_id, ep.position
                FROM events e
                JOIN event_photos ep ON ep.event_id = e.event_id
                WHERE ep.photo_id IN ({placeholders})
                ORDER BY e.start_at_sort IS NULL, e.start_at_sort, e.event_id, ep.position
                """,
                unique_ids,
            ).fetchall()

        selected = set(unique_ids)
        assigned: set[str] = set()
        partitions: list[dict[str, Any]] = []
        by_event: dict[str, dict[str, Any]] = {}
        for row in rows:
            photo_id = str(row["photo_id"])
            if photo_id not in selected or photo_id in assigned:
                continue
            event_id = str(row["event_id"])
            partition = by_event.get(event_id)
            if partition is None:
                partition = {
                    "event_id": event_id,
                    "title": str(row["title"] or event_id),
                    "photo_ids": [],
                }
                by_event[event_id] = partition
                partitions.append(partition)
            partition["photo_ids"].append(photo_id)
            assigned.add(photo_id)

        for photo_id in unique_ids:
            if photo_id not in assigned:
                partitions.append({"event_id": None, "title": "Unassigned photo", "photo_ids": [photo_id]})
        return partitions

    def event_photo_ids(self, event_id: str) -> list[str]:
        """Return one persisted event's photos in its frozen position order."""

        with self.transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT photo_id
                FROM event_photos
                WHERE event_id = ?
                ORDER BY position
                """,
                (event_id,),
            ).fetchall()
        return [str(row["photo_id"]) for row in rows]

    def add_story(
        self,
        *,
        content: str,
        model_name: str,
        event_id: str | None = None,
        story_id: str | None = None,
        title: str = "",
        style: str = "grounded",
        language: str = "zh",
        model_version: str = "",
        prompt_version: str = "",
        grounding: Mapping[str, Any] | None = None,
    ) -> str:
        story_id = str(uuid.UUID(story_id)) if story_id else str(uuid.uuid4())
        if not content.strip():
            raise ValueError("story content must not be empty")
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO stories(
                    story_id, event_id, title, content, style, language,
                    model_name, model_version, prompt_version, grounding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    story_id, event_id, title, content, style, language,
                    model_name, model_version, prompt_version,
                    json.dumps(grounding or {}, ensure_ascii=False, sort_keys=True), self._utc_now(),
                ),
            )
        return story_id

    def start_index_run(
        self,
        *,
        pipeline_version: str = "",
        config: Mapping[str, Any] | None = None,
        discovered_count: int = 0,
        run_id: str | None = None,
    ) -> str:
        run_id = str(uuid.UUID(run_id)) if run_id else str(uuid.uuid4())
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO index_runs(
                    run_id, status, pipeline_version, config_json, started_at, discovered_count
                ) VALUES (?, 'running', ?, ?, ?, ?)
                """,
                (
                    run_id, pipeline_version,
                    json.dumps(config or {}, ensure_ascii=False, sort_keys=True),
                    self._utc_now(), int(discovered_count),
                ),
            )
        return run_id

    def finish_index_run(
        self,
        run_id: str,
        *,
        status: str,
        indexed_count: int,
        skipped_count: int = 0,
        failed_count: int = 0,
        notes: str = "",
    ) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE index_runs SET
                    status = ?, completed_at = ?, indexed_count = ?,
                    skipped_count = ?, failed_count = ?, notes = ?
                WHERE run_id = ?
                """,
                (
                    status, self._utc_now(), int(indexed_count),
                    int(skipped_count), int(failed_count), notes, run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown index run: {run_id}")

    def record_index_failure(
        self,
        *,
        stage: str,
        error: BaseException | str,
        run_id: str | None = None,
        photo_id: str | None = None,
        relative_path: str | Path | None = None,
        retryable: bool = False,
    ) -> int:
        error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
        message = str(error)
        safe_path = self._normalize_relative_path(relative_path) if relative_path is not None else None
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO index_failures(
                    run_id, photo_id, relative_path, stage, error_type,
                    message, retryable, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, photo_id, safe_path, stage, error_type,
                    message, int(retryable), self._utc_now(),
                ),
            )
        return int(cursor.lastrowid)


__all__ = [
    "BM25Result",
    "MOOD_LABELS",
    "PhotoMood",
    "PhotoStorage",
    "SCHEMA_VERSION",
    "SceneGraphTriple",
    "StorageCapabilityError",
    "StorageConflictError",
    "StorageError",
]
