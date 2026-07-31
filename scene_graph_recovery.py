"""Explicit, audited recovery of a confirmed legacy personal Scene Graph batch.

The normal importer intentionally rejects legacy personal-photo JSONL because it
does not contain the generation-time manifest SHA-256.  This module does not
weaken that rule.  It provides a separate recovery path that must be explicitly
confirmed by the user and records the weaker provenance honestly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from scene_graph_io import (
    INDEXABLE_SOURCES,
    RESULT_SCHEMA_VERSION,
    photo_id_for_sha256,
    sha256_file,
)
from storage import PhotoStorage


RECOVERY_PROVENANCE = "legacy_path_time_recovered"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROMPT_VERSION = "legacy-snapseek-scene-graph-unversioned-20260717"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"})
_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_BLOCKING_CATEGORIES = frozenset(
    {
        "malformed",
        "duplicate",
        "unsafe_path",
        "local_missing",
        "storage_missing",
        "checksum_conflict",
        "modified_after_generation",
        "unsupported_source",
        "album_uncovered",
        "extra_result_path",
    }
)


@dataclass(frozen=True)
class LegacyRecoveryIssue:
    category: str
    line_number: int
    message: str
    image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LegacyPersonalRecoveryReport:
    dataset_id: str
    source_file: str
    source_file_sha256: str
    album_root: str
    database: str
    recovered_at: str
    batch_relative_root: str = ""
    provenance: str = RECOVERY_PROVENANCE
    generation_sha256_verified: bool = False
    user_confirmed_current_album_source: bool = True
    counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    album_snapshot_sha256: str = ""
    indexable_records: list[dict[str, Any]] = field(default_factory=list)
    issues: list[LegacyRecoveryIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> int:
        return sum(self.counts.get(category, 0) for category in _BLOCKING_CATEGORIES)

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        payload = {
            "dataset_id": self.dataset_id,
            "source_file": self.source_file,
            "source_file_sha256": self.source_file_sha256,
            "album_root": self.album_root,
            "batch_relative_root": self.batch_relative_root,
            "database": self.database,
            "recovered_at": self.recovered_at,
            "provenance": self.provenance,
            "generation_sha256_verified": self.generation_sha256_verified,
            "user_confirmed_current_album_source": self.user_confirmed_current_album_source,
            "counts": dict(sorted(self.counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "album_snapshot_sha256": self.album_snapshot_sha256,
            "blocking_issues": self.blocking_issues,
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if include_records:
            payload["indexable_records"] = self.indexable_records
        return payload


def _normalise_relative_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _resolve_inside(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes album root: {relative_path!r}") from exc
    return candidate


def _parse_generated_at(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("generated_at is missing")
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalise_triples(value: Any, limit: int = 15) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or item.get("relation") or "").strip()
        obj = str(item.get("object") or item.get("obj") or "").strip()
        if not subject or not predicate or not obj:
            continue
        if max(len(subject), len(predicate), len(obj)) > 240:
            continue
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append({"subject": subject, "predicate": predicate, "object": obj})
        if len(triples) >= limit:
            break
    return triples


def _album_files(root: Path, subdir: str | None = None) -> dict[str, Path]:
    """Return ``{relative_to_root: path}`` for image files under ``root``.

    When ``subdir`` is given the scan is restricted to ``root / subdir`` so an
    incremental batch can be recovered without the rest of the album counting as
    uncovered.  Keys remain relative to ``root`` (e.g. ``pic2/DSC00486.JPG``) so
    path matching against SQLite (which stores paths relative to the photo root)
    is unchanged.
    """
    scan_root = root
    if subdir:
        scan_root = (root / Path(*PurePosixPath(subdir).parts)).resolve()
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(scan_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _snapshot_sha256(paths: dict[str, Path], digest_cache: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(paths, key=str.casefold):
        file_digest = digest_cache.get(relative_path)
        if file_digest is None:
            file_digest = sha256_file(paths[relative_path])
            digest_cache[relative_path] = file_digest
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def recover_confirmed_legacy_personal_results(
    source_jsonl: str | Path,
    album_root: str | Path,
    storage: PhotoStorage,
    *,
    dataset_id: str,
    confirmed_generated_from_current_album: bool,
    model_id: str = DEFAULT_MODEL_ID,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    batch_relative_root: str | None = None,
) -> LegacyPersonalRecoveryReport:
    """Build indexable derivative records without claiming generation-time hashes.

    Exact relative path, current file SHA-256, the SQLite SHA-256 and the fact
    that the local file was not modified after ``generated_at`` must all agree.
    The returned records remain explicitly marked as recovered legacy evidence.
    """

    if not confirmed_generated_from_current_album:
        raise ValueError("explicit confirmation that this batch came from the current album is required")
    dataset_id = str(dataset_id).strip()
    if not _DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError("dataset_id must contain 1-64 letters, digits, dots, underscores or hyphens")
    source_path = Path(source_jsonl).expanduser().resolve()
    root = Path(album_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not root.is_dir():
        raise NotADirectoryError(root)

    scan_subdir: str | None = None
    if batch_relative_root is not None:
        candidate = str(batch_relative_root).strip().replace("\\", "/").strip("/")
        if candidate:
            sub = PurePosixPath(candidate)
            if sub.is_absolute() or any(part in {"", ".", ".."} for part in sub.parts):
                raise ValueError(f"unsafe batch_relative_root: {batch_relative_root!r}")
            scan_dir = _resolve_inside(root, candidate)
            if not scan_dir.is_dir():
                raise NotADirectoryError(scan_dir)
            scan_subdir = candidate

    source_digest = sha256_file(source_path)
    recovered_at = datetime.now(timezone.utc).isoformat()
    counts: Counter[str] = Counter(
        {
            "total_rows": 0,
            "recovered": 0,
            "failed": 0,
            "empty": 0,
            "malformed": 0,
            "duplicate": 0,
            "unsafe_path": 0,
            "local_missing": 0,
            "storage_missing": 0,
            "checksum_conflict": 0,
            "modified_after_generation": 0,
            "unsupported_source": 0,
            "album_image_files": 0,
            "covered_album_paths": 0,
            "album_uncovered": 0,
            "extra_result_path": 0,
        }
    )
    source_counts: Counter[str] = Counter()
    issues: list[LegacyRecoveryIssue] = []
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    result_paths: set[str] = set()
    digest_cache: dict[str, str] = {}
    album_files = _album_files(root, scan_subdir)
    album_paths_casefold = {path.casefold(): path for path in album_files}
    counts["album_image_files"] = len(album_files)

    def issue(category: str, line_number: int, message: str, image_path: str = "") -> None:
        counts[category] += 1
        if len(issues) < 1000:
            issues.append(LegacyRecoveryIssue(category, line_number, message, image_path))

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            counts["total_rows"] += 1
            try:
                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise ValueError("row must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                issue("malformed", line_number, f"{type(exc).__name__}: {exc}")
                continue

            try:
                relative_path = _normalise_relative_path(row.get("image_path"))
            except ValueError as exc:
                issue("unsafe_path", line_number, str(exc))
                continue
            path_key = relative_path.casefold()
            if path_key in seen_paths:
                issue("duplicate", line_number, "duplicate image_path", relative_path)
                continue
            seen_paths.add(path_key)
            result_paths.add(path_key)

            canonical_relative = album_paths_casefold.get(path_key, relative_path)
            try:
                current_path = _resolve_inside(root, canonical_relative)
            except ValueError as exc:
                issue("unsafe_path", line_number, str(exc), relative_path)
                continue
            if not current_path.is_file():
                issue("local_missing", line_number, "current album file is missing", relative_path)
                continue

            current_digest = sha256_file(current_path)
            digest_cache[canonical_relative] = current_digest
            photo = storage.get_photo_by_path(canonical_relative)
            if photo is None:
                issue("storage_missing", line_number, "no SQLite photo at this exact relative path", relative_path)
                continue
            if str(photo.get("content_sha256") or "").casefold() != current_digest.casefold():
                issue("checksum_conflict", line_number, "SQLite SHA-256 differs from current file", relative_path)
                continue

            try:
                generated_at = _parse_generated_at(row.get("generated_at"))
            except (TypeError, ValueError) as exc:
                issue("malformed", line_number, f"invalid generated_at: {exc}", relative_path)
                continue
            if current_path.stat().st_mtime > generated_at.timestamp() + 1.0:
                issue(
                    "modified_after_generation",
                    line_number,
                    "current file mtime is later than the Qwen result",
                    relative_path,
                )
                continue

            source = str(row.get("scene_graph_source") or row.get("source") or "").strip()
            source_counts[source or "unknown"] += 1
            triples = _normalise_triples(row.get("scene_graph"), limit=15)
            if source == "failed":
                counts["failed"] += 1
                issues.append(
                    LegacyRecoveryIssue(
                        "failed",
                        line_number,
                        str(row.get("remote_error") or "Qwen generation failed"),
                        relative_path,
                    )
                )
                continue
            if source not in INDEXABLE_SOURCES:
                issue("unsupported_source", line_number, f"unsupported source: {source!r}", relative_path)
                continue
            if not triples:
                counts["empty"] += 1
                issues.append(LegacyRecoveryIssue("empty", line_number, "scene graph is empty", relative_path))
                continue

            line_digest = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            recovered_photo_id = photo_id_for_sha256(current_digest)
            flat_text = " ; ".join(
                f"{triple['subject']} {triple['predicate']} {triple['object']}" for triple in triples
            )
            records.append(
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "dataset_id": dataset_id,
                    "photo_id": recovered_photo_id,
                    "relative_path": canonical_relative,
                    "sha256": current_digest,
                    "sha256_origin": "current_local_file_at_recovery",
                    "photo_id_origin": "reconstructed_from_current_sha256",
                    "local_sqlite_photo_id": str(photo["photo_id"]),
                    "model_id": model_id,
                    "prompt_version": prompt_version,
                    "max_triples": 15,
                    "source": source,
                    "status": "success",
                    "scene_graph": triples,
                    "scene_graph_flat_text": flat_text,
                    "generated_at": generated_at.isoformat(),
                    "recovered_at": recovered_at,
                    "provenance": RECOVERY_PROVENANCE,
                    "generation_sha256_verified": False,
                    "user_confirmed_current_album_source": True,
                    "identity_checks": [
                        "exact_relative_path",
                        "current_file_sha256_matches_sqlite",
                        "current_file_mtime_not_after_generated_at",
                    ],
                    "result_file_sha256": source_digest,
                    "legacy_line_sha256": line_digest,
                    "legacy_line_number": line_number,
                    "source_ref": f"legacy-recovered:{source_digest}:{line_number}",
                    "output_sha256": source_digest,
                }
            )
            counts["recovered"] += 1

    album_path_keys = set(album_paths_casefold)
    counts["covered_album_paths"] = len(result_paths & album_path_keys)
    counts["album_uncovered"] = len(album_path_keys - result_paths)
    counts["extra_result_path"] = len(result_paths - album_path_keys)
    if counts["album_uncovered"]:
        issues.append(
            LegacyRecoveryIssue(
                "album_uncovered",
                0,
                f"{counts['album_uncovered']} current album images have no result row",
            )
        )
    if counts["extra_result_path"]:
        issues.append(
            LegacyRecoveryIssue(
                "extra_result_path",
                0,
                f"{counts['extra_result_path']} result paths are outside the current album image set",
            )
        )

    return LegacyPersonalRecoveryReport(
        dataset_id=dataset_id,
        source_file=str(source_path),
        source_file_sha256=source_digest,
        album_root=str(root),
        batch_relative_root=scan_subdir or "",
        database=str(storage.database_path),
        recovered_at=recovered_at,
        counts=dict(counts),
        source_counts=dict(source_counts),
        album_snapshot_sha256=_snapshot_sha256(album_files, digest_cache),
        indexable_records=records,
        issues=issues,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_recovered_scene_graphs(
    report: LegacyPersonalRecoveryReport,
    output_jsonl: str | Path,
    report_json: str | Path,
) -> None:
    if report.blocking_issues:
        raise ValueError(f"recovery has {report.blocking_issues} blocking issue(s); refusing to write indexable JSONL")
    jsonl_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in report.indexable_records
    )
    _atomic_write_text(Path(output_jsonl), jsonl_text)
    _atomic_write_text(
        Path(report_json),
        json.dumps(report.to_dict(include_records=False), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_PROMPT_VERSION",
    "LegacyPersonalRecoveryReport",
    "RECOVERY_PROVENANCE",
    "recover_confirmed_legacy_personal_results",
    "write_recovered_scene_graphs",
]
