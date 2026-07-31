"""Safe Scene Graph export/import contracts for personal photo collections.

The module deliberately does not match records by basename or filename stem.  A
modern result is accepted only when its dataset id, stable photo id and SHA-256
agree with the manifest *and* the same bytes still exist in the local album.
Legacy SnapSeek JSONL can be audited only with an explicit source root; legacy
records are indexable only for a dataset whose id starts with ``geograph``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence


MANIFEST_SCHEMA_VERSION = "spn-scene-graph-manifest-v1"
RESULT_SCHEMA_VERSION = "spn-scene-graph-result-v1"
INDEXABLE_SOURCES = frozenset({"remote", "remote_repaired"})
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PHOTO_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PHOTO_NAMESPACE = uuid.UUID("509ca89c-7959-44a5-8f7b-405784d733f6")
_ISSUE_LIMIT = 1000


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a file without loading the whole image in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def photo_id_for_sha256(sha256: str) -> str:
    """Create a rename-stable, non-filename photo id from a verified hash."""

    sha256 = str(sha256).lower()
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return uuid.uuid5(_PHOTO_NAMESPACE, sha256).hex


def _validate_dataset_id(dataset_id: str) -> str:
    dataset_id = str(dataset_id).strip()
    if not _DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError(
            "dataset_id must contain 1-64 letters, digits, dots, underscores or hyphens"
        )
    return dataset_id


def _normalise_relative_path(value: str) -> str:
    raw = str(value).replace("\\", "/").strip()
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
    root = root.expanduser().resolve()
    relative = _normalise_relative_path(relative_path)
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes its declared root: {relative_path!r}") from exc
    return candidate


def _is_jpeg(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(3) == b"\xff\xd8\xff"
    except OSError:
        return False


@dataclass(frozen=True)
class SceneGraphManifestRecord:
    schema_version: str
    dataset_id: str
    photo_id: str
    relative_path: str
    sha256: str
    file_size: int

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        relative_path: str,
        sha256: str,
        file_size: int,
    ) -> "SceneGraphManifestRecord":
        sha256 = str(sha256).lower()
        return cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            dataset_id=_validate_dataset_id(dataset_id),
            photo_id=photo_id_for_sha256(sha256),
            relative_path=_normalise_relative_path(relative_path),
            sha256=sha256,
            file_size=int(file_size),
        ).validated()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SceneGraphManifestRecord":
        required = {
            "schema_version",
            "dataset_id",
            "photo_id",
            "relative_path",
            "sha256",
            "file_size",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"manifest record missing fields: {', '.join(missing)}")
        return cls(
            schema_version=str(payload["schema_version"]),
            dataset_id=str(payload["dataset_id"]),
            photo_id=str(payload["photo_id"]).lower(),
            relative_path=str(payload["relative_path"]),
            sha256=str(payload["sha256"]).lower(),
            file_size=int(payload["file_size"]),
        ).validated()

    def validated(self) -> "SceneGraphManifestRecord":
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema_version: {self.schema_version!r}")
        _validate_dataset_id(self.dataset_id)
        _normalise_relative_path(self.relative_path)
        if not _PHOTO_ID_RE.fullmatch(self.photo_id):
            raise ValueError(f"invalid photo_id: {self.photo_id!r}")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError(f"invalid sha256 for photo_id={self.photo_id}")
        if self.photo_id != photo_id_for_sha256(self.sha256):
            raise ValueError(f"photo_id is not derived from sha256 for {self.relative_path}")
        if self.file_size <= 0:
            raise ValueError(f"file_size must be positive for photo_id={self.photo_id}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneGraphAuditIssue:
    category: str
    source_file: str
    line_number: int
    message: str
    photo_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneGraphExportReport:
    dataset_id: str
    manifest_path: str
    image_dir: str
    exported: int = 0
    duplicate_content: int = 0
    reused_cloud_images: int = 0
    duplicate_paths: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_counts() -> dict[str, int]:
    return {
        "total_rows": 0,
        "matched": 0,
        "unmatched": 0,
        "checksum_conflict": 0,
        "failed": 0,
        "empty": 0,
        "duplicate": 0,
        "malformed": 0,
        "legacy_blocked": 0,
        "local_missing": 0,
        "indexable": 0,
    }


@dataclass
class SceneGraphImportReport:
    dataset_id: str
    dry_run: bool = True
    counts: dict[str, int] = field(default_factory=_empty_counts)
    indexable_records: list[dict[str, Any]] = field(default_factory=list)
    issues: list[SceneGraphAuditIssue] = field(default_factory=list)
    issues_truncated: int = 0

    def add_issue(
        self,
        category: str,
        source_file: Path,
        line_number: int,
        message: str,
        photo_id: str = "",
    ) -> None:
        if len(self.issues) < _ISSUE_LIMIT:
            self.issues.append(
                SceneGraphAuditIssue(
                    category=category,
                    source_file=source_file.name,
                    line_number=line_number,
                    message=message,
                    photo_id=photo_id,
                )
            )
        else:
            self.issues_truncated += 1

    def to_dict(
        self,
        *,
        include_records: bool = False,
        include_issues: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "dry_run": self.dry_run,
            "counts": dict(self.counts),
            "issues_truncated": self.issues_truncated,
        }
        if include_issues:
            payload["issues"] = [issue.to_dict() for issue in self.issues]
        if include_records:
            payload["indexable_records"] = list(self.indexable_records)
        return payload


def _iter_album_jpegs(root: Path, excluded_root: Path | None = None) -> Iterator[Path]:
    root = root.expanduser().resolve()
    excluded = excluded_root.expanduser().resolve() if excluded_root else None
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.lower() not in JPEG_SUFFIXES:
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if excluded is not None:
            try:
                resolved.relative_to(excluded)
            except ValueError:
                pass
            else:
                continue
        yield resolved


def _atomic_write_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)


def export_scene_graph_batch(
    album_root: str | Path,
    export_dir: str | Path,
    dataset_id: str,
    *,
    manifest_name: str = "manifest.jsonl",
) -> SceneGraphExportReport:
    """Export byte-identical JPEGs under stable ``photo_id.jpg`` cloud names.

    Non-JPEG files are intentionally not transcoded: transcoding would make the
    cloud bytes differ from the local bytes and defeat end-to-end hash checking.
    The project's HEIC/PNG normalisation step should run before this exporter.
    """

    album_root = Path(album_root).expanduser().resolve()
    export_dir = Path(export_dir).expanduser().resolve()
    dataset_id = _validate_dataset_id(dataset_id)
    if not album_root.is_dir():
        raise ValueError(f"album root does not exist or is not a directory: {album_root}")
    manifest_name = Path(_normalise_relative_path(manifest_name)).name
    manifest_path = export_dir / manifest_name
    image_dir = export_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    report = SceneGraphExportReport(
        dataset_id=dataset_id,
        manifest_path=str(manifest_path),
        image_dir=str(image_dir),
    )

    records_by_sha: dict[str, SceneGraphManifestRecord] = {}
    photo_id_to_sha: dict[str, str] = {}
    for image_path in _iter_album_jpegs(album_root, export_dir):
        if not _is_jpeg(image_path):
            raise ValueError(
                f"file has a JPEG extension but not JPEG bytes: "
                f"{image_path.relative_to(album_root).as_posix()}"
            )
        relative = image_path.relative_to(album_root).as_posix()
        digest = sha256_file(image_path)
        if digest in records_by_sha:
            report.duplicate_content += 1
            report.duplicate_paths.append(
                {
                    "canonical": records_by_sha[digest].relative_path,
                    "duplicate": relative,
                    "sha256": digest,
                }
            )
            continue
        record = SceneGraphManifestRecord.create(
            dataset_id=dataset_id,
            relative_path=relative,
            sha256=digest,
            file_size=image_path.stat().st_size,
        )
        previous_sha = photo_id_to_sha.get(record.photo_id)
        if previous_sha is not None and previous_sha != digest:
            raise RuntimeError("stable photo_id collision detected; export aborted")
        photo_id_to_sha[record.photo_id] = digest

        cloud_image = image_dir / f"{record.photo_id}.jpg"
        if cloud_image.exists():
            if cloud_image.stat().st_size != record.file_size or sha256_file(cloud_image) != digest:
                raise ValueError(f"existing cloud image has wrong bytes: {cloud_image.name}")
            report.reused_cloud_images += 1
        else:
            shutil.copy2(image_path, cloud_image)
            if cloud_image.stat().st_size != record.file_size or sha256_file(cloud_image) != digest:
                raise IOError(f"cloud image verification failed after copying: {cloud_image.name}")
        records_by_sha[digest] = record

    records = sorted(records_by_sha.values(), key=lambda item: item.photo_id)
    _atomic_write_jsonl((record.to_dict() for record in records), manifest_path)
    report.exported = len(records)
    return report


def load_scene_graph_manifest(
    manifest_path: str | Path,
) -> list[SceneGraphManifestRecord]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"manifest does not exist: {path}")
    records: list[SceneGraphManifestRecord] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    dataset_id = ""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            try:
                record = SceneGraphManifestRecord.from_dict(payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
            if dataset_id and record.dataset_id != dataset_id:
                raise ValueError("one manifest cannot contain multiple dataset_id values")
            dataset_id = record.dataset_id
            if record.photo_id in seen_ids:
                raise ValueError(f"duplicate photo_id in manifest: {record.photo_id}")
            if record.sha256 in seen_hashes:
                raise ValueError(f"duplicate sha256 in manifest: {record.sha256}")
            seen_ids.add(record.photo_id)
            seen_hashes.add(record.sha256)
            records.append(record)
    if not records:
        raise ValueError("manifest contains no records")
    return records


def _normalise_triples(value: Any, *, limit: int | None = None) -> list[dict[str, str]]:
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
        if limit is not None and len(triples) >= limit:
            break
    return triples


def _flat_text(triples: Sequence[dict[str, str]]) -> str:
    return " ".join(
        f"{triple['subject']} {triple['predicate']} {triple['object']}"
        for triple in triples
    )


def _build_album_catalog(album_root: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    by_sha: dict[str, list[str]] = {}
    sha_by_relative: dict[str, str] = {}
    for path in _iter_album_jpegs(album_root):
        relative = path.relative_to(album_root).as_posix()
        digest = sha256_file(path)
        by_sha.setdefault(digest, []).append(relative)
        sha_by_relative[relative] = digest
    for paths in by_sha.values():
        paths.sort(key=str.casefold)
    return by_sha, sha_by_relative


def _current_relative_path(
    record: SceneGraphManifestRecord,
    album_by_sha: dict[str, list[str]],
) -> str | None:
    candidates = album_by_sha.get(record.sha256, [])
    if record.relative_path in candidates:
        return record.relative_path
    return candidates[0] if candidates else None


def _iter_result_lines(paths: Sequence[Path]) -> Iterator[tuple[Path, int, dict[str, Any] | None, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield path, line_number, None, f"invalid JSON: {exc.msg}"
                    continue
                if not isinstance(payload, dict):
                    yield path, line_number, None, "result line must be a JSON object"
                    continue
                yield path, line_number, payload, ""


def _result_triples(payload: dict[str, Any]) -> list[dict[str, str]]:
    value = payload.get("scene_graph")
    if value is None:
        value = payload.get("scene graph")
    if value is None:
        value = payload.get("triples")
    return _normalise_triples(value)


def _safe_legacy_path(legacy_root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("legacy row has no image_path")
    return _resolve_inside(legacy_root, raw_path)


def _make_indexable_record(
    *,
    manifest: SceneGraphManifestRecord,
    current_relative_path: str,
    triples: list[dict[str, str]],
    source: str,
    model_id: str,
    prompt_version: str,
    max_triples: int,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset_id": manifest.dataset_id,
        "photo_id": manifest.photo_id,
        "relative_path": current_relative_path,
        "manifest_relative_path": manifest.relative_path,
        "sha256": manifest.sha256,
        "scene_graph": triples,
        "scene_graph_flat_text": _flat_text(triples),
        "source": source,
        "status": "success",
        "model_id": model_id,
        "prompt_version": prompt_version,
        "max_triples": max_triples,
        "generated_at": generated_at,
    }


def audit_scene_graph_results(
    manifest_path: str | Path,
    result_paths: Sequence[str | Path],
    album_root: str | Path,
    *,
    legacy_root: str | Path | None = None,
    dry_run: bool = True,
) -> SceneGraphImportReport:
    """Audit result JSONL and return only records safe for indexing.

    Legacy records (without modern identity fields) require ``legacy_root``.
    They are resolved by their exact relative path under that root, hashed on
    site, and matched to the manifest by full hash.  Basenames are used only to
    count dangerous collisions; they are never used to accept a record.
    """

    manifest_records = load_scene_graph_manifest(manifest_path)
    dataset_id = manifest_records[0].dataset_id
    report = SceneGraphImportReport(dataset_id=dataset_id, dry_run=dry_run)
    album_root_path = Path(album_root).expanduser().resolve()
    if not album_root_path.is_dir():
        raise ValueError(f"album root does not exist or is not a directory: {album_root_path}")
    legacy_root_path = Path(legacy_root).expanduser().resolve() if legacy_root else None
    if legacy_root_path is not None and not legacy_root_path.is_dir():
        raise ValueError(f"legacy root does not exist or is not a directory: {legacy_root_path}")

    paths = [Path(path).expanduser().resolve() for path in result_paths]
    if not paths:
        raise ValueError("at least one result JSONL path is required")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"result file does not exist: {missing[0]}")

    manifest_by_id = {record.photo_id: record for record in manifest_records}
    manifest_by_sha = {record.sha256: record for record in manifest_records}
    manifest_basenames = {
        PurePosixPath(record.relative_path).name.casefold() for record in manifest_records
    }
    album_by_sha, album_sha_by_relative = _build_album_catalog(album_root_path)
    accepted: dict[str, dict[str, Any]] = {}
    seen_logical_ids: set[str] = set()

    for source_file, line_number, payload, parse_error in _iter_result_lines(paths):
        report.counts["total_rows"] += 1
        if payload is None:
            report.counts["malformed"] += 1
            report.add_issue("malformed", source_file, line_number, parse_error)
            continue

        is_modern = any(
            key in payload for key in ("photo_id", "sha256", "dataset_id", "schema_version")
        )
        if is_modern:
            raw_photo_id = str(payload.get("photo_id") or "").lower()
            logical_id = raw_photo_id if _PHOTO_ID_RE.fullmatch(raw_photo_id) else ""
            if logical_id:
                if logical_id in seen_logical_ids:
                    report.counts["duplicate"] += 1
                    report.add_issue(
                        "duplicate",
                        source_file,
                        line_number,
                        "another result row already used this photo_id; latest valid row wins",
                        logical_id,
                    )
                seen_logical_ids.add(logical_id)

            source = str(payload.get("source") or "").strip()
            status = str(payload.get("status") or "").strip().lower()
            if source == "failed" or status == "failed":
                report.counts["failed"] += 1
                report.add_issue("failed", source_file, line_number, "remote generation failed", logical_id)
                continue
            triples = _result_triples(payload)
            if source in INDEXABLE_SOURCES and status in {"success", "empty"} and not triples:
                report.counts["empty"] += 1
                report.add_issue("empty", source_file, line_number, "remote result has no valid triples", logical_id)
                continue

            required_values = {
                "schema_version": payload.get("schema_version"),
                "dataset_id": payload.get("dataset_id"),
                "photo_id": payload.get("photo_id"),
                "sha256": payload.get("sha256"),
                "model_id": payload.get("model_id"),
                "prompt_version": payload.get("prompt_version"),
                "max_triples": payload.get("max_triples"),
                "source": payload.get("source"),
                "status": payload.get("status"),
            }
            missing_fields = [
                name
                for name, value in required_values.items()
                if value is None or value == ""
            ]
            if missing_fields:
                report.counts["malformed"] += 1
                report.add_issue(
                    "malformed",
                    source_file,
                    line_number,
                    f"modern result missing fields: {', '.join(missing_fields)}",
                    logical_id,
                )
                continue
            if str(payload.get("schema_version")) != RESULT_SCHEMA_VERSION:
                report.counts["malformed"] += 1
                report.add_issue("malformed", source_file, line_number, "unsupported result schema_version", logical_id)
                continue
            if str(payload.get("dataset_id")) != dataset_id:
                report.counts["unmatched"] += 1
                report.add_issue("unmatched", source_file, line_number, "dataset_id does not match manifest", logical_id)
                continue
            manifest = manifest_by_id.get(logical_id)
            if manifest is None:
                report.counts["unmatched"] += 1
                report.add_issue("unmatched", source_file, line_number, "photo_id is not in manifest", logical_id)
                continue
            result_sha = str(payload.get("sha256") or "").lower()
            if result_sha != manifest.sha256:
                report.counts["checksum_conflict"] += 1
                report.add_issue(
                    "checksum_conflict",
                    source_file,
                    line_number,
                    "result sha256 disagrees with manifest",
                    logical_id,
                )
                continue
            if source not in INDEXABLE_SOURCES or status != "success":
                report.counts["failed"] += 1
                report.add_issue(
                    "failed",
                    source_file,
                    line_number,
                    "only successful remote/remote_repaired results are indexable",
                    logical_id,
                )
                continue
            try:
                max_triples = int(payload.get("max_triples"))
            except (TypeError, ValueError):
                max_triples = 0
            if max_triples < 1 or max_triples > 15 or len(triples) > max_triples:
                report.counts["malformed"] += 1
                report.add_issue(
                    "malformed",
                    source_file,
                    line_number,
                    "max_triples must be 1-15 and cannot be smaller than the result",
                    logical_id,
                )
                continue
            current_relative = _current_relative_path(manifest, album_by_sha)
            if current_relative is None:
                actual_sha = album_sha_by_relative.get(manifest.relative_path)
                if actual_sha is not None and actual_sha != manifest.sha256:
                    report.counts["checksum_conflict"] += 1
                    category = "checksum_conflict"
                    message = "local manifest path now contains different bytes"
                else:
                    report.counts["unmatched"] += 1
                    report.counts["local_missing"] += 1
                    category = "unmatched"
                    message = "manifest bytes no longer exist in the local album"
                report.add_issue(category, source_file, line_number, message, logical_id)
                continue
            accepted[logical_id] = _make_indexable_record(
                manifest=manifest,
                current_relative_path=current_relative,
                triples=triples,
                source=source,
                model_id=str(payload.get("model_id")),
                prompt_version=str(payload.get("prompt_version")),
                max_triples=max_triples,
                generated_at=str(payload.get("generated_at") or ""),
            )
            continue

        # Legacy records are never trusted by filename.  The exact legacy file
        # is resolved under an explicitly supplied root and hashed on site.
        legacy_source = str(payload.get("scene_graph_source") or payload.get("source") or "").strip()
        legacy_triples = _result_triples(payload)
        if legacy_source == "failed":
            report.counts["failed"] += 1
            report.add_issue("failed", source_file, line_number, "legacy remote generation failed")
            continue
        if legacy_source in INDEXABLE_SOURCES and not legacy_triples:
            report.counts["empty"] += 1
            report.add_issue("empty", source_file, line_number, "legacy result has no valid triples")
            continue
        if legacy_root_path is None:
            report.counts["unmatched"] += 1
            report.add_issue(
                "unmatched",
                source_file,
                line_number,
                "legacy row requires an explicit legacy_root for on-site hashing",
            )
            continue
        try:
            legacy_image = _safe_legacy_path(legacy_root_path, payload.get("image_path"))
        except ValueError as exc:
            report.counts["malformed"] += 1
            report.add_issue("malformed", source_file, line_number, str(exc))
            continue
        if not legacy_image.is_file():
            report.counts["unmatched"] += 1
            report.add_issue("unmatched", source_file, line_number, "exact legacy image_path does not exist")
            continue
        legacy_sha = sha256_file(legacy_image)
        manifest = manifest_by_sha.get(legacy_sha)
        if manifest is None:
            if legacy_image.name.casefold() in manifest_basenames:
                report.counts["checksum_conflict"] += 1
                report.add_issue(
                    "checksum_conflict",
                    source_file,
                    line_number,
                    "same basename exists in target album but bytes differ; not matched",
                )
            else:
                report.counts["unmatched"] += 1
                report.add_issue("unmatched", source_file, line_number, "legacy image hash is not in manifest")
            continue
        if manifest.photo_id in seen_logical_ids:
            report.counts["duplicate"] += 1
            report.add_issue(
                "duplicate",
                source_file,
                line_number,
                "another legacy result already resolved to this photo_id; latest valid row wins",
                manifest.photo_id,
            )
        seen_logical_ids.add(manifest.photo_id)
        if not dataset_id.casefold().startswith("geograph"):
            report.counts["legacy_blocked"] += 1
            report.add_issue(
                "legacy_blocked",
                source_file,
                line_number,
                "legacy records are audit-only outside an explicitly named geograph dataset",
                manifest.photo_id,
            )
            continue
        if legacy_source not in INDEXABLE_SOURCES:
            report.counts["failed"] += 1
            report.add_issue(
                "failed",
                source_file,
                line_number,
                "legacy source is not remote or remote_repaired",
                manifest.photo_id,
            )
            continue
        current_relative = _current_relative_path(manifest, album_by_sha)
        if current_relative is None:
            report.counts["unmatched"] += 1
            report.counts["local_missing"] += 1
            report.add_issue(
                "unmatched",
                source_file,
                line_number,
                "verified legacy bytes no longer exist in target album",
                manifest.photo_id,
            )
            continue
        max_triples = min(15, max(1, len(legacy_triples)))
        accepted[manifest.photo_id] = _make_indexable_record(
            manifest=manifest,
            current_relative_path=current_relative,
            triples=legacy_triples[:15],
            source=legacy_source,
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            prompt_version="legacy-aire-guide-v1",
            max_triples=max_triples,
            generated_at=str(payload.get("generated_at") or ""),
        )

    report.indexable_records = [accepted[key] for key in sorted(accepted)]
    report.counts["matched"] = len(report.indexable_records)
    report.counts["indexable"] = len(report.indexable_records)
    return report


def write_indexable_scene_graphs(
    report: SceneGraphImportReport,
    output_path: str | Path,
) -> Path:
    """Atomically write the already-audited, deterministic index input JSONL."""

    path = Path(output_path).expanduser().resolve()
    _atomic_write_jsonl(report.indexable_records, path)
    report.dry_run = False
    return path


__all__ = [
    "INDEXABLE_SOURCES",
    "MANIFEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SceneGraphAuditIssue",
    "SceneGraphExportReport",
    "SceneGraphImportReport",
    "SceneGraphManifestRecord",
    "audit_scene_graph_results",
    "export_scene_graph_batch",
    "load_scene_graph_manifest",
    "photo_id_for_sha256",
    "sha256_file",
    "write_indexable_scene_graphs",
]
