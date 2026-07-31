"""Apply already-audited Qwen Scene Graphs to SQLite and the CLIP vector index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from config import APP_DB_PATH, PHOTOS_DIR
from scene_graph_io import sha256_file
from storage import PhotoStorage
from vector_store import ChromaVectorStore


@dataclass(frozen=True)
class SceneGraphApplySummary:
    total: int
    applied: int
    skipped: int
    checksum_conflicts: int
    malformed: int
    errors: tuple[str, ...]


def _triple_texts(triples: Iterable[dict[str, Any]]) -> list[str]:
    values = []
    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("predicate", triple.get("relation", ""))).strip()
        obj = str(triple.get("object", "")).strip()
        if subject and predicate and obj:
            values.append(f"{subject} {predicate} {obj}")
    return values[:15]


def apply_indexable_scene_graphs(
    indexable_jsonl: str | Path,
    *,
    storage: Optional[PhotoStorage] = None,
    vector_store: Optional[ChromaVectorStore] = None,
    clip_backend: Any = None,
    photos_root: str | Path = PHOTOS_DIR,
    dry_run: bool = False,
) -> SceneGraphApplySummary:
    """Import by current SQLite UUID after verifying full content SHA-256."""
    storage = storage or PhotoStorage(APP_DB_PATH)
    vector_store = vector_store or ChromaVectorStore()
    if clip_backend is None and not dry_run:
        from model_pipeline import CLIPModelManager

        clip_backend = CLIPModelManager()
    root = Path(photos_root).resolve()
    total = applied = skipped = conflicts = malformed = 0
    errors: list[str] = []

    with Path(indexable_jsonl).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                digest = str(row["sha256"]).casefold()
                triples = row.get("scene_graph") or []
                texts = _triple_texts(triples)
                if row.get("status") != "success" or row.get("source") not in {"remote", "remote_repaired"} or not texts:
                    skipped += 1
                    continue
                photo = storage.get_photo_by_hash(digest)
                if not photo:
                    conflicts += 1
                    errors.append(f"line {line_number}: no SQLite photo with matching SHA-256")
                    continue
                current_path = (root / photo["relative_path"]).resolve()
                try:
                    current_path.relative_to(root)
                except ValueError:
                    conflicts += 1
                    errors.append(f"line {line_number}: stored path escapes photo root")
                    continue
                if not current_path.is_file() or sha256_file(current_path).casefold() != digest:
                    conflicts += 1
                    errors.append(f"line {line_number}: current photo checksum changed")
                    continue
                if dry_run:
                    applied += 1
                    continue

                storage.upsert_scene_graph(
                    photo["photo_id"],
                    scene_graph_text=str(row.get("scene_graph_flat_text") or " ; ".join(texts)),
                    triples=triples[:15],
                    model_name=str(row.get("model_id") or "Qwen/Qwen2.5-VL-7B-Instruct"),
                    model_version=str(row.get("prompt_version") or "unknown"),
                    source=str(row["source"]),
                    source_ref=str(
                        row.get("source_ref")
                        or f"{row.get('dataset_id', '')}:{row.get('photo_id', '')}"
                    ),
                    output_sha256=(str(row.get("output_sha256") or "").strip() or None),
                    status="ok",
                    generated_at=str(row.get("generated_at") or "") or None,
                )
                if hasattr(clip_backend, "encode_texts_batch"):
                    embeddings = clip_backend.encode_texts_batch(texts)
                else:
                    embeddings = [clip_backend.encode_text(text) for text in texts]
                vector_store.replace_scene_graph(
                    photo["photo_id"],
                    texts,
                    embeddings,
                    content_sha256=digest,
                )
                applied += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                malformed += 1
                errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")

    if clip_backend is not None and hasattr(clip_backend, "unload_model"):
        clip_backend.unload_model()
    return SceneGraphApplySummary(total, applied, skipped, conflicts, malformed, tuple(errors))
