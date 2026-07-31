"""Freeze the retrieval/Story experiment inputs without copying private data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    APP_DB_PATH, BLIP_MODEL_NAME, CHROMA_PERSIST_DIR, CLIP_MODEL_NAME,
    INDEX_SCHEMA_VERSION, RRF_K, RETRIEVAL_CANDIDATE_K,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace",
    ).strip()


def database_counts() -> dict[str, int]:
    tables = {
        "photos": "photos", "captions": "captions", "scene_graphs": "scene_graphs",
        "scene_graph_triples": "scene_graph_triples", "events": "events",
    }
    with sqlite3.connect(APP_DB_PATH) as connection:
        return {name: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for name, table in tables.items()}


def file_manifest() -> dict[str, str]:
    files: list[Path] = []
    excluded_parts = {"__pycache__", ".pytest_cache", "private", "outputs"}
    for path in ROOT.rglob("*.py"):
        if path.is_file() and not excluded_parts.intersection(path.relative_to(ROOT).parts):
            files.append(path)
    for pattern in ("requirements*.txt", "*.md", "docs/*.md", "evaluation/*.md", "evaluation/*.json"):
        files.extend(
            path for path in ROOT.glob(pattern)
            if path.is_file() and path.name != "frozen_scope_v1.json"
        )
    files.extend(path for path in CHROMA_PERSIST_DIR.rglob("*") if path.is_file())
    files.append(APP_DB_PATH)
    unique = sorted(set(files), key=lambda path: path.as_posix().casefold())
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in unique}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "frozen_scope_v1.json")
    args = parser.parse_args()
    # ROOT is already the project subdirectory inside the parent repository;
    # the pathspec therefore needs to be relative to ROOT.
    status = git("status", "--porcelain", "--", ".")
    manifest = {
        "manifest_version": "retrieval-story-scope-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_entries": status.splitlines(),
        "database_counts": database_counts(),
        "models": {"clip": CLIP_MODEL_NAME, "caption": BLIP_MODEL_NAME, "scene_graph": "Qwen offline JSONL"},
        "retrieval": {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "rrf_k": RRF_K,
            "candidate_k": RETRIEVAL_CANDIDATE_K,
            "ablations": ["A0", "A1", "A2", "A3", "A4"],
        },
        "story": {
            "prompt_version": "grounded-story-v3-event",
            "variants": ["N0", "N1", "N2", "N3"],
        },
        "sha256": file_manifest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
