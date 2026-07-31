"""Checkpointed Qwen2.5-VL Scene Graph generation for an exported batch.

This script talks to a local OpenAI-compatible vLLM endpoint on the compute
node.  It never logs the API token.  Checkpointing skips only a successful,
non-empty result with exactly the same identity and generation provenance;
failed, malformed and empty rows are retried on the next run.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_graph_io import (  # noqa: E402
    INDEXABLE_SOURCES,
    RESULT_SCHEMA_VERSION,
    SceneGraphManifestRecord,
    load_scene_graph_manifest,
    sha256_file,
)


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROMPT_VERSION = "spn-scene-graph-v1"
SYSTEM_PROMPT = """You extract a factual scene graph from one photo.
Return one JSON object only: {{"scene_graph":[{{"subject":"...","predicate":"...","object":"..."}}]}}.
Use only visibly supported entities and relations. Do not infer names, identity,
intent, emotion, exact location, or events outside the image. Use short English
noun phrases. Remove duplicates. Return at most {max_triples} triples."""


def _normalise_triples(value: Any, max_triples: int) -> list[dict[str, str]]:
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
        if len(triples) >= max_triples:
            break
    return triples


def _parse_scene_graph(raw: str, max_triples: int) -> list[dict[str, str]]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    candidates = [cleaned]
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = ValueError("model response must be a JSON object")
            continue
        value = payload.get("scene_graph")
        if value is None:
            value = payload.get("scene graph")
        if value is None:
            value = payload.get("triples")
        return _normalise_triples(value, max_triples)
    raise ValueError(f"could not parse model JSON: {last_error}")


def _chat(
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read(600).decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {error_body}") from exc
    payload = json.loads(body)
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("vLLM response did not contain message content") from exc


def _generate_one(
    *,
    image_path: Path,
    base_url: str,
    api_key: str,
    model_id: str,
    max_triples: int,
    max_tokens: int,
    timeout: float,
) -> tuple[list[dict[str, str]], str]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    raw = _chat(
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(max_triples=max_triples)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the visible scene graph."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
        timeout=timeout,
    )
    try:
        return _parse_scene_graph(raw, max_triples), "remote"
    except ValueError:
        # Keep the repair input deliberately short so input + 640 output tokens
        # remains inside the guide-validated 1536-token vLLM context.
        repair_input = raw[:2600]
        repaired = _chat(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Repair malformed scene-graph JSON. Return only "
                        f"{chr(123)}\"scene_graph\":[subject/predicate/object objects]{chr(125)} "
                        f"with at most {max_triples} triples."
                    ),
                },
                {"role": "user", "content": repair_input},
            ],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return _parse_scene_graph(repaired, max_triples), "remote_repaired"


def checkpoint_key(
    *,
    dataset_id: str,
    photo_id: str,
    sha256: str,
    model_id: str,
    prompt_version: str,
    max_triples: int,
) -> tuple[str, str, str, str, str, int]:
    return (dataset_id, photo_id, sha256, model_id, prompt_version, max_triples)


def load_successful_checkpoint_keys(
    output_jsonl: Path,
) -> set[tuple[str, str, str, str, str, int]]:
    """Load only successful and non-empty checkpoint rows."""

    done: set[tuple[str, str, str, str, str, int]] = set()
    if not output_jsonl.exists():
        return done
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            source = str(row.get("source") or "")
            status = str(row.get("status") or "")
            try:
                max_triples = int(row.get("max_triples"))
            except (TypeError, ValueError):
                continue
            if not 1 <= max_triples <= 15:
                continue
            triples = _normalise_triples(row.get("scene_graph"), max_triples)
            if source not in INDEXABLE_SOURCES or status != "success" or not triples:
                continue
            values = [row.get(key) for key in ("dataset_id", "photo_id", "sha256", "model_id", "prompt_version")]
            if any(not isinstance(value, str) or not value for value in values):
                continue
            done.add(
                checkpoint_key(
                    dataset_id=values[0],
                    photo_id=values[1],
                    sha256=values[2],
                    model_id=values[3],
                    prompt_version=values[4],
                    max_triples=max_triples,
                )
            )
    return done


def _selected_records(
    records: list[SceneGraphManifestRecord],
    shard_index: int,
    num_shards: int,
    limit: int | None,
) -> list[SceneGraphManifestRecord]:
    selected = [record for index, record in enumerate(records) if index % num_shards == shard_index]
    return selected[:limit] if limit is not None else selected


def _append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _summary(output_jsonl: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError:
                    counts["malformed"] += 1
                    continue
                if isinstance(row, dict):
                    total += 1
                    counts[str(row.get("status") or "unknown")] += 1
    return {"rows": total, "status_counts": dict(sorted(counts.items()))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate hash-bound Qwen Scene Graph JSONL.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--max-triples", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=640)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.max_triples <= 15:
        raise SystemExit("--max-triples must be between 1 and 15")
    if args.max_tokens != 640:
        raise SystemExit("this AIRE workflow is validated with --max-tokens 640")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")

    manifest_path = Path(args.manifest).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    records = load_scene_graph_manifest(manifest_path)
    records = _selected_records(records, args.shard_index, args.num_shards, args.limit)
    completed = load_successful_checkpoint_keys(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    pending: list[SceneGraphManifestRecord] = []
    for record in records:
        key = checkpoint_key(
            dataset_id=record.dataset_id,
            photo_id=record.photo_id,
            sha256=record.sha256,
            model_id=args.model_id,
            prompt_version=args.prompt_version,
            max_triples=args.max_triples,
        )
        if key not in completed:
            pending.append(record)

    print(
        json.dumps(
            {
                "selected": len(records),
                "already_successful_nonempty": len(records) - len(pending),
                "pending": len(pending),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "model_id": args.model_id,
                "prompt_version": args.prompt_version,
                "max_triples": args.max_triples,
                "max_tokens": args.max_tokens,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with output_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(pending, start=1):
            image_path = image_dir / f"{record.photo_id}.jpg"
            started = time.perf_counter()
            triples: list[dict[str, str]] = []
            source = "failed"
            status = "failed"
            error = ""
            if not image_path.is_file():
                error = "cloud image is missing"
            elif image_path.stat().st_size != record.file_size or sha256_file(image_path) != record.sha256:
                error = "cloud image checksum or size does not match manifest"
            else:
                for attempt in range(1, args.attempts + 1):
                    try:
                        triples, source = _generate_one(
                            image_path=image_path,
                            base_url=args.base_url,
                            api_key=api_key,
                            model_id=args.model_id,
                            max_triples=args.max_triples,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                        )
                        if triples:
                            status = "success"
                            error = ""
                            break
                        status = "empty"
                        error = "model returned no valid triples"
                    except Exception as exc:  # one image must not abort the checkpoint
                        source = "failed"
                        status = "failed"
                        error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    if attempt < args.attempts:
                        time.sleep(min(2.0 * attempt, 5.0))

            elapsed = time.perf_counter() - started
            output = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "dataset_id": record.dataset_id,
                "photo_id": record.photo_id,
                "sha256": record.sha256,
                "model_id": args.model_id,
                "prompt_version": args.prompt_version,
                "max_triples": args.max_triples,
                "source": source,
                "status": status,
                "scene_graph": triples,
                "scene_graph_flat_text": " ".join(
                    f"{item['subject']} {item['predicate']} {item['object']}" for item in triples
                ),
                "error": error,
                "elapsed_seconds": round(elapsed, 4),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            _append_record(handle, output)
            print(
                f"[{index}/{len(pending)}] photo_id={record.photo_id} "
                f"status={status} source={source} triples={len(triples)} elapsed={elapsed:.2f}s",
                flush=True,
            )

    print(json.dumps(_summary(output_jsonl), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
