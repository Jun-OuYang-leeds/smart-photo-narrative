"""Run local Scene Graph retrieval smoke tests without Ollama or Qwen."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval_engine import MultimodalRetriever  # noqa: E402


DEFAULT_QUERIES = (
    "person wearing jacket",
    "tree beside road",
    "building with windows",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the independent Scene Graph channel.")
    parser.add_argument("--queries", nargs="+", default=list(DEFAULT_QUERIES))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    retriever = MultimodalRetriever()
    checks = []
    started = time.perf_counter()
    try:
        for query in args.queries:
            query_started = time.perf_counter()
            response = retriever.search_standard(
                query,
                top_k=args.top_k,
                enabled_channels=("scene_graph",),
            )
            results = [
                {
                    "photo_id": result.photo_id,
                    "relative_path": result.id,
                    "score": round(result.score, 6),
                    "matched_scene_graph_triples": result.matched_scene_graph_triples,
                    "modalities": result.matched_modalities,
                }
                for result in response.results
            ]
            checks.append(
                {
                    "query": query,
                    "elapsed_ms": round((time.perf_counter() - query_started) * 1000, 3),
                    "result_count": len(results),
                    "passed": bool(results)
                    and all("scene_graph" in result["modalities"] for result in results)
                    and any(result["matched_scene_graph_triples"] for result in results),
                    "results": results,
                }
            )
    finally:
        retriever.vector_store.close()
        if retriever._clip_backend is not None and hasattr(retriever._clip_backend, "unload_model"):
            retriever._clip_backend.unload_model()

    report = {
        "ollama_used": False,
        "qwen_used": False,
        "channel": "scene_graph",
        "query_count": len(checks),
        "all_passed": all(check["passed"] for check in checks),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "checks": checks,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
