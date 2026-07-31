"""Exploratory RRF weight-tuning ablation (NOT the pre-registered R0--R7).

The pre-registered main result uses FIXED spec weights (CLIP 1.0, caption 0.8,
scene 0.9, k=60) and stands unchanged. This script asks a separate question:
*given the same per-channel rankings*, is there a weighting under which the
3-channel RRF fusion BEATS CLIP-alone (R0)?

It reuses the already-generated valid queries (no new LLM calls): for each valid
query it fetches the three channel ranked lists once (CLIP / BLIP2 / SceneGraph),
then grid-searches the relative weights (w_clip fixed at 1.0; w_blip2, w_scene in
[0,1]) and reports Hit@1 / Hit@3 for each scheme, plus the grid optimum.

Example:
    python scripts/reweight_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH  # noqa: E402
from eval_corpus import frozen_corpus_photo_ids  # noqa: E402
from eval_metrics import hit_at_k  # noqa: E402
from eval_retrieval import CHANNEL_WEIGHT, RetrievalBackend, rrf_combine  # noqa: E402


def _load_valid_queries(results_path: Path) -> list[tuple[str, str]]:
    d = json.loads(results_path.read_text(encoding="utf-8"))
    return [
        (tid, o["query"]) for tid, o in d["query_outcomes"].items()
        if o.get("status") == "ok" and o.get("query")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "evaluation" / "private" / "retrieval_results_n300.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "evaluation" / "private" / "reweight_ablation.json")
    args = parser.parse_args()

    from model_pipeline import CLIPModelManager
    from vector_store import ChromaVectorStore

    valid = _load_valid_queries(args.results)
    print(f"valid queries: {len(valid)}", flush=True)
    conn = sqlite3.connect(APP_DB_PATH)
    corpus = frozen_corpus_photo_ids(conn)
    conn.close()

    backend = RetrievalBackend(
        clip_backend=CLIPModelManager(),
        vector_store=ChromaVectorStore(),
        caption_fts=__import__("eval_retrieval").EvalCaptionFTS(),
    )

    # Fetch the three channel ranked lists once per query (the expensive part).
    channels = ("clip", "blip2", "scene")
    print("fetching per-channel rankings...", flush=True)
    per_query: list[tuple[str, dict[str, list[str]]]] = []
    for i, (target_id, query) in enumerate(valid, 1):
        lists = {ch: backend._channel_list(ch, query, corpus) for ch in channels}
        per_query.append((target_id, lists))
        if i % 50 == 0:
            print(f"  {i}/{len(valid)}", flush=True)

    def evaluate(w_blip2: float, w_scene: float) -> tuple[float, float]:
        weights = {"clip": 1.0, "blip2": w_blip2, "scene": w_scene}
        h1 = h3 = 0
        for target_id, lists in per_query:
            ranked = rrf_combine(lists, weights=weights)
            if hit_at_k(ranked, target_id, 1):
                h1 += 1
            if hit_at_k(ranked, target_id, 3):
                h3 += 1
        n = len(per_query)
        return h1 / n, h3 / n

    n = len(per_query)

    # Named schemes.
    print("\n=== Named schemes (w_clip=1.0 fixed) ===", flush=True)
    print(f"{'scheme':<22}{'w_blip2':>8}{'w_scene':>8}{'Hit@1':>8}{'Hit@3':>8}", flush=True)
    named = []
    schemes = [
        ("R0 clip-only", 0.0, 0.0),
        ("R6 fixed (spec)", CHANNEL_WEIGHT["blip2"], CHANNEL_WEIGHT["scene"]),
        ("proportional Hit@1", 0.363 / 0.768, 0.262 / 0.768),
        ("proportional Hit@1^2", (0.363 / 0.768) ** 2, (0.262 / 0.768) ** 2),
        ("heavy-CLIP", 0.2, 0.1),
    ]
    for name, wb, ws in schemes:
        h1, h3 = evaluate(wb, ws)
        named.append({"scheme": name, "w_clip": 1.0, "w_blip2": round(wb, 3),
                      "w_scene": round(ws, 3), "hit_at_1": round(h1, 4), "hit_at_3": round(h3, 4)})
        print(f"{name:<22}{wb:>8.3f}{ws:>8.3f}{h1:>8.3f}{h3:>8.3f}", flush=True)

    # Grid search.
    print("\n=== Grid search (w_clip=1.0; w_blip2, w_scene in 0..1 step 0.1) ===", flush=True)
    best = None
    grid = []
    steps = [round(0.1 * i, 1) for i in range(11)]
    for wb in steps:
        for ws in steps:
            h1, h3 = evaluate(wb, ws)
            grid.append({"w_blip2": wb, "w_scene": ws,
                         "hit_at_1": round(h1, 4), "hit_at_3": round(h3, 4)})
            if best is None or h1 > best["hit_at_1"] or (h1 == best["hit_at_1"] and h3 > best["hit_at_3"]):
                best = {"w_blip2": wb, "w_scene": ws,
                        "hit_at_1": round(h1, 4), "hit_at_3": round(h3, 4)}
    r0_h1 = next(s["hit_at_1"] for s in named if s["scheme"] == "R0 clip-only")
    print(f"R0 (clip-only) Hit@1 = {r0_h1:.4f}", flush=True)
    print(f"grid-best: w_blip2={best['w_blip2']} w_scene={best['w_scene']} "
          f"Hit@1={best['hit_at_1']:.4f} Hit@3={best['hit_at_3']:.4f}", flush=True)
    print(f"beats R0? {'YES' if best['hit_at_1'] > r0_h1 else 'NO'} "
          f"(delta {best['hit_at_1']-r0_h1:+.4f})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_valid": n, "r0_hit_at_1": r0_h1, "named": named, "grid_best": best, "grid": grid,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}", flush=True)
    backend.vector_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
