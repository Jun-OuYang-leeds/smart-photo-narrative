"""Phase 3 re-run + RRF grid ablations using the HUMAN-corrected 300 queries.

Metrics: Hit@1 and Hit@5 (the @3 column was dropped per request). Because all
300 queries are valid (human-corrected), Success@k == Hit@k here (denominator =
300 either way); both are reported.

1. R0--R7 main result, fixed spec weights (k=60, CLIP 1.0, caption 0.8, SG 0.9).
2. Grid A -- CLIP+BLIP: k in {1,5,10,20,60} x w_caption in {0.05,0.1,0.2,0.4,0.8}.
3. Apply Grid-A optimum to CLIP+BLIP2.
4. Grid B -- CLIP+BLIP2+SG: k x w_blip2 x w_sg.

Per-channel ranked lists fetched ONCE per query; every grid point is a pure RRF
re-fusion. Results -> evaluation/private/retrieval_human_ablation.json.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH  # noqa: E402
from eval_corpus import frozen_corpus_photo_ids, seed_from_name  # noqa: E402
from eval_experiment import select_metadata_subset  # noqa: E402
from eval_metrics import exact_mcnemar_two_sided, hit_at_k, holm_correction  # noqa: E402
from eval_retrieval import (  # noqa: E402
    CHANNEL_WEIGHT, EvalCaptionFTS, RetrievalBackend, metadata_filter_ids, rrf_combine,
)

JSON_RESULTS = ROOT / "evaluation" / "private" / "retrieval_results_n300.json"
HUMAN_CSV = ROOT / "outputs" / "failed_review" / "descriptions.csv"
OUT = ROOT / "evaluation" / "private" / "retrieval_human_ablation.json"

KS = [1, 5, 10, 20, 60]
W_CAP = [0.05, 0.1, 0.2, 0.4, 0.8]
W_SG = [0.05, 0.1, 0.2, 0.4, 0.9]
METADATA_COUNT = 60


def _load_300_queries() -> list[tuple[str, str]]:
    d = json.loads(JSON_RESULTS.read_text(encoding="utf-8"))
    human = {}
    for r in csv.DictReader(HUMAN_CSV.open(encoding="utf-8-sig")):
        q = (r.get("query") or "").strip()
        if q and q.lower() not in ("(no query generated)", "(generation failed)"):
            human[r["photo_id"]] = q
    out = []
    for tid, o in d["query_outcomes"].items():
        q = human.get(tid) or o.get("query") or ""
        if not q:
            raise ValueError(f"no query for {tid}")
        out.append((tid, q))
    return out


def _load_target_rows(conn, targets):
    ph = ",".join("?" for _ in targets)
    return [dict(r) for r in conn.execute(
        f"SELECT photo_id, date_local, location, timestamp_confidence "
        f"FROM photos WHERE photo_id IN ({ph})", targets,
    )]


def _hit_rates(per_query_ranked):
    """Return (Hit@1, Hit@5) rates. With 300/300 valid these equal Success@1/5."""
    n = len(per_query_ranked)
    h1 = sum(hit_at_k(r, t, 1) for t, r in per_query_ranked)
    h5 = sum(hit_at_k(r, t, 5) for t, r in per_query_ranked)
    return round(h1 / n, 4), round(h5 / n, 4)


def main() -> int:
    from model_pipeline import CLIPModelManager
    from vector_store import ChromaVectorStore

    valid = _load_300_queries()
    print(f"queries: {len(valid)} (all valid, human-corrected) -> Hit@k == Success@k", flush=True)
    conn = sqlite3.connect(APP_DB_PATH); conn.row_factory = sqlite3.Row
    corpus = frozen_corpus_photo_ids(conn)
    targets = [t for t, _ in valid]
    target_rows = _load_target_rows(conn, targets)

    backend = RetrievalBackend(
        clip_backend=CLIPModelManager(),
        vector_store=ChromaVectorStore(),
        caption_fts=EvalCaptionFTS(),
    )

    channels = ("clip", "blip", "blip2", "scene")
    print("fetching per-channel rankings (4 channels x 300 queries)...", flush=True)
    lists_by_target: dict[str, dict[str, list[str]]] = {}
    for i, (tid, q) in enumerate(valid, 1):
        lists_by_target[tid] = {ch: backend._channel_list(ch, q, corpus) for ch in channels}
        if i % 50 == 0:
            print(f"  {i}/{len(valid)}", flush=True)

    def fuse(targets_subset, weight_map, k, restrict_pool=None):
        out = []
        for tid in targets_subset:
            lists = {}
            for ch, w in weight_map.items():
                base = lists_by_target[tid][ch]
                lists[ch] = [p for p in base if p in restrict_pool] if restrict_pool is not None else base
            out.append((tid, rrf_combine(lists, weights=weight_map, k=k)))
        return out

    def corr_vec(wm, k=60, kk=1):
        return [hit_at_k(r, t, kk) for t, r in fuse(targets, wm, k)]

    # ---------- 1. Main R0-R7 ----------
    print("\n=== Main R0-R7 (fixed: k=60, CLIP1.0/cap0.8/SG0.9) ===", flush=True)
    W = CHANNEL_WEIGHT
    main_variants = {
        "R0 clip": {"clip": W["clip"]},
        "R1 blip": {"blip": W["blip"]},
        "R2 blip2": {"blip2": W["blip2"]},
        "R3 scene": {"scene": W["scene"]},
        "R4 clip+blip": {"clip": W["clip"], "blip": W["blip"]},
        "R5 clip+blip2": {"clip": W["clip"], "blip2": W["blip2"]},
        "R6 all": {"clip": W["clip"], "blip2": W["blip2"], "scene": W["scene"]},
    }
    main_results = {}
    for name, wm in main_variants.items():
        h1, h5 = _hit_rates(fuse(targets, wm, 60))
        main_results[name] = {"hit_at_1": h1, "hit_at_5": h5}
        print(f"  {name:<16} Hit@1={h1:.3f}  Hit@5={h5:.3f}", flush=True)

    # McNemar at Hit@1 and Hit@5
    comparisons = []
    pairs = [("R0_vs_R4", main_variants["R0 clip"], main_variants["R4 clip+blip"]),
             ("R5_vs_R6", main_variants["R5 clip+blip2"], main_variants["R6 all"])]
    for label, wa, wb in pairs:
        for kk in (1, 5):
            mc = exact_mcnemar_two_sided(corr_vec(wa, 60, kk), corr_vec(wb, 60, kk))
            comparisons.append({"pair": label, "k_cutoff": kk, **mc})
    pvals = [c["p_value"] for c in comparisons]
    for c, adj in zip(comparisons, holm_correction(pvals)):
        c["holm_adjusted_p"] = adj["holm_adjusted_p"]

    # R6-M vs R7 on metadata subset
    meta_ids = select_metadata_subset(target_rows, count=METADATA_COUNT, seed=seed_from_name() + 1)
    meta_valid = [t for t in meta_ids if t in {x for x, _ in valid}]
    row_by_id = {r["photo_id"]: r for r in target_rows}
    r6m = _hit_rates(fuse(meta_valid, main_variants["R6 all"], 60))
    r7_ranked = []
    pool_sizes = []
    for tid in meta_valid:
        row = row_by_id.get(tid, {})
        elig = metadata_filter_ids(conn, date_local=row.get("date_local") or "",
                                   location=row.get("location") or "", corpus=corpus) if row.get("date_local") and row.get("location") else []
        pool_sizes.append(len(elig))
        elig_set = set(elig)
        ranked = rrf_combine({ch: [p for p in lists_by_target[tid][ch] if p in elig_set]
                              for ch in main_variants["R6 all"]}, weights=main_variants["R6 all"], k=60)
        r7_ranked.append((tid, ranked))
    r7 = _hit_rates(r7_ranked)
    import statistics
    r7_pool = {"pool_before": len(corpus), "n": len(pool_sizes),
               "after_median": statistics.median(pool_sizes) if pool_sizes else 0,
               "after_mean": round(statistics.mean(pool_sizes), 2) if pool_sizes else 0}
    print(f"  R6-M (subset)       Hit@1={r6m[0]:.3f}  Hit@5={r6m[1]:.3f}", flush=True)
    print(f"  R7 (subset+filter)  Hit@1={r7[0]:.3f}  Hit@5={r7[1]:.3f}  pool_median={r7_pool['after_median']}", flush=True)
    mc_r7 = {kk: exact_mcnemar_two_sided(
        [hit_at_k(r, t, kk) for t, r in fuse(meta_valid, main_variants["R6 all"], 60)],
        [hit_at_k(r, t, kk) for t, r in r7_ranked],
    ) for kk in (1, 5)}

    # ---------- 2. Grid A: CLIP+BLIP ----------
    print("\n=== Grid A: CLIP+BLIP (w_clip=1.0) ===", flush=True)
    grid_a, best_a = [], None
    for k in KS:
        for wc in W_CAP:
            h1, h5 = _hit_rates(fuse(targets, {"clip": 1.0, "blip": wc}, k))
            grid_a.append({"k": k, "w_blip": wc, "hit_at_1": h1, "hit_at_5": h5})
            if best_a is None or h1 > best_a["hit_at_1"] or (h1 == best_a["hit_at_1"] and h5 > best_a["hit_at_5"]):
                best_a = {"k": k, "w_blip": wc, "hit_at_1": h1, "hit_at_5": h5}
    print(f"  best: k={best_a['k']} w_blip={best_a['w_blip']} Hit@1={best_a['hit_at_1']:.3f} Hit@5={best_a['hit_at_5']:.3f}", flush=True)

    # ---------- 3. Apply Grid-A best to CLIP+BLIP2 ----------
    h1, h5 = _hit_rates(fuse(targets, {"clip": 1.0, "blip2": best_a["w_blip"]}, best_a["k"]))
    applied = {"k": best_a["k"], "w_blip2": best_a["w_blip"], "hit_at_1": h1, "hit_at_5": h5}
    print(f"\n=== CLIP+BLIP2 with Grid-A best (k={best_a['k']}, w={best_a['w_blip']}) ===\n  Hit@1={h1:.3f}  Hit@5={h5:.3f}", flush=True)

    # ---------- 4. Grid B: CLIP+BLIP2+SG ----------
    print("\n=== Grid B: CLIP+BLIP2+SG (w_clip=1.0) ===", flush=True)
    grid_b, best_b = [], None
    for k in KS:
        for wb in W_CAP:
            for ws in W_SG:
                h1, h5 = _hit_rates(fuse(targets, {"clip": 1.0, "blip2": wb, "scene": ws}, k))
                grid_b.append({"k": k, "w_blip2": wb, "w_scene": ws, "hit_at_1": h1, "hit_at_5": h5})
                if best_b is None or h1 > best_b["hit_at_1"] or (h1 == best_b["hit_at_1"] and h5 > best_b["hit_at_5"]):
                    best_b = {"k": k, "w_blip2": wb, "w_scene": ws, "hit_at_1": h1, "hit_at_5": h5}
    print(f"  best: k={best_b['k']} w_blip2={best_b['w_blip2']} w_scene={best_b['w_scene']} "
          f"Hit@1={best_b['hit_at_1']:.3f} Hit@5={best_b['hit_at_5']:.3f}", flush=True)
    print(f"  beats R0 (clip-only Hit@1={main_results['R0 clip']['hit_at_1']:.3f}, "
          f"Hit@5={main_results['R0 clip']['hit_at_5']:.3f})? "
          f"H@1 {'YES' if best_b['hit_at_1'] > main_results['R0 clip']['hit_at_1'] else 'NO'}, "
          f"H@5 {'YES' if best_b['hit_at_5'] > main_results['R0 clip']['hit_at_5'] else 'NO'}", flush=True)

    OUT.write_text(json.dumps({
        "n_queries": len(valid),
        "note": "all 300 valid -> Hit@k == Success@k",
        "main": main_results,
        "comparisons": comparisons,
        "r6m_subset": {"hit_at_1": r6m[0], "hit_at_5": r6m[1]},
        "r7_subset": {"hit_at_1": r7[0], "hit_at_5": r7[1], "pool": r7_pool, "mcnemar_vs_r6m": mc_r7},
        "grid_a_clip_blip": {"best": best_a, "all": grid_a},
        "clip_blip2_applied_grid_a_best": applied,
        "grid_b_all_channels": {"best": best_b, "all": grid_b},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten -> {OUT}", flush=True)
    backend.vector_store.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
