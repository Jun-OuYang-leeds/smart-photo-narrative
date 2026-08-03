"""Re-run the 60-target metadata subset (Table V) with the searched-optimal weights.

The deployed-weight Table V (R6-M vs R7 on the 60-target metadata subset) lives in
``evaluation/private/retrieval_human_ablation.json`` under ``r6m_subset`` /
``r7_subset`` and uses the shipped (deployed) RRF settings k=60, caption 0.8,
scene-graph 0.9. Section V-B's grid search found those weights to be poor; the
best three-channel setting is k=1, BLIP2 0.4, SG 0.2 (``grid_b_all_channels.best``).

Two corpora are reported, because they disagree and the choice matters:

* ``frozen`` (default, the basis used by the paper's deployed Table V): reuses the
  FROZEN per-channel ranked lists in ``retrieval_results_n300.json`` (R0=CLIP,
  R2=BLIP2, R3=Scene), produced on the frozen 1,596-photo corpus. Byte-for-byte
  identical to the deployed-weight run; the ONLY variable is the RRF weights/k.
  This is the only basis on which optimal-vs-deployed is a fair comparison.

* ``live``: re-queries the current database (now 1,599 photos) with CLIP/FTS/scene.
  This matches what an independent re-run from scratch produces today, but it is
  NOT comparable to the paper's deployed numbers (which are on 1,596). Reported
  only to show the corpus drift.

Output -> ``evaluation/private/retrieval_metadata_optimal_weights.json``.
Nothing else is modified: no .docx, no build_final_report.py, no existing JSON.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH  # noqa: E402
from eval_corpus import frozen_corpus_photo_ids  # noqa: E402
from eval_metrics import (  # noqa: E402
    exact_mcnemar_two_sided,
    hit_at_k,
    wilson_interval,
)
from eval_retrieval import metadata_filter_ids, rrf_combine  # noqa: E402

N300_JSON = ROOT / "evaluation" / "private" / "retrieval_results_n300.json"
ABLATION_JSON = ROOT / "evaluation" / "private" / "retrieval_human_ablation.json"
OUT = ROOT / "evaluation" / "private" / "retrieval_metadata_optimal_weights.json"

# Frozen per-channel variant -> channel name.
CHANNEL_OF = {"clip": "R0", "blip2": "R2", "scene": "R3"}


def _rate_ci(ranked_pairs, kk):
    """Return (hits, rate, [lo, hi]) Wilson 95% CI for Hit@kk over the paired list."""
    n = len(ranked_pairs)
    hits = sum(hit_at_k(r, t, kk) for t, r in ranked_pairs)
    lo, hi = wilson_interval(hits, n)
    return hits, round(hits / n, 4), [round(lo, 3), round(hi, 3)]


def _frozen_lists(n300):
    """Per-channel top-100 ranked lists from the frozen 1,596-corpus run."""
    rankings = n300["rankings"]
    return {
        ch: {x["target_id"]: x["ranked_ids"] for x in rankings[var]}
        for ch, var in CHANNEL_OF.items()
    }


def _live_lists(meta_ids, query_of, corpus):
    """Re-query the current DB (1,599) with CLIP/FTS/scene for the subset only.
    Returns {channel: {target_id: [ranked_ids]}}, matching _frozen_lists."""
    from eval_retrieval import EvalCaptionFTS, RetrievalBackend
    backend = RetrievalBackend(
        clip_backend=__import__("model_pipeline").CLIPModelManager(),
        vector_store=__import__("vector_store").ChromaVectorStore(),
        caption_fts=EvalCaptionFTS(),
    )
    out = {ch: {} for ch in CHANNEL_OF}
    for i, tid in enumerate(meta_ids, 1):
        lists = {ch: backend._channel_list(ch, query_of[tid], corpus) for ch in CHANNEL_OF}
        for ch in CHANNEL_OF:
            out[ch][tid] = lists[ch]
        if i % 20 == 0:
            print(f"  live fetch {i}/{len(meta_ids)}", flush=True)
    return out


def _evaluate(lists_by_target, meta_ids, w_opt, k_opt, rows, conn, corpus):
    """R6-M / R7 / McNemar / Wilson for one set of per-channel lists."""
    r6m_ranked, r7_ranked, pool_sizes = [], [], []
    for tid in meta_ids:
        lists = {ch: lists_by_target[ch][tid] for ch in w_opt}
        r6m_ranked.append((tid, rrf_combine(lists, weights=w_opt, k=k_opt)))
        row = rows.get(tid, {})
        date_local = row.get("date_local") or ""
        location = row.get("location") or ""
        elig = (
            metadata_filter_ids(conn, date_local=date_local, location=location, corpus=corpus)
            if date_local and location
            else []
        )
        pool_sizes.append(len(elig))
        elig_set = set(elig)
        r7_ranked.append(
            (tid, rrf_combine(
                {ch: [p for p in lists[ch] if p in elig_set] for ch in w_opt},
                weights=w_opt, k=k_opt))
        )
    import statistics
    r6m_h1, r6m_r1, r6m_c1 = _rate_ci(r6m_ranked, 1)
    r6m_h5, r6m_r5, r6m_c5 = _rate_ci(r6m_ranked, 5)
    r7_h1, r7_r1, r7_c1 = _rate_ci(r7_ranked, 1)
    r7_h5, r7_r5, r7_c5 = _rate_ci(r7_ranked, 5)
    pool = {
        "pool_before": len(corpus), "n": len(pool_sizes),
        "after_median": statistics.median(pool_sizes) if pool_sizes else 0,
        "after_mean": round(statistics.mean(pool_sizes), 2) if pool_sizes else 0,
    }
    mc = {
        str(kk): exact_mcnemar_two_sided(
            [hit_at_k(r, t, kk) for t, r in r6m_ranked],
            [hit_at_k(r, t, kk) for t, r in r7_ranked],
        )
        for kk in (1, 5)
    }
    return {
        "r6m": {"hits_1": r6m_h1, "hit_at_1": r6m_r1, "ci_95_hit_at_1": r6m_c1,
                "hits_5": r6m_h5, "hit_at_5": r6m_r5, "ci_95_hit_at_5": r6m_c5},
        "r7": {"hits_1": r7_h1, "hit_at_1": r7_r1, "ci_95_hit_at_1": r7_c1,
               "hits_5": r7_h5, "hit_at_5": r7_r5, "ci_95_hit_at_5": r7_c5,
               "pool": pool},
        "mcnemar_r6m_vs_r7": mc,
    }


def main() -> int:
    n300 = json.loads(N300_JSON.read_text(encoding="utf-8"))
    abl = json.loads(ABLATION_JSON.read_text(encoding="utf-8"))
    best = abl["grid_b_all_channels"]["best"]
    w_opt = {"clip": 1.0, "blip2": best["w_blip2"], "scene": best["w_scene"]}
    k_opt = int(best["k"])
    print(f"optimal weights (from grid_b_all_channels.best): {w_opt}, k={k_opt}",
          flush=True)

    meta_ids = list(n300["metadata_subset"])  # frozen 60-target subset
    query_of = {tid: o["query"] for tid, o in n300["query_outcomes"].items()}
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    corpus_live = frozen_corpus_photo_ids(conn)  # current DB: 1,599
    ph = ",".join("?" for _ in meta_ids)
    rows = {r["photo_id"]: dict(r) for r in conn.execute(
        f"SELECT photo_id, date_local, location FROM photos WHERE photo_id IN ({ph})",
        meta_ids)}
    print(f"subset={len(meta_ids)}  live corpus={len(corpus_live)}", flush=True)

    # --- FROZEN: 1,596-corpus ranked lists (the paper's deployed basis) ---
    print("\n=== FROZEN (1,596 corpus, same lists as deployed Table V) ===", flush=True)
    frozen = _evaluate(_frozen_lists(n300), meta_ids, w_opt, k_opt, rows, conn, corpus_live)

    # --- LIVE: re-query current 1,599 DB (what a fresh run produces today) ---
    print("\n=== LIVE (1,599 corpus, re-queried from scratch) ===", flush=True)
    live = _evaluate(_live_lists(meta_ids, query_of, corpus_live), meta_ids, w_opt, k_opt,
                     rows, conn, corpus_live)

    out = {
        "note": (
            "R6-M vs R7 on the 60-target metadata subset, re-fused with the "
            "searched-optimal three-channel weights. Two bases are reported: "
            "'frozen' reuses the 1,596-corpus per-channel lists from "
            "retrieval_results_n300.json (identical to the deployed-weight run; "
            "the only variable is the RRF weights/k, so it is the fair "
            "optimal-vs-deployed comparison). 'live' re-queries the current "
            "1,599-photo DB and is NOT comparable to the paper's deployed numbers."
        ),
        "weight_source": "retrieval_human_ablation.json::grid_b_all_channels.best",
        "weights": w_opt,
        "k": k_opt,
        "n_subset": len(meta_ids),
        "frozen_1596_corpus": frozen,
        "live_1599_corpus": live,
        "comparison_to_deployed": {
            "r6m_deployed": {"hit_at_1": abl["r6m_subset"]["hit_at_1"],
                             "hit_at_5": abl["r6m_subset"]["hit_at_5"]},
            "r7_deployed": {"hit_at_1": abl["r7_subset"]["hit_at_1"],
                            "hit_at_5": abl["r7_subset"]["hit_at_5"]},
            "mcnemar_deployed": abl["r7_subset"]["mcnemar_vs_r6m"],
            "source": "retrieval_human_ablation.json (1,596 corpus)",
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for label, res in (("FROZEN 1596", frozen), ("LIVE 1599", live)):
        r6m, r7 = res["r6m"], res["r7"]
        print(f"\n[{label}]  R6-M Hit@1={r6m['hit_at_1']:.3f} {r6m['ci_95_hit_at_1']}  "
              f"Hit@5={r6m['hit_at_5']:.3f} {r6m['ci_95_hit_at_5']}", flush=True)
        print(f"[{label}]  R7   Hit@1={r7['hit_at_1']:.3f} {r7['ci_95_hit_at_1']}  "
              f"Hit@5={r7['hit_at_5']:.3f} {r7['ci_95_hit_at_5']}  "
              f"pool={r7['pool']}", flush=True)
        print(f"[{label}]  McNemar {res['mcnemar_r6m_vs_r7']}", flush=True)
    dep = out["comparison_to_deployed"]
    print(f"\n[deployed 1596] R6-M={dep['r6m_deployed']} R7={dep['r7_deployed']} "
          f"McNemar_H1_p={dep['mcnemar_deployed']['1']['p_value']:.2g}", flush=True)
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
