"""Run the R0--R7 retrieval experiment end to end.

Two modes:

* ``--prepare`` (offline, no LLM, no keys): freezes the 1,596-photo corpus,
  samples the deterministic 300 targets (near-duplicates / events / metadata
  stratification), and builds the two read-only BLIP/BLIP2 FTS indexes. Writes a
  freeze manifest. Use this to validate the offline pipeline against the live DB.

* ``--full``: everything in ``--prepare`` PLUS Qwen3.7 query generation (needs
  the eval keys in .env and a frozen region lock) and the R0--R7 retrieval run,
  then writes the private results + public aggregate summary.

  ``--limit N`` runs only the first N sampled targets (smoke-test the paid path
  before committing to all 300).

Examples:
    python scripts/run_retrieval_experiment.py --prepare
    python scripts/run_retrieval_experiment.py --full --limit 5
    python scripts/run_retrieval_experiment.py --full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH, DATA_DIR, PHOTOS_DIR  # noqa: E402
from eval_corpus import (  # noqa: E402
    compute_near_duplicate_groups,
    frozen_corpus_photo_ids,
    metadata_tier,
    near_duplicate_index,
    seed_from_name,
    stratified_sample,
    valid_target_pool_photo_ids,
)
from eval_experiment import ExperimentConfig, run_experiment  # noqa: E402
from eval_retrieval import (  # noqa: E402
    EVAL_CAPTION_FTS_PATH,
    EvalCaptionFTS,
    RetrievalBackend,
    metadata_filter_ids,
    run_r7,
    run_variant,
)

PRIVATE_DIR = ROOT / "evaluation" / "private"
PUBLIC_DIR = ROOT / "evaluation" / "public"
MANIFEST_PATH = PRIVATE_DIR / "retrieval_freeze_manifest.json"


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_target_rows(conn: sqlite3.Connection, target_ids: list[str]) -> list[dict]:
    placeholders = ",".join("?" for _ in target_ids)
    rows = conn.execute(
        f"""
        SELECT photo_id, relative_path, captured_at_sort, timestamp_confidence,
               date_local, location
        FROM photos WHERE photo_id IN ({placeholders})
        """,
        target_ids,
    ).fetchall()
    return [dict(r) for r in rows]


def _load_timestamps(conn: sqlite3.Connection) -> dict[str, float]:
    return {
        str(r[0]): float(r[1] or 0.0)
        for r in conn.execute("SELECT photo_id, captured_at_sort FROM photos")
    }


def _load_events(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        str(r[0]): str(r[1])
        for r in conn.execute("SELECT photo_id, event_id FROM event_photos")
    }


def build_sample(conn: sqlite3.Connection, vector_store) -> dict:
    """Freeze corpus + targets and draw the deterministic 300-target sample."""
    corpus = frozen_corpus_photo_ids(conn)
    target_pool = valid_target_pool_photo_ids(conn)
    if len(target_pool) < 300:
        raise SystemExit(f"target pool has only {len(target_pool)} photos (need 300)")

    clip_vectors = vector_store.get_image_embeddings(target_pool)
    timestamps = _load_timestamps(conn)
    events = _load_events(conn)
    rows = _load_target_rows(conn, target_pool)
    row_by_id = {r["photo_id"]: r for r in rows}

    groups = compute_near_duplicate_groups(target_pool, clip_vectors, timestamps)
    pid_to_group, _ = near_duplicate_index(groups)

    stratum_of = {
        r["photo_id"]: f"{metadata_tier(r)}_{'nd' if r['photo_id'] in pid_to_group else 'single'}"
        for r in rows
    }
    sample = stratified_sample(
        target_pool, stratum_of, pid_to_group, events,
        count=300, seed=seed_from_name(),
    )
    return {
        "corpus": corpus,
        "target_pool": target_pool,
        "sample": sample,
        "row_by_id": row_by_id,
        "events": events,
    }


def cmd_prepare(args: argparse.Namespace) -> int:
    from vector_store import ChromaVectorStore

    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    vs = ChromaVectorStore()
    try:
        sample = build_sample(conn, vs)
    finally:
        vs.close()
        conn.close()

    fts = EvalCaptionFTS()
    conn = sqlite3.connect(APP_DB_PATH)
    try:
        fts_counts = fts.build(conn)
    finally:
        conn.close()

    corpus, target_pool, samp = sample["corpus"], sample["target_pool"], sample["sample"]
    sample_hash = _sha256_bytes("\n".join(samp).encode("utf-8"))
    manifest = {
        "corpus_count": len(corpus),
        "target_pool_count": len(target_pool),
        "sample_count": len(samp),
        "sample_sha256": sample_hash,
        "caption_fts_counts": fts_counts,
        "caption_fts_db": str(EVAL_CAPTION_FTS_PATH.relative_to(ROOT)),
        "caption_fts_db_sha256": (
            _sha256_bytes(EVAL_CAPTION_FTS_PATH.read_bytes())
            if EVAL_CAPTION_FTS_PATH.exists() else None
        ),
        "seed_name": "spn-known-item-full-corpus-v1",
        "seed_int": seed_from_name(),
    }
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Private: the sampled target ids (needed to reproduce the exact run).
    (PRIVATE_DIR / "retrieval_sample_targets.json").write_text(
        json.dumps(samp, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        "Retrieval experiment -- prepare\n"
        f"  corpus            = {manifest['corpus_count']}\n"
        f"  target pool       = {manifest['target_pool_count']}\n"
        f"  sampled targets   = {manifest['sample_count']}  (sha256 {sample_hash[:12]}...)\n"
        f"  caption FTS (blip)= {fts_counts.get('eval_blip_fts', 0)}\n"
        f"  caption FTS (blip2)= {fts_counts.get('eval_blip2_fts', 0)}\n"
        f"  manifest          = {MANIFEST_PATH}"
    )
    return 0


def cmd_full(args: argparse.Namespace) -> int:
    from model_pipeline import CLIPModelManager
    from vector_store import ChromaVectorStore

    from eval_llm import EvalLLMClient, EvalConfigError, read_region_lock
    from eval_query_gen import QueryGenOutcome, generate_and_audit

    # The formal run requires a frozen region lock (no mid-run region mixing).
    if read_region_lock() is None and not args.skip_lock:
        raise SystemExit(
            "Region lock not frozen. Run: python scripts/eval_lock_region.py freeze"
        )

    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    vs = ChromaVectorStore()
    try:
        sample = build_sample(conn, vs)
    finally:
        vs.close()

    targets = sample["sample"][: args.limit] if args.limit else sample["sample"]
    corpus = sample["corpus"]
    row_by_id = sample["row_by_id"]
    # Query-gen distractors come from the WHOLE corpus, so the image resolver
    # must map any corpus photo_id -> file path (not just the 300 targets).
    placeholders = ",".join("?" for _ in corpus)
    rel_rows = conn.execute(
        f"SELECT photo_id, relative_path FROM photos WHERE photo_id IN ({placeholders})",
        corpus,
    ).fetchall()
    _path_of = {str(r["photo_id"]): str(PHOTOS_DIR / str(r["relative_path"]))
                for r in rel_rows}
    image_path_of = lambda pid: _path_of[pid]

    # Eval LLM (active frozen region). check_lock enforces no region mixing.
    try:
        client = EvalLLMClient(check_lock=not args.skip_lock)
    except EvalConfigError as exc:
        raise SystemExit(f"Eval LLM not configured: {exc}")

    # Reload a fresh vector store + CLIP backend for retrieval.
    vector_store = ChromaVectorStore()
    backend = RetrievalBackend(
        clip_backend=CLIPModelManager(),
        vector_store=vector_store,
        caption_fts=EvalCaptionFTS(),
    )

    base_seed = seed_from_name()

    # If retrying, keep the queries that already passed; only re-run failures
    # (with a shifted seed so the model gets a fresh attempt).
    prior_outcomes: dict[str, dict] = {}
    if args.retry_failed_from:
        prior = json.loads(Path(args.retry_failed_from).read_text(encoding="utf-8"))
        prior_outcomes = dict(prior.get("query_outcomes", {}))
        kept = sum(1 for o in prior_outcomes.values() if o.get("status") == "ok")
        print(f"[retry] keeping {kept} prior 'ok' queries; re-running the rest.", flush=True)

    def query_gen_fn(target_id):
        cached = prior_outcomes.get(target_id)
        if cached and cached.get("status") == "ok" and cached.get("query"):
            return QueryGenOutcome(
                target_id=target_id, status="ok", query=cached["query"],
                cues=list(cached.get("cues") or []), panel=list(cached.get("panel") or []),
                attempts=int(cached.get("attempts") or 1),
                repair_used=bool(cached.get("repair_used")),
            )
        # failed or absent -> re-run with a seed shift for a fresh attempt
        query_gen_fn._done = getattr(query_gen_fn, "_done", 0) + 1
        print(f"[retry {query_gen_fn._done}] re-running {target_id[:8]}", flush=True)
        shift = 500000 if cached else 0
        return generate_and_audit(
            target_id, image_path_of, vector_store, corpus, client,
            base_seed=base_seed + shift + _stable_int(target_id),
        )

    def rank_fn(variant, query, target_id):
        if variant == "R7":
            row = row_by_id.get(target_id, {})
            eligible = metadata_filter_ids(
                conn, date_local=row.get("date_local") or "",
                location=row.get("location") or "", corpus=corpus,
            )
            do = lambda: run_r7(query, backend, corpus, eligible_ids=eligible).ranked_ids
        else:
            do = lambda: run_variant(variant, query, backend, corpus).ranked_ids
        do()  # warmup (excluded from timing)
        ranked = None
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            ranked = do()
            times.append(time.perf_counter() - t0)
        return ranked, (statistics.median(times) if times else None)

    target_rows = _load_target_rows(conn, targets)
    try:
        config = ExperimentConfig()
        results = run_experiment(
            targets, corpus, conn, query_gen_fn=query_gen_fn, rank_fn=rank_fn,
            target_rows=target_rows, config=config, base_seed=base_seed,
        )
    finally:
        vector_store.close()
        conn.close()

    _write_outputs(results, args)
    _print_summary(results)
    return 0


def _write_outputs(results: dict, args: argparse.Namespace) -> None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_n{results['n_targets']}"

    # Private: queries, target uuids, full rankings, query-gen outcomes.
    private = {
        "n_targets": results["n_targets"],
        "variants": results["variants"],
        "query_outcomes": {
            tid: {
                "target_id": tid, "status": o.status, "query": o.query,
                "cues": o.cues, "panel": o.panel, "attempts": o.attempts,
                "repair_used": o.repair_used,
            }
            for tid, o in results["query_outcomes"].items()
        },
        "rankings": {
            v: [{"target_id": r.target_id, "ranked_ids": list(r.ranked_ids),
                  "valid": r.valid} for r in results["per_variant"][v]]
            for v in results["variants"]
        },
        "summaries": results["summaries"],
        "metadata_subset": results["metadata_subset"],
        "metadata_r6m_summary": results["metadata_r6m_summary"],
        "metadata_r7_summary": results["metadata_r7_summary"],
        "comparisons": results["comparisons"],
    }
    (PRIVATE_DIR / f"retrieval_results{suffix}.json").write_text(
        json.dumps(private, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Public: aggregate metrics + comparison table only (no queries, no uuids).
    public = {
        "n_targets": results["n_targets"],
        "variants": {
            v: {
                "hit_at_1_valid": results["summaries"][v]["hit_at_1_valid"],
                "hit_at_3_valid": results["summaries"][v]["hit_at_3_valid"],
                "success_at_1_all": results["summaries"][v]["success_at_1_all"],
                "success_at_3_all": results["summaries"][v]["success_at_3_all"],
                "query_generation_error": results["summaries"][v]["query_generation_error"],
                "latency_p95_seconds": results["summaries"][v]["latency_p95_seconds"],
                "rank_buckets": results["summaries"][v]["rank_buckets"],
            }
            for v in results["variants"]
        },
        "metadata_r6m_vs_r7": {
            "r6m": results["metadata_r6m_summary"]["hit_at_1_valid"],
            "r7": results["metadata_r7_summary"]["hit_at_1_valid"],
        },
        "metadata_r7_pool": results["metadata_r7_pool"],
        "comparisons": results["comparisons"],
    }
    (PUBLIC_DIR / f"retrieval_summary{suffix}.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _print_summary(results: dict) -> None:
    print(f"\nRetrieval experiment -- {results['n_targets']} targets")
    print(f"{'variant':<6}{'Hit@1':>8}{'Hit@3':>8}{'Succ@1':>8}{'P95(s)':>9}")
    for v in results["variants"]:
        s = results["summaries"][v]
        h1 = s["hit_at_1_valid"]["rate"]
        h3 = s["hit_at_3_valid"]["rate"]
        s1 = s["success_at_1_all"]["rate"]
        p95 = s["latency_p95_seconds"]
        print(f"{v:<6}{h1:>8.3f}{h3:>8.3f}{s1:>8.3f}{(p95 or 0):>9.3f}")
    print("\nPre-registered comparisons (exact McNemar, Holm-adjusted):")
    for rec in results["comparisons"]:
        print(f"  {rec['pair']:<12} b={rec['b']:<3} c={rec['c']:<3} "
              f"p={rec['p_value']:.4f} adj={rec['holm_adjusted_p']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true",
                      help="offline: freeze corpus + sample + build FTS indexes")
    mode.add_argument("--full", action="store_true",
                      help="full run: prepare + Qwen3.7 query gen + R0-R7 retrieval")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N sampled targets (smoke test)")
    parser.add_argument("--skip-lock", action="store_true",
                        help="allow running without a frozen region lock (dev only)")
    parser.add_argument("--retry-failed-from", type=Path, default=None,
                        help="prior results JSON: keep 'ok' queries, re-run failures")
    args = parser.parse_args()
    if args.prepare:
        return cmd_prepare(args)
    return cmd_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
