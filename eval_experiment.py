"""Orchestration logic for the R0--R7 retrieval experiment.

Ties together the frozen corpus, query generation, the R0--R7 variants, and the
metric/statistic layer into one deterministic loop, then runs the pre-registered
McNemar + Holm comparisons.

The orchestration is fully decoupled from the real LLM / CLIP / Chroma backends:
callers inject ``query_gen_fn`` (target_id -> :class:`QueryGenOutcome`) and
``rank_fn`` (variant, query, target_id -> (ranked_ids, latency)). The real run
script wires production backends; the unit test wires fakes. This keeps the
experiment *logic* auditable independently of the slow, paid real run.

Outputs (returned as a plain dict, written to disk by the script):

* per-variant :class:`QueryResult` lists and :func:`summarize_variant` metrics;
* the deterministic 60-target metadata subset (R6-M vs R7 comparison);
* pre-registered two-sided exact McNemar p-values with Holm adjustment.
"""

from __future__ import annotations

import random
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from eval_corpus import METADATA_TIME_CONFIDENCE, seed_from_name
from eval_metrics import (
    QueryResult,
    exact_mcnemar_two_sided,
    hit_at_k,
    holm_correction,
    summarize_variant,
)
from eval_query_gen import QueryGenOutcome
from eval_retrieval import metadata_filter_ids

METADATA_SUBSET_COUNT = 60
DEFAULT_VARIANTS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")
# Pre-registered pairwise comparisons (Hit@1 correctness, paired by target).
# Holm correction is applied across all of these plus the R6-M vs R7 comparison.
DEFAULT_COMPARISONS = (("R0", "R4"), ("R4", "R6"), ("R5", "R6"))
HIT_K_FOR_COMPARISON = 1


RankFn = Callable[[str, str, str], "tuple[list[str], Optional[float]]"]
QueryGenFn = Callable[[str], QueryGenOutcome]


@dataclass
class ExperimentConfig:
    variants: Sequence[str] = DEFAULT_VARIANTS
    metadata_subset_count: int = METADATA_SUBSET_COUNT
    comparisons: Sequence[tuple[str, str]] = DEFAULT_COMPARISONS
    hit_k: int = HIT_K_FOR_COMPARISON


def metadata_complete(row: dict) -> bool:
    """A target qualifies for the metadata subset when it has a reliable
    timestamp (confidence >= 0.6), a local date, and a full geocoded location."""
    try:
        conf_ok = float(row.get("timestamp_confidence") or 0.0) >= METADATA_TIME_CONFIDENCE
    except (TypeError, ValueError):
        conf_ok = False
    return conf_ok and bool(row.get("date_local")) and bool(row.get("location"))


def select_metadata_subset(
    target_rows: Sequence[dict],
    *,
    count: int = METADATA_SUBSET_COUNT,
    seed: Optional[int] = None,
) -> list[str]:
    """Deterministically pick ``count`` metadata-complete targets.

    Filters to targets with reliable time + date + location, then takes a seeded
    deterministic sample (sorted by photo_id, then a seeded shuffle). Query-
    generation errors are NOT replaced -- a target selected here stays even if
    its query later fails.
    """
    if seed is None:
        seed = seed_from_name("spn-metadata-subset-v1")
    eligible = sorted(r["photo_id"] for r in target_rows if metadata_complete(r))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return sorted(eligible[:count])


def _correctness_vector(results: Sequence[QueryResult], target_of: dict[str, str], k: int) -> list[bool]:
    """Hit@k correctness per query, ordered by the results list (paired)."""
    return [hit_at_k(r.ranked_ids, target_of.get(r.query_id, r.target_id), k)
            if r.valid else False for r in results]


def run_experiment(
    targets: Sequence[str],
    corpus: Sequence[str],
    conn: sqlite3.Connection,
    *,
    query_gen_fn: QueryGenFn,
    rank_fn: RankFn,
    target_rows: Sequence[dict],
    config: ExperimentConfig = ExperimentConfig(),
    base_seed: Optional[int] = None,
) -> dict:
    """Run the full R0--R7 experiment over ``targets``.

    ``target_rows`` provides each target's metadata (photo_id, date_local,
    location, timestamp_confidence) for the metadata subset and R7 filter.
    Returns a dict with per-variant results, summaries, the metadata subset, and
    the McNemar + Holm comparison table.
    """
    if base_seed is None:
        base_seed = seed_from_name()

    # 1. Generate queries (one per target).
    outcomes: dict[str, QueryGenOutcome] = {}
    for target_id in targets:
        outcomes[target_id] = query_gen_fn(target_id)

    # 2. Run every variant for every target.
    per_variant: dict[str, list[QueryResult]] = {v: [] for v in config.variants}
    for target_id in targets:
        outcome = outcomes[target_id]
        query = outcome.query or ""
        valid = outcome.status == "ok" and bool(query)
        for variant in config.variants:
            if valid:
                ranked, latency = rank_fn(variant, query, target_id)
            else:
                ranked, latency = [], None
            per_variant[variant].append(
                QueryResult(
                    query_id=target_id, target_id=target_id,
                    ranked_ids=ranked, valid=valid, latency_seconds=latency,
                )
            )

    summaries = {v: summarize_variant(per_variant[v]) for v in config.variants}
    target_of = {t: t for t in targets}

    # 3. Metadata subset: R6-M (R6, unfiltered) vs R7 (R6 + date/location filter).
    metadata_ids = select_metadata_subset(
        target_rows, count=config.metadata_subset_count, seed=base_seed + 1
    )
    meta_index = {tid: i for i, tid in enumerate(targets) if tid in set(metadata_ids)}
    r6m_results = [per_variant["R6"][i] for tid, i in meta_index.items()]
    row_by_id = {r["photo_id"]: r for r in target_rows}
    r7_results: list[QueryResult] = []
    pool_after_sizes: list[int] = []  # R7 candidate-pool size per metadata target
    for tid in sorted(meta_index):
        # R7's metadata filter (date + location) is applied INSIDE rank_fn; here
        # we additionally measure the candidate-pool reduction (spec: report
        # pool sizes before/after so a smaller pool is never misread as a
        # semantic-ranking gain).
        row = row_by_id.get(tid, {})
        date_local = row.get("date_local") or ""
        location = row.get("location") or ""
        if date_local and location:
            eligible = metadata_filter_ids(
                conn, date_local=date_local, location=location, corpus=corpus,
            )
        else:
            eligible = []
        pool_after_sizes.append(len(eligible))
        query = outcomes[tid].query or ""
        if outcomes[tid].status == "ok" and query:
            ranked, _ = rank_fn("R7", query, tid)
        else:
            ranked = []
        r7_results.append(QueryResult(query_id=tid, target_id=tid, ranked_ids=ranked,
                                      valid=(outcomes[tid].status == "ok")))
    metadata_r7_pool = {
        "pool_before": len(corpus),
        "n_targets": len(pool_after_sizes),
        "after_median": statistics.median(pool_after_sizes) if pool_after_sizes else 0,
        "after_mean": statistics.mean(pool_after_sizes) if pool_after_sizes else 0,
        "after_min": min(pool_after_sizes) if pool_after_sizes else 0,
        "after_max": max(pool_after_sizes) if pool_after_sizes else 0,
    }

    # 4. Pre-registered McNemar comparisons + Holm correction.
    vectors = {v: _correctness_vector(per_variant[v], target_of, config.hit_k) for v in config.variants}
    comparison_records = []
    p_values = []
    for a, b in config.comparisons:
        mc = exact_mcnemar_two_sided(vectors[a], vectors[b])
        comparison_records.append({"pair": f"{a}_vs_{b}", "scope": "all_300", **mc})
        p_values.append(mc["p_value"])
    # R6-M vs R7 on the metadata subset.
    vec_r6m = _correctness_vector(r6m_results, {t: t for t in metadata_ids}, config.hit_k)
    vec_r7 = _correctness_vector(r7_results, {t: t for t in metadata_ids}, config.hit_k)
    mc_r7 = exact_mcnemar_two_sided(vec_r6m, vec_r7)
    comparison_records.append({"pair": "R6M_vs_R7", "scope": "metadata_60", **mc_r7})
    p_values.append(mc_r7["p_value"])

    holm = holm_correction(p_values)
    for rec, h in zip(comparison_records, holm):
        rec["holm_adjusted_p"] = h["holm_adjusted_p"]

    return {
        "n_targets": len(targets),
        "variants": list(config.variants),
        "per_variant": per_variant,
        "summaries": summaries,
        "metadata_subset": metadata_ids,
        "metadata_r6m_summary": summarize_variant(r6m_results),
        "metadata_r7_summary": summarize_variant(r7_results),
        "metadata_r7_pool": metadata_r7_pool,
        "comparisons": comparison_records,
        "query_outcomes": outcomes,
    }


__all__ = [
    "DEFAULT_COMPARISONS",
    "DEFAULT_VARIANTS",
    "ExperimentConfig",
    "HIT_K_FOR_COMPARISON",
    "METADATA_SUBSET_COUNT",
    "metadata_complete",
    "run_experiment",
    "select_metadata_subset",
]
