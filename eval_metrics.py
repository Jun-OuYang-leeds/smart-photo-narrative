"""Retrieval-experiment metrics and statistics.

Implements exactly the protocol's metric set -- nothing more:

* **Hit@1 / Hit@3** over valid queries. Only the *specified* target photo counts
  as a hit; a near-duplicate of the target in the top-k is NOT a hit.
* **End-to-end Success@k** = hits / 300 over ALL queries (a query with a
  generation error counts as a miss, not a dropped query).
* **query_generation_error** count and rate.
* **P95 retrieval latency** -- each query is warmed up once then timed 3 times;
  the per-query latency is the median of the 3, and P95 is taken across queries.
* **95% Wilson score intervals** for every reported proportion.
* **Pre-registered comparisons**: two-sided exact McNemar with Holm correction.
  No Recall / MRR / nDCG are computed (the protocol dropped them).

The main figure is a Rank 1 / Rank 2--3 / Miss stacked bar, built from
:func:`stacked_rank_counts`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


# ============================ Hit / rank ============================


def rank_of_target(ranked_ids: Sequence[str], target_id: str) -> Optional[int]:
    """1-based rank of the exact target in the ranked list, or ``None`` if absent.

    Near-duplicates of the target are NOT substitutes: only an exact id match
    counts (the protocol: "只认指定目标照片，近重复照片不算命中").
    """
    for i, pid in enumerate(ranked_ids):
        if pid == target_id:
            return i + 1
    return None


def hit_at_k(ranked_ids: Sequence[str], target_id: str, k: int) -> bool:
    """True iff the exact target appears in the top-k (near-duplicates excluded)."""
    rank = rank_of_target(ranked_ids, target_id)
    return rank is not None and rank <= k


def rank_bucket(rank: Optional[int]) -> str:
    """Bucket for the stacked-bar figure: rank1 / rank2-3 / miss."""
    if rank is None:
        return "miss"
    if rank == 1:
        return "rank1"
    if rank <= 3:
        return "rank2-3"
    return "miss"  # present but below the Hit@3 cutoff -> counts as a miss in the figure


@dataclass
class QueryResult:
    """One query's retrieval outcome across a single variant."""

    query_id: str
    target_id: str
    ranked_ids: Sequence[str]
    valid: bool = True            # False when query_generation_error occurred
    latency_seconds: Optional[float] = None  # median of the 3 timed runs


def stacked_rank_counts(results: Sequence[QueryResult]) -> dict[str, int]:
    counts = {"rank1": 0, "rank2-3": 0, "miss": 0}
    for r in results:
        rank = rank_of_target(r.ranked_ids, r.target_id) if r.valid else None
        counts[rank_bucket(rank)] += 1
    return counts


# ============================ Wilson interval ============================


def wilson_interval(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (hits/n).

    Returns (0.0, 0.0) when n == 0. Clamped to [0, 1].
    """
    if n <= 0:
        return 0.0, 0.0
    phat = hits / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    spread = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, centre - spread), min(1.0, centre + spread)


def proportion_summary(hits: int, n: int, z: float = Z_95) -> dict:
    low, high = wilson_interval(hits, n, z)
    return {
        "hits": hits,
        "n": n,
        "rate": hits / n if n else 0.0,
        "wilson95_low": low,
        "wilson95_high": high,
    }


# ============================ Variant summary ============================


def summarize_variant(results: Sequence[QueryResult]) -> dict:
    """Aggregate one R-variant's per-query results into the protocol's metrics."""
    n_all = len(results)
    valid = [r for r in results if r.valid]
    n_valid = len(valid)
    errors = n_all - n_valid

    hit1 = sum(1 for r in valid if hit_at_k(r.ranked_ids, r.target_id, 1))
    hit3 = sum(1 for r in valid if hit_at_k(r.ranked_ids, r.target_id, 3))

    latencies = [r.latency_seconds for r in results if r.latency_seconds is not None]
    p95 = _percentile(latencies, 95) if latencies else None

    return {
        # valid-query Hit@k (denominator = valid queries)
        "hit_at_1_valid": proportion_summary(hit1, n_valid),
        "hit_at_3_valid": proportion_summary(hit3, n_valid),
        # end-to-end Success@k (denominator = ALL 300 queries)
        "success_at_1_all": proportion_summary(hit1, n_all),
        "success_at_3_all": proportion_summary(hit3, n_all),
        # query generation errors
        "query_generation_error": proportion_summary(errors, n_all),
        # latency
        "latency_p95_seconds": p95,
        "latency_count": len(latencies),
        # stacked-bar figure
        "rank_buckets": stacked_rank_counts(results),
    }


def _percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy default method)."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


# ============================ Exact McNemar + Holm ============================


def exact_mcnemar_two_sided(
    correct_a: Sequence[bool], correct_b: Sequence[bool]
) -> dict:
    """Two-sided exact McNemar test on paired Hit@1 correctness vectors.

    ``correct_a[i]`` / ``correct_b[i]`` are whether variant A / B hit query i at
    the chosen cutoff. Uses the exact binomial form (valid for small discordant
    counts); returns b, c, n_discordant, and the two-sided p-value.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("correctness vectors must be paired and equal length")
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)   # A right, B wrong
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)   # A wrong, B right
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        # two-sided: twice the smaller binomial tail, capped at 1.0
        tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
        p = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p}


def holm_correction(p_values: Sequence[float]) -> list[dict]:
    """Holm-Bonferroni step-down correction.

    Returns one entry per input p-value (in the ORIGINAL order) with the raw
    and Holm-adjusted p-value. Adjusted values are monotone non-decreasing in
    sorted order and never below the raw value.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank_pos, idx in enumerate(order):
        raw = p_values[idx]
        adj = min((m - rank_pos) * raw, 1.0)
        running_max = max(running_max, adj)  # enforce monotonicity
        adjusted[idx] = running_max
    return [
        {"index": i, "raw_p": p_values[i], "holm_adjusted_p": adjusted[i]}
        for i in range(m)
    ]


__all__ = [
    "QueryResult",
    "Z_95",
    "exact_mcnemar_two_sided",
    "hit_at_k",
    "holm_correction",
    "proportion_summary",
    "rank_bucket",
    "rank_of_target",
    "stacked_rank_counts",
    "summarize_variant",
    "wilson_interval",
]
