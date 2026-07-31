"""Validate a private A0-A4 report and publish privacy-safe aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VARIANTS = ("A0", "A1", "A2", "A3", "A4")
EXPECTED_CATEGORIES = (
    "scene", "object_attribute", "relation_semantic", "relation_exact",
    "caption_lexical", "metadata",
)
METRICS = ("MRR", "Recall@1", "nDCG@1", "Recall@5", "nDCG@5", "Recall@10", "nDCG@10")
PAIRWISE = (
    ("A1_minus_A0", "A1", "A0", "Caption/BM25 contribution relative to CLIP"),
    ("A2_minus_A0", "A2", "A0", "Scene Graph contribution relative to CLIP"),
    ("A3_minus_A0", "A3", "A0", "Caption plus Scene Graph contribution relative to CLIP"),
    ("A4_minus_A0", "A4", "A0", "Complete system contribution relative to CLIP"),
    ("A4_minus_A3", "A4", "A3", "Metadata contribution relative to the same three-channel fusion"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * percentage / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("aggregate requires at least one query")
    return {
        "query_count": len(rows),
        "metrics_macro": {
            metric: statistics.fmean(float(row["metrics"][metric]) for row in rows)
            for metric in METRICS
        },
    }


def paired_bootstrap(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    metric: str,
    *,
    seed_label: str,
    iterations: int = 10000,
) -> dict[str, Any]:
    query_ids = sorted(set(left) & set(right))
    if set(left) != set(right) or not query_ids:
        raise ValueError("paired variants must contain the same non-empty query IDs")
    deltas = [
        float(left[query_id]["metrics"][metric]) - float(right[query_id]["metrics"][metric])
        for query_id in query_ids
    ]
    seed = int(hashlib.sha256(seed_label.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    means = [
        statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(iterations)
    ]
    low, high = percentile(means, 2.5), percentile(means, 97.5)
    return {
        "query_count": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "bootstrap_ci95": [low, high],
        "ci_excludes_zero": low > 0.0 or high < 0.0,
        "positive_queries": sum(value > 1e-12 for value in deltas),
        "unchanged_queries": sum(abs(value) <= 1e-12 for value in deltas),
        "negative_queries": sum(value < -1e-12 for value in deltas),
        "bootstrap_iterations": iterations,
    }


def validate_and_summarize(report: Mapping[str, Any], *, report_hash: str) -> dict[str, Any]:
    errors: list[str] = []
    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if report.get("accuracy_status") != "computed_from_supplied_qrels":
        errors.append("accuracy_status is not computed_from_supplied_qrels")
    if report.get("query_count") != 48 or report.get("photo_query_count") != 48:
        errors.append("report must contain exactly 48 photo queries")
    if report.get("evaluation_depth") != 10 or report.get("cutoffs") != [1, 5, 10]:
        errors.append("evaluation depth/cutoffs do not match the frozen protocol")
    if report.get("repeats") != 3 or report.get("warmup_calls_per_variant") != 1:
        errors.append("repeats/warmup do not match the frozen protocol")
    if report.get("ollama_parser_enabled") is not False:
        errors.append("Ollama parser must be disabled")
    variants = report.get("variants") or {}
    if tuple(variants) != EXPECTED_VARIANTS:
        errors.append(f"variant order mismatch: {tuple(variants)}")

    rows_by_variant: dict[str, list[Mapping[str, Any]]] = {}
    maps: dict[str, dict[str, Mapping[str, Any]]] = {}
    for variant_id in EXPECTED_VARIANTS:
        variant = variants.get(variant_id) or {}
        rows = list(variant.get("per_query") or [])
        rows_by_variant[variant_id] = rows
        maps[variant_id] = {str(row.get("query_id")): row for row in rows}
        if variant.get("status") != "computed" or variant.get("query_count") != 48 or len(rows) != 48:
            errors.append(f"{variant_id} does not contain 48 computed queries")
            continue
        if variant.get("all_rankings_deterministic") is not True:
            errors.append(f"{variant_id} rankings are not deterministic")
        if any(row.get("ranking_deterministic_across_repeats") is not True for row in rows):
            errors.append(f"{variant_id} has a non-deterministic per-query ranking")
        if any(row.get("judgment_source") != "human_validated" for row in rows):
            errors.append(f"{variant_id} includes non-human-validated judgments")
        recomputed = aggregate(rows)["metrics_macro"]
        stored = variant.get("metrics_macro") or {}
        for metric in METRICS:
            if not math.isclose(float(stored.get(metric, math.nan)), recomputed[metric], abs_tol=1e-12):
                errors.append(f"{variant_id} {metric} macro value does not recompute")

    reference_rows = rows_by_variant.get("A0", [])
    category_counts = Counter(str(row.get("category")) for row in reference_rows)
    language_counts = Counter(str(row.get("language")) for row in reference_rows)
    split_counts = Counter(str(row.get("split")) for row in reference_rows)
    if category_counts != Counter({category: 8 for category in EXPECTED_CATEGORIES}):
        errors.append(f"category quotas mismatch: {dict(category_counts)}")
    if language_counts != Counter({"en": 36, "zh": 12}):
        errors.append(f"language quotas mismatch: {dict(language_counts)}")
    if split_counts != Counter({"dev": 12, "test": 36}):
        errors.append(f"split quotas mismatch: {dict(split_counts)}")
    reference_ids = set(maps.get("A0", {}))
    if any(set(maps[variant_id]) != reference_ids for variant_id in EXPECTED_VARIANTS):
        errors.append("variants do not contain identical query IDs")
    if errors:
        raise ValueError("; ".join(errors))

    def grouped(field: str, values: Iterable[str]) -> dict[str, Any]:
        return {
            value: {
                variant_id: aggregate([
                    row for row in rows_by_variant[variant_id] if row.get(field) == value
                ])
                for variant_id in EXPECTED_VARIANTS
            }
            for value in values
        }

    def pairwise_summary(
        query_maps: Mapping[str, Mapping[str, Mapping[str, Any]]], *, scope: str,
    ) -> dict[str, Any]:
        values = {}
        for label, left_id, right_id, interpretation in PAIRWISE:
            values[label] = {
                "left": left_id,
                "right": right_id,
                "interpretation": interpretation,
                "metrics": {
                    metric: paired_bootstrap(
                        query_maps[left_id], query_maps[right_id], metric,
                        seed_label=f"retrieval-ablation-v1|{scope}|{label}|{metric}",
                    )
                    for metric in ("MRR", "Recall@5", "Recall@10", "nDCG@10")
                },
            }
        return values

    test_rows = {
        variant_id: [row for row in rows_by_variant[variant_id] if row.get("split") == "test"]
        for variant_id in EXPECTED_VARIANTS
    }
    test_maps = {
        variant_id: {str(row["query_id"]): row for row in rows}
        for variant_id, rows in test_rows.items()
    }

    return {
        "protocol_version": "retrieval-ablation-analysis-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "private_report_sha256": report_hash,
        "qrels_sha256": report["qrels"]["sha256"],
        "privacy": "aggregate only; query text, photo IDs, qrels and rankings omitted",
        "validation": {
            "passed": True,
            "query_count": 48,
            "variant_count": 5,
            "all_rankings_deterministic": True,
            "macro_metrics_recomputed": True,
            "judgment_source": "human_validated",
        },
        "parameters": {
            "cutoffs": report["cutoffs"],
            "evaluation_depth": report["evaluation_depth"],
            "repeats": report["repeats"],
            "warmup_calls_per_variant": report["warmup_calls_per_variant"],
            "ollama_parser_enabled": report["ollama_parser_enabled"],
            "primary_effectiveness_scope": "36 held-out test queries",
        },
        "overall_all_48_descriptive": {
            variant_id: {
                **aggregate(rows_by_variant[variant_id]),
                "latency_ms": variants[variant_id]["latency_ms"],
            }
            for variant_id in EXPECTED_VARIANTS
        },
        "primary_test_36": {
            variant_id: aggregate(test_rows[variant_id])
            for variant_id in EXPECTED_VARIANTS
        },
        "by_category_all_48_descriptive": grouped("category", EXPECTED_CATEGORIES),
        "by_category_test_36": {
            category: {
                variant_id: aggregate([
                    row for row in test_rows[variant_id] if row.get("category") == category
                ])
                for variant_id in EXPECTED_VARIANTS
            }
            for category in EXPECTED_CATEGORIES
        },
        "by_language": grouped("language", ("en", "zh")),
        "by_language_test_36": {
            language: {
                variant_id: aggregate([
                    row for row in test_rows[variant_id] if row.get("language") == language
                ])
                for variant_id in EXPECTED_VARIANTS
            }
            for language in ("en", "zh")
        },
        "by_split": grouped("split", ("dev", "test")),
        "paired_bootstrap": {
            "all_48_descriptive": pairwise_summary(maps, scope="all-48"),
            "test_36_primary": pairwise_summary(test_maps, scope="test-36"),
        },
        "runtime": report.get("runtime", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report", type=Path,
        nargs="?",
        default=ROOT / "outputs" / "experiments" / "retrieval_ablation_A0_A4_qrels_v1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "evaluation" / "retrieval_ablation_v1_summary.json",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    summary = validate_and_summarize(report, report_hash=sha256(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "private_report_sha256": summary["private_report_sha256"],
        "qrels_sha256": summary["qrels_sha256"],
        "validation": summary["validation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
