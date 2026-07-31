"""Aggregate automatic metrics, claim audit and finalized blind preferences."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from story_evaluation import (  # noqa: E402
    STORY_VARIANTS,
    aggregate_metrics,
    exact_two_sided_binomial,
    paired_bootstrap_difference,
    reveal_blind_preferences,
)


PRIVATE = ROOT / "evaluation" / "private"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _claim_rates(packet_path: Path, mapping_path: Path) -> tuple[dict[tuple[str, str], float], str, dict]:
    if not packet_path.exists() or not mapping_path.exists():
        return {}, "not_created", {}
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    claims = {item["audit_id"]: item for item in packet.get("claims", [])}
    allowed = {"supported", "unsupported", "uncertain", "non_factual"}
    if any(item.get("status") not in allowed for item in claims.values()):
        return {}, "pending_agent_evidence_audit", {}
    grouped = defaultdict(list)
    by_variant = defaultdict(Counter)
    for item in mapping.get("mappings", []):
        claim = claims[item["audit_id"]]
        by_variant[item["variant_id"]][claim["status"]] += 1
        if claim["status"] != "non_factual":
            grouped[(item["case_id"], item["variant_id"])].append(claim["status"])
    rates = {
        key: values.count("unsupported") / len(values) if values else 0.0
        for key, values in grouped.items()
    }
    summary = {}
    for variant, counts in by_variant.items():
        factual = sum(counts[value] for value in ("supported", "unsupported", "uncertain"))
        summary[variant] = {
            "status_counts": dict(counts),
            "factual_unit_count": factual,
            "supported_fraction": counts["supported"] / factual if factual else None,
            "unsupported_fraction": counts["unsupported"] / factual if factual else None,
            "uncertain_fraction": counts["uncertain"] / factual if factual else None,
        }
    return rates, "complete_agent_evidence_audit", summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=PRIVATE / "story_ablation_n0_n3_v1.jsonl")
    parser.add_argument("--claim-audit", type=Path, default=PRIVATE / "story_claim_audit_v1.json")
    parser.add_argument("--claim-mapping", type=Path, default=PRIVATE / "story_claim_audit_mapping_v1.json")
    parser.add_argument("--responses", type=Path, default=PRIVATE / "story_n0_n3_blind_responses.json")
    parser.add_argument("--blind-mapping", type=Path, default=PRIVATE / "story_n0_n3_blind_mapping.json")
    parser.add_argument("--private-reveal", type=Path, default=PRIVATE / "story_n0_n3_blind_revealed.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "story_ablation_v1_summary.json")
    args = parser.parse_args()

    records = _read_jsonl(args.results)
    keys = {(item["case_id"], item["variant_id"]) for item in records}
    if len(records) != 48 or len(keys) != 48:
        raise SystemExit("Formal Story results must contain exactly 48 unique case/variant records")
    claim_rates, audit_status, audit_summary = _claim_rates(args.claim_audit, args.claim_mapping)
    metrics_by_variant = {}
    status_by_variant = {}
    latency_by_variant = {}
    metric_records = []
    for record in records:
        metrics = dict(record["automatic_metrics"])
        key = (record["case_id"], record["variant_id"])
        if key in claim_rates:
            metrics["unsupported_claim_rate"] = claim_rates[key]
        metric_records.append({**record, "automatic_metrics": metrics})
    for variant in STORY_VARIANTS:
        selected_records = [item for item in metric_records if item["variant_id"] == variant.variant_id]
        selected = [item["automatic_metrics"] for item in selected_records]
        metrics_by_variant[variant.variant_id] = aggregate_metrics(selected)
        status_by_variant[variant.variant_id] = dict(Counter(item["status"] for item in selected_records))
        latencies = [float(item["latency_ms"]) for item in selected_records]
        latency_by_variant[variant.variant_id] = {
            "mean_ms": sum(latencies) / len(latencies),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "max_ms": max(latencies),
        }

    blind_status = "not_started"
    preference = None
    if args.responses.exists() and args.blind_mapping.exists():
        responses = json.loads(args.responses.read_text(encoding="utf-8"))
        if responses.get("status") == "locked":
            mapping = json.loads(args.blind_mapping.read_text(encoding="utf-8"))
            revealed = reveal_blind_preferences(responses, mapping)
            args.private_reveal.write_text(
                json.dumps({"protocol_version": "story-blind-reveal-v1", "results": revealed}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            counts = Counter(item["preferred_variant"] for item in revealed)
            preference = {
                "N0_wins": counts["N0"], "N3_wins": counts["N3"], "ties": counts["tie"],
                "non_tie_exact_binomial_p": exact_two_sided_binomial(counts["N3"], counts["N0"]),
            }
            blind_status = "complete"
        else:
            blind_status = "in_progress"

    comparisons = {
        "N1_to_N2_duplicate_text_unit_rate": paired_bootstrap_difference(
            metric_records, "N1", "N2", "duplicate_text_unit_rate",
        ),
        "N2_to_N3_risk_term_rate": paired_bootstrap_difference(
            metric_records, "N2", "N3", "risk_term_rate",
        ),
        "N2_to_N3_language_compliant": paired_bootstrap_difference(
            metric_records, "N2", "N3", "language_compliant",
        ),
        "N2_to_N3_unsupported_claim_rate": paired_bootstrap_difference(
            metric_records, "N2", "N3", "unsupported_claim_rate",
        ),
    }
    public = {
        "protocol_version": "story-ablation-summary-v1",
        "record_count": len(records),
        "case_count": 12,
        "automatic_metric_means": metrics_by_variant,
        "status_counts": status_by_variant,
        "latency_distribution": latency_by_variant,
        "paired_bootstrap": comparisons,
        "claim_audit_status": audit_status,
        "claim_audit_micro_summary": audit_summary,
        "blind_review_status": blind_status,
        "blind_N0_vs_N3": preference,
        "privacy": "aggregate only; no query, UUID, path, citation, or story text",
    }
    args.output.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = ROOT / "evaluation" / "story_ablation_v1_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["claim_audit_status"] = audit_status
        manifest["blind_review_status"] = blind_status
        manifest["analysis_summary_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
