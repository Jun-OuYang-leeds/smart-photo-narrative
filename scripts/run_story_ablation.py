"""Run the frozen 12-case N0--N3 Story experiment with checkpoint/resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from story_agent import ContextAggregator  # noqa: E402
from story_evaluation import (  # noqa: E402
    EXPECTED_MODEL_DIGEST,
    EXPERIMENT_MODEL,
    EXPERIMENT_PROTOCOL,
    STORY_VARIANTS,
    OllamaExperimentBackend,
    StoryExperimentRunner,
    aggregate_metrics,
    assert_model_digest,
    build_blind_review_files,
    build_claim_audit_files,
    load_story_cases,
)


PRIVATE = ROOT / "evaluation" / "private"
DEFAULT_CASES = PRIVATE / "story_cases_v1.json"
DEFAULT_RESULTS = PRIVATE / "story_ablation_n0_n3_v1.jsonl"
DEFAULT_PACKET = PRIVATE / "story_n0_n3_blind_packet.json"
DEFAULT_MAPPING = PRIVATE / "story_n0_n3_blind_mapping.json"
DEFAULT_AUDIT = PRIVATE / "story_claim_audit_v1.json"
DEFAULT_AUDIT_MAPPING = PRIVATE / "story_claim_audit_mapping_v1.json"
PUBLIC_MANIFEST = ROOT / "evaluation" / "story_ablation_v1_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_public_manifest(records: list[dict], private_path: Path) -> dict:
    previous = {}
    if PUBLIC_MANIFEST.exists():
        try:
            previous = json.loads(PUBLIC_MANIFEST.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    per_variant = {}
    for variant in STORY_VARIANTS:
        selected = [item for item in records if item["variant_id"] == variant.variant_id]
        per_variant[variant.variant_id] = {
            "count": len(selected),
            "prompt_version": variant.prompt_version,
            "status_counts": {
                status: sum(item["status"] == status for item in selected)
                for status in sorted({item["status"] for item in selected})
            },
            "automatic_metric_means": aggregate_metrics([
                item["automatic_metrics"] for item in selected
            ]),
        }
    payload = {
        "protocol_version": EXPERIMENT_PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_generation_complete": len(records) == 48,
        "record_count": len(records),
        "unique_case_count": len({item["case_id"] for item in records}),
        "variant_counts": {
            variant.variant_id: sum(item["variant_id"] == variant.variant_id for item in records)
            for variant in STORY_VARIANTS
        },
        "language_counts_per_variant": {
            variant.variant_id: {
                language: sum(
                    item["variant_id"] == variant.variant_id and item["language"] == language
                    for item in records
                ) for language in ("zh", "en")
            } for variant in STORY_VARIANTS
        },
        "source_kind_counts_per_variant": {
            variant.variant_id: {
                kind: sum(
                    item["variant_id"] == variant.variant_id and item["source_kind"] == kind
                    for item in records
                ) for kind in ("single", "event", "date")
            } for variant in STORY_VARIANTS
        },
        "model_name": EXPERIMENT_MODEL,
        "model_digest": records[0]["model_digest"] if records else None,
        "variants": per_variant,
        "private_results_sha256": _sha256(private_path),
        "private_content_committed": False,
        "blind_review_status": previous.get("blind_review_status", "pending_user_review"),
        "claim_audit_status": previous.get("claim_audit_status", "pending_agent_evidence_audit"),
    }
    if previous.get("analysis_summary_sha256"):
        payload["analysis_summary_sha256"] = previous["analysis_summary_sha256"]
    PUBLIC_MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    backend = OllamaExperimentBackend(EXPERIMENT_MODEL)
    try:
        info = backend.model_info()
        assert_model_digest(info, EXPECTED_MODEL_DIGEST)
    except Exception as exc:
        raise SystemExit(
            "Ollama preflight failed. Start Ollama manually, then rerun this command. "
            f"No formal output was generated. Details: {type(exc).__name__}: {exc}"
        ) from exc
    if args.check_only:
        print(json.dumps({"available": True, "model": info, "digest_matches": True}, indent=2))
        return

    cases = load_story_cases(args.cases)
    aggregator = ContextAggregator()
    contexts = {}

    def context_factory(case):
        if case.case_id not in contexts:
            contexts[case.case_id] = aggregator.aggregate_by_photo_ids(
                case.photo_ids, label=case.label, event_id=case.event_id,
                source_kind=case.source_kind,
            )
        return contexts[case.case_id]

    runner = StoryExperimentRunner(
        backend, args.output, context_factory=context_factory,
        expected_digest=EXPECTED_MODEL_DIGEST,
        progress=lambda record, resumed: print(
            f"[story-ablation] {'resumed' if resumed else 'completed'} "
            f"{record['case_id']}/{record['variant_id']} status={record['status']} "
            f"latency_ms={record['latency_ms']:.0f}",
            flush=True,
        ),
    )
    records = runner.run(cases)
    for case in cases:
        context_factory(case)
    if not DEFAULT_PACKET.exists() and not DEFAULT_MAPPING.exists():
        build_blind_review_files(records, contexts, DEFAULT_PACKET, DEFAULT_MAPPING)
    elif not DEFAULT_PACKET.exists() or not DEFAULT_MAPPING.exists():
        raise RuntimeError("Only one blind-review file exists; refusing to replace the private review freeze")
    if not DEFAULT_AUDIT.exists() and not DEFAULT_AUDIT_MAPPING.exists():
        build_claim_audit_files(records, DEFAULT_AUDIT, DEFAULT_AUDIT_MAPPING)
    elif not DEFAULT_AUDIT.exists() or not DEFAULT_AUDIT_MAPPING.exists():
        raise RuntimeError("Only one claim-audit file exists; refusing to replace the private audit freeze")
    public = _write_public_manifest(records, args.output)
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
