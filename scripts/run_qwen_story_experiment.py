"""Run the Qwen-only Faithful/Creative Story experiment with resume support."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_story_evaluation import (  # noqa: E402
    BLIND_MAPPING_PATH,
    BLIND_RESPONSES_PATH,
    CLAIM_AUDIT_PATH,
    CREATIVE_RESULTS_PATH,
    FAITHFUL_RESULTS_PATH,
    QwenCreativeRunner,
    QwenFaithfulRunner,
    QwenOllamaBackend,
    assert_qwen_digest,
    build_claim_audit_packet,
    build_dual_blind_files,
    build_public_summary,
    default_context_factory,
    freeze_creative_cases,
    load_creative_case_manifest,
    load_faithful_cases,
    privacy_scan_public_summary,
    reveal_dual_blind,
)
from config import APP_DB_PATH  # noqa: E402


def _story_count() -> int:
    connection = sqlite3.connect(f"file:{Path(APP_DB_PATH).resolve().as_posix()}?mode=ro", uri=True)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM stories").fetchone()[0])
    finally:
        connection.close()


def _progress(record: dict, resumed: bool) -> None:
    suffix = "resume" if resumed else "generated"
    print(
        f"[{suffix}] {record['case_id']}/{record['variant_id']} "
        f"status={record['status']} latency_ms={record['latency_ms']:.1f}",
        flush=True,
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _contexts(cases: list, *, creative: bool) -> dict:
    return {case.case_id: default_context_factory(case, creative=creative) for case in cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("faithful", "creative", "all"), default="all")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reveal", action="store_true")
    args = parser.parse_args()

    manifest = freeze_creative_cases()
    faithful_cases = load_faithful_cases()
    creative_main, creative_reserve, manifest = load_creative_case_manifest()
    backend = QwenOllamaBackend()
    model_info = backend.model_info()
    assert_qwen_digest(model_info)
    if args.check_only:
        print(json.dumps({
            "status": "ready", "model": model_info,
            "faithful_cases": len(faithful_cases), "creative_main": len(creative_main),
            "creative_reserve": len(creative_reserve), "case_manifest_sha256": manifest["freeze_sha256"],
            "production_story_count": _story_count(),
        }, ensure_ascii=False, indent=2))
        return

    stories_before = _story_count()
    faithful_records = _read_jsonl(FAITHFUL_RESULTS_PATH)
    creative_records = _read_jsonl(CREATIVE_RESULTS_PATH)
    creative_pair_ids: list[str] = []
    if args.track in {"faithful", "all"} and not args.reveal:
        faithful_records = QwenFaithfulRunner(backend, progress=_progress).run(faithful_cases)
    if args.track in {"creative", "all"} and not args.reveal:
        creative_runner = QwenCreativeRunner(backend, progress=_progress)
        try:
            creative_records, creative_pair_ids = creative_runner.run_with_reserves(
                creative_main, creative_reserve,
            )
        except RuntimeError as exc:
            if "valid Creative blind pairs remain after all frozen reserves" not in str(exc):
                raise
            creative_records = _read_jsonl(CREATIVE_RESULTS_PATH)
            creative_pair_ids = [
                case.case_id for case in [*creative_main, *creative_reserve]
                if creative_runner.valid_blind_pair(creative_records, case.case_id)
            ][:12]
            print(f"[incomplete] {exc}", flush=True)

    if not faithful_records or not creative_records:
        if args.reveal:
            raise RuntimeError("Both formal result files are required before reveal")
        print("The selected track is complete; run --track all to build the combined blind packet.")
        return

    if not creative_pair_ids:
        runner = QwenCreativeRunner(backend)
        all_cases = [*creative_main, *creative_reserve]
        creative_pair_ids = [
            case.case_id for case in all_cases
            if runner.valid_blind_pair(creative_records, case.case_id)
        ][:12]
    if not creative_pair_ids:
        raise RuntimeError("Creative results do not contain any valid frozen blind pairs")

    faithful_contexts = _contexts(faithful_cases, creative=False)
    creative_case_lookup = {case.case_id: case for case in [*creative_main, *creative_reserve]}
    creative_contexts = _contexts(
        [creative_case_lookup[case_id] for case_id in creative_pair_ids], creative=True,
    )
    build_dual_blind_files(
        faithful_records, creative_records, faithful_contexts, creative_contexts, creative_pair_ids,
    )
    all_contexts = {**faithful_contexts, **{
        case.case_id: default_context_factory(case, creative=True)
        for case in [*creative_main, *creative_reserve]
        if any(record.get("case_id") == case.case_id for record in creative_records)
    }}
    if not CLAIM_AUDIT_PATH.exists() and not args.reveal:
        build_claim_audit_packet([*faithful_records, *creative_records], all_contexts)

    claim_audit = None
    if CLAIM_AUDIT_PATH.exists():
        candidate = json.loads(CLAIM_AUDIT_PATH.read_text(encoding="utf-8"))
        if candidate.get("claims") and all(
            item.get("status") in set(candidate.get("allowed_statuses", []))
            and str(item.get("reason") or "").strip()
            for item in candidate["claims"]
        ):
            claim_audit = candidate
    blind_reveal = None
    if args.reveal:
        if not BLIND_RESPONSES_PATH.exists():
            raise RuntimeError("Blind responses have not been submitted")
        responses = json.loads(BLIND_RESPONSES_PATH.read_text(encoding="utf-8"))
        mapping = json.loads(BLIND_MAPPING_PATH.read_text(encoding="utf-8"))
        blind_reveal = reveal_dual_blind(responses, mapping)
        private_reveal = ROOT / "evaluation" / "private" / "story_qwen_dual_blind_revealed_v1.json"
        from qwen_story_evaluation import atomic_write_json
        atomic_write_json(private_reveal, blind_reveal["private"])
    summary = build_public_summary(
        faithful_records, creative_records, manifest, creative_pair_ids,
        claim_audit=claim_audit, blind_reveal=blind_reveal,
    )
    privacy_errors = privacy_scan_public_summary(summary)
    if privacy_errors:
        raise RuntimeError("Public summary privacy check failed: " + "; ".join(privacy_errors))
    stories_after = _story_count()
    if stories_after != stories_before:
        raise RuntimeError(f"Production stories changed during evaluation: {stories_before} -> {stories_after}")
    print(json.dumps({
        "status": (
            ("complete" if len(creative_pair_ids) == 12 else "incomplete")
            + ("_pending_blind_review" if blind_reveal is None else "_revealed")
        ),
        "faithful_records": len(faithful_records), "creative_records": len(creative_records),
        "creative_blind_pairs": len(creative_pair_ids), "claim_count": len(
            json.loads(CLAIM_AUDIT_PATH.read_text(encoding="utf-8")).get("claims", [])
        ),
        "production_story_count": stories_after, "privacy_errors": privacy_errors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
