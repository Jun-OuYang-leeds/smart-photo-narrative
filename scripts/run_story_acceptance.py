"""Run the three frozen real-Ollama Story v3 acceptance cases exactly once.

The detailed evidence and outputs remain under evaluation/private.  The public
manifest contains only aggregate pass/fail information and content hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from story_agent import GroundingValidator, OllamaGenerator, StoryGenerator  # noqa: E402
from story_evaluation import EXPERIMENT_MODEL, StoryCase, load_story_cases, score_story  # noqa: E402


PRIVATE_CASES = ROOT / "evaluation" / "private" / "story_cases_v1.json"
PRIVATE_OUTPUT = ROOT / "evaluation" / "private" / "ollama_acceptance_v1.json"
PUBLIC_OUTPUT = ROOT / "evaluation" / "story_acceptance_v1_manifest.json"
FROZEN_ACCEPTANCE_MODEL = EXPERIMENT_MODEL


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_cases(cases: list[StoryCase]) -> list[StoryCase]:
    selected = [
        next(case for case in cases if case.source_kind == "single"),
        next(case for case in cases if case.source_kind == "event" and case.label == "2026-07-12"),
        next(case for case in cases if case.source_kind == "date"),
    ]
    if len(selected) != 3 or len({case.case_id for case in selected}) != 3:
        raise RuntimeError("Acceptance protocol must select exactly three distinct frozen cases")
    return selected


def _model_record(model: str) -> dict[str, Any]:
    try:
        import ollama

        listing = ollama.list()
        models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
        for item in models:
            value = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: {})()
            name = str(value.get("model") or value.get("name") or "")
            if name == model or name.startswith(model + ":"):
                return {
                    "requested_model": model,
                    "resolved_model": name,
                    "digest": value.get("digest"),
                    "modified_at": str(value.get("modified_at") or ""),
                    "size": value.get("size"),
                }
    except Exception as exc:
        return {"requested_model": model, "inspection_error": f"{type(exc).__name__}: {exc}"}
    return {"requested_model": model, "resolved_model": None}


def _acceptance_pass(metrics: dict[str, Any], final_errors: list[str]) -> bool:
    return bool(
        metrics["json_valid"]
        and metrics["citation_valid_rate"] == 1.0
        and metrics["evidence_group_coverage"] == 1.0
        and metrics["evidence_id_accounting_rate"] == 1.0
        and metrics["language_compliant"]
        and metrics["duplicate_paragraph_rate"] == 0.0
        and metrics["unsupported_claim_rate"] == 0.0
        and metrics["conflict_handling_rate"] == 1.0
        and not final_errors
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Check model availability without generating a story")
    args = parser.parse_args()

    cases = load_story_cases(PRIVATE_CASES)
    selected = _selected_cases(cases)
    backend = OllamaGenerator(FROZEN_ACCEPTANCE_MODEL)
    available = backend.is_available()
    check = {
        "available": available,
        "error": backend.get_error_message(),
        "model": _model_record(FROZEN_ACCEPTANCE_MODEL),
        "selected_case_ids": [case.case_id for case in selected],
    }
    if args.check_only:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return
    if not available:
        raise SystemExit(check["error"] or "Ollama is unavailable")
    if PRIVATE_OUTPUT.exists():
        raise SystemExit(f"Refusing to rerun the three real cases: {PRIVATE_OUTPUT} already exists")

    generator = StoryGenerator(model=FROZEN_ACCEPTANCE_MODEL, generator=backend)
    records: list[dict[str, Any]] = []
    for case in selected:
        context = generator.aggregator.aggregate_by_photo_ids(
            case.photo_ids,
            label=case.label,
            event_id=case.event_id,
            source_kind=case.source_kind,
        )
        started = time.perf_counter()
        story = generator.generate_story_with_context(
            context, mode="faithful", language=case.language, save=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        final_errors = GroundingValidator.validate(
            story.to_dict(), context, "faithful", case.language,
        )
        metrics = score_story(story, context, latency_ms=latency_ms)
        records.append({
            "case": {
                "case_id": case.case_id,
                "source_kind": case.source_kind,
                "label": case.label,
                "language": case.language,
                "photo_ids": list(case.photo_ids),
            },
            "evidence_plan": {
                "photo_count": context.photo_count,
                "groups": [group.to_dict() for group in context.groups],
            },
            "story": story.to_dict(),
            "story_metadata": {
                "status": story.status,
                "model": story.model,
                "prompt_hash": story.prompt_hash,
                "validation_history": story.validation_codes,
                "final_validation_errors": final_errors,
            },
            "metrics": metrics,
            "passed": _acceptance_pass(metrics, final_errors),
        })

    private_payload = {
        "protocol_version": "story-acceptance-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": check["model"],
        "real_case_count": len(records),
        "cases": records,
    }
    PRIVATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUTPUT.write_text(
        json.dumps(private_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    public_payload = {
        "protocol_version": "story-acceptance-v1",
        "real_case_count": len(records),
        "all_constraint_checks_passed": all(record["passed"] for record in records),
        "all_model_outputs_accepted": all(record["story_metadata"]["status"] != "fallback" for record in records),
        "fallback_count": sum(record["story_metadata"]["status"] == "fallback" for record in records),
        "case_results": [{
            "case_id": record["case"]["case_id"],
            "source_kind": record["case"]["source_kind"],
            "language": record["case"]["language"],
            "status": record["story_metadata"]["status"],
            "constraint_checks_passed": record["passed"],
            "model_output_accepted": record["story_metadata"]["status"] != "fallback",
        } for record in records],
        "private_acceptance_sha256": _sha256(PRIVATE_OUTPUT),
        "private_content_committed": False,
    }
    PUBLIC_OUTPUT.write_text(
        json.dumps(public_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(public_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
