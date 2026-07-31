"""CLI for human-labelled multimodal retrieval ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import (  # noqa: E402
    DEFAULT_ABLATIONS,
    QrelsValidationError,
    RetrievalEvaluator,
    load_qrels,
    qrels_sha256,
    runtime_metadata,
    write_report,
)
from retrieval_engine import get_multimodal_retriever  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate A0-A4 retrieval ablations using human qrels. "
            "Accuracy is never emitted for empty or placeholder judgments."
        )
    )
    parser.add_argument("qrels", type=Path, help="Human-labelled .jsonl/.ndjson/.json qrels file")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_ablation_report.json",
        help="Destination JSON report",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10], help="Metric cutoffs")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions per query/variant")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup calls per variant")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[spec.variant_id for spec in DEFAULT_ABLATIONS],
        default=[spec.variant_id for spec in DEFAULT_ABLATIONS],
        help="Ablation variants to run",
    )
    parser.add_argument(
        "--ollama-parser",
        action="store_true",
        help=(
            "Enable optional Ollama parsing in the separate temporal mode "
            "(off by default for reproducibility)"
        ),
    )
    parser.add_argument(
        "--include-temporal-mode",
        action="store_true",
        help=(
            "Evaluate task_type=temporal_pair qrels separately with full before-after logic; "
            "this does not change A0-A4"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        judgments = load_qrels(args.qrels)
        selected = tuple(spec for spec in DEFAULT_ABLATIONS if spec.variant_id in args.variants)
        report = RetrievalEvaluator(get_multimodal_retriever()).evaluate(
            judgments,
            ks=args.ks,
            repeats=args.repeats,
            warmup=args.warmup,
            specs=selected,
            use_ollama_parser=args.ollama_parser,
            include_temporal_mode=args.include_temporal_mode,
        )
        report["qrels"] = {
            "path": str(args.qrels.resolve()),
            "sha256": qrels_sha256(args.qrels),
            "provenance": "human_supplied",
        }
        report["runtime"] = runtime_metadata()
        destination = write_report(report, args.output)
    except (QrelsValidationError, ValueError) as exc:
        print(f"Evaluation refused: {exc}", file=sys.stderr)
        return 2

    summary = {
        "report": str(destination.resolve()),
        "query_count": report["query_count"],
        "accuracy_status": report["accuracy_status"],
        "metrics": {
            variant_id: value["metrics_macro"]
            for variant_id, value in report["variants"].items()
            if value["status"] == "computed"
        },
        "optional_mode_metrics": {
            mode: (
                value["metrics_macro"]
                if value["status"] == "computed"
                else {"status": value["status"]}
            )
            for mode, value in report["optional_modes"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
