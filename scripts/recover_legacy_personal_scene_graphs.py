"""Create an explicitly labelled, hash-bound derivative of a confirmed legacy batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APP_DB_PATH, PHOTOS_DIR  # noqa: E402
from scene_graph_recovery import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_VERSION,
    recover_confirmed_legacy_personal_results,
    write_recovered_scene_graphs,
)
from storage import PhotoStorage  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a user-confirmed legacy personal Scene Graph batch without "
            "claiming that generation-time SHA-256 provenance exists."
        )
    )
    parser.add_argument("source_jsonl", type=Path)
    parser.add_argument("--album-root", type=Path, default=PHOTOS_DIR)
    parser.add_argument(
        "--batch-subdir",
        default=None,
        help=(
            "Optional path (relative to --album-root) restricting the album scan and "
            "coverage check to a single incremental batch, e.g. 'pic2'. Keys stay "
            "relative to the album root so SQLite path matching is unchanged."
        ),
    )
    parser.add_argument("--database", type=Path, default=APP_DB_PATH)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument(
        "--confirm-generated-from-current-album",
        action="store_true",
        help="Required acknowledgement that the legacy results came from the unchanged current album.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the recovered derivative and report. Without this flag the command is read-only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage = PhotoStorage(args.database)
    report = recover_confirmed_legacy_personal_results(
        args.source_jsonl,
        args.album_root,
        storage,
        dataset_id=args.dataset_id,
        confirmed_generated_from_current_album=args.confirm_generated_from_current_album,
        model_id=args.model_id,
        prompt_version=args.prompt_version,
        batch_relative_root=args.batch_subdir,
    )
    payload = report.to_dict(include_records=False)
    payload["dry_run"] = not args.apply
    payload["output_jsonl"] = str(args.output_jsonl.resolve())
    payload["report_json"] = str(args.report_json.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if report.blocking_issues:
        return 2
    if args.apply:
        write_recovered_scene_graphs(report, args.output_jsonl, args.report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
