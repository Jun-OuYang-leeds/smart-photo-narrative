"""Dry-run and apply hash-verified Scene Graph JSONL results."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_graph_io import (  # noqa: E402
    audit_scene_graph_results,
    write_indexable_scene_graphs,
)


def _expand_result_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value)) if any(token in value for token in "*?[") else [value]
        paths.extend(Path(match) for match in matches)
    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduplicated.append(resolved)
    return deduplicated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Scene Graph results using dataset_id + photo_id + SHA-256 + local bytes. "
            "The default is dry-run and never changes an index."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--album-root", required=True)
    parser.add_argument(
        "--results",
        required=True,
        nargs="+",
        help="One or more JSONL files; quoted glob patterns are supported.",
    )
    parser.add_argument(
        "--legacy-root",
        default=None,
        help=(
            "Explicit original Geograph image root for legacy SnapSeek JSONL. "
            "Legacy paths are resolved exactly and hashed on site; no basename fallback."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write only verified, non-empty remote results to --output-jsonl.",
    )
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--report-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.apply and not args.output_jsonl:
        raise SystemExit("--apply requires --output-jsonl")
    if not args.apply and args.output_jsonl:
        raise SystemExit("--output-jsonl is only written with explicit --apply")
    result_paths = _expand_result_paths(args.results)
    if not result_paths:
        raise SystemExit("--results did not match any files")

    report = audit_scene_graph_results(
        manifest_path=args.manifest,
        result_paths=result_paths,
        album_root=args.album_root,
        legacy_root=args.legacy_root,
        dry_run=not args.apply,
    )
    if args.apply:
        write_indexable_scene_graphs(report, args.output_jsonl)

    payload = report.to_dict(include_records=False, include_issues=True)
    if args.apply:
        payload["output_jsonl"] = str(Path(args.output_jsonl).expanduser().resolve())
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
