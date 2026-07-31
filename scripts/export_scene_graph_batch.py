"""Export a hash-bound JPEG batch for offline/AIRE Scene Graph generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_graph_io import export_scene_graph_batch  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export byte-identical JPEGs as photo_id.jpg plus a SHA-256 manifest. "
            "Run the existing JPG normaliser first for HEIC/PNG inputs."
        )
    )
    parser.add_argument("--album-root", required=True, help="Local JPEG album directory.")
    parser.add_argument("--export-dir", required=True, help="Batch output directory.")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable dataset label, for example personal-main-v1 or geograph-v1.",
    )
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export_scene_graph_batch(
        album_root=args.album_root,
        export_dir=args.export_dir,
        dataset_id=args.dataset_id,
        manifest_name=args.manifest_name,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
