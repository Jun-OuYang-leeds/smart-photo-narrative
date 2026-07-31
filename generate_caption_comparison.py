"""
Generate BLIP and BLIP2 caption files for offline comparison.

Examples:
    python generate_caption_comparison.py --models blip --limit 3
    python generate_caption_comparison.py --models blip2 --limit 1
    python generate_caption_comparison.py --models blip,blip2 --limit 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from caption_comparison import DEFAULT_OUTPUT_DIR, DEFAULT_PHOTOS_DIR, run_caption_comparison


def parse_models(value: str) -> list[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("At least one model is required")
    unknown = [item for item in models if item not in {"blip", "blip2"}]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unsupported model(s): {', '.join(unknown)}")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate BLIP/BLIP2 caption comparison files.")
    parser.add_argument("--models", type=parse_models, default=parse_models("blip,blip2"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing JSONL files.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = run_caption_comparison(
        model_types=args.models,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        device=args.device,
    )
    print("\nGenerated files:")
    for key, path in written.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
