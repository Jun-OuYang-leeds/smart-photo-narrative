"""Build pooled and independently shuffled private qrels judging packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import DEFAULT_ABLATIONS, load_qrels  # noqa: E402
from retrieval_engine import SearchFilters, get_multimodal_retriever  # noqa: E402
from config import PHOTOS_DIR  # noqa: E402


def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def result_ids(response) -> list[str]:
    return [item.photo_id for item in response.results[:20]]


def render_packet(query_id: str, pass_name: str, candidates: list[dict], output: Path) -> None:
    columns, cell_width, cell_height, per_page = 5, 250, 190, 30
    font = ImageFont.load_default()
    for offset in range(0, len(candidates), per_page):
        values = candidates[offset:offset + per_page]
        row_count = (len(values) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_width, row_count * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, candidate in enumerate(values):
            x, y = (index % columns) * cell_width, (index // columns) * cell_height
            try:
                with Image.open(PHOTOS_DIR / candidate["relative_path"]) as source:
                    thumb = ImageOps.exif_transpose(source).convert("RGB")
                    thumb.thumbnail((cell_width - 10, 145))
                    sheet.paste(thumb, (x + (cell_width - thumb.width) // 2, y + 3))
            except Exception:
                draw.text((x + 10, y + 60), "IMAGE ERROR", fill="red", font=font)
            draw.text(
                (x + 5, y + 151),
                f"J{offset + index + 1:03d}  {candidate['photo_id'][:8]}\n{Path(candidate['relative_path']).name[:34]}",
                fill="black", font=font,
            )
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#aaaaaa")
        page = offset // per_page + 1
        sheet.save(output / f"{query_id}_{pass_name}_{page:02d}.jpg", quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, default=ROOT / "evaluation" / "private" / "qrels_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "private" / "candidate_pool")
    parser.add_argument(
        "--render-sheets", action="store_true",
        help="Optionally decode source images into per-query sheets; the full-album sheets are normally faster.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    packets = args.output / "blind_sheets"
    if args.render_sheets:
        packets.mkdir(parents=True, exist_ok=True)
    judgments = load_qrels(args.qrels)
    retriever = get_multimodal_retriever()
    all_ids = retriever.storage.metadata_eligible_ids()
    docs = retriever._documents(all_ids)
    pool_records = []
    blind_records = {"pass1": [], "pass2": []}
    for number, judgment in enumerate(judgments, 1):
        rankings: dict[str, list[str]] = {}
        rankings["CLIP-only"] = result_ids(retriever.search_standard(
            judgment.query, top_k=20, enabled_channels=("clip",), filters=SearchFilters(),
        ))
        rankings["Caption-only"] = result_ids(retriever.search_standard(
            judgment.query, top_k=20, enabled_channels=("caption",), filters=SearchFilters(),
        ))
        rankings["SG-only"] = result_ids(retriever.search_standard(
            judgment.query, top_k=20, enabled_channels=("scene_graph",), filters=SearchFilters(),
        ))
        for spec in DEFAULT_ABLATIONS:
            filters = judgment.search_filters(enabled=spec.use_metadata)
            rankings[spec.variant_id] = result_ids(retriever.search_standard(
                judgment.query, top_k=20, enabled_channels=spec.channels, filters=filters,
            ))
        candidate_ids = list(dict.fromkeys(
            photo_id for values in rankings.values() for photo_id in values
        ))
        # Full-contact-sheet positives supplement any system miss.
        candidate_ids.extend(
            photo_id for photo_id in judgment.positive_relevance if photo_id not in candidate_ids
        )
        pool_records.append({
            "query_id": judgment.query_id, "query": judgment.query,
            "candidate_ids": candidate_ids, "rankings": rankings,
        })
        for pass_name in ("pass1", "pass2"):
            shuffled = list(candidate_ids)
            random.Random(stable_seed(f"{judgment.query_id}:{pass_name}")).shuffle(shuffled)
            candidates = [{
                "photo_id": photo_id,
                "relative_path": docs[photo_id]["relative_path"],
            } for photo_id in shuffled if photo_id in docs]
            blind_records[pass_name].append({
                "query_id": judgment.query_id, "query": judgment.query,
                "candidates": candidates,
            })
            if args.render_sheets:
                render_packet(judgment.query_id, pass_name, candidates, packets)
        print(f"[{number:02d}/48] {judgment.query_id}: {len(candidate_ids)} pooled candidates", flush=True)
    (args.output / "pool.json").write_text(
        json.dumps({"queries": pool_records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    for pass_name, records in blind_records.items():
        (args.output / f"blind_{pass_name}.json").write_text(
            json.dumps({"queries": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
