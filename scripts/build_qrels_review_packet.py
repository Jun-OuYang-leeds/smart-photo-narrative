"""Render only qrels positives/ambiguous items for the user's terminal review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH, PHOTOS_DIR  # noqa: E402
from evaluation import load_qrels  # noqa: E402
from retrieval_engine import MultimodalRetriever  # noqa: E402
from storage import PhotoStorage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, default=ROOT / "evaluation" / "private" / "qrels_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "private" / "human_review")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    judgments = load_qrels(args.qrels)
    storage = PhotoStorage(APP_DB_PATH)
    retriever = MultimodalRetriever(storage=storage)
    positive_ids = list(dict.fromkeys(
        photo_id for judgment in judgments for photo_id in judgment.positive_relevance
    ))
    docs = retriever._documents(positive_ids)
    font = ImageFont.load_default()
    categories = sorted({item.category for item in judgments if item.category != "metadata"})
    thumb_width, row_height, label_width, max_thumbs = 165, 190, 380, 8
    review_index = []
    for category in categories:
        items = [item for item in judgments if item.category == category]
        sheet = Image.new("RGB", (label_width + max_thumbs * thumb_width, len(items) * row_height), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, judgment in enumerate(items):
            y = row_index * row_height
            draw.multiline_text(
                (8, y + 8), f"{judgment.query_id}\n{judgment.query[:52]}",
                fill="black", font=font, spacing=3,
            )
            positives = sorted(judgment.positive_relevance.items(), key=lambda item: (-item[1], item[0]))
            review_index.append({
                "query_id": judgment.query_id, "query": judgment.query,
                "positives": [{"photo_id": photo_id, "grade": grade} for photo_id, grade in positives],
            })
            for column, (photo_id, grade) in enumerate(positives[:max_thumbs]):
                x = label_width + column * thumb_width
                doc = docs[photo_id]
                try:
                    with Image.open(PHOTOS_DIR / doc["relative_path"]) as source:
                        thumb = ImageOps.exif_transpose(source).convert("RGB")
                        thumb.thumbnail((thumb_width - 8, 145))
                        sheet.paste(thumb, (x + (thumb_width - thumb.width) // 2, y + 3))
                except Exception:
                    draw.text((x + 5, y + 60), "IMAGE ERROR", fill="red", font=font)
                color = "#ff8c00" if float(grade) == 1.0 else "#008000"
                draw.text(
                    (x + 4, y + 151), f"grade {int(grade)}  {photo_id[:8]}\n{Path(doc['relative_path']).name[:23]}",
                    fill=color, font=font,
                )
                draw.rectangle((x, y, x + thumb_width - 1, y + row_height - 1), outline=color, width=2)
            draw.line((0, y + row_height - 1, sheet.width, y + row_height - 1), fill="#999999")
        sheet.save(args.output / f"review_{category}.jpg", quality=90)
    metadata = [{
        "query_id": item.query_id, "query": item.query, "filters": dict(item.filters),
        "positive_count": len(item.positive_relevance),
    } for item in judgments if item.category == "metadata"]
    (args.output / "review_index.json").write_text(
        json.dumps({"visual_queries": review_index, "metadata_queries": metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(categories)} visual review sheets and {len(metadata)} metadata summaries")


if __name__ == "__main__":
    main()
