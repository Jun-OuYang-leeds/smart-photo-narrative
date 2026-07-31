"""Build private, full-album contact sheets for system-blind qrels authoring."""

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
from storage import PhotoStorage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "private" / "contact_sheets")
    parser.add_argument("--per-page", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage = PhotoStorage(APP_DB_PATH)
    with storage.transaction(write=False) as connection:
        rows = connection.execute(
            """
            SELECT photo_id, relative_path, captured_at, date_local, timestamp_source,
                   timestamp_confidence, location, image_width, image_height
            FROM photos
            ORDER BY captured_at_sort IS NULL, captured_at_sort, relative_path COLLATE NOCASE
            """
        ).fetchall()
    records = [dict(row) for row in rows]
    for index, record in enumerate(records, 1):
        record["ordinal"] = index
    (args.output_dir / "catalog.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    columns, cell_width, cell_height = 5, 280, 220
    font = ImageFont.load_default()
    for offset in range(0, len(records), args.per_page):
        page_records = records[offset:offset + args.per_page]
        rows_count = (len(page_records) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for local_index, record in enumerate(page_records):
            column, row = local_index % columns, local_index // columns
            x, y = column * cell_width, row * cell_height
            path = PHOTOS_DIR / record["relative_path"]
            try:
                with Image.open(path) as image:
                    thumb = ImageOps.exif_transpose(image).convert("RGB")
                    thumb.thumbnail((cell_width - 12, 165))
                    image_x = x + (cell_width - thumb.width) // 2
                    sheet.paste(thumb, (image_x, y + 4))
            except Exception:
                draw.rectangle((x + 5, y + 5, x + cell_width - 5, y + 165), outline="red")
                draw.text((x + 12, y + 70), "IMAGE ERROR", fill="red", font=font)
            label = (
                f"#{record['ordinal']:03d} {record['photo_id'][:8]}  {record['date_local'] or '?'}\n"
                f"{Path(record['relative_path']).name[:38]}"
            )
            draw.multiline_text((x + 5, y + 172), label, fill="black", font=font, spacing=2)
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#bbbbbb")
        page = offset // args.per_page + 1
        sheet.save(args.output_dir / f"album_contact_{page:02d}.jpg", quality=90)
    print(f"Wrote {len(records)} catalog rows and {(len(records) + args.per_page - 1) // args.per_page} sheets")


if __name__ == "__main__":
    main()
