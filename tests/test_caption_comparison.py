from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from caption_comparison import (
    CaptionRecord,
    generate_records_for_backend,
    merge_caption_records,
    scan_image_paths,
    write_jsonl,
)


class MockBackend:
    model_type = "blip"
    model_name = "mock-blip"

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.loaded = False

    def load_model(self) -> None:
        self.loaded = True

    def generate_caption(self, image):
        if self.fail_on == getattr(image, "filename", None):
            raise RuntimeError("mock failure")
        return "mock caption"

    def close(self) -> None:
        pass


class CaptionComparisonTests(unittest.TestCase):
    def make_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path)

    def test_scan_image_paths_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_image(root / "a.jpg")
            self.make_image(root / "nested" / "b.png")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            paths = scan_image_paths(root)

            self.assertEqual([path.name for path in paths], ["a.jpg", "b.png"])

    def test_write_jsonl_has_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "blip_captions.jsonl"
            record = CaptionRecord(
                image_id="a.jpg",
                image_path="a.jpg",
                model_type="blip",
                model_name="mock",
                caption="a caption",
                status="ok",
                error="",
                elapsed_seconds=0.1,
                generated_at="2026-01-01T00:00:00+00:00",
            )

            write_jsonl([record], output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

            self.assertEqual(payload["image_id"], "a.jpg")
            self.assertEqual(payload["model_type"], "blip")
            self.assertEqual(payload["caption"], "a caption")
            self.assertIn("elapsed_seconds", payload)

    def test_merge_caption_records_creates_comparison_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_jsonl(
                [
                    CaptionRecord("a.jpg", "a.jpg", "blip", "mock-blip", "blip caption", "ok", "", 0.1, "now"),
                ],
                output_dir / "blip_captions.jsonl",
            )
            write_jsonl(
                [
                    CaptionRecord("a.jpg", "a.jpg", "blip2", "mock-blip2", "blip2 caption", "ok", "", 0.2, "now"),
                ],
                output_dir / "blip2_captions.jsonl",
            )

            csv_path = merge_caption_records(output_dir)
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["blip_caption"], "blip caption")
            self.assertEqual(rows[0]["blip2_caption"], "blip2 caption")

    def test_generation_records_errors_do_not_stop_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.jpg"
            second = root / "b.jpg"
            self.make_image(first)
            self.make_image(second)

            class FailingSecondBackend(MockBackend):
                def __init__(self):
                    super().__init__()
                    self.calls = 0

                def generate_caption(self, image):
                    self.calls += 1
                    if self.calls == 2:
                        raise RuntimeError("mock failure")
                    return "mock caption"

            records = generate_records_for_backend(FailingSecondBackend(), [first, second], root)

            self.assertEqual([record.status for record in records], ["ok", "error"])
            self.assertEqual(records[1].error, "mock failure")


if __name__ == "__main__":
    unittest.main()
