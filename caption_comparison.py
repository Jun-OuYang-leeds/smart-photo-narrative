"""
Offline caption generation helpers for comparing BLIP and BLIP2.

This module is intentionally separate from the Streamlit indexing pipeline so
comparison runs do not alter ChromaDB or the existing gallery/search behavior.
"""

from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from PIL import Image


BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_PHOTOS_DIR = BASE_DIR / "photos"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "captions"
DEFAULT_BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEFAULT_BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

JSONL_FIELDS = [
    "image_id",
    "image_path",
    "model_type",
    "model_name",
    "caption",
    "status",
    "error",
    "elapsed_seconds",
    "generated_at",
]

CSV_FIELDS = [
    "image_id",
    "image_path",
    "blip_caption",
    "blip2_caption",
    "blip_status",
    "blip2_status",
    "blip_error",
    "blip2_error",
]


@dataclass
class CaptionRecord:
    image_id: str
    image_path: str
    model_type: str
    model_name: str
    caption: str
    status: str
    error: str
    elapsed_seconds: float
    generated_at: str


class CaptionBackend(Protocol):
    model_type: str
    model_name: str

    def load_model(self) -> None:
        ...

    def generate_caption(self, image: Image.Image) -> str:
        ...

    def close(self) -> None:
        ...


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def scan_image_paths(image_dir: Path, limit: int | None = None) -> list[Path]:
    """Recursively scan supported images in a deterministic order."""
    paths = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    paths = sorted(paths)
    if limit is not None:
        return paths[: max(limit, 0)]
    return paths


def image_id_for_path(image_path: Path, image_dir: Path) -> str:
    try:
        return image_path.relative_to(image_dir).as_posix()
    except ValueError:
        return image_path.name


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def validate_caption(caption: str) -> bool:
    caption = caption.strip()
    if len(caption) < 3:
        return False

    words = caption.lower().split()
    if len(words) >= 3:
        for word in set(words):
            if len(word) >= 3 and words.count(word) / len(words) > 0.6:
                return False

    if len(caption) >= 10:
        for pattern_len in range(3, min(20, len(caption) // 3)):
            pattern = caption[:pattern_len]
            if pattern * 3 in caption:
                return False

    return True


def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BLIPCaptionBackend:
    model_type = "blip"

    def __init__(self, model_name: str = DEFAULT_BLIP_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = self.device or resolve_device()
        print(f"[BLIP] Loading {self.model_name} on {self.device}")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._torch = torch

    def generate_caption(self, image: Image.Image) -> str:
        self.load_model()
        params_list = [
            {"max_length": 30, "num_beams": 1, "do_sample": False},
            {"max_length": 50, "num_beams": 3, "do_sample": False},
            {"max_length": 40, "num_beams": 1, "do_sample": True, "temperature": 0.7},
        ]
        assert self.processor is not None and self.model is not None
        for params in params_list:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                output = self.model.generate(**inputs, **params)
            caption = self.processor.decode(output[0], skip_special_tokens=True).strip()
            if validate_caption(caption):
                return caption
        return "an image"

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class BLIP2CaptionBackend:
    model_type = "blip2"

    def __init__(self, model_name: str = DEFAULT_BLIP2_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.device = self.device or resolve_device()
        kwargs = {}
        if self.device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        else:
            print("[BLIP2] CUDA is not available; BLIP2 generation may be very slow.")

        print(f"[BLIP2] Loading {self.model_name} on {self.device}")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name, **kwargs).to(self.device)
        self.model.eval()
        self._torch = torch

    def generate_caption(self, image: Image.Image) -> str:
        self.load_model()
        assert self.processor is not None and self.model is not None
        inputs = self.processor(images=image, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to(self.device, self._torch.float16)
        else:
            inputs = inputs.to(self.device)
        with self._torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
        caption = self.processor.decode(output[0], skip_special_tokens=True).strip()
        return caption if validate_caption(caption) else "an image"

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def create_backend(model_type: str, device: str | None = None) -> CaptionBackend:
    model_type = model_type.strip().lower()
    blip_model = DEFAULT_BLIP_MODEL_NAME
    blip2_model = DEFAULT_BLIP2_MODEL_NAME
    try:
        from config import BLIP_MODEL_NAME, BLIP2_MODEL_NAME

        blip_model = BLIP_MODEL_NAME
        blip2_model = BLIP2_MODEL_NAME
    except Exception:
        pass

    if model_type == "blip":
        return BLIPCaptionBackend(model_name=blip_model, device=device)
    if model_type == "blip2":
        return BLIP2CaptionBackend(model_name=blip2_model, device=device)
    raise ValueError(f"Unsupported model type: {model_type}")


def generate_records_for_backend(
    backend: CaptionBackend,
    image_paths: Iterable[Path],
    image_dir: Path,
) -> list[CaptionRecord]:
    records: list[CaptionRecord] = []
    backend.load_model()
    for image_path in image_paths:
        started = time.perf_counter()
        status = "ok"
        caption = ""
        error = ""
        try:
            image = load_rgb_image(image_path)
            caption = backend.generate_caption(image)
            if not validate_caption(caption):
                caption = "an image"
        except Exception as exc:
            status = "error"
            error = str(exc)
        elapsed = round(time.perf_counter() - started, 4)
        records.append(
            CaptionRecord(
                image_id=image_id_for_path(image_path, image_dir),
                image_path=str(image_path),
                model_type=backend.model_type,
                model_name=backend.model_name,
                caption=caption,
                status=status,
                error=error,
                elapsed_seconds=elapsed,
                generated_at=utc_now_iso(),
            )
        )
        print(f"[{backend.model_type}] {records[-1].image_id}: {status}")
    return records


def write_jsonl(records: Iterable[CaptionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            payload = asdict(record)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(output_path: Path) -> list[dict[str, object]]:
    if not output_path.exists():
        return []
    rows = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def captions_jsonl_path(output_dir: Path, model_type: str) -> Path:
    return output_dir / f"{model_type}_captions.jsonl"


def comparison_csv_path(output_dir: Path) -> Path:
    return output_dir / "caption_comparison.csv"


def merge_caption_records(output_dir: Path, csv_path: Path | None = None) -> Path:
    blip_rows = {str(row["image_id"]): row for row in read_jsonl(captions_jsonl_path(output_dir, "blip"))}
    blip2_rows = {str(row["image_id"]): row for row in read_jsonl(captions_jsonl_path(output_dir, "blip2"))}
    all_ids = sorted(set(blip_rows) | set(blip2_rows))
    csv_path = csv_path or comparison_csv_path(output_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for image_id in all_ids:
            blip = blip_rows.get(image_id, {})
            blip2 = blip2_rows.get(image_id, {})
            writer.writerow(
                {
                    "image_id": image_id,
                    "image_path": blip.get("image_path") or blip2.get("image_path") or "",
                    "blip_caption": blip.get("caption", ""),
                    "blip2_caption": blip2.get("caption", ""),
                    "blip_status": blip.get("status", ""),
                    "blip2_status": blip2.get("status", ""),
                    "blip_error": blip.get("error", ""),
                    "blip2_error": blip2.get("error", ""),
                }
            )
    return csv_path


def run_caption_comparison(
    model_types: list[str],
    image_dir: Path = DEFAULT_PHOTOS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    overwrite: bool = False,
    device: str | None = None,
) -> dict[str, Path]:
    image_paths = scan_image_paths(image_dir, limit=limit)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    written: dict[str, Path] = {}
    for model_type in model_types:
        normalized = model_type.strip().lower()
        output_path = captions_jsonl_path(output_dir, normalized)
        if output_path.exists() and not overwrite:
            print(f"[{normalized}] Reusing existing file: {output_path}")
            written[normalized] = output_path
            continue

        backend = create_backend(normalized, device=device)
        try:
            records = generate_records_for_backend(backend, image_paths, image_dir)
            write_jsonl(records, output_path)
            written[normalized] = output_path
        finally:
            backend.close()

    written["comparison"] = merge_caption_records(output_dir)
    return written
