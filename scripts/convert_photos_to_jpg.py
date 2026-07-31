"""Safely normalize a photo directory to JPG without overwriting originals.

The command works in two phases:

1. Convert every non-JPEG image into a staging directory and verify dimensions,
   visual fidelity, EXIF dates, GPS, ICC, and XMP metadata.
2. Only after every staged file passes verification, move the source files into
   a rollback backup and place the verified JPG files in the photo directory.

Existing .jpg files are never re-encoded. A JSON manifest and Markdown report
are retained alongside the original non-JPEG files.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import sys
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, ImageChops, ImageOps, ImageStat
from pillow_heif import register_heif_opener


register_heif_opener()

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".heic",
    ".heif",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}

EXIF_IFD = 34665
GPS_IFD = 34853


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def raw_profile_exif(image: Image.Image) -> bytes | None:
    """Decode ImageMagick/Picasa's PNG `Raw profile type APP1` text chunk."""
    text = getattr(image, "text", {}).get("Raw profile type APP1")
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return None

    hex_payload = "".join(re.findall(r"[0-9A-Fa-f]+", lines[-1]))
    try:
        tiff_payload = bytes.fromhex(hex_payload)
    except ValueError:
        return None

    if not tiff_payload.startswith((b"II*\x00", b"MM\x00*")):
        return None
    return b"Exif\x00\x00" + tiff_payload


def effective_exif_bytes(image: Image.Image) -> bytes | None:
    payload = image.info.get("exif")
    if payload:
        return bytes(payload)
    return raw_profile_exif(image)


def load_exif(image: Image.Image, payload: bytes | None = None) -> Image.Exif:
    exif = image.getexif()
    if len(exif) or not payload:
        return exif

    parsed = Image.Exif()
    try:
        parsed.load(payload)
    except Exception:
        if payload.startswith(b"Exif\x00\x00"):
            parsed.load(payload[6:])
        else:
            raise
    return parsed


def get_ifd(exif: Image.Exif, ifd_id: int) -> dict[int, Any]:
    try:
        return dict(exif.get_ifd(ifd_id))
    except Exception:
        return {}


def exif_value(exif: Image.Exif, tag_id: int) -> Any:
    if tag_id in exif:
        return exif.get(tag_id)
    return get_ifd(exif, EXIF_IFD).get(tag_id)


def metadata_summary(image: Image.Image, exif_payload: bytes | None = None) -> dict[str, Any]:
    exif_payload = exif_payload or effective_exif_bytes(image)
    exif = load_exif(image, exif_payload)
    gps = get_ifd(exif, GPS_IFD)
    exif_ifd = get_ifd(exif, EXIF_IFD)

    return {
        "format": image.format,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "frames": int(getattr(image, "n_frames", 1)),
        "has_alpha": "A" in image.getbands() or "transparency" in image.info,
        "bit_depth": json_safe(image.info.get("bit_depth")),
        "exif_present": bool(exif_payload),
        "exif_sha256": sha256_bytes(exif_payload),
        "icc_sha256": sha256_bytes(image.info.get("icc_profile")),
        "xmp_sha256": sha256_bytes(image.info.get("xmp")),
        "make": json_safe(exif_value(exif, 271)),
        "model": json_safe(exif_value(exif, 272)),
        "software": json_safe(exif_value(exif, 305)),
        "orientation": json_safe(exif_value(exif, 274)),
        "datetime": json_safe(exif_value(exif, 306)),
        "datetime_original": json_safe(exif_value(exif, 36867)),
        "datetime_digitized": json_safe(exif_value(exif, 36868)),
        "offset_time": json_safe(exif_value(exif, 36880)),
        "offset_time_original": json_safe(exif_value(exif, 36881)),
        "offset_time_digitized": json_safe(exif_value(exif, 36882)),
        "lens_make": json_safe(exif_value(exif, 42035)),
        "lens_model": json_safe(exif_value(exif, 42036)),
        "maker_note_sha256": sha256_bytes(exif_ifd.get(37500)),
        "gps": json_safe(gps),
    }


def file_snapshot(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    with Image.open(path) as image:
        image.load()
        image_metadata = metadata_summary(image)

    return {
        "relative_path": path.relative_to(root).as_posix(),
        "suffix": path.suffix,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "created_ns": stat.st_ctime_ns,
        "accessed_ns": stat.st_atime_ns,
        "modified_ns": stat.st_mtime_ns,
        "image": image_metadata,
    }


def to_rgb(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, background)
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return image.convert("RGB")


def calculate_psnr(reference: Image.Image, converted: Image.Image) -> float:
    difference = ImageChops.difference(reference, converted)
    channel_rms = ImageStat.Stat(difference).rms
    rms = math.sqrt(sum(value * value for value in channel_rms) / max(len(channel_rms), 1))
    if rms == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / rms)


def verify_metadata(source: dict[str, Any], output: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = [
        "make",
        "model",
        "software",
        "datetime",
        "datetime_original",
        "datetime_digitized",
        "offset_time",
        "offset_time_original",
        "offset_time_digitized",
        "lens_make",
        "lens_model",
        "maker_note_sha256",
        "gps",
        "icc_sha256",
        "xmp_sha256",
    ]

    for field in required_fields:
        source_value = source.get(field)
        if source_value not in (None, {}, [], "") and output.get(field) != source_value:
            errors.append(f"metadata mismatch: {field}")
    return errors


def unique_target(relative_path: Path, reserved: set[str], source_suffix: str) -> Path:
    candidate = relative_path.with_suffix(".jpg")
    if candidate.as_posix().casefold() not in reserved:
        reserved.add(candidate.as_posix().casefold())
        return candidate

    base = f"{relative_path.stem}__from_{source_suffix.lstrip('.').lower()}"
    counter = 1
    while True:
        suffix = "" if counter == 1 else f"_{counter}"
        candidate = relative_path.with_name(f"{base}{suffix}.jpg")
        key = candidate.as_posix().casefold()
        if key not in reserved:
            reserved.add(key)
            return candidate
        counter += 1


def build_plan(photos_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    images = sorted(
        (path for path in photos_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(photos_dir).as_posix().casefold(),
    )
    existing_jpg = [path for path in images if path.suffix.lower() == ".jpg"]
    reserved = {path.relative_to(photos_dir).as_posix().casefold() for path in existing_jpg}

    plan: list[dict[str, str]] = []
    for source in images:
        if source.suffix.lower() == ".jpg":
            continue
        relative = source.relative_to(photos_dir)
        target = unique_target(relative, reserved, source.suffix)
        plan.append(
            {
                "source": relative.as_posix(),
                "target": target.as_posix(),
                "operation": "copy_without_reencode" if source.suffix.lower() == ".jpeg" else "convert",
            }
        )
    return existing_jpg, plan


def save_jpeg(
    source: Path,
    destination: Path,
    quality: int,
    background: tuple[int, int, int],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        image.load()
        original_exif = effective_exif_bytes(image)
        if original_exif and not image.info.get("exif"):
            image.info["exif"] = original_exif

        source_metadata = metadata_summary(image, original_exif)
        oriented = ImageOps.exif_transpose(image)
        reference = to_rgb(oriented, background)

        save_options: dict[str, Any] = {
            "quality": quality,
            "subsampling": 0,
            "optimize": True,
            "progressive": True,
        }
        for key in ("exif", "icc_profile", "xmp"):
            value = oriented.info.get(key)
            if value:
                save_options[key] = value

        if original_exif and "exif" not in save_options:
            save_options["exif"] = original_exif
        if image.info.get("icc_profile") and "icc_profile" not in save_options:
            save_options["icc_profile"] = image.info["icc_profile"]
        if image.info.get("xmp") and "xmp" not in save_options:
            save_options["xmp"] = image.info["xmp"]
        if image.info.get("dpi"):
            save_options["dpi"] = image.info["dpi"]

        reference.save(destination, "JPEG", **save_options)
        shutil.copystat(source, destination)

        with Image.open(destination) as output_image:
            output_image.load()
            output_metadata = metadata_summary(output_image)
            output_rgb = output_image.convert("RGB")
            psnr = calculate_psnr(reference, output_rgb)

        if output_metadata["width"] != reference.width or output_metadata["height"] != reference.height:
            raise RuntimeError("output dimensions differ from the display-oriented source")

        metadata_errors = verify_metadata(source_metadata, output_metadata)
        if metadata_errors:
            raise RuntimeError("; ".join(metadata_errors))

        if psnr < 35.0:
            raise RuntimeError(f"visual quality verification failed: PSNR={psnr:.2f} dB")

        return source_metadata, output_metadata, psnr


def copy_without_reencode(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def filetime_from_ns(timestamp_ns: int) -> wintypes.FILETIME:
    ticks = timestamp_ns // 100 + 116444736000000000
    return wintypes.FILETIME(ticks & 0xFFFFFFFF, ticks >> 32)


def apply_windows_file_times(path: Path, source_stat: os.stat_result) -> None:
    if os.name != "nt":
        os.utime(path, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE

    set_file_time = kernel32.SetFileTime
    set_file_time.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    set_file_time.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        0x0100,  # FILE_WRITE_ATTRIBUTES
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,  # OPEN_EXISTING
        0x00000080,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())

    creation = filetime_from_ns(source_stat.st_ctime_ns)
    access = filetime_from_ns(source_stat.st_atime_ns)
    modified = filetime_from_ns(source_stat.st_mtime_ns)
    try:
        if not set_file_time(handle, ctypes.byref(creation), ctypes.byref(access), ctypes.byref(modified)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Photo conversion report",
        "",
        f"- Status: **{report['status']}**",
        f"- Started: {report['started_at']}",
        f"- Finished: {report['finished_at']}",
        f"- Photo directory: `{report['photos_dir']}`",
        f"- Initial images: {report['initial_images']}",
        f"- Existing JPG files left byte-identical: {report['existing_jpg_unchanged']}",
        f"- Converted or normalized files: {report['converted']}",
        f"- Final JPG files: {report['final_jpg']}",
        f"- Original non-JPG backup files: {report['backup_files']}",
        f"- Source files with EXIF capture dates: {report['source_datetime_count']}",
        f"- Source files with GPS metadata: {report['source_gps_count']}",
        f"- EXIF date checks passed: {report['datetime_checks_passed']}",
        f"- GPS checks passed: {report['gps_checks_passed']}",
        f"- Images originally containing alpha: {report['alpha_images']}",
        f"- Minimum visual PSNR: {report['minimum_psnr_db']} dB",
        "",
        "Original non-JPG files are retained in `original_non_jpeg/`. Existing JPG files were not re-encoded.",
        "The manifest records SHA-256 hashes, dimensions, filesystem timestamps, and metadata verification results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_safe_paths(photos_dir: Path, run_dir: Path) -> None:
    photos_dir = photos_dir.resolve()
    run_dir = run_dir.resolve()
    if not photos_dir.is_dir():
        raise FileNotFoundError(f"photo directory does not exist: {photos_dir}")
    if photos_dir == run_dir or photos_dir in run_dir.parents:
        raise ValueError("run directory must not be inside the photo directory")
    if run_dir in photos_dir.parents:
        raise ValueError("photo directory must not be inside the run directory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--quality", type=int, default=98)
    parser.add_argument("--background", default="255,255,255")
    parser.add_argument("--execute", action="store_true", help="perform the staged conversion and commit")
    args = parser.parse_args()

    photos_dir = args.photos_dir.resolve()
    run_dir = args.run_dir.resolve()
    ensure_safe_paths(photos_dir, run_dir)

    if not 90 <= args.quality <= 100:
        raise ValueError("quality must be between 90 and 100")
    background_values = tuple(int(item) for item in args.background.split(","))
    if len(background_values) != 3 or any(item < 0 or item > 255 for item in background_values):
        raise ValueError("background must be R,G,B with values from 0 to 255")
    background = (background_values[0], background_values[1], background_values[2])

    existing_jpg, plan = build_plan(photos_dir)
    all_images = existing_jpg + [photos_dir / item["source"] for item in plan]
    summary = {
        "photos_dir": str(photos_dir),
        "images": len(all_images),
        "existing_jpg": len(existing_jpg),
        "to_convert_or_normalize": len(plan),
        "collisions_resolved": sum(Path(item["source"]).with_suffix(".jpg").as_posix() != item["target"] for item in plan),
    }

    if not args.execute:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    staging_dir = run_dir / "staging"
    backup_dir = run_dir / "original_non_jpeg"
    run_dir.mkdir(parents=True)
    staging_dir.mkdir()

    started_at = datetime.now(timezone.utc).isoformat()
    existing_hashes = {
        path.relative_to(photos_dir).as_posix(): sha256_file(path)
        for path in existing_jpg
    }
    write_json(run_dir / "conversion_plan.json", {"summary": summary, "plan": plan})
    write_json(run_dir / "existing_jpg_hashes_before.json", existing_hashes)

    records: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(plan, 1):
            source = photos_dir / item["source"]
            target = photos_dir / item["target"]
            staged = staging_dir / item["target"]
            backup = backup_dir / item["source"]
            source_snapshot = file_snapshot(source, photos_dir)

            print(f"[{index}/{len(plan)}] {item['source']} -> {item['target']}", flush=True)
            if item["operation"] == "copy_without_reencode":
                copy_without_reencode(source, staged)
                with Image.open(source) as source_image, Image.open(staged) as output_image:
                    source_metadata = metadata_summary(source_image)
                    output_metadata = metadata_summary(output_image)
                psnr = float("inf")
            else:
                source_metadata, output_metadata, psnr = save_jpeg(
                    source,
                    staged,
                    args.quality,
                    background,
                )

            staged_snapshot = file_snapshot(staged, staging_dir)
            records.append(
                {
                    **item,
                    "source_absolute": str(source),
                    "target_absolute": str(target),
                    "backup_absolute": str(backup),
                    "source_snapshot": source_snapshot,
                    "staged_snapshot": staged_snapshot,
                    "source_metadata": source_metadata,
                    "output_metadata": output_metadata,
                    "psnr_db": None if math.isinf(psnr) else round(psnr, 4),
                    "status": "staged_and_verified",
                }
            )
    except Exception as error:
        write_json(
            run_dir / "conversion_failed.json",
            {"error": repr(error), "records": records, "plan": plan},
        )
        print(f"Conversion aborted before commit: {error}", file=sys.stderr)
        return 1

    write_json(run_dir / "manifest_staged.json", records)

    committed: list[dict[str, Any]] = []
    try:
        for record in records:
            source = Path(record["source_absolute"])
            target = Path(record["target_absolute"])
            backup = Path(record["backup_absolute"])
            staged = staging_dir / record["target"]

            if target.exists():
                raise FileExistsError(f"target appeared after planning: {target}")
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(source), str(backup))
            shutil.move(str(staged), str(target))
            shutil.copystat(backup, target)
            apply_windows_file_times(target, backup.stat())
            record["status"] = "committed"
            committed.append(record)
    except Exception as error:
        rollback_errors: list[str] = []
        for record in reversed(committed):
            source = Path(record["source_absolute"])
            target = Path(record["target_absolute"])
            backup = Path(record["backup_absolute"])
            try:
                if target.exists():
                    staged = staging_dir / record["target"]
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(target), str(staged))
                if backup.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(backup), str(source))
            except Exception as rollback_error:
                rollback_errors.append(f"{record['source']}: {rollback_error!r}")

        write_json(
            run_dir / "commit_failed.json",
            {"error": repr(error), "rollback_errors": rollback_errors, "records": records},
        )
        print(f"Commit failed and rollback was attempted: {error}", file=sys.stderr)
        if rollback_errors:
            print("Rollback errors: " + "; ".join(rollback_errors), file=sys.stderr)
        return 1

    final_images = sorted(
        path for path in photos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(final_images) != len(all_images):
        raise RuntimeError(f"final image count mismatch: {len(final_images)} != {len(all_images)}")
    if any(path.suffix.lower() != ".jpg" for path in final_images):
        raise RuntimeError("non-JPG images remain in the photo directory")

    for path in existing_jpg:
        relative = path.relative_to(photos_dir).as_posix()
        if sha256_file(path) != existing_hashes[relative]:
            raise RuntimeError(f"existing JPG changed unexpectedly: {relative}")

    for record in records:
        source = Path(record["source_absolute"])
        target = Path(record["target_absolute"])
        backup = Path(record["backup_absolute"])
        if source.exists() or not target.exists() or not backup.exists():
            raise RuntimeError(f"commit path verification failed: {record['source']}")
        if sha256_file(backup) != record["source_snapshot"]["sha256"]:
            raise RuntimeError(f"backup hash mismatch: {record['source']}")
        if sha256_file(target) != record["staged_snapshot"]["sha256"]:
            raise RuntimeError(f"output hash mismatch after commit: {record['target']}")

        with Image.open(backup) as source_image, Image.open(target) as target_image:
            source_meta = metadata_summary(source_image)
            output_meta = metadata_summary(target_image)
        metadata_errors = verify_metadata(source_meta, output_meta)
        if metadata_errors:
            raise RuntimeError(f"post-commit metadata mismatch {record['target']}: {metadata_errors}")

        if target.stat().st_mtime_ns != backup.stat().st_mtime_ns:
            raise RuntimeError(f"filesystem modified time mismatch: {record['target']}")
        if backup.stat().st_ctime_ns != record["source_snapshot"]["created_ns"]:
            raise RuntimeError(f"backup creation time mismatch: {record['source']}")
        if target.stat().st_ctime_ns != backup.stat().st_ctime_ns:
            raise RuntimeError(f"output creation time mismatch: {record['target']}")

        record["final_output_sha256"] = sha256_file(target)
        record["backup_sha256"] = sha256_file(backup)
        record["status"] = "final_verified"

    final_hashes = {
        path.relative_to(photos_dir).as_posix(): sha256_file(path)
        for path in existing_jpg
    }
    write_json(run_dir / "existing_jpg_hashes_after.json", final_hashes)
    write_json(run_dir / "manifest_final.json", records)

    source_datetime_count = sum(
        bool(record["source_metadata"].get("datetime_original") or record["source_metadata"].get("datetime"))
        for record in records
    )
    source_gps_count = sum(bool(record["source_metadata"].get("gps")) for record in records)
    alpha_images = sum(bool(record["source_metadata"].get("has_alpha")) for record in records)
    finite_psnr = [record["psnr_db"] for record in records if record["psnr_db"] is not None]

    report = {
        "status": "success",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "photos_dir": str(photos_dir),
        "initial_images": len(all_images),
        "existing_jpg_unchanged": len(existing_jpg),
        "converted": len(records),
        "final_jpg": len(final_images),
        "backup_files": sum(1 for path in backup_dir.rglob("*") if path.is_file()),
        "source_datetime_count": source_datetime_count,
        "source_gps_count": source_gps_count,
        "datetime_checks_passed": source_datetime_count,
        "gps_checks_passed": source_gps_count,
        "alpha_images": alpha_images,
        "minimum_psnr_db": round(min(finite_psnr), 4) if finite_psnr else "lossless-copy",
    }
    write_json(run_dir / "conversion_report.json", report)
    write_report(run_dir / "conversion_report.md", report)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
