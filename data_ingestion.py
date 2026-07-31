"""
Smart Photo Narrative - Data Ingestion & Preprocessing Module
====================
Features:
1. Image loading (skips corrupted files)
2. EXIF data extraction (datetime, GPS)
3. GPS coordinate conversion (DMS -> decimal)
4. Tiered reverse geocoding (offline-first + online fallback)
"""

import hashlib
import os
import re
import exifread
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
from functools import lru_cache
from dataclasses import dataclass, asdict

import reverse_geocoder as rg
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from tqdm import tqdm

from config import (
    PHOTOS_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    NOMINATIM_USER_AGENT,
    GEOCODER_CACHE_SIZE
)


# ==================== Data Structures ====================
@dataclass
class PhotoMetadata:
    """Photo metadata structure"""
    file_path: str              # relative path (used as ID)
    absolute_path: str          # absolute path
    photo_id: Optional[str] = None       # stable UUID assigned by SQLite
    content_sha256: str = ""             # file identity / cloud round-trip guard
    mtime_ns: int = 0
    datetime_original: Optional[str] = None  # capture datetime (ISO format)
    captured_at_sort: Optional[int] = None   # sortable naive-local seconds
    date_local: Optional[str] = None
    timestamp_source: str = "missing"       # exif_original|exif_digitized|filename|mtime
    timestamp_confidence: str = "low"       # high|medium|low
    location: Optional[str] = None           # reverse geocoding result
    gps_coords: Optional[Tuple[float, float]] = None  # (latitude, longitude)
    file_size: int = 0          # file size in bytes
    image_width: int = 0        # image width
    image_height: int = 0       # image height
    error: Optional[str] = None  # error message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# ==================== Image Loader ====================
def load_image(image_path: Path) -> Optional[Image.Image]:
    """
    Safely load an image file.
    Skips corrupted or unsupported files.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB (handles RGBA, grayscale, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            return img.copy()  # return a copy to avoid file handle issues
    except UnidentifiedImageError:
        print(f"[Warning] Unrecognized image format: {image_path}")
        return None
    except Exception as e:
        print(f"[Error] Failed to load image {image_path}: {e}")
        return None


def scan_photos_directory(photos_dir: Path = PHOTOS_DIR) -> List[Path]:
    """
    Scan a directory for all supported image files.
    Returns a list of file paths.
    """
    image_files = []

    if not photos_dir.exists():
        print(f"[Warning] Photo directory does not exist: {photos_dir}")
        return image_files

    for root, _, files in os.walk(photos_dir):
        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.append(Path(root) / file)

    print(f"[Info] Scan complete, found {len(image_files)} photos")
    return sorted(image_files)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the photo into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ==================== EXIF Extractor ====================
def extract_exif_data(image_path: Path) -> Dict[str, Any]:
    """
    Extract EXIF data from an image.
    Returns: {
        'datetime_original': str or None,
        'gps_coords': (lat, lon) or None
    }
    """
    exif_data = {
        'datetime_original': None,
        'gps_coords': None
    }

    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        # Prefer capture time, then digitized time, then the generic EXIF time.
        for tag_name, source in [
            ('EXIF DateTimeOriginal', 'exif_original'),
            ('EXIF DateTimeDigitized', 'exif_digitized'),
            ('Image DateTime', 'exif_datetime'),
        ]:
            datetime_tag = tags.get(tag_name)
            if datetime_tag:
                parsed = parse_exif_datetime(str(datetime_tag))
                if parsed:
                    exif_data['datetime_original'] = parsed
                    exif_data['timestamp_source'] = source
                    break

        # Extract GPS coordinates
        gps_coords = extract_gps_coords(tags)
        if gps_coords:
            exif_data['gps_coords'] = gps_coords

    except Exception as e:
        print(f"[Warning] Failed to extract EXIF {image_path}: {e}")

    return exif_data


def parse_exif_datetime(datetime_str: str) -> Optional[str]:
    """
    Parse EXIF datetime format to ISO format.
    Input: '2024:01:15 14:30:25'
    Output: '2024-01-15T14:30:25'
    """
    try:
        # EXIF format: YYYY:MM:DD HH:MM:SS
        dt = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
        return dt.isoformat()
    except ValueError:
        # Try other common formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        return None


_FILENAME_DATETIME_PATTERNS = (
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})[_ -]?(?P<time>\d{6})"),
    re.compile(r"(?P<date>\d{8})[_ -]?(?P<time>\d{6})"),
)


def parse_filename_datetime(filename: str) -> Optional[str]:
    """Parse common camera/screenshot timestamps as a medium-confidence fallback."""
    for pattern in _FILENAME_DATETIME_PATTERNS:
        match = pattern.search(filename)
        if not match:
            continue
        date_part = match.group("date").replace("-", "")
        value = f"{date_part}{match.group('time')}"
        try:
            return datetime.strptime(value, "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            continue
    return None


def datetime_sort_value(value: Optional[str]) -> Optional[int]:
    """Convert a naive local ISO time to a timezone-independent ordering integer."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value).replace(tzinfo=None)
        return int((parsed - datetime(1970, 1, 1)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


# ==================== GPS Coordinate Processing ====================
def extract_gps_coords(tags: Dict) -> Optional[Tuple[float, float]]:
    """
    Extract GPS coordinates from EXIF tags.
    Returns: (latitude, longitude) in decimal format
    """
    try:
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lat = tags.get('GPS GPSLatitude')
        lon_ref = tags.get('GPS GPSLongitudeRef')
        lon = tags.get('GPS GPSLongitude')

        if not all([lat_ref, lat, lon_ref, lon]):
            return None

        # Convert DMS to decimal
        lat_decimal = dms_to_decimal(lat, lat_ref)
        lon_decimal = dms_to_decimal(lon, lon_ref)

        if lat_decimal is None or lon_decimal is None:
            return None

        return (lat_decimal, lon_decimal)

    except Exception as e:
        print(f"[Warning] Failed to extract GPS coordinates: {e}")
        return None


def dms_to_decimal(dms_value, ref_value) -> Optional[float]:
    """
    Convert DMS (degrees, minutes, seconds) format to decimal.
    EXIF GPS format: [degrees, minutes, seconds] (each a Ratio object)
    ref: 'N'/'S' or 'E'/'W'
    """
    try:
        # Handle Ratio objects returned by exifread
        degrees = float(dms_value.values[0].num) / float(dms_value.values[0].den)
        minutes = float(dms_value.values[1].num) / float(dms_value.values[1].den)
        seconds = float(dms_value.values[2].num) / float(dms_value.values[2].den)

        decimal = degrees + minutes / 60 + seconds / 3600

        # Southern latitude and western longitude are negative
        ref_str = str(ref_value).upper()
        if ref_str in ['S', 'W']:
            decimal = -decimal

        return round(decimal, 6)

    except (AttributeError, IndexError, TypeError, ZeroDivisionError) as e:
        print(f"[Warning] DMS conversion failed: {e}")
        return None


# ==================== Tiered Reverse Geocoding ====================
class GeocodingService:
    """
    Tiered reverse geocoding service.
    Level 1: reverse_geocoder (offline, fast, city-level)
    Level 2: geopy Nominatim (online, detailed, with cache)
    """

    def __init__(self):
        self._nominatim = None

    @property
    def nominatim(self):
        """Lazy-initialize Nominatim"""
        if self._nominatim is None:
            self._nominatim = Nominatim(
                user_agent=NOMINATIM_USER_AGENT,
                timeout=10
            )
        return self._nominatim

    def reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
        """
        Tiered reverse geocoding.
        1. Try offline library for city-level location first
        2. Fall back to Nominatim for detailed address (with cache)
        """
        # Level 1: offline fast query (city-level)
        location_str = self._offline_reverse(lat, lon)
        if location_str:
            return location_str

        # Level 2: online detailed query
        return self._online_reverse(lat, lon)

    def _offline_reverse(self, lat: float, lon: float) -> Optional[str]:
        """
        Fast city-level lookup using the reverse_geocoder offline library.
        """
        try:
            result = rg.search((lat, lon), mode=1)
            if result:
                # Return city, region, country
                city = result[0].get('name', '')
                admin = result[0].get('admin1', '')
                cc = result[0].get('cc', '')

                parts = [p for p in [city, admin, cc] if p]
                return ', '.join(parts) if parts else None
        except Exception as e:
            print(f"[Warning] Offline geocoding failed: {e}")
        return None

    @lru_cache(maxsize=GEOCODER_CACHE_SIZE)
    def _online_reverse(self, lat: float, lon: float) -> Optional[str]:
        """
        Detailed address lookup via Nominatim online service.
        LRU-cached to avoid excessive requests.
        """
        try:
            location = self.nominatim.reverse((lat, lon), language='en')
            if location and location.address:
                return location.address
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"[Warning] Online geocoding timed out: {e}")
        except Exception as e:
            print(f"[Warning] Online geocoding failed: {e}")
        return None


# Global geocoding service singleton
_geocoding_service = None


def get_geocoding_service() -> GeocodingService:
    """Get the geocoding service singleton"""
    global _geocoding_service
    if _geocoding_service is None:
        _geocoding_service = GeocodingService()
    return _geocoding_service


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Convenience function: reverse geocode coordinates.
    """
    service = get_geocoding_service()
    return service.reverse_geocode(lat, lon)


# ==================== Main Metadata Extraction ====================
def extract_photo_metadata(
    image_path: Path,
    base_dir: Path = PHOTOS_DIR
) -> PhotoMetadata:
    """
    Extract complete metadata for a single photo.
    """
    # Basic info
    relative_path = image_path.relative_to(base_dir).as_posix()
    stat = image_path.stat() if image_path.exists() else None
    file_size = stat.st_size if stat else 0

    metadata = PhotoMetadata(
        file_path=relative_path,
        absolute_path=str(image_path),
        file_size=file_size,
        mtime_ns=stat.st_mtime_ns if stat else 0,
    )

    try:
        metadata.content_sha256 = sha256_file(image_path)
    except OSError as exc:
        metadata.error = f"Hashing failed: {exc}"
        return metadata

    # Load image to get dimensions
    try:
        with Image.open(image_path) as img:
            metadata.image_width, metadata.image_height = img.size
    except Exception as e:
        metadata.error = f"Image loading failed: {e}"
        return metadata

    # Extract EXIF data
    exif_data = extract_exif_data(image_path)
    metadata.datetime_original = exif_data.get('datetime_original')
    if metadata.datetime_original:
        metadata.timestamp_source = exif_data.get('timestamp_source', 'exif_original')
        metadata.timestamp_confidence = 'high'

    # Filename timestamps are useful for exported screenshots, but remain less
    # trustworthy than camera EXIF.
    if not metadata.datetime_original:
        filename_time = parse_filename_datetime(image_path.name)
        if filename_time:
            metadata.datetime_original = filename_time
            metadata.timestamp_source = 'filename'
            metadata.timestamp_confidence = 'medium'

    # Fall back to file modification time if no EXIF date
    if not metadata.datetime_original:
        try:
            mtime = image_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            metadata.datetime_original = dt.isoformat()
            metadata.timestamp_source = 'mtime'
            metadata.timestamp_confidence = 'low'
            print(f"[Info] No EXIF date, using file modification time: {relative_path} -> {metadata.datetime_original}")
        except Exception as e:
            print(f"[Warning] Failed to get file modification time: {e}")

    metadata.captured_at_sort = datetime_sort_value(metadata.datetime_original)
    metadata.date_local = metadata.datetime_original[:10] if metadata.datetime_original else None

    # Geocoding
    gps_coords = exif_data.get('gps_coords')
    if gps_coords:
        metadata.gps_coords = gps_coords
        location = reverse_geocode(gps_coords[0], gps_coords[1])
        metadata.location = location

    return metadata


def batch_extract_metadata(
    image_paths: List[Path],
    base_dir: Path = PHOTOS_DIR,
    show_progress: bool = True
) -> List[PhotoMetadata]:
    """
    Batch-extract metadata from a list of photos.
    Shows a progress bar if enabled.
    """
    results = []

    iterator = tqdm(image_paths, desc="Extracting metadata") if show_progress else image_paths

    for path in iterator:
        metadata = extract_photo_metadata(path, base_dir)
        results.append(metadata)

    return results


# ==================== Test Entry ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Smart Photo Narrative - Data Ingestion Module Test")
    print("=" * 50)

    # Scan for photos
    photos = scan_photos_directory()
    print(f"\nPhotos found: {len(photos)}")

    if photos:
        # Test extracting metadata from the first photo
        print(f"\nTest photo: {photos[0]}")
        metadata = extract_photo_metadata(photos[0])
        print(f"\nMetadata:")
        for key, value in metadata.to_dict().items():
            print(f"  {key}: {value}")
