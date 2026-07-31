"""Label the 12 Event-1 photos with photographer Mood (paris_20251218_mood_v1).

This is the human-labeling step the Story case study depends on. Mood is the
photographer's OWN recollection at the moment of capture -- never the emotion of
anyone in the frame. Labels: neutral / calm / happy / excited / tense / sad.

Two ways to supply labels:

* Interactive (default): for each photo the path is printed (and optionally
  opened with ``--open``); type one of the six labels (Enter to skip).
* Batch: ``--from-file moods.json`` where the JSON maps photo_id -> label.

Labels are written via ``storage.save_photo_moods`` (annotation_set
``paris_20251218_mood_v1``); the production Stories table is untouched.

Examples:
    python scripts/label_event_moods.py --open
    python scripts/label_event_moods.py --from-file my_moods.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH, PHOTOS_DIR  # noqa: E402
from eval_story_case import DEFAULT_MOOD_SET, EVENT_ID  # noqa: E402

LABELS = ("neutral", "calm", "happy", "excited", "tense", "sad")


def _event_rows(conn: sqlite3.Connection, event_id: str):
    return conn.execute(
        "SELECT ep.photo_id, p.relative_path, ep.position "
        "FROM event_photos ep JOIN photos p ON p.photo_id=ep.photo_id "
        "WHERE ep.event_id=? ORDER BY ep.position, ep.photo_id",
        (event_id,),
    ).fetchall()


def _open_image(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            import os
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", str(path)], check=False)
        else:
            import subprocess
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass  # opening is a convenience, not a requirement


def _label_interactively(rows, *, open_images: bool) -> dict[str, str]:
    moods: dict[str, str] = {}
    print(f"Labels: {', '.join(LABELS)}  (Enter to skip a photo)\n")
    for photo_id, relative_path, position in rows:
        path = PHOTOS_DIR / str(relative_path)
        print(f"[{position}] {photo_id}  {path}")
        if open_images:
            _open_image(path)
        while True:
            choice = input("  mood> ").strip().casefold()
            if not choice:
                break  # skip
            if choice in LABELS:
                moods[str(photo_id)] = choice
                break
            print(f"  choose one of: {', '.join(LABELS)}")
    return moods


def main() -> int:
    from storage import PhotoStorage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default=EVENT_ID)
    parser.add_argument("--mood-set", default=DEFAULT_MOOD_SET)
    parser.add_argument("--open", action="store_true", help="open each photo as you label it")
    parser.add_argument("--from-file", type=Path, default=None,
                        help="JSON mapping photo_id -> label (batch mode)")
    args = parser.parse_args()

    conn = sqlite3.connect(APP_DB_PATH)
    rows = _event_rows(conn, args.event_id)
    if not rows:
        print(f"event {args.event_id!r} has no photos", file=sys.stderr)
        return 2

    if args.from_file:
        moods = {str(k): str(v).strip().casefold() for k, v in
                 json.loads(args.from_file.read_text(encoding="utf-8")).items()}
        bad = {k: v for k, v in moods.items() if v not in LABELS}
        if bad:
            print(f"invalid labels (must be one of {LABELS}): {bad}", file=sys.stderr)
            return 2
    else:
        moods = _label_interactively(rows, open_images=args.open)

    if not moods:
        print("no labels provided; nothing written.", file=sys.stderr)
        return 0

    storage = PhotoStorage(connection=conn)
    saved = storage.save_photo_moods(moods, annotation_set=args.mood_set)
    conn.close()
    print(f"\nSaved {len(saved)} mood labels as annotation_set {args.mood_set!r}.")
    for photo_id, mood in saved.items():
        print(f"  {photo_id}: {mood.mood_label}")
    print("\nNext: python scripts/run_story_case_experiment.py --status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
