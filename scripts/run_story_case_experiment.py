"""Run the single-event Story case study (Event 1) end to end.

Modes:

* ``--status`` (offline, no LLM): freezes Event 1 (12 photos, SHA-256), reports
  whether the human Mood set is complete, prints the production Stories count,
  and the eval region-lock state. Use this to check readiness.

* ``--run``: generates S-F / S-C0 / S-C1 over the PRODUCTION remote qwen3.5-27b
  (save=False, temperature 0.25, max_tokens 2048, fixed seed, no fallback), then
  runs the Qwen3.7 blinded review (eval config), verifies the production Stories
  count is unchanged, and writes the private results + public summary.

The 12-photo human Mood labeling is a prerequisite for S-C1; label the photos
first (annotation_set ``paris_20251218_mood_v1``) -- see
``scripts/label_event_moods.py``.

Examples:
    python scripts/run_story_case_experiment.py --status
    python scripts/run_story_case_experiment.py --run
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

from config import APP_DB_PATH, REMOTE_LLM_MODEL  # noqa: E402
from eval_llm import EVAL_LLM_DEFAULT_SEED, EvalLLMClient, read_region_lock  # noqa: E402
from eval_story_case import (  # noqa: E402
    DEFAULT_MOOD_SET,
    EVENT_ID,
    STORY_LANGUAGE,
    STORY_MAX_TOKENS,
    STORY_TEMPERATURE,
    freeze_event,
    generate_three,
    story_text,
    validate_event_moods,
)
from eval_story_review import ReviewInput, review_all

PRIVATE_DIR = ROOT / "evaluation" / "private"
PUBLIC_DIR = ROOT / "evaluation" / "public"


def _stories_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0]


def _event_photo_ids(conn: sqlite3.Connection, event_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT photo_id FROM event_photos WHERE event_id=? ORDER BY position, photo_id",
        (event_id,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def cmd_status(args: argparse.Namespace) -> int:
    from storage import PhotoStorage

    storage = PhotoStorage(APP_DB_PATH)
    conn = sqlite3.connect(APP_DB_PATH)
    try:
        frozen = freeze_event(conn, args.event_id)
        photo_ids = frozen.photo_ids
        mood_status = "complete"
        mood_map: dict[str, str] = {}
        try:
            mood_map = validate_event_moods(storage, photo_ids, args.mood_set)
        except ValueError as exc:
            mood_status = f"incomplete ({exc})"
    finally:
        conn.close()

    lock = read_region_lock()
    conn = sqlite3.connect(APP_DB_PATH)
    try:
        stories_n = _stories_count(conn)
    finally:
        conn.close()

    print(
        "Story case study -- status\n"
        f"  event            = {args.event_id}\n"
        f"  photos (frozen)  = {len(photo_ids)}  (order sha256 {frozen.order_hash[:12]}...)\n"
        f"  mood set         = {args.mood_set!r}: {mood_status}\n"
        f"  labelled so far  = {len(mood_map)}/{len(photo_ids)}\n"
        f"  prod Stories     = {stories_n}  (must be unchanged after --run)\n"
        f"  eval region lock = {'frozen: ' + str(lock.get('region')) if lock else 'not frozen'}"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from story_agent import ContextAggregator, DashScopeGenerator, StoryGenerator
    from storage import PhotoStorage

    storage = PhotoStorage(APP_DB_PATH)
    conn = sqlite3.connect(APP_DB_PATH)
    stories_before = _stories_count(conn)
    try:
        frozen = freeze_event(conn, args.event_id)
        # S-C1 needs the complete human Mood set; fail loud if it is missing.
        mood_map = validate_event_moods(storage, frozen.photo_ids, args.mood_set)
    finally:
        conn.close()

    # Reliable metadata for the reviewer.
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        ev = conn.execute("SELECT * FROM events WHERE event_id=?", (args.event_id,)).fetchone()
        image_paths = [
            str(ROOT / "photos" / r["relative_path"])
            for r in conn.execute(
                "SELECT p.relative_path FROM event_photos ep JOIN photos p "
                "ON p.photo_id=ep.photo_id WHERE ep.event_id=? ORDER BY ep.position",
                (args.event_id,),
            )
        ]
    finally:
        conn.close()

    base_seed = args.seed if args.seed is not None else EVAL_LLM_DEFAULT_SEED

    # 1. Generate S-F / S-C0 / S-C1 over the production remote qwen3.5-27b.
    story_generator = StoryGenerator(
        model=REMOTE_LLM_MODEL,
        generator=DashScopeGenerator(REMOTE_LLM_MODEL),
        storage=storage,
    )
    aggregator = ContextAggregator(storage=storage)
    stories = generate_three(story_generator, aggregator, event_id=args.event_id, base_seed=base_seed)

    # 2. Qwen3.7 blinded review (eval config). check_lock enforces no region mixing.
    client = EvalLLMClient(check_lock=not args.skip_lock)
    review_input = ReviewInput(
        images=image_paths,
        metadata={
            "date": str(ev["date_local"]) if ev else "",
            "time_range": (
                f"{str(ev['start_at'])[11:16]}-{str(ev['end_at'])[11:16]}" if ev else ""
            ),
            "location": "",  # Event 1 verified_context is empty by design
        },
        moods=mood_map,
    )
    reviews = review_all(client, stories, review_input, base_seed=base_seed + 1000)

    # 3. Safety: production Stories table must be unchanged.
    conn = sqlite3.connect(APP_DB_PATH)
    try:
        stories_after = _stories_count(conn)
    finally:
        conn.close()
    if stories_after != stories_before:
        raise SystemExit(
            f"PRODUCTION STORIES CHANGED: {stories_before} -> {stories_after}. "
            "save=False was violated; aborting."
        )

    _write_outputs(frozen, stories, reviews, mood_map, args, stories_before)
    _print_summary(stories, reviews)
    return 0


def _write_outputs(frozen, stories, reviews, mood_map, args, stories_count) -> None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    private = {
        "event_id": args.event_id,
        "mood_set": args.mood_set,
        "frozen": {
            "photo_ids": frozen.photo_ids,
            "photo_hashes": frozen.photo_hashes,
            "order_hash": frozen.order_hash,
        },
        "generation": {
            "model": REMOTE_LLM_MODEL,
            "temperature": STORY_TEMPERATURE,
            "max_tokens": STORY_MAX_TOKENS,
            "language": STORY_LANGUAGE,
        },
        "moods": mood_map,
        "stories": {
            key: {
                "status": getattr(s, "status", "?"),
                "mode": getattr(s, "mode", "?"),
                "use_photographer_mood": getattr(s, "use_photographer_mood", None),
                "title": getattr(s, "title", ""),
                "text": story_text(s),
                "paragraphs": [getattr(p, "text", str(p)) for p in (getattr(s, "paragraphs", []) or [])],
                "creative_transitions": [str(t) for t in (getattr(s, "creative_transitions", []) or [])],
                "mood_reflections": [
                    (dict(r) if isinstance(r, dict) else {"text": str(r)})
                    for r in (getattr(s, "mood_reflections", []) or [])
                ],
                "photographer_moods": list(getattr(s, "photographer_moods", []) or []),
                "warnings": [str(w) for w in (getattr(s, "warnings", []) or [])],
            }
            for key, s in stories.items()
        },
        "reviews": reviews,
        "production_stories_count": stories_count,
    }
    (PRIVATE_DIR / "story_case_event1.json").write_text(
        json.dumps(private, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # Public: aggregate only (scores table, claim-audit counts, anonymized).
    public = {
        "event_id": args.event_id,
        "variants": list(stories.keys()),
        "statuses": {k: getattr(s, "status", "?") for k, s in stories.items()},
        "review_scores": {
            r["comparison_id"]: r["scores"] for r in reviews
        },
        "claim_audit_counts": {
            r["comparison_id"]: {
                key: {cat: len(vals) for cat, vals in audit.items()}
                for key, audit in r["claim_audit"].items()
            }
            for r in reviews
        },
        "diff_notes": {r["comparison_id"]: r["diff_note"] for r in reviews},
        "order_hash": frozen.order_hash,
    }
    (PUBLIC_DIR / "story_case_event1_summary.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n  private -> {PRIVATE_DIR / 'story_case_event1.json'}")
    print(f"  public  -> {PUBLIC_DIR / 'story_case_event1_summary.json'}")


def _print_summary(stories, reviews) -> None:
    print("\nStory variants:")
    for key, s in stories.items():
        print(f"  {key}: status={getattr(s, 'status', '?')} "
              f"mode={getattr(s, 'mode', '?')} "
              f"use_mood={getattr(s, 'use_photographer_mood', None)}")
    print("\nReview scores (1-5):")
    for r in reviews:
        print(f"  {r['comparison_id']}: {r['scores']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true", help="offline readiness check")
    mode.add_argument("--run", action="store_true", help="generate + review + outputs")
    parser.add_argument("--event-id", default=EVENT_ID)
    parser.add_argument("--mood-set", default=DEFAULT_MOOD_SET)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-lock", action="store_true", help="dev: skip region-lock check")
    args = parser.parse_args()
    if args.status:
        return cmd_status(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
