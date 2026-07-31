"""Run or reveal the private five-photo M0/M1 mood case study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mood_evaluation import reveal_and_publish, run_m0_m1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reveal",
        action="store_true",
        help="Reveal a completed blind response and write the privacy-safe public summary.",
    )
    args = parser.parse_args()
    try:
        payload = reveal_and_publish() if args.reveal else run_m0_m1()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Mood experiment not run: {error}", file=sys.stderr)
        return 2
    safe = {
        "protocol_version": payload.get("protocol_version"),
        "case_id": payload.get("case_id"),
        "record_count": len(payload.get("records", [])),
        "status": {
            row["variant_id"]: row["status"] for row in payload.get("records", [])
        },
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
