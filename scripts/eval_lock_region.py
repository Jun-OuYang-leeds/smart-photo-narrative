"""Freeze / inspect / release the experimental-eval region lock.

After the formal experiment starts the active region + fixed model snapshot are
frozen so a later run cannot silently mix in a different region or model. This
script is the ONLY intended way to write or remove that lock.

    python scripts/eval_lock_region.py freeze     # freeze current region+model
    python scripts/eval_lock_region.py show       # print the lock (or "not frozen")
    python scripts/eval_lock_region.py unlock     # remove the lock

No secrets are printed. The lock record stores the region, the model, and a
SHA-256 fingerprint of (region, model, base_url); the api key is never written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_llm import (  # noqa: E402
    EvalConfigError,
    EvalLLMClient,
    freeze_region_lock,
    read_region_lock,
    release_region_lock,
)


def _build_client() -> EvalLLMClient:
    # check_lock=False: the lock tool itself must be usable before/after freeze.
    return EvalLLMClient(check_lock=False)


def cmd_freeze(args: argparse.Namespace) -> int:
    client = _build_client()
    record = freeze_region_lock(client.region, client.model, client.base_url)
    print(
        "Frozen experimental-eval region lock:\n"
        f"  region     = {record['region']}\n"
        f"  model      = {record['model']}\n"
        f"  base_url   = {client.base_url}\n"
        f"  fingerprint= {record['base_url_fingerprint']}\n"
        f"  key hint   = {client.key_hint()}\n"
        f"  lock file  = {args.lock_path}"
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    client = _build_client()
    record = read_region_lock(args.lock_path)
    if record is None:
        print("No region lock present (the experiment is not frozen yet).")
    else:
        current_fp_ok = record.get("base_url_fingerprint") and client.base_url
        print(
            "Frozen experimental-eval region lock:\n"
            f"  region     = {record.get('region')}\n"
            f"  model      = {record.get('model')}\n"
            f"  fingerprint= {record.get('base_url_fingerprint')}"
        )
    print(
        "Current client:\n"
        f"  region     = {client.region}\n"
        f"  model      = {client.model}\n"
        f"  base_url   = {client.base_url}\n"
        f"  key hint   = {client.key_hint()}"
    )
    if record is not None and current_fp_ok:
        from eval_llm import config_fingerprint  # noqa: E402

        match = config_fingerprint(client.region, client.model, client.base_url)
        print(f"  matches    = {match == record.get('base_url_fingerprint')}")
    return 0


def cmd_unlock(args: argparse.Namespace) -> int:
    removed = release_region_lock(args.lock_path)
    print("Region lock removed." if removed else "No region lock to remove.")
    return 0


def main() -> int:
    from config import EVAL_REGION_LOCK_PATH  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "show", "unlock"),
        help="freeze | show | unlock the experimental-eval region lock",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=EVAL_REGION_LOCK_PATH,
        help="Path to the lock file (default: the configured EVAL_REGION_LOCK_PATH).",
    )
    args = parser.parse_args()

    try:
        if args.command == "freeze":
            return cmd_freeze(args)
        if args.command == "show":
            return cmd_show(args)
        return cmd_unlock(args)
    except EvalConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
