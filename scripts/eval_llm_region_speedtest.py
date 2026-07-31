"""Region speed test for the experimental eval LLM (Qwen3.7).

Runs 1 warmup + N fixed five-image structured requests against the ACTIVE
region and reports P50 / P95 latency, success rate, and an error-type
breakdown. The region is read from the active config (or ``--region``) and is
NEVER switched automatically: cross-region selection is manual only.

If the fixed model snapshot is unavailable in the region the script stops
immediately and does NOT fall back to another model or rolling alias.

No secrets are written: the summary records region, model, base_url, run
timings and per-image SHA-256, but only a masked key hint.

Examples:
    python scripts/eval_llm_region_speedtest.py
    python scripts/eval_llm_region_speedtest.py --runs 10 --warmup 1
    python scripts/eval_llm_region_speedtest.py --region frankfurt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVAL_REGION_LOCK_PATH, PHOTOS_DIR  # noqa: E402
from eval_llm import (  # noqa: E402
    EVAL_LLM_DEFAULT_SEED,
    EvalLLMClient,
    EvalModelUnavailableError,
    read_region_lock,
)

PRIVATE_DIR = ROOT / "evaluation" / "private"

SYSTEM_PROMPT = (
    "You are a strict photo analyst. You choose exactly one image by its "
    "display index. You never invent indices outside the provided range."
)
TASK_PROMPT = (
    "From the 5 photos (display index 0..4 in the order shown), pick the single "
    "one that best matches: 'a clear outdoor scene photographed during daytime.' "
    'Return JSON: {"chosen_index": <0-4>, "reason": "<= 20 words"}. Choose by '
    "display index only."
)


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy default method)."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _default_images(count: int) -> list[Path]:
    images = sorted(
        p for p in PHOTOS_DIR.iterdir() if p.suffix.lower() == ".jpg"
    )
    if len(images) < count:
        raise SystemExit(
            f"Need at least {count} images in {PHOTOS_DIR}; found {len(images)}. "
            "Pass --images path1,path2,..."
        )
    return images[:count]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_run(client: EvalLLMClient, images: list[Path], seed: int) -> dict:
    start = time.perf_counter()
    try:
        response = client.complete(
            SYSTEM_PROMPT,
            TASK_PROMPT,
            images=images,
            seed=seed,
            max_tokens=256,
            structured=True,
        )
        elapsed = time.perf_counter() - start
        return {
            "ok": True,
            "elapsed_seconds": elapsed,
            "error_type": None,
            "finish_reason": response.finish_reason,
            "model_echoed": response.model_echoed,
            "seed": response.request_seed,
        }
    except EvalModelUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - record error type, keep going
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "elapsed_seconds": elapsed,
            "error_type": client.last_error_type or type(exc).__name__,
            "finish_reason": None,
            "model_echoed": None,
            "seed": seed,
        }


def run_speedtest(
    client: EvalLLMClient,
    images: list[Path],
    *,
    runs: int,
    warmup: int,
    seed: int,
) -> dict:
    # Warmup (not counted in percentiles; errors are logged but non-fatal).
    for i in range(warmup):
        result = _one_run(client, images, seed=seed + i)
        status = "ok" if result["ok"] else f"error={result['error_type']}"
        print(f"[warmup {i + 1}/{warmup}] {result['elapsed_seconds']:.2f}s  {status}")

    measured: list[dict] = []
    for i in range(runs):
        result = _one_run(client, images, seed=seed + warmup + i)
        measured.append(result)
        status = "ok" if result["ok"] else f"error={result['error_type']}"
        print(
            f"[run {i + 1}/{runs}] {result['elapsed_seconds']:.2f}s  {status}"
            f"  model={result['model_echoed']}"
        )

    latencies = [r["elapsed_seconds"] for r in measured]
    successes = sum(1 for r in measured if r["ok"])
    error_counts: dict[str, int] = {}
    for r in measured:
        if not r["ok"]:
            error_counts[r["error_type"] or "unknown"] = (
                error_counts.get(r["error_type"] or "unknown", 0) + 1
            )
    echoed_models = sorted({r["model_echoed"] for r in measured if r["model_echoed"]})

    return {
        "region": client.region,
        "model_requested": client.model,
        "models_echoed": echoed_models,
        "base_url": client.base_url,
        "runs": runs,
        "warmup": warmup,
        "success_count": successes,
        "success_rate": successes / runs if runs else 0.0,
        "latency_seconds": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "error_types": error_counts,
        "image_fingerprints": [
            {"path": str(p.relative_to(ROOT)), "sha256": _sha256_file(p)}
            for p in images
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="measured runs (default 10)")
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs (default 1)")
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="comma-separated list of 5 image paths (default: first 5 photos)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="override the active region (manual only; never auto-switched)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=EVAL_LLM_DEFAULT_SEED,
        help="base seed for the fixed structured task",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the JSON summary here (default: evaluation/private/)",
    )
    args = parser.parse_args()

    if args.images:
        images = [Path(p) for p in args.images.split(",") if p.strip()]
    else:
        images = _default_images(5)
    if len(images) != 5:
        raise SystemExit(f"Expected exactly 5 images, got {len(images)}.")

    try:
        client = EvalLLMClient(region=args.region, check_lock=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    lock = read_region_lock(EVAL_REGION_LOCK_PATH)
    print(
        "Experimental-eval region speed test\n"
        f"  region   = {client.region}\n"
        f"  model    = {client.model}\n"
        f"  base_url = {client.base_url}\n"
        f"  key hint = {client.key_hint()}\n"
        f"  lock     = {'frozen: ' + str(lock.get('region')) if lock else 'not frozen'}\n"
        f"  images   = {[str(p.name) for p in images]}"
    )
    error = client.get_config_error()
    if error:
        print(f"\nNot configured: {error}", file=sys.stderr)
        return 2

    try:
        summary = run_speedtest(
            client, images, runs=args.runs, warmup=args.warmup, seed=args.seed
        )
    except EvalModelUnavailableError as exc:
        print(f"\nSTOP: fixed model snapshot unavailable: {exc}", file=sys.stderr)
        return 3

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (PRIVATE_DIR / f"eval_region_speedtest_{client.region}.json")
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lat = summary["latency_seconds"]
    print(
        "\nSummary\n"
        f"  success   = {summary['success_count']}/{summary['runs']} "
        f"({summary['success_rate'] * 100:.1f}%)\n"
        f"  P50       = {lat['p50']:.2f}s\n"
        f"  P95       = {lat['p95']:.2f}s\n"
        f"  mean/min/max = {lat['mean']:.2f}/{lat['min']:.2f}/{lat['max']:.2f}s\n"
        f"  errors    = {summary['error_types'] or 'none'}\n"
        f"  models_echoed = {summary['models_echoed']}\n"
        f"  written   = {out_path}"
    )
    # Warn (do not stop) if the endpoint echoed a different model name than the
    # pinned snapshot -- that is a silent-swap signal the operator must review.
    if summary["models_echoed"] and client.model not in summary["models_echoed"]:
        print(
            f"\nWARNING: endpoint echoed {summary['models_echoed']} but the pinned "
            f"snapshot is {client.model!r}. Review before freezing.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
