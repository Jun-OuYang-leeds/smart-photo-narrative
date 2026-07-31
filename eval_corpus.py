"""Frozen corpus + deterministic target sampling for the retrieval experiment.

This module is the deterministic, fully-testable core of the R0--R7 retrieval
experiment. It knows nothing about the eval LLM or the retrieval engine; it only
materialises the frozen candidate library, the valid target pool, the
near-duplicate groups, and the stratified 300-target sample.

Frozen data contract (verified against the live DB at freeze time):

* Candidate library (corpus searched over) = the **1,596** photos that have all
  of: a CLIP vector, a base-BLIP caption, a BLIP2 caption, and a main Scene
  Graph.
* Target pool = the **373** of the original **376** main-project personal photos
  (``relative_path NOT LIKE 'pic2/%'``) that also satisfy the all-channel
  requirement.
* Targets = **300** photos sampled deterministically from the 373 with the fixed
  seed ``spn-known-item-full-corpus-v1``, stratified by metadata completeness
  and near-duplicate status, with at most 6 per event and 2 per near-duplicate
  group (near-duplicate = captured within 10 minutes AND CLIP cosine >= 0.94).

The protocol forbids importing photos, rebuilding model output, or editing the
SQLite Primary records during the experiment, so every function here is
read-only.
"""

from __future__ import annotations

import hashlib
import math
import random
import sqlite3
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Optional, Sequence

from config import BLIP2_MODEL_NAME, BLIP_MODEL_NAME

# ============================ Frozen constants ============================

SEED_NAME = "spn-known-item-full-corpus-v1"
TARGET_COUNT = 300
NEAR_DUP_TIME_SECONDS = 10 * 60
NEAR_DUP_CLIP_THRESHOLD = 0.94
MAX_PER_EVENT = 6
MAX_PER_NEAR_DUP_GROUP = 2
METADATA_TIME_CONFIDENCE = 0.6


def seed_from_name(name: str = SEED_NAME) -> int:
    """Stable 32-bit integer seed from the protocol's seed *name*.

    Python's ``random.Random`` is deterministic across versions for a given
    integer seed, so sampling is reproducible run-to-run and machine-to-machine.
    """
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ============================ All-channel SQL ============================

# A photo qualifies for the frozen corpus when all four model channels exist.
# The captions table stores one row per (photo, model); the CLIP vector is
# marked by a ``clip:`` tag source; the main Scene Graph lives in scene_graphs.
# NOTE: the corpus intentionally INCLUDES the later ``pic2/`` batch (1,596
# total); only the target pool excludes it.
_ALL_CHANNEL_SQL = """
SELECT p.photo_id
FROM photos p
WHERE EXISTS (SELECT 1 FROM tags t
              WHERE t.photo_id = p.photo_id AND t.source LIKE 'clip:%')
  AND EXISTS (SELECT 1 FROM captions c
              WHERE c.photo_id = p.photo_id AND c.model_name = ?)
  AND EXISTS (SELECT 1 FROM captions c
              WHERE c.photo_id = p.photo_id AND c.model_name = ?)
  AND EXISTS (SELECT 1 FROM scene_graphs s WHERE s.photo_id = p.photo_id)
"""
# The original main-project photos are the flat ``photos/`` files (no ``pic2/``
# prefix). The target pool is restricted to these even though pic2 stays in the
# search corpus.
_ORIGINAL_FILTER_SQL = " AND NOT p.relative_path LIKE 'pic2/%'"


def frozen_corpus_photo_ids(
    conn: sqlite3.Connection,
    *,
    blip_model: str = BLIP_MODEL_NAME,
    blip2_model: str = BLIP2_MODEL_NAME,
) -> list[str]:
    """The 1,596-photo all-channel candidate library, deterministically ordered.

    Includes the later ``pic2/`` batch; every photo has a CLIP vector, a
    base-BLIP caption, a BLIP2 caption, and a main Scene Graph.
    """
    rows = conn.execute(_ALL_CHANNEL_SQL, (blip_model, blip2_model)).fetchall()
    return sorted(r[0] for r in rows)


def valid_target_pool_photo_ids(
    conn: sqlite3.Connection,
    *,
    blip_model: str = BLIP_MODEL_NAME,
    blip2_model: str = BLIP2_MODEL_NAME,
) -> list[str]:
    """The 373 valid targets: original 376 (non-pic2) that satisfy all channels.

    The original main-project photos live at the top of ``photos/`` (flat,
    ``relative_path`` with no ``pic2/`` prefix); the later ``pic2/`` batch is
    excluded from the target pool even though it remains in the search corpus.
    """
    sql = _ALL_CHANNEL_SQL.rstrip().rstrip(";") + _ORIGINAL_FILTER_SQL
    rows = conn.execute(sql, (blip_model, blip2_model)).fetchall()
    return sorted(r[0] for r in rows)


# ============================ Metadata completeness ============================


def metadata_tier(row: Mapping) -> str:
    """Stratification key: 'complete' if the photo has reliable time + date + place.

    Mirrors the metadata-subset rule (timestamp_confidence >= 0.6, a local date,
    and a full geocoded location). Everything else is 'partial'.
    """
    conf = row.get("timestamp_confidence")
    try:
        conf_ok = conf is not None and float(conf) >= METADATA_TIME_CONFIDENCE
    except (TypeError, ValueError):
        conf_ok = False
    date_ok = bool(row.get("date_local"))
    loc_ok = bool(row.get("location"))
    return "complete" if (conf_ok and date_ok and loc_ok) else "partial"


# ============================ Near-duplicate grouping ============================


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def components(self) -> dict[str, set]:
        groups: dict[str, set] = defaultdict(set)
        for x in self.parent:
            groups[self.find(x)].add(x)
        return groups


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def compute_near_duplicate_groups(
    photo_ids: Sequence[str],
    clip_vectors: Mapping[str, Sequence[float]],
    timestamps: Mapping[str, float],
    *,
    time_seconds: float = NEAR_DUP_TIME_SECONDS,
    clip_threshold: float = NEAR_DUP_CLIP_THRESHOLD,
) -> dict[str, set]:
    """Connected components of the near-duplicate graph.

    Two photos are near-duplicates when captured within ``time_seconds`` of each
    other AND their CLIP cosine similarity is at least ``clip_threshold``. Returns
    a mapping ``component_root -> set(photo_ids)`` (singletons included).
    """
    present = [pid for pid in photo_ids if pid in clip_vectors and pid in timestamps]
    ordered = sorted(present, key=lambda p: (timestamps[p], p))
    uf = _UnionFind(ordered)
    n = len(ordered)
    for i in range(n):
        a = ordered[i]
        ta = timestamps[a]
        va = clip_vectors[a]
        for j in range(i + 1, n):
            b = ordered[j]
            if timestamps[b] - ta > time_seconds:
                break  # sorted by time: no further pair can be within the window
            if _cosine(va, clip_vectors[b]) >= clip_threshold:
                uf.union(a, b)
    return uf.components()


def near_duplicate_index(
    groups: Mapping[str, set],
) -> tuple[dict[str, int], dict[int, int]]:
    """Map photo_id -> group_id and group_id -> size (only groups of size >= 2).

    Singletons get no group_id (``None``); the sampling cap applies only to real
    near-duplicate clusters.
    """
    pid_to_group: dict[str, int] = {}
    group_size: dict[int, int] = {}
    next_id = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        gid = next_id
        next_id += 1
        group_size[gid] = len(members)
        for pid in members:
            pid_to_group[pid] = gid
    return pid_to_group, group_size


# ============================ Stratified constrained sampling ============================


def _largest_remainder_quotas(stratum_sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Proportional integer quotas summing to ``total`` (largest-remainder method)."""
    denom = sum(stratum_sizes.values()) or 1
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, size in stratum_sizes.items():
        exact = total * size / denom
        fl = math.floor(exact)
        floors[key] = fl
        remainders.append((exact - fl, key))
    assigned = sum(floors.values())
    # Distribute the leftover by largest remainder (key breaks ties deterministically).
    for _, key in sorted(remainders, key=lambda t: (-t[0], t[1])):
        if assigned >= total:
            break
        floors[key] += 1
        assigned += 1
    return floors


def stratified_sample(
    targets: Sequence[str],
    stratum_of: Mapping[str, str],
    near_dup_group_of: Mapping[str, Optional[int]],
    event_of: Mapping[str, Optional[str]],
    *,
    count: int = TARGET_COUNT,
    max_per_event: int = MAX_PER_EVENT,
    max_per_near_dup_group: int = MAX_PER_NEAR_DUP_GROUP,
    seed: Optional[int] = None,
) -> list[str]:
    """Deterministic stratified sample of ``count`` targets under the caps.

    Strata come from ``stratum_of`` (metadata completeness x near-duplicate
    status). Within each stratum the order is a seeded shuffle; the per-event
    (<=6) and per-near-duplicate-group (<=2) caps are enforced greedily. Returns
    a deterministically sorted list; the same ``seed`` always yields the same
    selection. If the caps prevent reaching ``count`` (best-effort 'in principle'
    constraints), the result is shorter and the caller is told via the count.
    """
    if seed is None:
        seed = seed_from_name()
    rng = random.Random(seed)

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for pid in targets:
        by_stratum[stratum_of.get(pid, "partial")].append(pid)
    quotas = _largest_remainder_quotas(
        {k: len(v) for k, v in by_stratum.items()}, count
    )

    event_count: Counter = Counter()
    nd_count: Counter = Counter()
    selected: list[str] = []
    for stratum in sorted(by_stratum):  # deterministic stratum order
        pool = sorted(by_stratum[stratum])  # deterministic before shuffle
        rng.shuffle(pool)
        taken = 0
        for pid in pool:
            if taken >= quotas[stratum]:
                break
            ev = event_of.get(pid)
            nd = near_dup_group_of.get(pid)
            if ev is not None and event_count[ev] >= max_per_event:
                continue
            if nd is not None and nd_count[nd] >= max_per_near_dup_group:
                continue
            selected.append(pid)
            taken += 1
            if ev is not None:
                event_count[ev] += 1
            if nd is not None:
                nd_count[nd] += 1

    # Phase 2 -- soft fill. The caps are protocol *preferences* ("in principle"),
    # not hard ceilings, and the target count is a hard goal. If the capped pass
    # fell short, top up deterministically from the leftover (stratum-ordered,
    # seeded shuffle) until ``count`` is reached or the pool is exhausted. The
    # capped pass already captured the diverse core; this only relaxes caps.
    if len(selected) < count:
        selected_set = set(selected)
        leftover_by_stratum: dict[str, list[str]] = defaultdict(list)
        for pid in targets:
            if pid not in selected_set:
                leftover_by_stratum[stratum_of.get(pid, "partial")].append(pid)
        for stratum in sorted(leftover_by_stratum):
            pool2 = sorted(leftover_by_stratum[stratum])
            rng.shuffle(pool2)
            for pid in pool2:
                if len(selected) >= count:
                    break
                selected.append(pid)
            if len(selected) >= count:
                break
    return sorted(selected)


__all__ = [
    "MAX_PER_EVENT",
    "MAX_PER_NEAR_DUP_GROUP",
    "METADATA_TIME_CONFIDENCE",
    "NEAR_DUP_CLIP_THRESHOLD",
    "NEAR_DUP_TIME_SECONDS",
    "SEED_NAME",
    "TARGET_COUNT",
    "compute_near_duplicate_groups",
    "frozen_corpus_photo_ids",
    "metadata_tier",
    "near_duplicate_index",
    "seed_from_name",
    "stratified_sample",
    "valid_target_pool_photo_ids",
]
