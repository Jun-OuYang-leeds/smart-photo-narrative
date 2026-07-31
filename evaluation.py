"""Reproducible retrieval evaluation and controlled ablation experiments.

The evaluator deliberately separates human judgments from system output.  It
will not compute retrieval accuracy unless every enabled query has at least one
positive qrel.  The canonical qrels formats are documented in
``evaluation/QRELS_FORMAT.md``.

The paper ablation variants are:

* A0: CLIP visual-semantic recall only.
* A1: CLIP + caption recall through SQLite FTS5/BM25.
* A2: CLIP + Qwen Scene Graph triple recall (without caption recall).
* A3: CLIP + caption/BM25 + Scene Graph recall.
* A4: A3 + explicit metadata filters supplied by the qrels.

The full before-after query path is reported as an optional, separate temporal
task.  It is intentionally not folded into A4 because that would change two
experimental variables at once.  Filter-free queries can legitimately produce
identical A3/A4 rankings; metadata qrels are needed to measure that component.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from retrieval_engine import MultimodalRetriever, SearchFilters, SearchResponse


SCHEMA_VERSION = 1
PAIR_SEPARATOR = "->"
SUPPORTED_TASK_TYPES = frozenset({"photo", "temporal_pair"})
SUPPORTED_CATEGORIES = frozenset({
    "scene", "object_attribute", "relation_semantic", "relation_exact",
    "caption_lexical", "metadata",
})
SUPPORTED_LANGUAGES = frozenset({"en", "zh"})
SUPPORTED_SPLITS = frozenset({"dev", "test"})
SUPPORTED_JUDGMENT_SOURCES = frozenset({
    "legacy", "agent_pass1", "agent_pass2", "human_validated",
})
SUPPORTED_FILTERS = frozenset(
    {
        "start_date",
        "end_date",
        "tags",
        "location",
        "min_timestamp_confidence",
        "event_id",
    }
)


class QrelsValidationError(ValueError):
    """Raised when relevance judgments are absent, ambiguous, or malformed."""


@dataclass(frozen=True)
class QueryJudgment:
    """One query and its human relevance judgments.

    ``relevance`` maps stable SQLite ``photo_id`` values to non-negative grades.
    For ``temporal_pair`` tasks, keys use ``before_id->after_id``.
    """

    query_id: str
    query: str
    relevance: Mapping[str, float]
    filters: Mapping[str, Any] = field(default_factory=dict)
    task_type: str = "photo"
    notes: str = ""
    category: str = ""
    language: str = ""
    split: str = ""
    judgment_source: str = "legacy"

    @property
    def positive_relevance(self) -> dict[str, float]:
        return {item_id: grade for item_id, grade in self.relevance.items() if grade > 0}

    def search_filters(self, *, enabled: bool) -> SearchFilters:
        if not enabled:
            return SearchFilters()
        values = dict(self.filters)
        values["tags"] = tuple(values.get("tags") or ())
        return SearchFilters(**values)


@dataclass(frozen=True)
class AblationSpec:
    variant_id: str
    label: str
    channels: tuple[str, ...]
    use_metadata: bool = False
    use_full_query_logic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "channels": list(self.channels),
            "use_metadata": self.use_metadata,
            "use_full_query_logic": self.use_full_query_logic,
        }


DEFAULT_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("A0", "CLIP", ("clip",)),
    AblationSpec("A1", "CLIP + Caption/BM25", ("clip", "caption")),
    AblationSpec(
        "A2",
        "CLIP + Scene Graph",
        ("clip", "scene_graph"),
    ),
    AblationSpec(
        "A3",
        "CLIP + Caption/BM25 + Scene Graph",
        ("clip", "caption", "scene_graph"),
    ),
    AblationSpec(
        "A4",
        "Complete fusion (CLIP + Caption/BM25 + Scene Graph + Metadata)",
        ("clip", "caption", "scene_graph"),
        use_metadata=True,
    ),
)

TEMPORAL_MODE_SPEC = AblationSpec(
    "temporal_pair",
    "Optional full before-after query mode",
    ("clip", "caption", "scene_graph"),
    use_metadata=True,
    use_full_query_logic=True,
)


def temporal_pair_id(before_photo_id: str, after_photo_id: str) -> str:
    """Return the stable qrels identifier for an ordered temporal pair."""
    return f"{before_photo_id}{PAIR_SEPARATOR}{after_photo_id}"


def _read_qrel_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise QrelsValidationError(f"Qrels file does not exist: {path}")
    try:
        if path.suffix.casefold() in {".jsonl", ".ndjson"}:
            records: list[Mapping[str, Any]] = []
            for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                line = raw_line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise QrelsValidationError(
                        f"JSONL line {line_number} must be an object, not {type(value).__name__}"
                    )
                records.append(value)
            return records

        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, Mapping):
            payload = payload.get("queries")
        if not isinstance(payload, list):
            raise QrelsValidationError("JSON qrels must be an array or an object with a 'queries' array")
        if not all(isinstance(record, Mapping) for record in payload):
            raise QrelsValidationError("Every JSON qrels entry must be an object")
        return list(payload)
    except json.JSONDecodeError as exc:
        raise QrelsValidationError(
            f"Invalid JSON in qrels at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _parse_relevance(value: Any, *, query_id: str) -> dict[str, float]:
    if isinstance(value, list):
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise QrelsValidationError(
                f"Query {query_id!r}: relevance list must contain non-empty item IDs"
            )
        if len(set(value)) != len(value):
            raise QrelsValidationError(f"Query {query_id!r}: relevance list contains duplicate IDs")
        return {item: 1.0 for item in value}
    if not isinstance(value, Mapping):
        raise QrelsValidationError(
            f"Query {query_id!r}: relevance must be an object of graded judgments or a list"
        )
    relevance: dict[str, float] = {}
    for raw_item_id, raw_grade in value.items():
        item_id = str(raw_item_id).strip()
        if not item_id:
            raise QrelsValidationError(f"Query {query_id!r}: relevance contains an empty item ID")
        if isinstance(raw_grade, bool) or not isinstance(raw_grade, (int, float)):
            raise QrelsValidationError(
                f"Query {query_id!r}: grade for {item_id!r} must be a number"
            )
        grade = float(raw_grade)
        if not math.isfinite(grade) or grade < 0:
            raise QrelsValidationError(
                f"Query {query_id!r}: grade for {item_id!r} must be finite and non-negative"
            )
        relevance[item_id] = grade
    return relevance


def _validate_filters(value: Any, *, query_id: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise QrelsValidationError(f"Query {query_id!r}: filters must be an object")
    unknown = set(value) - SUPPORTED_FILTERS
    if unknown:
        raise QrelsValidationError(
            f"Query {query_id!r}: unsupported filters: {', '.join(sorted(unknown))}"
        )
    result = dict(value)
    tags = result.get("tags")
    if tags is not None and (
        not isinstance(tags, list)
        or not all(isinstance(tag, str) and tag.strip() for tag in tags)
    ):
        raise QrelsValidationError(f"Query {query_id!r}: filters.tags must be a list of strings")
    confidence = result.get("min_timestamp_confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
    ):
        raise QrelsValidationError(
            f"Query {query_id!r}: min_timestamp_confidence must be between 0 and 1"
        )
    for key in ("start_date", "end_date", "location", "event_id"):
        if key in result and result[key] is not None and not isinstance(result[key], str):
            raise QrelsValidationError(f"Query {query_id!r}: filters.{key} must be a string")
    parsed_dates: dict[str, date] = {}
    for key in ("start_date", "end_date"):
        if result.get(key):
            try:
                parsed_dates[key] = date.fromisoformat(result[key])
            except ValueError as exc:
                raise QrelsValidationError(
                    f"Query {query_id!r}: filters.{key} must use YYYY-MM-DD"
                ) from exc
    if (
        "start_date" in parsed_dates
        and "end_date" in parsed_dates
        and parsed_dates["start_date"] > parsed_dates["end_date"]
    ):
        raise QrelsValidationError(f"Query {query_id!r}: start_date is after end_date")
    return result


def load_qrels(path: str | Path) -> list[QueryJudgment]:
    """Load JSON/JSONL qrels and reject unjudged enabled queries.

    Disabled template records are ignored.  If no enabled, positively judged
    records remain, evaluation stops instead of emitting made-up zero accuracy.
    """
    source = Path(path)
    records = _read_qrel_records(source)
    queries: list[QueryJudgment] = []
    seen_query_ids: set[str] = set()
    for index, record in enumerate(records, 1):
        enabled = record.get("enabled", True)
        if not isinstance(enabled, bool):
            raise QrelsValidationError(f"Record {index}: enabled must be true or false")
        if not enabled:
            continue
        query_id = str(record.get("query_id", "")).strip()
        if not query_id:
            raise QrelsValidationError(f"Record {index}: query_id is required")
        if query_id in seen_query_ids:
            raise QrelsValidationError(f"Duplicate query_id: {query_id!r}")
        seen_query_ids.add(query_id)
        query = str(record.get("query", "")).strip()
        if not query:
            raise QrelsValidationError(f"Query {query_id!r}: query text is required")
        task_type = str(record.get("task_type", "photo")).strip()
        if task_type not in SUPPORTED_TASK_TYPES:
            raise QrelsValidationError(
                f"Query {query_id!r}: task_type must be one of {sorted(SUPPORTED_TASK_TYPES)}"
            )
        category = str(record.get("category", "")).strip()
        language = str(record.get("language", "")).strip()
        split = str(record.get("split", "")).strip()
        judgment_source = str(record.get("judgment_source", "legacy")).strip()
        if category and category not in SUPPORTED_CATEGORIES:
            raise QrelsValidationError(
                f"Query {query_id!r}: category must be one of {sorted(SUPPORTED_CATEGORIES)}"
            )
        if language and language not in SUPPORTED_LANGUAGES:
            raise QrelsValidationError(
                f"Query {query_id!r}: language must be one of {sorted(SUPPORTED_LANGUAGES)}"
            )
        if split and split not in SUPPORTED_SPLITS:
            raise QrelsValidationError(
                f"Query {query_id!r}: split must be one of {sorted(SUPPORTED_SPLITS)}"
            )
        if judgment_source not in SUPPORTED_JUDGMENT_SOURCES:
            raise QrelsValidationError(
                f"Query {query_id!r}: judgment_source must be one of {sorted(SUPPORTED_JUDGMENT_SOURCES)}"
            )
        relevance = _parse_relevance(record.get("relevance"), query_id=query_id)
        if not any(grade > 0 for grade in relevance.values()):
            raise QrelsValidationError(
                f"Query {query_id!r} has no positive human relevance judgments; "
                "disable it or label at least one relevant item"
            )
        if task_type == "temporal_pair" and any(
            len(item_id.split(PAIR_SEPARATOR)) != 2
            or not all(part.strip() for part in item_id.split(PAIR_SEPARATOR))
            for item_id, grade in relevance.items()
            if grade > 0
        ):
            raise QrelsValidationError(
                f"Query {query_id!r}: temporal_pair qrel IDs must use 'before_id{PAIR_SEPARATOR}after_id'"
            )
        queries.append(
            QueryJudgment(
                query_id=query_id,
                query=query,
                relevance=relevance,
                filters=_validate_filters(record.get("filters", {}), query_id=query_id),
                task_type=task_type,
                notes=str(record.get("notes", "")),
                category=category,
                language=language,
                split=split,
                judgment_source=judgment_source,
            )
        )
    if not queries:
        raise QrelsValidationError(
            "No enabled queries with positive human judgments were found; accuracy was not computed"
        )
    return queries


def validate_frozen_qrels(
    judgments: Sequence[QueryJudgment],
    *,
    existing_photo_ids: Iterable[str],
    photo_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Enforce the dissertation query/qrels v1 quotas and metadata semantics."""
    if len(judgments) != 48:
        raise QrelsValidationError(f"Frozen qrels must contain exactly 48 queries, got {len(judgments)}")
    existing = set(existing_photo_ids)
    normalized_queries = [item.query.casefold().strip() for item in judgments]
    if len(set(normalized_queries)) != len(normalized_queries):
        raise QrelsValidationError("Frozen qrels contains duplicate query text")

    category_counts = {category: 0 for category in SUPPORTED_CATEGORIES}
    split_counts = {"dev": 0, "test": 0}
    language_counts = {"en": 0, "zh": 0}
    per_category_split = {
        category: {"dev": 0, "test": 0} for category in SUPPORTED_CATEGORIES
    }
    for judgment in judgments:
        if not judgment.category or not judgment.language or not judgment.split:
            raise QrelsValidationError(f"Query {judgment.query_id!r} is missing category/language/split")
        category_counts[judgment.category] += 1
        split_counts[judgment.split] += 1
        language_counts[judgment.language] += 1
        per_category_split[judgment.category][judgment.split] += 1
        unknown = set(judgment.relevance) - existing
        if unknown:
            raise QrelsValidationError(
                f"Query {judgment.query_id!r} references unknown photo IDs: {sorted(unknown)}"
            )
        if not any(float(grade) == 2.0 for grade in judgment.relevance.values()):
            raise QrelsValidationError(f"Query {judgment.query_id!r} has no grade-2 positive")

        if judgment.category == "metadata":
            forbidden = set(judgment.filters) - {
                "start_date", "end_date", "location", "min_timestamp_confidence",
            }
            if forbidden:
                raise QrelsValidationError(
                    f"Metadata query {judgment.query_id!r} uses forbidden filters: {sorted(forbidden)}"
                )
            if not judgment.filters:
                raise QrelsValidationError(f"Metadata query {judgment.query_id!r} has no metadata filter")
            for photo_id, grade in judgment.positive_relevance.items():
                metadata = photo_metadata.get(photo_id)
                if metadata is None:
                    raise QrelsValidationError(
                        f"Metadata query {judgment.query_id!r}: no metadata for positive {photo_id}"
                    )
                date_local = str(metadata.get("date_local") or "")
                location = str(metadata.get("location") or "")
                confidence = float(metadata.get("timestamp_confidence") or 0.0)
                if judgment.filters.get("start_date") and date_local < judgment.filters["start_date"]:
                    raise QrelsValidationError(
                        f"Metadata positive {photo_id} is before {judgment.filters['start_date']}"
                    )
                if judgment.filters.get("end_date") and date_local > judgment.filters["end_date"]:
                    raise QrelsValidationError(
                        f"Metadata positive {photo_id} is after {judgment.filters['end_date']}"
                    )
                expected_location = str(judgment.filters.get("location") or "").casefold()
                if expected_location and expected_location not in location.casefold():
                    raise QrelsValidationError(
                        f"Metadata positive {photo_id} does not match location {expected_location!r}"
                    )
                minimum = float(judgment.filters.get("min_timestamp_confidence") or 0.0)
                if confidence < minimum:
                    raise QrelsValidationError(
                        f"Metadata positive {photo_id} has timestamp confidence below {minimum}"
                    )

    expected_categories = {category: 8 for category in SUPPORTED_CATEGORIES}
    if category_counts != expected_categories:
        raise QrelsValidationError(f"Category quota mismatch: {category_counts}")
    if split_counts != {"dev": 12, "test": 36}:
        raise QrelsValidationError(f"Split quota mismatch: {split_counts}")
    if language_counts != {"en": 36, "zh": 12}:
        raise QrelsValidationError(f"Language quota mismatch: {language_counts}")
    for category, counts in per_category_split.items():
        if counts != {"dev": 2, "test": 6}:
            raise QrelsValidationError(f"Split quota mismatch for {category}: {counts}")
    return {
        "queries": len(judgments),
        "category_counts": category_counts,
        "split_counts": split_counts,
        "language_counts": language_counts,
        "judgment_sources": dict(sorted({
            source: sum(1 for item in judgments if item.judgment_source == source)
            for source in {item.judgment_source for item in judgments}
        }.items())),
    }


def _unique_ranking(ranked_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item_id) for item_id in ranked_ids))


def recall_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    positive = {item_id for item_id, grade in relevance.items() if grade > 0}
    if not positive:
        raise QrelsValidationError("Recall cannot be computed without positive judgments")
    retrieved = set(_unique_ranking(ranked_ids)[:k])
    return len(retrieved & positive) / len(positive)


def reciprocal_rank(ranked_ids: Sequence[str], relevance: Mapping[str, float]) -> float:
    positive = {item_id for item_id, grade in relevance.items() if grade > 0}
    if not positive:
        raise QrelsValidationError("MRR cannot be computed without positive judgments")
    for rank, item_id in enumerate(_unique_ranking(ranked_ids), 1):
        if item_id in positive:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    def dcg(grades: Sequence[float]) -> float:
        return sum((2.0**grade - 1.0) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))

    ranking = _unique_ranking(ranked_ids)[:k]
    observed = [float(relevance.get(item_id, 0.0)) for item_id in ranking]
    ideal = sorted((float(grade) for grade in relevance.values() if grade > 0), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        raise QrelsValidationError("nDCG cannot be computed without positive judgments")
    return dcg(observed) / ideal_dcg


def ranking_metrics(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    ks: Sequence[int],
) -> dict[str, float]:
    normalized_ks = _validate_ks(ks)
    metrics: dict[str, float] = {"MRR": reciprocal_rank(ranked_ids, relevance)}
    for k in normalized_ks:
        metrics[f"Recall@{k}"] = recall_at_k(ranked_ids, relevance, k)
        metrics[f"nDCG@{k}"] = ndcg_at_k(ranked_ids, relevance, k)
    return metrics


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("At least one value is required")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float | int]:
    if not latencies_ms:
        raise ValueError("At least one latency measurement is required")
    values = [float(value) for value in latencies_ms]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "P50": _percentile(values, 50),
        "P95": _percentile(values, 95),
        "max": max(values),
    }


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    if not ks:
        raise ValueError("At least one cutoff k is required")
    normalized = tuple(sorted(set(int(k) for k in ks)))
    if normalized[0] <= 0:
        raise ValueError("All cutoff values must be positive")
    return normalized


def _ranked_ids(response: SearchResponse, task_type: str) -> list[str]:
    if task_type == "temporal_pair":
        return _unique_ranking(
            temporal_pair_id(pair.before.photo_id, pair.after.photo_id)
            for pair in response.temporal_pairs
        )
    if response.results:
        return _unique_ranking(result.photo_id for result in response.results)
    # The existing UI represents a before-after result by its later image.
    return _unique_ranking(pair.after.photo_id for pair in response.temporal_pairs)


class RetrievalEvaluator:
    """Run paper ablations against a ``MultimodalRetriever``-compatible object."""

    def __init__(
        self,
        retriever: MultimodalRetriever,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.retriever = retriever
        self.clock = clock

    def _search(
        self,
        spec: AblationSpec,
        judgment: QueryJudgment,
        *,
        top_k: int,
        use_ollama_parser: bool,
    ) -> SearchResponse:
        filters = judgment.search_filters(enabled=spec.use_metadata)
        if spec.use_full_query_logic:
            return self.retriever.search(
                judgment.query,
                top_k=top_k,
                filters=filters,
                enabled_channels=spec.channels,
                use_ollama_parser=use_ollama_parser,
            )
        return self.retriever.search_standard(
            judgment.query,
            top_k=top_k,
            filters=filters,
            enabled_channels=spec.channels,
        )

    def evaluate(
        self,
        judgments: Sequence[QueryJudgment],
        *,
        ks: Sequence[int] = (1, 5, 10),
        repeats: int = 3,
        warmup: int = 1,
        specs: Sequence[AblationSpec] = DEFAULT_ABLATIONS,
        use_ollama_parser: bool = False,
        include_temporal_mode: bool = False,
    ) -> dict[str, Any]:
        if not judgments:
            raise QrelsValidationError("No judged queries supplied; accuracy was not computed")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        normalized_ks = _validate_ks(ks)
        depth = max(normalized_ks)
        variants: dict[str, Any] = {}
        optional_modes: dict[str, Any] = {}
        active_specs = list(specs)
        if include_temporal_mode:
            active_specs.append(TEMPORAL_MODE_SPEC)
        computed_spec_count = 0

        for spec in active_specs:
            required_task_type = "temporal_pair" if spec is TEMPORAL_MODE_SPEC else "photo"
            spec_judgments = [
                judgment for judgment in judgments if judgment.task_type == required_task_type
            ]
            destination = optional_modes if spec is TEMPORAL_MODE_SPEC else variants
            destination_key = "temporal_pair" if spec is TEMPORAL_MODE_SPEC else spec.variant_id
            if not spec_judgments:
                destination[destination_key] = {
                    "status": "not_computed_no_matching_qrels",
                    "spec": spec.to_dict(),
                    "query_count": 0,
                }
                continue
            computed_spec_count += 1
            for _ in range(warmup):
                self._search(
                    spec,
                    spec_judgments[0],
                    top_k=depth,
                    use_ollama_parser=use_ollama_parser,
                )

            all_latencies: list[float] = []
            per_query: list[dict[str, Any]] = []
            for judgment in spec_judgments:
                query_latencies: list[float] = []
                rankings: list[list[str]] = []
                for _ in range(repeats):
                    started = self.clock()
                    response = self._search(
                        spec,
                        judgment,
                        top_k=depth,
                        use_ollama_parser=use_ollama_parser,
                    )
                    elapsed_ms = (self.clock() - started) * 1000
                    query_latencies.append(elapsed_ms)
                    all_latencies.append(elapsed_ms)
                    rankings.append(_ranked_ids(response, judgment.task_type))
                metrics = ranking_metrics(rankings[0], judgment.relevance, normalized_ks)
                per_query.append(
                    {
                        "query_id": judgment.query_id,
                        "query": judgment.query,
                        "task_type": judgment.task_type,
                        "category": judgment.category,
                        "language": judgment.language,
                        "split": judgment.split,
                        "judgment_source": judgment.judgment_source,
                        "qrels_filters": dict(judgment.filters),
                        "metadata_filters_applied": spec.use_metadata,
                        "relevance": dict(judgment.relevance),
                        "relevant_count": len(judgment.positive_relevance),
                        "ranked_ids": rankings[0],
                        "ranking_deterministic_across_repeats": all(
                            ranking == rankings[0] for ranking in rankings[1:]
                        ),
                        "metrics": metrics,
                        "latency_ms": latency_summary(query_latencies),
                    }
                )

            metric_names = list(per_query[0]["metrics"])
            macro_metrics = {
                name: statistics.fmean(item["metrics"][name] for item in per_query)
                for name in metric_names
            }
            destination[destination_key] = {
                "status": "computed",
                "spec": spec.to_dict(),
                "query_count": len(spec_judgments),
                "metrics_macro": macro_metrics,
                "latency_ms": latency_summary(all_latencies),
                "all_rankings_deterministic": all(
                    item["ranking_deterministic_across_repeats"] for item in per_query
                ),
                "per_query": per_query,
            }

        if not computed_spec_count:
            raise QrelsValidationError(
                "No qrels match the requested evaluation modes; photo qrels drive A0-A4, "
                "and temporal_pair qrels require include_temporal_mode=True"
            )

        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "accuracy_status": "computed_from_supplied_qrels",
            "query_count": len(judgments),
            "photo_query_count": sum(item.task_type == "photo" for item in judgments),
            "temporal_pair_query_count": sum(
                item.task_type == "temporal_pair" for item in judgments
            ),
            "evaluation_depth": depth,
            "cutoffs": list(normalized_ks),
            "repeats": repeats,
            "warmup_calls_per_variant": warmup,
            "ollama_parser_enabled": use_ollama_parser,
            "ablation_protocol": [spec.to_dict() for spec in specs],
            "temporal_mode_separate_from_ablation": True,
            "metric_definitions": {
                "Recall@K": "macro mean of relevant-item coverage at cutoff K",
                "MRR": f"macro mean reciprocal rank within retrieved depth {depth}",
                "nDCG@K": "macro mean graded normalized discounted cumulative gain",
                "latency": "search-call wall time; warmup calls excluded; milliseconds",
            },
            "variants": variants,
            "optional_modes": optional_modes,
        }


def qrels_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "chromadb", "streamlit", "numpy"):
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def write_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination
