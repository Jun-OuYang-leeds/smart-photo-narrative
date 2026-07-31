# Retrieval query and qrels protocol v1

The private dissertation file contains 48 enabled photo queries. It is never
committed because it contains personal photo UUIDs, dates, locations and query
text. `evaluation/private/` is Git-ignored. Public artifacts contain only this
protocol, aggregate counts and SHA-256 hashes.

Each JSON/JSONL query record has:

```json
{
  "query_id": "scene_en_dev_01",
  "query": "an indoor room with a desk",
  "task_type": "photo",
  "category": "scene",
  "language": "en",
  "split": "dev",
  "judgment_source": "human_validated",
  "filters": {},
  "relevance": {"stable-photo-uuid": 2},
  "notes": "system-blind visual judgment"
}
```

The new fields are optional when loading legacy templates, but all are required
by `validate_frozen_qrels()` for the frozen v1 set. Allowed categories are
`scene`, `object_attribute`, `relation_semantic`, `relation_exact`,
`caption_lexical`, and `metadata`; languages are `en`/`zh`; splits are
`dev`/`test`. Judgment sources progress from `agent_pass1` to `agent_pass2`, and
only become `human_validated` after the user's positive/ambiguous review.

The frozen v1 set reached `human_validated` on 2026-07-20. The terminal review
changed one proposed grade-2 positive to grade 0 for
`relation_semantic_zh_dev_02`; the correction is retained in the private query
notes and reflected in the public manifest hash and grade counts.

## Quotas

- 48 queries: 36 English and 12 Chinese.
- 12 development and 36 test queries.
- Eight per category, with two development and six test queries per category.
- Every query has at least one grade-2 positive and no duplicate query text.

## Relevance

- `2`: clearly and completely satisfies every query condition.
- `1`: satisfies every condition but the subject is small, occluded or less
  representative.
- `0`: does not satisfy the query. For `relation_exact`, the wrong relation or
  direction is always zero.

Queries and initial positives are authored from the original-image contact
sheets and reliable EXIF before system rankings are opened. Candidate pools are
the de-duplicated union of the top 20 from CLIP-only, Caption-only, SG-only and
A0–A4, supplemented by the full album contact sheet. During both labeling
passes, channel names and scores are hidden and candidate order is independently
shuffled.

Metadata queries may use only `start_date`, `end_date`, `location`, and
`min_timestamp_confidence`. Tags and Event IDs are forbidden as metadata qrels.
The evaluator checks every positive against the corresponding stored metadata.
