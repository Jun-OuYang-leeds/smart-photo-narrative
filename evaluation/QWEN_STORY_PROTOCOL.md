# Qwen3:4B Story dual-track protocol

## Scope and research questions

This protocol evaluates the local `qwen3:4b` model only. The earlier Llama
N0--N3 experiment remains a historical result and is neither overwritten nor
included in the new primary tables.

- **RQ2a:** Under Qwen3:4B, how do event-level evidence organisation and strict
  validation affect factual support, structural compliance, fallback rate,
  latency and readability?
- **RQ2b:** Can first-person event narration improve coherence and
  personalisation over a legacy free-form story while preserving evidence
  trustworthiness?

The registered model digest is
`359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`.
All calls use temperature `0.25`, context length `8192`, at most `1200`
prediction tokens, `think=false`, and a seed derived deterministically from the
case ID. Formal outputs are generated once; poor content is not resampled.

## Faithful track

The twelve previously frozen Story cases are reused: two single-photo cases,
eight Events and two Dates, with six Chinese and six English cases.

| Variant | Frozen intervention |
|---|---|
| QF0 | Caption, CLIP tags and reliable metadata; free story, no citations |
| QF1 | Per-photo multimodal evidence; one cited paragraph per photo; no repair |
| QF2 | Evidence grouping, near-duplicate compression and conflict separation; one draft, no repair |
| QF3 | Exact QF2 draft plus strict validation; at most one repair, then grouped deterministic fallback |

QF2 and QF3 share the same prompt, seed and raw first draft. This isolates the
contribution of validation, repair and fallback from sampling variation.

## Creative track

The Creative track uses twelve newly frozen primary cases and six reserves.
It excludes all old Story cases, photos already used by saved production
stories, the Mood experiment, and the development dates 2026-07-07,
2026-07-10, 2026-07-11 and 2026-07-12. Cases do not share photos and require a
valid image, primary Caption and primary Scene Graph.

All Creative cases use observer narration, empty verified context and disabled
Mood metadata.

| Variant | Frozen intervention |
|---|---|
| QC0 | Caption, CLIP tags and reliable metadata; free first-person observer memoir |
| QC1 | Current v6.2 grouped multimodal evidence first draft; no repair |
| QC2 | Exact QC1 draft plus the current validator and at most one repair; failure returns `error`, never fallback |

A primary case with no readable QC0/QC2 pair remains in the automatic failure
rate. Reserves replace it for blind text comparison only, in frozen order, up
to twelve readable Creative pairs.

## Records, privacy and checkpointing

Each private JSONL record stores the case, variant, model and digest, seed,
prompt/context/output hashes, raw attempts, normalised output, status,
validation codes, repair/fallback state, latency and automatic metrics. Writes
are atomic and resume only when all identity fields match. All formal calls use
`save=False`; the production `stories` table count must be unchanged.

Private files may contain photo IDs, evidence, mappings and story text and are
kept under the Git-ignored `evaluation/private/` directory. Public summaries
contain only aggregate metrics, quotas and SHA-256 values and are scanned for
UUIDs, paths, story text and mapping fields.

## Evaluation

Automatic metrics cover status, JSON/language compliance, repair,
fallback/error, latency, citation and group coverage, conflict leakage,
repetition, length, compression, risk terms, observer-role compliance,
metadata repetition, cliches and structure. Individual factual claims are
audited as `supported`, `unsupported`, `uncertain` or `non_factual`, with source
`agent_evidence_audit`.

The preregistered blind-review target was 24 valid pairs:

- 12 QF0 versus QF3 comparisons, scored for coherence, informativeness and
  evidence consistency;
- 12 QC0 versus QC2 comparisons, scored for coherence, personalisation and
  trustworthiness.

Variant identity, prompts, validation, repair, fallback and A/B mappings remain
hidden until all responses are locked. Results are a single-user exploratory
evaluation across personal-photo cases and are not evidence of general user
preference.

## Development preflight

Two development-only cases were run before the formal experiment. An initial
candidate from 2026-07-10 was rejected before any model call because it
overlapped the historical formal cases. It was replaced by a non-overlapping
2026-07-11 Event according to the development-only rule.

The final preflight report hash is
`39e533c637f2bbc5ddc3060eb14b510f2d191dd44026f8914fe0592bed2d3bee`.
Both tracks completed without modifying the production Story table. The
Faithful run exercised QF0--QF3, and the Creative run confirmed that QC2 reused
the exact QC1 first draft.

## Formal one-shot result

The formal generation completed with 48 Faithful primary records and 36
Creative primary records. All six Creative reserves were then used according
to their frozen order, producing 18 additional records. The production
`stories` count remained 20.

| Faithful variant | Status | Language compliance | Mean latency |
|---|---|---:|---:|
| QF0 | 12 `ok` | 0.500 | 106.47 s |
| QF1 | 12 `invalid` | 0.000 | 101.43 s |
| QF2 | 10 `unvalidated`, 2 `invalid` | 0.667 | 82.66 s |
| QF3 | 7 `ok`, 2 `repaired`, 3 `fallback` | 1.000 | 127.81 s |

QF3 achieved citation validity and evidence-group coverage of 1.000, with a
fallback rate of 0.250. Its language-compliance change from QF2 was +0.333
with a 10,000-sample paired-bootstrap 95% interval of [0.083, 0.583]. The
QF1-to-QF2 repetition comparison was not estimable because no QF1 record
produced a valid comparable text metric.

| Creative primary variant | Status | Mean duplicate rate | Mean latency |
|---|---|---:|---:|
| QC0 | 12 `ok` | 0.0452 | 108.65 s |
| QC1 | 12 `unvalidated` | 0.0139 | 30.06 s |
| QC2 | 6 `ok`, 2 `repaired`, 4 `error` | 0.0000 | 42.01 s |

QC0-to-QC1 duplicate-rate change was -0.0313, interval [-0.0523, -0.0132].
However, QC2's strict gate produced a 4/12 primary error rate. Language and
structure compliance therefore changed by -0.333 when errors were retained as
system outcomes, interval [-0.583, -0.083]. No Creative fallback was used.

Only three of the six reserves yielded a valid QC0/QC2 pair. Consequently the
frozen set produced 11 Creative blind pairs, not 12. The system stopped instead
of selecting a seventh post-hoc reserve. The available blind packet is
therefore explicitly marked `incomplete` and contains 12 Faithful plus 11
Creative comparisons. It must not be reported as a completed 24-pair review.

The conservative agent evidence audit contains 2,500 text units: 577
supported, 853 unsupported, 494 uncertain and 576 non-factual. This audit is a
reproducible lexical/concept rubric against frozen BLIP/Qwen/CLIP observations
and metadata, not human multi-rater ground truth. It also records extensive
prompt-planning leakage in free-text QF0/QC0 outputs. Upstream visual-model
observations may themselves be wrong.

The user completed and locked all 23 available comparisons before the separate
mapping was read. Faithful preference was QF0 12, QF3 0, ties 0 (two-sided
exact binomial `p=0.00048828125`). Creative preference was QC0 11, QC2 0, ties
0 (`p=0.0009765625`). QF3-minus-QF0 mean score differences were -0.333 for
coherence, -0.583 for informativeness and -0.333 for evidence consistency.
QC2-minus-QC0 differences were -0.273 for coherence, -0.273 for
personalisation and -0.455 for trustworthiness. The trustworthiness interval
was [-0.909, -0.091]; the coherence and personalisation intervals ended at
zero.

No optional textual reasons were supplied, so the preference pattern cannot be
attributed to one specific defect. Together with the automatic metrics, the
result documents a safety--readability/usability trade-off rather than a
general improvement from stricter validation. It remains a single-user
personal-album study.

Final public summary SHA-256:
`87e35cf8bc1aab9364bf91d74fa214f04ea9daadf78b3058132c558f9b9d477d`.
Private result, response and reveal hashes are recorded in the tracking
document. The available 23-pair blind review is complete; the preregistered
24-pair target remains incomplete because only 11 Creative pairs were valid.
