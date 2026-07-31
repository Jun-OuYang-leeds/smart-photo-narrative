# Story experiment protocol v1

RQ2 tests whether event-level evidence organization and strict validation reduce
repetition and unsupported statements while improving coherence. The private
case file contains two single-photo notes, eight stored Events, and two
multi-event Dates; six are Chinese and six English. The large `2026-07-12` Event
is a mandatory regression case.

| Variant | Frozen definition |
|---|---|
| N0 | Legacy Caption + Tags + Metadata prompt, without evidence citations. |
| N1 | `grounded-story-v2`: raw per-photo evidence and citations; one-photo-one-sentence baseline. |
| N2 | Story v3 evidence grouping, near-duplicate compression and conflict separation, without the N3 repair gate. |
| N3 | N2 plus language, citation, group coverage, repetition and unsupported-claim validation; one repair, then grouped deterministic fallback. |

The initial three frozen `llama3:latest` acceptance cases ran exactly once. All
three raw outputs failed the strict contract and entered grouped fallback; the
final artifacts passed the automated language, citation, coverage, repetition,
Faithful and conflict checks. Those cases were not rerun or deleted.

The later formal experiment ran all 48 N0–N3 outputs exactly once with
`llama3:latest`, temperature 0.25, `num_predict=1200`, and a deterministic seed
derived from the case ID. N2 and N3 share the exact same first prompt and raw
draft; N3 alone may make one repair call and then use grouped fallback. Results
are checkpointed outside the production Story database.

Automatic outcomes are JSON validity, citation validity, evidence-group
coverage, evidence-ID accounting, language compliance, text-unit duplicate
rate, fallback rate, paragraph compression, latency, claim count, risk-term
rate, independently audited unsupported-claim rate and conflict handling.
Unavailable metrics remain N/A rather than becoming zero.

The formal run produced 12 N0, 12 N1, 12 N2 and 12 N3 records. N3 accepted one
raw draft and used deterministic fallback in eleven cases. This is a safety and
structure-compliance result, not evidence that the local model generated better
prose. The independent `agent_evidence_audit` found five unsupported statements
in the sole accepted N3 draft, exposing validator vocabulary gaps. The audit is
not presented as human multi-rater annotation and upstream visual observations
remain fallible.

The user completed and locked all 12 blind N0-vs-N3 A/B/tie preferences before
the separately stored mapping was revealed. N0 won 12 cases, N3 won none and
there were no ties (two-sided exact binomial `p = 0.00048828125`). This is one
user's paired assessment of 12 private-album cases, not a multi-participant user
study. Together with the 11/12 N3 fallback rate, the result shows that improved
formal compliance and detectable claim safety came with a severe
informativeness/readability trade-off in this implementation.
