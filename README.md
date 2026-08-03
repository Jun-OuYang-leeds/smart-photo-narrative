# Smart Photo Narrative v2

This is a local-first system for multimodal retrieval, event organisation and trustworthy narration over a personal photo collection. The current version reuses the CLIP and BLIP models already present in the `torchtest` Conda environment, so no new vision models are introduced. Qwen scene graphs are generated offline on AIRE and safely imported back, because the local GPU has insufficient memory.

The main pipeline is:

```text
Personal photos
  -> CLIP visual semantics + BLIP caption + Qwen scene graph + EXIF time/location
  -> independent CLIP / BM25 / scene-graph recall + metadata pre-filtering + RRF fusion
  -> text retrieval / image search / date and relation queries / before-after queries
  -> deterministic event organisation
  -> memory, diary or story with photographic evidence citations
```

## Current real data state

The table below is the measured state reported by `scripts/report_index_state.py` after merging the second batch (pic2) (last verified 2026-08-02), not example numbers:

| Item | Count | Notes |
|---|---:|---|
| SQLite photo records | 1599 | 376 main album + 1223 pic2 second batch |
| CLIP image vectors | 1599 | Chroma `photo_image_v2` |
| BLIP captions / FTS5 documents | 1599 | also written to SQLite FTS5 |
| CLIP-index marker photos | 1599 | non-semantic sentinel `__clip_indexed__` carries CLIP completion / version identity; not in FTS, not in story evidence |
| Scene graphs (succeeded) | 1599 | 376 (main album) + 1223 (pic2); all are identity/hash-bound legacy recoveries |
| Scene-graph triples / vectors | 22230 | SQLite and Chroma `photo_scene_graph_v2` counts agree |
| Automatic events | 226 | 1579 photos assigned; 20 low time-confidence photos are not force-assigned to an event |
| Photos with GPS | 317 | — |
| Indexing failures | 0 | — |

Evolution: 2026-07-16 completed the full 376-photo main-album v2 baseline (~9 min 02 s); 2026-07-17 recovered 374 personal scene graphs into the library; 2026-07-25 merged the pic2 second batch of 1223 photos + scene graphs (incremental CLIP+BLIP ~28 min 46 s, 0 failures; pic2 and the main album share one byte-identical duplicate, deduplicated by content hash). The incremental fast path skips SHA-256, EXIF and model inference for unchanged photos. These timings describe only this batch on this machine and must not be treated as performance claims for other devices or datasets.

On 2026-07-27 the 72 CLIP zero-shot preset labels (§35) were removed: semantic labels are no longer generated, and the `tags` table keeps only one non-semantic sentinel row per photo to carry the CLIP completion / version-identity marker (incremental fast path and v1→v2→v1 protection unchanged); tags also left story evidence and the Creative first-person eligibility check. Retrieval ranking is unaffected (tags never took part in scoring, and frozen qrels have no tag filter).

**Important boundary: the frozen A0–A4 v1 retrieval ablation and the N0–N3 / Qwen / Mood formal story experiments were all completed on the 376-photo album on 2026-07-16 and have not been re-run on the expanded 1599-photo album.** The expanded album is used for functional demos and later experiments; the formal conclusions in each experimental section below still rest on the 376-photo case study. The pic2 batch is likewise a legacy recovery, flagged `generation_sha256_verified=false`, and does not claim Manifest-hash verification at generation time.

## Runtime environment

This project already runs in `torchtest` with the following core components:

- Python 3.10
- PyTorch 2.10.0 + CUDA 12.6
- Transformers 5.2.0
- CLIP: `openai/clip-vit-base-patch32`
- BLIP: `Salesforce/blip-image-captioning-base`
- ChromaDB 1.5.1
- Streamlit 1.54.0
- SQLite FTS5 (provided by the Python built-in SQLite)

The host machine is an RTX 3050 Ti Laptop GPU with 4 GB. The default CLIP batch is 8 and the BLIP batch is 4; on CUDA OOM the cache is cleared and the current batch is halved, down to a minimum of 1. The CLIP stage completes and is unloaded before the BLIP stage begins, so the two models do not occupy GPU memory together for long. An ordinary batch failure also degrades to per-image processing, and failures and checkpoints are recorded.

If `torchtest` already exists, use it directly — do not reinstall CLIP, BLIP or overwrite the existing CUDA build of PyTorch just to start the project. Refer to [requirements.txt](requirements.txt) only when building a fresh environment, and first install a PyTorch build that matches the machine's CUDA version.

## Quick start

Run from the project root:

```powershell
cd "E:\leeds\smart-photo-narrative"
conda run -n torchtest python -m streamlit run app.py
```

A browser usually opens `http://localhost:8501`. The page offers album and index, text/image retrieval, events, story, scene-graph workflow and diagnostics.

Ollama is an optional capability, not a startup condition, and does not have to be tested before using the project:

- The default rule parser handles ordinary, date, relation and before-after queries;
- Production uses the local `qwen3:4b` by default, and every chat request explicitly sets `think=False`;
- Only when checked does the system try the same Qwen model to assist query parsing;
- Story generation may call Qwen, but if it is unavailable or the output fails evidence validation it falls back to deterministic text with photo citations;
- CLIP, BLIP, FTS5, Chroma, event organisation and scene-graph import do not depend on Ollama.

The model can be overridden with the environment variable `SMART_PHOTO_OLLAMA_MODEL`; the default value and context settings are in
[config.py](config.py). The frozen N0--N3 formal experiments still pin the original `llama3:latest` and its digest in
[story_evaluation.py](story_evaluation.py), so switching the production model does not rewrite the existing experimental results.

## Indexing: incremental, forced recompute and status report

Photos live in `photos/`. Indexing keys on file-content SHA-256 and a stable `photo_id`, so renaming does not depend on basename matching. SQLite is the canonical source of truth; Chroma is a rebuildable derived vector index.

Daily incremental indexing processes only new, content-changed or stage-missing photos, and reorganises events afterwards:

```powershell
conda run -n torchtest python scripts/rebuild_multimodal_index.py --organize-events
```

The no-change fast path first compares exact relative path, file size, mtime and ctime, then reuses the saved SHA-256 and EXIF; vectors, the current CLIP identity tag and the specified caption version are still checked individually and back-filled per missing stage. Before recomputing CLIP, the old `clip:*` completion markers are revoked, and after the vector write succeeds the current identity and FTS are written transactionally, so a version switch or mid-run failure cannot mix old tags with new vectors. A transient metadata failure does not write a "complete" fingerprint, and a forced-recompute failure is retained as a pending retry state. File-system timestamps are not cryptographic proof: in an extreme case a content replacement at the same path, same size and with all timestamps deliberately preserved could evade the daily fast path, so before freezing data for a formal experiment run one `--force` strong check and save the final status report.

Use `--force` to recompute CLIP and BLIP for all discovered photos:

```powershell
conda run -n torchtest python scripts/rebuild_multimodal_index.py --force --organize-events
```

`--force` means recompute model results for all photos; it does not delete original photos and does not delete old Chroma backups. The script ends by printing `FINAL_REBUILD_REPORT=...` containing index counts, failure count, elapsed time, vector counts and event statistics, which can be saved straight to the experiment log.

To inspect current SQLite/Chroma consistency without running CLIP or BLIP:

```powershell
conda run -n torchtest python scripts/report_index_state.py
```

The report covers photos, captions, FTS5, CLIP index markers, scene graphs, events, stories, failures, GPS, time source, latest indexing run and Chroma collection counts.

## Database and versioned paths

| Purpose | Current path |
|---|---|
| v2 SQLite canonical store | `data/smart_photo.db` |
| v2 Chroma vector store | `indexes/chroma_v2/` |
| Original v1 Chroma path | `chroma_db/` |
| Pre-rebuild frozen backup | `chroma_db_legacy_20260716_pre_multimodal/` |

SQLite uses migrations, foreign keys, WAL and FTS5, and stores photo identity and hashes, EXIF/file-time confidence, captions, tags, scene graphs and triples, events, stories, indexing runs and failure records. Chroma 1.5.1 maintains versioned image-vector and scene-graph-triple collections separately.

Do not copy `chroma_db/` or the frozen backup over `indexes/chroma_v2/`. The old stores are for rollback and comparison; v2 SQLite and Chroma should be checked as a pair.

## How retrieval works

A standard text retrieval proceeds in this order:

1. SQLite first produces the qualifying `photo_id` set from date, tag, location, time-confidence or event;
2. CLIP performs visual-semantic recall between the query and image vectors;
3. SQLite FTS5 performs BM25 recall over text such as BLIP captions;
4. Chroma independently recalls each Qwen scene-graph triple;
5. The three channels are fused by weighted reciprocal rank fusion rather than concatenating everything into one text up front;
6. Each photo is returned with the hit channels, per-channel rank, caption words and scene-graph triple explanations.

The current weights are CLIP `1.0`, Caption/BM25 `0.8`, Scene Graph `0.9`, and the RRF constant is `60`. Metadata is a pre-recall constraint, not another vector, so "semantically relevant" and "time/location satisfied" can be told apart cleanly.

Supported tasks include:

- natural-language text retrieval;
- uploading a query image for image search;
- date-range, tag, location, minimum-time-confidence and event filtering;
- subject–relation–object relation queries;
- "what happened before/after A" temporal-neighbour queries;
- before-after pair queries of the form "B appears after A", which are ordered, same-day and constrained by a time window;
- organising events by date, time gap, CLIP similarity and GPS distance.

The order of a before-after pair is verified by reliable timestamps, not inferred from mere semantic similarity of two images. Low-confidence file times are not treated as evidence equivalent to reliable EXIF.

## Trustworthy narration

A story can be generated from a single photo, a selected photo basket, a date or a saved event. Story v3 first turns photos into stably-numbered model observations, then organises them into up to 5 evidence groups by event, time and similarity; CLIP near-duplicates compress observations but keep all evidence numbers, and obvious BLIP/Qwen conflicts move to an uncertain-observation area.

Multi-photo stories are generated by evidence group rather than by photo, and each paragraph must cite the evidence in its group. The Faithful default mode also validates Chinese/English, citations, group coverage, duplicates, first-person and purpose/emotion/causal speculation; it repairs at most once and then uses a deterministic per-group fallback. Only user-entered `verified_context` may support personal identity or background facts. A citation means "which photos / model observations this text is supported by"; it does not promote a BLIP, CLIP or Qwen prediction to a human-confirmed fact.

Ollama is used only for optional language generation or query parsing. Without Ollama the user can still complete all retrieval, event organisation and evidence-fallback story flows.

## Qwen scene graph: local export, AIRE three-GPU, strict import

Personal photos cannot run Qwen2.5-VL-7B on the local 4 GB GPU, so an offline batch is used. The old `qwen_scene_graph_outputs_full.jsonl` belongs to the Geograph dataset; even with the same filename, a differing full SHA-256 bars it from being imported into the personal album.

### 1. Generate an identity- and hash-bound batch locally

Scene-graph export accepts only files whose actual content is JPEG. HEIC, PNG and others should first be normalised with `scripts/convert_photos_to_jpg.py`, which has backup and verification.

```powershell
conda run -n torchtest python scripts/export_scene_graph_batch.py `
  --album-root photos `
  --export-dir outputs/scene_graph_batch `
  --dataset-id personal-main-v1
```

The output includes `manifest.jsonl` and `images/<photo_id>.jpg`. The manifest always contains schema, `dataset_id`, `photo_id`, relative path, SHA-256 and file size; exported images keep JPEG bytes unchanged.

### 2. Upload to AIRE and submit the three-GPU array job

Upload the code and `outputs/scene_graph_batch/` following `aire_hpc_upload_guide.md` one level above the repo. In the AIRE project directory run:

```bash
mkdir -p /mnt/scratch/$USER/smart-photo-narrative/logs
mkdir -p /mnt/scratch/$USER/smart-photo-narrative/outputs
sbatch scripts/cloud/run_scene_graph_array_3gpu.sbatch
```

The Slurm config is an array `0-2%3`; each task requests one GPU and outputs respectively:

```text
qwen_scene_graph_shard_0.jsonl
qwen_scene_graph_shard_1.jsonl
qwen_scene_graph_shard_2.jsonl
```

The default uses the verified Qwen2.5-VL-7B parameters: vLLM `max-model-len=1536`, generation `max_tokens=640`, at most 15 triples per image. The checkpoint only skips successful non-empty lines whose identity, hash, model and prompt version all match; failures, empty results and bad lines keep being processed on re-runs. The script only connects to the local vLLM on the compute node, defaults to `OPENAI_API_KEY=EMPTY`, and must not have a real key written into code, the manifest, JSONL or the Slurm script.

### 3. Audit locally first, then explicitly emit the indexable JSONL

After downloading the three shards, dry-run first; the wildcard in quotes is expanded by the script:

```powershell
conda run -n torchtest python scripts/import_scene_graph_results.py `
  --manifest outputs/scene_graph_batch/manifest.jsonl `
  --album-root photos `
  --results "outputs/qwen_scene_graph_shard_*.jsonl"
```

Only rows where `dataset_id + photo_id + manifest SHA-256 + current local file SHA-256` all match, the source is `remote` or `remote_repaired`, the status is successful and the triples are non-empty may proceed. The system never accepts results by basename or stem.

After confirming the audit report, explicitly write the clean results:

```powershell
conda run -n torchtest python scripts/import_scene_graph_results.py `
  --manifest outputs/scene_graph_batch/manifest.jsonl `
  --album-root photos `
  --results "outputs/qwen_scene_graph_shard_*.jsonl" `
  --apply `
  --output-jsonl outputs/scene_graph_indexable.jsonl `
  --report-json outputs/scene_graph_import_report.json
```

### 4. Dry-run then write to SQLite/Chroma

The audit script does not change the main index. Dry-run the mapping and conflicts first:

```powershell
conda run -n torchtest python scripts/index_scene_graph_results.py `
  outputs/scene_graph_indexable.jsonl --dry-run
```

After confirming `checksum_conflicts` and `malformed` are both 0, index for real:

```powershell
conda run -n torchtest python scripts/index_scene_graph_results.py `
  outputs/scene_graph_indexable.jsonl
```

This again maps cloud identity to the SQLite stable UUID by full content hash, stores normalised triples, and writes each triple into the scene-graph collection using the existing CLIP text encoder. After it finishes, run `scripts/report_index_state.py` to reconcile the database and vector counts.

For fuller cloud steps see [scripts/cloud/README_scene_graph_aire.md](scripts/cloud/README_scene_graph_aire.md).

### 5. Recovery of a confirmed personal batch that lacks an old manifest

The strict manifest above must always be tried first. The standalone recovery entry is permitted only when the user can confirm that old results did come from the current, unmodified personal album; the personal-album ban on the ordinary legacy importer is not relaxed by this.

Dry-run read-only first:

```powershell
conda run -n torchtest python scripts/recover_legacy_personal_scene_graphs.py `
  outputs/scene_graph/raw/qwen_scene_graph_photos_full.jsonl `
  --album-root photos `
  --database data/smart_photo.db `
  --dataset-id personal-main-v1-legacy-recovered-20260717 `
  --output-jsonl outputs/scene_graph/recovered/scene_graph_indexable_recovered_20260717.jsonl `
  --report-json outputs/scene_graph/recovered/scene_graph_recovery_report_20260717.json `
  --confirm-generated-from-current-album
```

After confirming `blocking_issues=0`, append `--apply` to emit the recovery JSONL, then do the indexing dry-run and real write per section 4. Recovery records are fixed to write `provenance=legacy_path_time_recovered`, the original result-file hash, the current photo hash, the SQLite UUID, the row hash and the verification conditions, and explicitly flag `generation_sha256_verified=false`. This is an honest route that records the evidence gap; it must not be described in the thesis as having Manifest-hash verification at generation time.

In the retriever the `caption` channel queries only the BLIP caption column of FTS5 and never reads the scene graph, tags or location from the same FTS document; otherwise A1 would leak A2's information. The generic `PhotoStorage.search_bm25(..., fields=...)` may still select these fields explicitly in non-ablation settings.

## A0–A4 retrieval ablation evaluation

The evaluation definition is fixed as:

| Variant | Enabled content |
|---|---|
| A0 | CLIP |
| A1 | CLIP + BLIP caption/BM25 |
| A2 | CLIP + Qwen scene graph |
| A3 | CLIP + caption/BM25 + scene graph |
| A4 | A3 + the metadata conditions explicitly given in qrels |

The frozen qrels v1 has 48 queries: 8 per each of six classes, 36 English / 12 Chinese, 12 dev / 36 test. Private queries, UUIDs and locations live in a Git-ignored directory; the public repo keeps only the [annotation protocol](evaluation/QRELS_FORMAT.md), class statistics and SHA-256. After two rounds of order-shuffled agent annotation, the user final-reviewed positives and hard cases on 2026-07-20, correcting one wrong relation positive; all 48 are now `human_validated`. The system never used its own ranking as ground truth, so formal Recall, MRR or nDCG can only come from subsequent A0–A4 runs.

Run the full A0–A4:

```powershell
conda run -n torchtest python scripts/evaluate_retrieval.py `
  evaluation/my_qrels.jsonl `
  --output outputs/retrieval_ablation_report.json `
  --ks 1 5 10 `
  --repeats 3 `
  --warmup 1
```

Run only selected variants:

```powershell
conda run -n torchtest python scripts/evaluate_retrieval.py `
  evaluation/my_qrels.jsonl --variants A0 A1 A3 A4
```

Before-after is a separate temporal task and is not mixed into A0–A4. Run it only when qrels contain `task_type: temporal_pair` and the relevant item uses `before_photo_id->after_photo_id`:

```powershell
conda run -n torchtest python scripts/evaluate_retrieval.py `
  evaluation/my_qrels.jsonl --include-temporal-mode
```

The Ollama parser is off by default for reproducibility; pass `--ollama-parser` only when actually evaluating it. The report records the qrels SHA-256, the run environment, per-query rankings, Recall@K, MRR, nDCG@K, P50/P95 latency and cross-run consistency.

The full annotation format is in [evaluation/QRELS_FORMAT.md](evaluation/QRELS_FORMAT.md).

### A0–A4 formal v1 results

On 2026-07-20 the formal run used the 48 `human_validated` qrels; the main effectiveness results report only the 36 held-out test queries, and the 12 dev do not enter the main conclusions. Each query was repeated 3 times, all rankings were deterministically identical, and neither Ollama nor before-after was enabled.

| Variant | MRR | Recall@5 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| A0 | 0.650 | **0.621** | 0.677 | 0.622 |
| A1 | 0.508 | 0.456 | 0.570 | 0.481 |
| A2 | 0.561 | 0.547 | 0.636 | 0.548 |
| A3 | 0.512 | 0.464 | 0.626 | 0.495 |
| A4 | **0.652** | 0.514 | **0.728** | **0.648** |

The A4-over-A0 nDCG@10 gain is only +0.026, paired bootstrap 95% CI `[-0.113, 0.171]`, so full fusion cannot be claimed to be significantly better than CLIP overall. The A4-over-A3 metadata increment is +0.153, CI `[0.049, 0.275]`; all 6 metadata test queries' nDCG@10 rose from A3's 0.082 to A4's 1.000. The fixed-RRF fusion of caption/scene graph brought no overall gain and even lowered ranking quality on some relation queries.

Public overall, per-class, language, latency and confidence-interval figures with no UUID/query text are in [evaluation/retrieval_ablation_v1_summary.json](evaluation/retrieval_ablation_v1_summary.json). The private per-query report lives in Git-ignored `outputs/experiments/`.

## Story N0–N3 formal experiment and blind review

The formal 12×4 generation is complete; the private raw text, photo paths, UUIDs, claim audits and A/B mappings are all in Git-ignored `evaluation/private/`. The public aggregation is in `evaluation/story_ablation_v1_summary.json` and contains no full story text or personal photo identifiers.

Continue or check the formal checkpoint (identity-exact matches only restore, never re-sample):

```powershell
conda run -n torchtest python scripts/run_story_ablation.py --check-only
conda run -n torchtest python scripts/run_story_ablation.py
```

Launch the standalone blind-review page:

```powershell
conda run -n torchtest streamlit run scripts/story_blind_review_app.py --server.port 8502
```

After the user finishes and locks all 12 A/B/tie choices, unblind and aggregate:

```powershell
conda run -n torchtest python scripts/analyze_story_ablation.py
```

In the formal results N3 had 11/12 fallbacks; the only directly accepted N3 draft was then found by the independent claim audit to have 5 inferences the validator did not catch. The user has finished and locked all 12 blind reviews; the unblinded result is N0 wins 12, N3 wins 0, ties 0 (two-sided exact binomial `p = 0.00048828125`). The current result supports "conservative gating improves formal compliance and detectable claim safety, but at a clear cost in informativeness, latency and subjective readability"; it does not support "N3 model text quality improves across the board".

## Qwen3:4B story dual-track experiment

The new experiment evaluates only the local `qwen3:4b` with a frozen digest; the old Llama N0--N3 remain as historical results, neither overwritten nor entering the new main results. The Faithful track is QF0--QF3 and the Creative track is QC0--QC2; QF2/QF3 and QC1/QC2 respectively share an identical first draft, to isolate the contribution of validation, repair and fallback. The full pre-registered protocol is in [evaluation/QWEN_STORY_PROTOCOL.md](evaluation/QWEN_STORY_PROTOCOL.md).

Check the model and frozen case list, or run the formal generation by checkpoint:

```powershell
conda run -n torchtest python scripts/run_qwen_story_experiment.py --check-only
conda run -n torchtest python scripts/run_qwen_story_experiment.py --track faithful
conda run -n torchtest python scripts/run_qwen_story_experiment.py --track creative
```

Generation and claim audits for both tracks are complete. The pre-registered target was 24 pairs, but the Creative main case had 4/12 `error`, and 3/6 frozen backups also had `error`, leaving only 11 valid Creative pairs; the system did not keep picking new cases after the fact. The actual page count is therefore **23 pairs** (12 Faithful + 11 Creative), and the experiment is explicitly marked incomplete:

```powershell
conda run -n torchtest streamlit run scripts/qwen_story_blind_review_app.py --server.port 8504
```

The blind review contains 12 QF0/QF3 pairs and 11 readable QC0/QC2 pairs; unblinding is forbidden until all choices and scores are locked. The user has now finished and locked all 23 actual pairs, after which the independent mapping is read: Faithful is QF0 wins 12, QF3 wins 0, ties 0 (two-sided exact binomial `p=0.00048828125`); Creative is QC0 wins 11, QC2 wins 0, ties 0 (`p=0.0009765625`). This result comes only from one user's personal-album cases, and Creative was one pair short of the pre-registered target, so it cannot be generalised to overall user preference or written up as 24/24 complete.

The full formal-generation and blind-review text, photo IDs and mappings are all in Git-ignored `evaluation/private/`; the public summary contains no UUID, image path, full story text or A/B mapping.

Automatic results: Faithful QF3 is 7 `ok`, 2 `repaired`, 3 `fallback`, with language, citation and evidence-group coverage all 1.000 and a mean latency of 127.81 s; QF1 is 12/12 `invalid`. Creative main sample QC2 is 6 `ok`, 2 `repaired`, 4 `error`, mean latency 42.01 s; the QC0→QC1 repetition-rate change is -0.0313, 95% paired-bootstrap interval `[-0.0523, -0.0132]`, but QC2's errors lowered language and structural compliance relative to QC1 by 0.333 each. In blind review QF3's coherence/informativeness/evidence-consistency are 0.333/0.583/0.333 lower than QF0; QC2's coherence/personalisation/credibility are 0.273/0.273/0.455 lower than QC0. The conservative agent evidence audit over 2,500 claim units and full results are in [evaluation/story_qwen_dual_v1_summary.json](evaluation/story_qwen_dual_v1_summary.json). This result is not used to claim Qwen beats the old Llama.

## Photographer-mood metadata single case

Creative v6.2 can optionally use human-confirmed photographer mood; this field does not enter retrieval, event grouping or Faithful story, and does not represent a heart-rate/HRV inference. To stop a small model from ignoring or mis-copying the label, Qwen handles scene narration while the application deterministically renders the human mood as a first-person photographer-mood sentence with an evidence ID, marked in the audit area as `verified_photographer_mood`. The experiment is limited to the 5 hot-air-balloon photos of `2026-07-07 · Event 1`. On the Story page select that event, switch to Creative, choose and save one of `neutral/calm/happy/excited/tense/sad` for each photo, then run:

```powershell
conda run -n torchtest python scripts/run_mood_story_experiment.py
conda run -n torchtest streamlit run scripts/mood_story_blind_review_app.py --server.port 8503
conda run -n torchtest python scripts/run_mood_story_experiment.py --reveal
```

The first two stories are generated with `save=False` and do not write the production story table. M0 supplies no mood, M1 supplies the frozen human mood; the model, seed, temperature and system prompt are identical. The private full text, UUIDs, mappings and manifest are in Git-ignored `evaluation/private/`; the public summary stores only aggregate metrics and hashes.

## Main code

- [app.py](app.py): Streamlit v2 interface
- [storage.py](storage.py): SQLite schema, migrations, FTS5/BM25 and audit records
- [indexing_service.py](indexing_service.py): staged incremental CLIP/BLIP indexing, checkpoints and OOM fallback
- [vector_store.py](vector_store.py): versioned Chroma image and triple collections
- [retrieval_engine.py](retrieval_engine.py): independent recall, metadata pre-filtering, RRF, image and temporal retrieval
- [event_organizer.py](event_organizer.py) / [event_service.py](event_service.py): event segmentation and persistence
- [story_agent.py](story_agent.py): evidence aggregation, citation validation, Ollama and deterministic fallback
- [story_evaluation.py](story_evaluation.py): N0–N3 frozen definitions, story cases and automatic metrics
- [qwen_story_evaluation.py](qwen_story_evaluation.py): Qwen QF0–QF3/QC0–QC2 freezing, generation records, automatic metrics, blind review and privacy summary
- [mood_evaluation.py](mood_evaluation.py): photographer-mood M0/M1 freezing, generation records, blind review and public summary
- [scene_graph_io.py](scene_graph_io.py) / [scene_graph_service.py](scene_graph_service.py): safe scene-graph round-trip and indexing
- [evaluation.py](evaluation.py): A0–A4 and standalone temporal-task evaluation
- [论文思路与实验跟踪.md](论文思路与实验跟踪.md): design decisions, failure points and experiment-process log (in Chinese)

## Known limitations

- The library now has 1,599 scene graphs and 22,230 triples (376 main album + 1223 pic2), but semantic neighbours of relation vectors are not strict S–P–O hits; the actual A2/A3/A4 gains must be proven by the formal ablation on frozen qrels.
- A0–A4 v1 is complete but covers only the 376 personal photos from 2026-07-16 and one human final-reviewer; the main album has since grown to 1599 photos, but the formal ablation was not re-run on them, so the conclusion remains a 376-photo personal-album case study and does not represent all users or general image retrieval.
- BLIP captions and Qwen scene graphs are model observations that may miss small objects, person relations, text and fine-grained actions; the exact-term power of FTS5 is bounded by caption quality. CLIP zero-shot tags were removed on 2026-07-27 (§35) and are no longer retrieval or narration evidence.
- Chinese queries pass through the project's deterministic parsing/translation rules, whose coverage is not full machine translation; complex colloquial input may optionally use Ollama, but that lowers strict reproducibility.
- On the 6 Chinese test queries of v1, A0/A1/A3 nDCG@10 is 0; the frozen lexicon does not fully translate concepts like "hot-air balloon, heron, gluttonous snake, sit, by the lake". This failure cannot be fixed by re-tuning words in place after seeing test results while still claiming an unseen test set.
- Before-after depends on reliable, same-day timestamps; photos lacking EXIF are not forced into a trustworthy event purely from file modification time.
- Event thresholds are deterministic heuristics suited to an explainable baseline, but still need a human sample evaluation and may need tuning for a personal album.
- Story citations and high-risk-word validation reduce unsupported narration but cannot guarantee the upstream vision-model observations are absolutely correct; the thesis should describe this as grounding/traceability, not proven fact.
- The formal N0–N3 and single-user 12-pair N0/N3 blind reviews are complete: N3 had 11/12 fallbacks, mean latency ~273 s, max ~751 s; the only directly accepted draft still had 5 inferences not caught by the word-list validator. The unblinded result is N0 wins 12, N3 wins 0, ties 0 (two-sided exact binomial `p = 0.00048828125`). This shows that current N3, while improving formal compliance and detectable factual safety, paid a clear cost in informativeness and readability; the result comes only from one user's paired evaluation of 12 cases and cannot be generalised to overall user preference.
- The photographer-mood M0/M1 single-case blind review is complete: overall preference picked M1; M0/M1 coherence is 4/3, personalisation 3/4, credibility 3/3. The result speaks only to an exploratory personalisation–coherence trade-off in one hot-air-balloon event and is neither statistically significant nor a general improvement; M1 is a hybrid of Qwen narration and programmatically injected human-confirmed mood, not an emotion recognised from the photo or a physiological signal.
- A 4 GB GPU should keep the conservative CLIP 8, BLIP 4 defaults. Forcing larger batches may trigger OOM; automatic fallback keeps running but increases total time.
- SQLite and Chroma target a single-machine personal album and a thesis prototype; large-scale multi-user concurrency, permission isolation, online sync and distributed indexing are out of scope.
