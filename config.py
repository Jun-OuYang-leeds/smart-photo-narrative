"""
Smart Photo Narrative - Global Configuration Module
====================
System config: model paths, batch sizes, device detection, CLIP index marker
"""

import os
import torch
from pathlib import Path

# ==================== Path Configuration ====================
# Base directory
BASE_DIR = Path(__file__).parent.resolve()


def _load_env_file(env_path: Path) -> None:
    """Load a minimal ``.env`` file into ``os.environ`` without overriding.

    Reads ``KEY=VALUE`` lines (optional ``export `` prefix), strips surrounding
    quotes, skips blank/``#`` lines, and strips ``# inline`` comments on
    unquoted values. Existing environment variables always win (matches
    python-dotenv's default non-override behaviour), so a value exported in the
    shell takes precedence over the file. The DashScope API key is intended to
    live in this file rather than code or the UI.
    """
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


# Load .env (if present) before any os.getenv() below reads config values.
_load_env_file(BASE_DIR / ".env")

# Photo storage directory
PHOTOS_DIR = BASE_DIR / "photos"

# Versioned application data and vector indexes.  The original `chroma_db`
# directory is deliberately left untouched as a rollback source.
DATA_DIR = BASE_DIR / "data"
INDEXES_DIR = BASE_DIR / "indexes"
APP_DB_PATH = DATA_DIR / "smart_photo.db"
CHROMA_PERSIST_DIR = INDEXES_DIR / "chroma_v2"
LEGACY_CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Cache directory (geocoding, etc.)
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
for dir_path in [PHOTOS_DIR, DATA_DIR, CHROMA_PERSIST_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ==================== Device Configuration ====================
def get_device() -> str:
    """
    Auto-detect the best compute device.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[Device] Detected NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("[Device] Detected Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("[Device] Using CPU mode (slower inference)")
    return device


# Runtime device
DEVICE = get_device()


# ==================== Model Configuration ====================
# CLIP model (for vector extraction and zero-shot classification)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512  # CLIP ViT-B/32 output dimension

# BLIP model (for image captioning)
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# BLIP2 model (for offline caption comparison)
BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"

# Ollama LLM configuration.  The production app uses the local bilingual Qwen
# model; the frozen N0--N3 experiment keeps its own llama3 model and digest in
# story_evaluation.py so that the completed dissertation run remains reproducible.
OLLAMA_MODEL = os.getenv("SMART_PHOTO_OLLAMA_MODEL", "qwen3:4b")
OLLAMA_THINK = False
OLLAMA_QUERY_NUM_CTX = 4096
OLLAMA_QUERY_NUM_PREDICT = 300
OLLAMA_STORY_NUM_CTX = 8192
# Frozen default used by the completed N0--N3 / QF--QC / Mood experiments and by
# any story generated without an explicit override. Do NOT change: the frozen
# experiments log and reproduce against this value.
OLLAMA_STORY_NUM_PREDICT = 1200
# Production-only budget. The interactive app passes this so a multi-group
# Faithful narrative (one full paragraph per evidence group) does not overrun the
# generation budget and truncate to invalid JSON. Threaded per-call, so the
# frozen experiments above stay at OLLAMA_STORY_NUM_PREDICT.
STORY_PRODUCTION_NUM_PREDICT = 2048


# ==================== Remote LLM (DashScope OpenAI-compatible) ====================
# Switchable remote story-generation backend. The API key is read ONLY from the
# DASHSCOPE_API_KEY environment variable; it is never written into code or
# exposed in the UI. base_url / model live here and can be overridden by env
# vars. The frontend only flips the local/remote switch.
REMOTE_LLM_BACKENDS = ("local", "remote")
# Default backend for the interactive app; frozen experiments still call
# StoryGenerator(model) directly and never read this value.
REMOTE_LLM_DEFAULT_BACKEND = os.getenv("SMART_PHOTO_STORY_BACKEND", "local")

# DashScope / Bailian OpenAI-compatible endpoint (EU region by default).
# Replace {WorkspaceId} with the real workspace id, or set the env var below.
REMOTE_LLM_BASE_URL = os.getenv(
    "SMART_PHOTO_REMOTE_BASE_URL",
    "https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
)
REMOTE_LLM_MODEL = os.getenv("SMART_PHOTO_REMOTE_MODEL", "qwen3.5-27b")
REMOTE_LLM_TIMEOUT = float(os.getenv("SMART_PHOTO_REMOTE_TIMEOUT", "120"))
# Same per-call generation budget the production local path uses so the remote
# model has the same room to emit a multi-group Faithful narrative.
REMOTE_LLM_NUM_PREDICT = STORY_PRODUCTION_NUM_PREDICT


# ==================== Experimental Eval LLM (separate from production Story) ====================
# A SECOND, fully independent remote LLM used ONLY by the new dissertation
# retrieval experiment (R0--R7) and the single-event Story case study. It never
# replaces the production Story backend (REMOTE_LLM_MODEL = qwen3.5-27b); Story
# generation is untouched. The production app never reads any of these values.
#
# Region discipline (enforced in eval_llm.py):
#   * Only the two Bailian MaaS regions below are supported.
#   * The active region is chosen MANUALLY by the operator and must pair its OWN
#     base_url with its OWN api_key. There is NO automatic cross-region failover:
#     an active beijing client reads only the BEIJING_* env vars, never the
#     FRANKFURT_* ones, and vice versa.
#       Beijing   -> https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
#       Frankfurt -> https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/compatible-mode/v1
#   * Once the formal experiment starts the region+model are frozen by a lock
#     file (EVAL_REGION_LOCK_PATH) and any later run in a different region or
#     with a different model is refused rather than silently mixed in.
#
# Secret discipline: the two region api_keys live ONLY in the git-ignored .env.
# They are never written into code, logs, or experiment results. .env.example
# carries placeholders only. eval_llm exposes only a masked hint, never the raw
# key.
EVAL_LLM_REGIONS = ("beijing", "frankfurt")
EVAL_LLM_ENDPOINT_TEMPLATES = {
    "beijing": "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    "frankfurt": "https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
}
# Region-paired env-var names. The active region selects exactly one pair; the
# other region's env is never consulted for that client.
EVAL_LLM_REGION_ENV = {
    "beijing": {
        "base_url": "SMART_PHOTO_EVAL_BEIJING_BASE_URL",
        "api_key": "SMART_PHOTO_EVAL_BEIJING_API_KEY",
    },
    "frankfurt": {
        "base_url": "SMART_PHOTO_EVAL_FRANKFURT_BASE_URL",
        "api_key": "SMART_PHOTO_EVAL_FRANKFURT_API_KEY",
    },
}
EVAL_LLM_ACTIVE_REGION_ENV = "SMART_PHOTO_EVAL_ACTIVE_REGION"
EVAL_LLM_MODEL_ENV = "SMART_PHOTO_EVAL_MODEL"
EVAL_LLM_TIMEOUT_ENV = "SMART_PHOTO_EVAL_TIMEOUT"
# Default active region. Operators switch to frankfurt MANUALLY (set the env
# var) only if Beijing latency is unacceptable, then re-run the speed test and
# freeze the German config. No code path flips this automatically.
EVAL_LLM_DEFAULT_REGION = "beijing"
# Fixed model snapshot for the WHOLE experiment. If this exact snapshot is
# unavailable in the chosen region the run stops -- it never silently falls back
# to a rolling alias or a different model.
EVAL_LLM_DEFAULT_MODEL = "qwen3.7-plus-2026-05-26"
EVAL_LLM_DEFAULT_TIMEOUT = 180.0
# Qwen3.7 generation contract: deterministic, non-thinking structured output.
# temperature 0 + a fixed seed + thinking disabled so query generation, the
# neighbour audit, and the Story review are reproducible run-to-run.
EVAL_LLM_TEMPERATURE = 0.0
EVAL_LLM_THINKING_ENABLED = False
# Lock file recording the frozen region/model/base_url fingerprint. Written once
# the formal experiment starts; its presence + a mismatch blocks any later run.
EVAL_REGION_LOCK_PATH = DATA_DIR / "eval_region_lock.json"


# ==================== Batch Processing Configuration ====================
# Conservative defaults for the local RTX 3050 Ti Laptop GPU (4 GB).  The
# indexing pipeline halves a batch automatically after CUDA OOM.
BATCH_SIZE = 8
BLIP_BATCH_SIZE = 4
MIN_GPU_BATCH_SIZE = 1
INDEX_CHECKPOINT_SIZE = 8


# ==================== CLIP Index Identity Marker ====================
# Non-semantic sentinel persisted in the ``tags`` table to mark that a photo's
# CLIP image vector has been written and to carry its model-version identity
# (``clip:<model>:<version>`` is the tag *source*). It is NOT a user-visible
# tag, it carries no visual meaning, it is excluded from the FTS5 ``tags``
# column, and it never enters Story evidence. The 72 preset CLIP zero-shot
# tags were removed; this single marker reuses the existing
# replace_tags / list_tag_sources / clear_source_prefix machinery so the
# incremental fast-path and the v1->v2->v1 version-identity guarantees are
# preserved without a schema migration.
CLIP_INDEX_MARKER_TAG = "__clip_indexed__"


# ==================== Geocoding Configuration ====================
# Nominatim API configuration (respect QPS limits)
NOMINATIM_USER_AGENT = "smart-photo-narrative/1.0"
GEOCODER_CACHE_SIZE = 1000  # LRU cache size


# ==================== ChromaDB Configuration ====================
COLLECTION_NAME = "photo_image_v2"
SCENE_GRAPH_COLLECTION_NAME = "photo_scene_graph_v2"
INDEX_SCHEMA_VERSION = 2
# RRF constant. Was 60 (intuitive shipped baseline). Set to 1, the optimum from
# the systematic search in retrieval_human_ablation.json::grid_b_all_channels.best
# (Hit@1 0.720). The report still describes the 1.0/0.8/0.9, k=60 baseline as the
# shipped config; this changes only the live demo, not the frozen experimental record.
RRF_K = 1
RETRIEVAL_CANDIDATE_K = 100


# ==================== Supported Image Formats ====================
SUPPORTED_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'
}


# ==================== UI Configuration ====================
# Pagination
IMAGES_PER_PAGE = 50

# Search result count
SEARCH_TOP_K = 20

# Hybrid search configuration
SEARCH_MODE_VECTOR = "vector"
SEARCH_MODE_KEYWORD = "keyword"
SEARCH_MODE_HYBRID = "hybrid"
HYBRID_VECTOR_WEIGHT = 0.7  # default weight for vector score in hybrid mode

# Narrative generation settings
STORY_MIN_PHOTOS = 1  # one photo produces a photo note, not a full-day diary
STORY_MAX_PHOTOS = 50  # maximum photos per narrative
STORY_MAX_WORDS = 500  # maximum word count for diary
STORY_WORDS_PER_PHOTO = 60  # approximate words per photo for dynamic length
# Single decoding-temperature source for production Story generation. Both the
# Faithful and Creative modes run at this value, identical to every frozen
# A0--A4 / N0--N3 / QF--QC / Mood evaluation, so deployed behaviour matches the
# evaluated behaviour. The mode (grounding contract + output structure), not a
# randomness slider, is the user-facing control; the temperature slider was
# removed because high values overran num_predict or failed strict validation.
STORY_DEFAULT_TEMPERATURE = 0.25

# Exploratory photographer-mood case. Mood is manually confirmed metadata,
# never a physiological or medical inference.
MOOD_EXPERIMENT_EVENT_ID = "58db14d5-e7ad-5975-87a0-59456560e09a"
MOOD_EXPERIMENT_ANNOTATION_SET = "hot_air_balloon_mood_v1"
MOOD_EXPERIMENT_SEED = 20260707

# Story v3 evidence grouping. These values are frozen for the dissertation
# experiments; changing one requires a new prompt/experiment manifest.
STORY_GROUP_MAX_GAP_SECONDS = 10 * 60
STORY_GROUP_CLIP_THRESHOLD = 0.88
STORY_GROUP_JACCARD_THRESHOLD = 0.60
STORY_NEAR_DUPLICATE_CLIP_THRESHOLD = 0.94
STORY_MAX_EVIDENCE_GROUPS = 5

# Coherent first-person memoir (production Creative). Heterogeneous selections
# are chronologically merged down to at most this many evidence groups so the
# memoir weaves scenes into a few flowing paragraphs instead of one paragraph
# per photo. This does not affect the frozen Faithful (v3) / Creative (v4)
# prompts or the N0--N3 experiment definitions.
STORY_NARRATIVE_MAX_PARAGRAPHS = 4

# Production Faithful speculation ban. When False (default) the production
# generator passes speculation_terms=() so GroundingValidator no longer hard-
# rejects the 46 speculation/emotion/causal/kinship terms (E_UNSUPPORTED_CLAIM).
# Faithful then relies on prompt guidance plus traceable, deterministically
# injected evidence citations instead of a validator-enforced low-hallucination
# gate. Set True to restore the hard ban in the production app. The frozen
# N0--N3 / QF--QC experiments call GroundingValidator.validate directly (no
# speculation_terms override) and therefore keep the full ban regardless of this
# switch, so their saved results stay reproducible.
STORY_FAITHFUL_SPECULATION_BAN = False
