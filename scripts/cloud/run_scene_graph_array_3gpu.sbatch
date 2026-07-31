#!/bin/bash
# Three independent one-GPU jobs.  The array is data parallel; each task owns
# one JSONL checkpoint, so a task can be safely resubmitted without losing work.
#SBATCH --job-name=spn_qwen_sg
#SBATCH --partition=gpu
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/scratch/%u/smart-photo-narrative/logs/scene_graph_%A_%a.out
#SBATCH --error=/mnt/scratch/%u/smart-photo-narrative/logs/scene_graph_%A_%a.err

set -euo pipefail

module purge
module load cuda/12.6.2
module load miniforge/24.7.1

PROJECT_DIR="${PROJECT_DIR:-/users/$USER/projects/smart-photo-narrative}"
SCRATCH_DIR="${SCRATCH_DIR:-/mnt/scratch/$USER/smart-photo-narrative}"
CONDA_ENV="${CONDA_ENV:-/mnt/scratch/$USER/snapseek/conda_py311}"
MODEL_PATH="${MODEL_PATH:-/mnt/scratch/$USER/snapseek/models/Qwen2.5-VL-7B-Instruct}"
MANIFEST="${MANIFEST:-$SCRATCH_DIR/batch/manifest.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-$SCRATCH_DIR/batch/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_DIR/outputs}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
PORT="$((8000 + TASK_ID))"
OUTPUT_JSONL="$OUTPUT_DIR/qwen_scene_graph_shard_${TASK_ID}.jsonl"
VLLM_LOG="$SCRATCH_DIR/logs/vllm_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_API_KEY=EMPTY

mkdir -p "$SCRATCH_DIR/logs" "$OUTPUT_DIR"

vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 1536 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image": 1}' \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

python scripts/cloud/generate_scene_graph_jsonl.py \
  --manifest "$MANIFEST" \
  --image-dir "$IMAGE_DIR" \
  --output-jsonl "$OUTPUT_JSONL" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --prompt-version spn-scene-graph-v1 \
  --max-triples 15 \
  --max-tokens 640 \
  --attempts 2 \
  --shard-index "$TASK_ID" \
  --num-shards 3

# Import all three shard files together on the local machine.  Concatenation is
# unnecessary: scripts/import_scene_graph_results.py accepts multiple JSONLs.
