#!/bin/bash
# FastWAM Cloth Folding — Serve Policy Launcher
#
# Usage:
#   bash experiments/piper/serve_policy.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Conda environment -----------------------------------------------
eval "$(conda shell.bash hook)"
conda activate fastwam-xjx

# Reduce fragmentation on 24GB cards. This does not lower model weight memory,
# but it helps avoid allocator failures after warmup/inference churn.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- DiffSynth / Wan2.2 model weights --------------------------------
export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_ROOT}/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD="true"

# ---- HuggingFace cache (local if available) --------------------------
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/hf}"

# ---- Launch -----------------------------------------------------------
echo "[serve_policy] config=experiments/piper/serve_cloth_folding.yaml"
python experiments/piper/serve_policy.py \
    --config experiments/piper/serve_cloth_folding.yaml
