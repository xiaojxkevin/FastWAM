#!/bin/bash
trap 'rc=$?; echo "[fatal] host=$(hostname) line=${LINENO} exit_code=${rc} time=$(date -Is)" >&2; exit ${rc}' ERR

# ========================================================
# Environment: debugging & error reporting
# ========================================================
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL

# ========================================================
# Proxy (platform internal network)
# ========================================================
export http_proxy=http://10.2.83.188:3128
export https_proxy=http://10.2.83.188:3128

# ========================================================
# WandB — replace with your own API key
# ========================================================
export WANDB_API_KEY="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"

# ========================================================
# HuggingFace & Model cache → all under ./.cache/
# ========================================================
export HF_HOME="./.cache/hf"

# DiffSynth / Wan2.2 model weights
export DIFFSYNTH_MODEL_BASE_PATH="./checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"
# Set DIFFSYNTH_SKIP_DOWNLOAD="true" if weights are already in ./checkpoints/
export DIFFSYNTH_SKIP_DOWNLOAD="true"

# ========================================================
# Multi-process safety: avoid shared-filesystem lock
# contention for Triton / JIT / Numba compilation caches
# ========================================================
export OMP_NUM_THREADS=8

export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export NUMBA_CACHE_DIR=/tmp/numba_cache
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $NUMBA_CACHE_DIR

# ========================================================
# Working directory & conda environment
# ========================================================
cd /workspace/mnt/sealab/xiaojx/FastWAM

source /opt/conda/etc/profile.d/conda.sh
conda activate fastwam

# ========================================================
# Multi-node configuration
#
#   NNODES              — total number of nodes (default 2 for production)
#   NPROC_PER_NODE      — GPUs per node (default 8)
#   NODE_RANK           — this node's rank (0 for master, platform injects via $RANK)
#   MASTER_ADDR         — rank-0 node IP (platform injects)
#   MASTER_PORT         — rendezvous port
#
# Platform (DLC) injects: RANK / MASTER_ADDR / DLB_JOB_ID
# ========================================================
echo "load nodes info..."

export NNODES="${NNODES:-2}"
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export MASTER_PORT="${MASTER_PORT:-38654}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  export NPROC_PER_NODE=8
fi

export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR="${MASTER_ADDR}"

if [[ "${NNODES}" =~ ^[0-9]+$ ]] && (( NNODES > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "Error: MASTER_ADDR is required for multi-node training. DLC should inject it, or set it to the rank-0 node address." >&2
  exit 1
fi

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "[dlc] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"

python - <<'PY'
import os
import torch

print(f"[cuda] available={torch.cuda.is_available()} device_count={torch.cuda.device_count()} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
PY

# ========================================================
# Accelerate multi-node env vars
# ========================================================
export ACCELERATE_NUM_MACHINES=${NNODES}
export ACCELERATE_MACHINE_RANK=${NODE_RANK}
export ACCELERATE_MAIN_PROCESS_IP=${MASTER_ADDR}
export ACCELERATE_MAIN_PROCESS_PORT=${MASTER_PORT}

export no_proxy="localhost,127.0.0.1,::1,${MASTER_ADDR},.svc,.local,.cluster.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

# Network interface hints — uncomment if NCCL autodetection fails
# export NCCL_SOCKET_IFNAME=eth0,bond0
# export GLOO_SOCKET_IFNAME=eth0,bond0

echo "[Safe Launch] Using MASTER_ADDR=${MASTER_ADDR}, NODE_RANK=${NODE_RANK}"

# ========================================================
# Logging: tee stdout/stderr to a log file per node
# ========================================================
LOG_ROOT="${LOG_ROOT:-./runs/_logs}"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/train_cloth_node${NODE_RANK}_$(hostname).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'rc=$?; echo "[fatal] host=$(hostname) line=${LINENO} exit_code=${rc} time=$(date -Is)" >&2; exit "${rc}"' ERR

echo "[log] file=${LOG_FILE}"
echo "[start] host=$(hostname) time=$(date -Is)"

# RUN_ID: platform-provided DLB_JOB_ID, or auto-generated by train_zero1.sh
export RUN_ID="run_job_${DLB_JOB_ID:-$(date +%Y-%m-%d-%H)}"

# ========================================================
# Launch training via train_zero1.sh
#
# train_zero1.sh handles:
#   - Multi-node RUN_ID synchronisation (TCPStore)
#   - accelerate launch with DeepSpeed ZeRO-1
#   - output_dir = ./runs/<task_basename>/<run_id>
# ========================================================
bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
    task=cloth_folding_3cam_384_1e-4 \
    wandb.enabled=true \
    wandb.project=fastwam-cloth-fold \
    wandb.name=0602-cloth-fold
