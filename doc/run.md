# FastWAM Training Guide

## Cloth Folding Dataset

Dataset: `data/0321_merged_dagger_anno/` (67 episodes, 560K frames, 30fps, 3 cameras, 14-DoF bimanual)

### 1. Precompute Text Embeddings (requires GPU)

```bash
export HF_HOME=./.tmp/hf

conda run -n fastwam python scripts/precompute_text_embeds.py \
    data=cloth_folding model=fastwam_joint overwrite=true
```

This encodes the task instruction "take the cloth from the basket and fold the cloth." via the Wan2.2 T5 text encoder and caches it to `data/text_embeds_cache/cloth_folding/`.

### 2. Launch Training

Single-GPU:
```bash
export HF_HOME=./.tmp/hf

conda run -n fastwam python scripts/train.py \
    data=cloth_folding \
    task=cloth_folding_joint_3cam_384_1e-4 \
    model=fastwam_joint
```

Multi-GPU (e.g., 4 GPUs with DeepSpeed ZeRO-1):
```bash
export HF_HOME=./.tmp/hf

bash scripts/train_zero1.sh 4 \
    data=cloth_folding \
    task=cloth_folding_joint_3cam_384_1e-4 \
    model=fastwam_joint
```

### 3. Monitor Training

Training logs are written to the Hydra output directory (`./outputs/<date>/<time>/` by default, or `output_dir=<path>` override). Key metrics:
- `loss`: combined video + action flow-matching loss
- `grad_norm`: gradient norm (clipped to `max_grad_norm=1.0`)
- `lr`: learning rate (cosine schedule from 1e-4)

On first run, dataset normalization stats are computed automatically and saved to `dataset_stats.json` in the output directory.

### Config Files

| Config | Path | Purpose |
|--------|------|---------|
| Data | `configs/data/cloth_folding.yaml` | Dataset paths, camera layout, normalization |
| Task | `configs/task/cloth_folding_joint_3cam_384_1e-4.yaml` | Training hyperparameters |

### Training Settings

| Parameter | Value |
|-----------|-------|
| Model | FastWAMJoint (action attends full video) |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Epochs | 5 |
| Weight decay | 1e-2 |
| LR schedule | Cosine |
| Optimizer | AdamW (betas=0.9, 0.95) |
| Mixed precision | bf16 |
| Gradient accumulation | 1 |
| Max grad norm | 1.0 |
| Video frames | 33 obs → 9 video frames (ratio=4) |
| Action steps | 32 (num_frames - 1) |
| Camera layout | robotwin (384×320, 3 cameras) |
| Normalization | Action+State: z-score, Images: [-1, 1] |
| Text encoder | Wan2.2 T5, context_len=128 |
