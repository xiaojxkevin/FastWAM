#!/usr/bin/env python
"""Offline video generation using the FastWAM video DiT to predict future frames.

Loads a finetuned FastWAM checkpoint, reads rollout data, and generates
a continuous predicted video using a GT-conditioned sliding window:
at each replan interval a GT frame conditions the model, 32 future frames
are predicted, and the first *stride* predictions are kept.  Predictions
are stitched into one full-length video alongside ground truth.

Joint action+video inference is used (``infer_joint``) so both modalities
are denoised simultaneously through the MoT.  Generated actions are saved
alongside the video for downstream analysis.

Usage::

    conda activate fastwam
    export DIFFSYNTH_MODEL_BASE_PATH=/home/xiaojx/sea/cloth/FastWAM/checkpoints
    export DIFFSYNTH_SKIP_DOWNLOAD=true

    # Predict using the verified 0603 checkpoint directory
    python experiments/piper/generate_rollout_video.py \\
        --model-dir pretrained/0603_67_base

    # Quick test: only 32 frames (one 1s inference step)
    python experiments/piper/generate_rollout_video.py \\
        --model-dir pretrained/0603_67_base --max-frames 32

    # Dense prediction with 0.8s replan interval
    python experiments/piper/generate_rollout_video.py \\
        --model-dir pretrained/0603_67_base --replan-interval 0.8

"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
import yaml
from PIL import Image
from tqdm import tqdm

# -- project root on sys.path ------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastwam.runtime import create_fastwam
from fastwam.utils.video_io import save_mp4
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign

logger = logging.getLogger(__name__)

# -- constants ---------------------------------------------------------------
DEFAULT_TASK = "take the cloth from the basket and fold the cloth."
DEFAULT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view "
    "executing the following instruction: {task}"
)
FRONT_SIZE = (320, 256)   # (W, H) for fixed_front after resize
ARM_SIZE = (160, 128)     # (W, H) for left_arm / right_arm after resize
CANVAS_W = 320
CANVAS_H = 384            # 256 (top) + 128 (bottom)
NUM_VIDEO_FRAMES = 33     # 1 input + 32 predicted
ACTION_HORIZON = 32


# ---------------------------------------------------------------------------
# Helpers: model loading
# ---------------------------------------------------------------------------

def load_config_dict(config_path: str | Path) -> dict:
    """Load the training config YAML as a plain dict."""
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)  # returns dict


def build_model_kwargs(
    train_config: dict,
    device: str,
    model_dtype: torch.dtype,
) -> dict:
    """Extract model kwargs from a training config.

    The finetuned checkpoint is a full MoT checkpoint, so inference defaults to
    skipping duplicate Wan/ActionDiT pretrain loads before applying it.  This
    keeps peak memory low enough for 24GB cards.
    """
    mc = train_config["model"]
    return dict(
        model_id=mc["model_id"],
        tokenizer_model_id=mc["tokenizer_model_id"],
        tokenizer_max_len=int(mc.get("tokenizer_max_len", 128)),
        video_dit_config=dict(mc["video_dit_config"]),
        load_text_encoder=bool(mc.get("load_text_encoder", True)),
        proprio_dim=int(mc.get("proprio_dim", 14)),
        action_dit_config=dict(mc.get("action_dit_config", {})),
        skip_dit_load_from_pretrain=True,
        action_dit_pretrained_path="",
        video_scheduler=dict(mc.get("video_scheduler", {})),
        action_scheduler=dict(mc["action_scheduler"]),
        loss=dict(mc.get("loss", {"lambda_action": 1.0})),
        mot_checkpoint_mixed_attn=bool(mc.get("mot_checkpoint_mixed_attn", True)),
        redirect_common_files=bool(mc.get("redirect_common_files", True)),
        model_dtype=model_dtype,
        device=device,
    )


def load_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    model_dtype: torch.dtype = torch.bfloat16,
):
    """Create FastWAM model and load the finetuned checkpoint."""
    train_config = load_config_dict(config_path)
    kwargs = build_model_kwargs(train_config, device, model_dtype)
    logger.info("Creating FastWAM model (model_id=%s) …", kwargs["model_id"])
    logger.info("Deployment-style load: skipping duplicate pretrained DiT/ActionDiT before checkpoint.")
    model = create_fastwam(**kwargs)

    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = _PROJECT_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    logger.info("Loading checkpoint: %s (%.1f GB)", ckpt, ckpt.stat().st_size / 1e9)
    model.load_checkpoint(str(ckpt), mmap=True, return_payload=False)
    model = model.to(device).eval()
    logger.info("Model ready.")
    return model


def build_state_processor(train_config: dict, stats_path: str | Path) -> FastWAMProcessor:
    """Build the same state normalizer path used by ``fastwam_server.py``."""
    train_data = train_config.get("data", {}).get("train", {})
    processor_cfg = train_data.get("processor", {})
    processor = FastWAMProcessor(
        shape_meta=train_data.get("shape_meta", processor_cfg.get("shape_meta", {})),
        num_obs_steps=int(train_data.get("num_frames", processor_cfg.get("num_obs_steps", 33))),
        num_output_cameras=int(processor_cfg.get("num_output_cameras", 3)),
        action_output_dim=int(processor_cfg.get("action_output_dim", 14)),
        proprio_output_dim=int(processor_cfg.get("proprio_output_dim", 14)),
        action_state_transforms=None,
        use_stepwise_action_norm=bool(processor_cfg.get("use_stepwise_action_norm", False)),
        norm_default_mode=str(processor_cfg.get("norm_default_mode", "z-score")),
        norm_exception_mode=processor_cfg.get("norm_exception_mode", None),
        action_state_merger=ConcatLeftAlign(),
        train_transforms=None,
        val_transforms=None,
    ).eval()
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor.set_normalizer_from_stats(dataset_stats)
    return processor


def normalize_state_with_processor(processor: FastWAMProcessor, state: np.ndarray) -> torch.Tensor:
    """Normalize one raw rollout state exactly like the websocket server."""
    state_key = processor.shape_meta["state"][0]["key"]
    state_batch = {
        "state": {
            state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0),
        }
    }
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


# ---------------------------------------------------------------------------
# Helpers: cached text embeddings
# ---------------------------------------------------------------------------

def load_cached_text_embedding(
    prompt: str,
    cache_dir: str | Path,
    context_len: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a precomputed T5 text embedding for *prompt* from *cache_dir*.

    Returns ``(context, context_mask)`` with shapes ``[1, L, D]`` and ``[1, L]``.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = _PROJECT_ROOT / cache_dir

    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Text embedding cache not found for prompt. "
            f"Hash={hashed[:16]}…  Expected: {cache_file}"
        )

    logger.info("Loading text cache: %s", cache_file.name)
    payload = torch.load(str(cache_file), map_location="cpu", weights_only=False)
    context = payload["context"].clone()    # [L, D]
    mask = payload["mask"]                  # [L] bool

    # Zero out padding positions, then broadcast mask to all-True
    context[~mask] = 0.0
    mask_ones = torch.ones_like(mask, dtype=torch.bool)

    context = context.unsqueeze(0).to(device=device, dtype=dtype)       # [1, L, D]
    mask_ones = mask_ones.unsqueeze(0).to(device=device, dtype=torch.bool)  # [1, L]
    return context, mask_ones


# ---------------------------------------------------------------------------
# Helpers: frame I/O  (reads from rollout MP4s)
# ---------------------------------------------------------------------------

def _hwc_uint8_to_chw_float(image_uint8: np.ndarray) -> torch.Tensor:
    """Convert a decoded uint8 HWC RGB frame to CHW float in [0, 1]."""
    return torch.from_numpy(image_uint8).permute(2, 0, 1).to(dtype=torch.float32) / 255.0


def _resize_chw_float(image: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """Resize CHW float tensor with the same antialiased bilinear path as training."""
    return transforms_F.resize(
        image,
        size=list(size_hw),
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )


def compose_robotwin_chw_float(
    fixed_uint8: np.ndarray,
    left_uint8: np.ndarray,
    right_uint8: np.ndarray,
) -> torch.Tensor:
    """Compose three raw camera frames into robotwin CHW float in [0, 1]."""
    fixed = _hwc_uint8_to_chw_float(fixed_uint8)
    left = _hwc_uint8_to_chw_float(left_uint8)
    right = _hwc_uint8_to_chw_float(right_uint8)

    top = _resize_chw_float(fixed, (FRONT_SIZE[1], FRONT_SIZE[0]))  # [3,256,320]
    bot_l = _resize_chw_float(left, (ARM_SIZE[1], ARM_SIZE[0]))     # [3,128,160]
    bot_r = _resize_chw_float(right, (ARM_SIZE[1], ARM_SIZE[0]))    # [3,128,160]
    bottom = torch.cat([bot_l, bot_r], dim=-1)                      # [3,128,320]
    return torch.cat([top, bottom], dim=-2).contiguous()            # [3,384,320]


def chw_float_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert CHW float [0, 1] image to RGB PIL."""
    arr = (image.clamp(0.0, 1.0).permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


def normalized_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a normalized image tensor [1,3,H,W] or [3,H,W] in [-1, 1] to RGB PIL."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for debug image, got {tuple(image.shape)}")
        image = image[0]
    image = image.detach().to(device="cpu", dtype=torch.float32)
    return chw_float_to_pil((image + 1.0) * 0.5)


@torch.no_grad()
def vae_roundtrip_first_frame(model, input_tensor: torch.Tensor) -> Image.Image:
    """Encode/decode the condition frame without DiT denoising."""
    latents = model._encode_input_image_latents_tensor(input_tensor, tiled=False)
    frames = model._decode_latents(latents, tiled=False)
    if not frames:
        raise RuntimeError("VAE roundtrip returned no frames")
    return frames[0]


def save_debug_frames(
    debug_dir: Path,
    cond_frame: int,
    input_tensor: torch.Tensor,
    vae_roundtrip: Image.Image,
    joint_frames: list[Image.Image],
) -> None:
    """Save diagnostic PNGs for the first offline joint inference step."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    normalized_tensor_to_pil(input_tensor).save(debug_dir / f"cond_{cond_frame:06d}_input_condition.png")
    vae_roundtrip.save(debug_dir / f"cond_{cond_frame:06d}_vae_roundtrip.png")

    for frame_idx in (0, 1, 8, 16):
        if frame_idx < len(joint_frames):
            joint_frames[frame_idx].save(debug_dir / f"cond_{cond_frame:06d}_joint_video_{frame_idx:02d}.png")


def read_robotwin_tensor(
    video_paths: dict[str, str],
    frame_idx: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Read three camera frames and compose the robotwin 384×320 input tensor.

    Returns ``[1, 3, 384, 320]`` normalised to ``[-1, 1]``.
    """
    fixed = decode_frames_sequential(video_paths["fixed_front"], [frame_idx])[0]
    left = decode_frames_sequential(video_paths["left_arm"], [frame_idx])[0]
    right = decode_frames_sequential(video_paths["right_arm"], [frame_idx])[0]

    image = compose_robotwin_chw_float(fixed, left, right).unsqueeze(0)
    image = image.to(device=device, dtype=dtype)
    return image * 2.0 - 1.0


def read_robotwin_pil(
    video_paths: dict[str, str],
    frame_idx: int,
) -> Image.Image:
    """Like ``read_robotwin_tensor`` but returns a PIL Image (uint8, for GT)."""
    fixed = decode_frames_sequential(video_paths["fixed_front"], [frame_idx])[0]
    left = decode_frames_sequential(video_paths["left_arm"], [frame_idx])[0]
    right = decode_frames_sequential(video_paths["right_arm"], [frame_idx])[0]
    return chw_float_to_pil(compose_robotwin_chw_float(fixed, left, right))


def read_robotwin_pil_batch(
    video_paths: dict[str, str],
    frame_indices: list[int],
) -> list[Image.Image]:
    """Read multiple robotwin frames at once — single decode call per camera."""
    fixed = decode_frames_sequential(video_paths["fixed_front"], frame_indices)
    left = decode_frames_sequential(video_paths["left_arm"], frame_indices)
    right = decode_frames_sequential(video_paths["right_arm"], frame_indices)

    result: list[Image.Image] = []
    for i in range(len(frame_indices)):
        result.append(chw_float_to_pil(compose_robotwin_chw_float(fixed[i], left[i], right[i])))
    return result


# ---------------------------------------------------------------------------
# Video frame decoding (torchcodec)
# ---------------------------------------------------------------------------

def decode_frames_sequential(
    video_path: str,
    indices: list[int],
) -> np.ndarray:
    """Decode frames at given indices from *video_path*.

    Returns uint8 ndarray ``[N, H, W, 3]``.
    """
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(str(video_path), device="cpu")
    batch = decoder.get_frames_at(indices=indices)
    frames = batch.data.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
    return frames.numpy()


# ---------------------------------------------------------------------------
# Model directory auto-discovery
# ---------------------------------------------------------------------------

def _resolve_model_dir(model_dir: str | Path) -> dict[str, Path]:
    """Auto-discover config, checkpoint, and stats from a single model directory.

    Expects a folder containing ``config.yaml``, ``step_*.pt``, and
    ``dataset_stats.json``.  Returns a dict with keys ``config``,
    ``checkpoint``, ``stats``.
    """
    md = Path(model_dir)
    if not md.is_absolute():
        md = _PROJECT_ROOT / md
    if not md.is_dir():
        raise FileNotFoundError(f"Model directory not found: {md}")

    # -- config -----------------------------------------------------------
    config_path = md / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {md}")

    # -- checkpoint (pick the latest step_*.pt) ---------------------------
    ckpt_candidates = sorted(md.glob("step_*.pt"))
    if not ckpt_candidates:
        raise FileNotFoundError(f"No step_*.pt checkpoint found in {md}")
    checkpoint_path = ckpt_candidates[-1]  # latest by lexical order
    if len(ckpt_candidates) > 1:
        logger.info(
            "Found %d checkpoints in %s; using latest: %s",
            len(ckpt_candidates), md.name, checkpoint_path.name,
        )

    # -- dataset stats ----------------------------------------------------
    stats_path = md / "dataset_stats.json"
    if not stats_path.exists():
        logger.warning("dataset_stats.json not found in %s — proprio conditioning disabled.", md)
        stats_path = None

    return {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "stats": stats_path,
    }


# ---------------------------------------------------------------------------
# Video inference (joint action + video)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_joint_inference(
    model,
    input_tensor: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    num_inference_steps: int = 10,
    seed: int = 42,
    tiled: bool = False,
    quiet: bool = False,
    proprio: torch.Tensor | None = None,
) -> dict:
    """Run joint video+action inference through the MoT.

    Calls ``model.infer_joint()`` directly with
    ``test_action_with_infer_action=False`` to avoid the redundant
    standalone ``infer_action`` pre-call.  Both video frames and action
    tokens are denoised simultaneously through shared MoT layers.

    Returns ``{"video": list[PIL.Image], "action": Tensor[T, D]}``.
    """
    if not quiet:
        logger.info(
            "Running joint video+action inference (%d steps, seed=%d) …",
            num_inference_steps, seed,
        )
    output = model.infer_joint(
        prompt=None,
        context=context,
        context_mask=context_mask,
        input_image=input_tensor,
        num_video_frames=NUM_VIDEO_FRAMES,
        action_horizon=ACTION_HORIZON,
        action=None,          # no GT action — video conditioned on first frame + text
        proprio=proprio,
        text_cfg_scale=1.0,
        num_inference_steps=num_inference_steps,
        seed=seed,
        tiled=tiled,
        test_action_with_infer_action=False,  # ← skip redundant pre-call
    )
    return output


# ---------------------------------------------------------------------------
# Frame stitching helpers
# ---------------------------------------------------------------------------

def stitch_pred_gt(pred_frame: Image.Image, gt_frame: Image.Image) -> Image.Image:
    """Stitch a single pred (top) / GT (bottom) pair vertically."""
    pw, ph = pred_frame.size
    gw, gh = gt_frame.size
    if pw != gw:
        new_w = max(pw, gw)
        pred_frame = pred_frame.resize((new_w, ph), Image.BILINEAR)
        gt_frame = gt_frame.resize((new_w, gh), Image.BILINEAR)
        pw, ph = pred_frame.size
        gw, gh = gt_frame.size
    raw_h = ph + gh + 4
    padded_h = (raw_h + 15) // 16 * 16
    canvas = Image.new("RGB", (pw, padded_h), (0, 0, 0))
    canvas.paste(pred_frame, (0, 0))
    canvas.paste(gt_frame, (0, ph + 4))
    return canvas


# ---------------------------------------------------------------------------
# Video path discovery
# ---------------------------------------------------------------------------

def discover_video_paths(dataset_path: Path) -> dict[str, str]:
    """Locate the three camera MP4 files in a LeRobot-format dataset."""
    video_base = dataset_path / "videos" / "chunk-000"
    paths = {}
    for cam in ("fixed_front", "left_arm", "right_arm"):
        cam_dir = video_base / f"observation.images.{cam}"
        candidates = sorted(cam_dir.glob("episode_*.mp4"))
        if not candidates:
            raise FileNotFoundError(f"No MP4 for camera '{cam}' in {cam_dir}")
        paths[cam] = str(candidates[0])
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline video generation via FastWAM MoT joint inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ---- model ---------------------------------------------------------
    parser.add_argument(
        "--model-dir", type=Path, default=Path("pretrained/0603_67_base"),
        help="Directory containing config.yaml, step_*.pt, and dataset_stats.json.",
    )
    # ---- data ----------------------------------------------------------
    parser.add_argument(
        "--dataset-path", type=Path,
        default=Path("data/eval_0622_fastwam_67_pretrain_rtc"),
        help="LeRobot-format rollout directory.",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Task prompt for text embedding lookup (default: built-in cloth folding).",
    )
    parser.add_argument(
        "--text-cache-dir", type=str,
        default="data/text_embeds_cache/cloth_folding",
        help="Directory of precomputed T5 embeddings.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--debug-frames", type=Path, nargs="?", const=Path("debug_frames"), default=None,
        help="Save condition, VAE roundtrip, and selected joint video PNGs. "
             "Defaults to output-dir/debug_frames when no path is provided.",
    )
    # ---- inference -----------------------------------------------------
    parser.add_argument(
        "--replan-interval", type=float, default=None,
        help="Seconds between inference starts (default: matches prediction length = 32/30 s).",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=10,
        help="Diffusion denoising steps (default: 10).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Max number of frames to predict (default: entire rollout).",
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="First frame index to start inference from (default: 0).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for diffusion noise. Default None matches the old joint server config.",
    )
    # ---- system --------------------------------------------------------
    parser.add_argument(
        "--device", type=str, default="cuda",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # -- resolve model paths ----------------------------------------------
    resolved = _resolve_model_dir(args.model_dir)
    config_path = resolved["config"]
    checkpoint_path = resolved["checkpoint"]
    stats_path = resolved["stats"]
    logger.info("Using model dir: %s", args.model_dir)

    logger.info("Config:  %s", config_path)
    logger.info("Checkpoint: %s", checkpoint_path)
    if stats_path:
        logger.info("Stats:   %s", stats_path)
    else:
        logger.info("Stats:   (none — proprio conditioning disabled)")

    # -- resolve output / dataset paths -----------------------------------
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.is_dir():
        parser.error(f"Dataset not found: {dataset_path}")

    if args.output_dir is None:
        output_dir = dataset_path / "predicted_videos"
    else:
        output_dir = args.output_dir.resolve()
    debug_frames_dir = None
    if args.debug_frames is not None:
        debug_frames_dir = args.debug_frames
        if not debug_frames_dir.is_absolute():
            debug_frames_dir = output_dir / debug_frames_dir

    prompt = args.prompt or DEFAULT_PROMPT_TEMPLATE.format(task=DEFAULT_TASK)
    logger.info("Prompt: %s", prompt)

    # -- locate videos ---------------------------------------------------
    video_paths = discover_video_paths(dataset_path)

    # -- load model ------------------------------------------------------
    dtype = torch.bfloat16
    train_config = load_config_dict(config_path)
    model = load_model(
        config_path,
        checkpoint_path,
        device=args.device,
        model_dtype=dtype,
    )

    # -- load text embeddings --------------------------------------------
    context, context_mask = load_cached_text_embedding(
        prompt, args.text_cache_dir, device=args.device, dtype=dtype,
    )

    # -- load state normalizer (for proprio conditioning) -----------------
    state_processor = None
    if stats_path is not None and stats_path.exists():
        logger.info("Loading dataset stats via FastWAMProcessor: %s", stats_path)
        state_processor = build_state_processor(train_config, stats_path)
        logger.info("State processor ready.")

    # -- read rollout states from parquet (small: ~0.5 MB) ------------------
    import pandas as pd
    parquet_path = dataset_path / "data" / "chunk-000" / "episode_000000.parquet"
    rollout_states = None
    if parquet_path.exists() and state_processor is not None:
        df = pd.read_parquet(parquet_path)
        rollout_states = np.stack(df["observation.state"].values)  # [N, 14]
        logger.info("Loaded %d state vectors from parquet.", len(rollout_states))
    elif state_processor is None:
        logger.info("No state processor — skipping proprio conditioning.")
    else:
        logger.warning("Parquet not found: %s", parquet_path)

    # -- determine prediction plan ---------------------------------------
    # Read episode metadata for total frame count
    meta_dir = dataset_path / "meta"
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    total_frames = info["total_frames"]
    fps = info["fps"]

    pred_per_step = NUM_VIDEO_FRAMES - 1  # 32 predicted frames per inference

    # Stride defaults to pred_per_step (each prediction covers exactly
    # the gap to the next condition frame).
    if args.replan_interval is not None:
        stride_frames = int(args.replan_interval * fps)
    else:
        stride_frames = pred_per_step

    # We predict from start_frame+1 onwards.  Each inference step
    # generates 32 frames; we keep the first `stride_frames` of them
    # and advance the condition frame by `stride_frames`.
    target_pred_frames = total_frames - args.start_frame - 1  # frames to predict
    if args.max_frames is not None:
        target_pred_frames = min(target_pred_frames, args.max_frames)

    num_steps = (target_pred_frames + stride_frames - 1) // stride_frames
    logger.info(
        "Replan: %d frames stride (%.2f s) | %d predicted per step | %d total steps | %d target frames",
        stride_frames, stride_frames / fps, pred_per_step, num_steps, target_pred_frames,
    )

    # -- run inference loop: accumulate predicted + GT frames -------------
    all_pred_frames: list[Image.Image] = []
    all_gt_frames: list[Image.Image] = []
    all_actions: list[np.ndarray] = []  # collect per-step actions
    frames_generated = 0

    for step_idx in tqdm(range(num_steps), desc="Inference", unit="step"):
        cond_frame = args.start_frame + step_idx * stride_frames
        if cond_frame + pred_per_step >= total_frames:
            break

        # Read condition frame (GT) as model input
        input_tensor = read_robotwin_tensor(
            video_paths, cond_frame,
            device=args.device, dtype=dtype,
        )
        vae_roundtrip = None
        if debug_frames_dir is not None and step_idx == 0:
            vae_roundtrip = vae_roundtrip_first_frame(model, input_tensor)

        # Normalize proprio state (if available)
        proprio = None
        if rollout_states is not None and state_processor is not None:
            raw_state = rollout_states[cond_frame]  # [14]
            proprio = normalize_state_with_processor(state_processor, raw_state)
            proprio = proprio.to(device=args.device, dtype=dtype)

        # Run joint video+action inference → 33 frames (1 recon + 32 pred)
        # + 32 action steps
        output = run_joint_inference(
            model, input_tensor, context, context_mask,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            quiet=True,
            proprio=proprio,
        )

        # Discard index 0 = reconstructed input frame
        pred_frames = output["video"][1:]   # list[32] of PIL Image
        pred_action = output["action"]       # Tensor [32, 14] float32
        if debug_frames_dir is not None and step_idx == 0:
            save_debug_frames(
                debug_frames_dir,
                cond_frame,
                input_tensor,
                vae_roundtrip,
                output["video"],
            )
            logger.info("Saved debug PNG frames to %s", debug_frames_dir)

        # How many predicted frames to keep from this step
        keep_n = min(stride_frames, target_pred_frames - frames_generated)
        if keep_n <= 0:
            break

        all_pred_frames.extend(pred_frames[:keep_n])
        all_actions.append(pred_action[:keep_n].cpu().numpy())
        frames_generated += keep_n

    # -- save generated actions ------------------------------------------
    if all_actions:
        actions_array = np.concatenate(all_actions, axis=0)  # [total_pred, 14]
        action_path = output_dir / "predicted_actions.npy"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(action_path), actions_array.astype(np.float32))
        logger.info(
            "Saved predicted actions: %s (shape=%s)",
            action_path, actions_array.shape,
        )

    # -- read GT frames for the same range (batched) ----------------------
    gt_start = args.start_frame + 1  # first predicted frame position
    gt_count = frames_generated
    logger.info("Reading %d GT frames (starting at frame %d, batched) …", gt_count, gt_start)
    gt_indices = list(range(gt_start, gt_start + gt_count))
    GT_CHUNK = 300
    for chunk_start in tqdm(range(0, len(gt_indices), GT_CHUNK), desc="GT frames", unit="chunk"):
        chunk = gt_indices[chunk_start : chunk_start + GT_CHUNK]
        all_gt_frames.extend(read_robotwin_pil_batch(video_paths, chunk))

    # -- stitch and save --------------------------------------------------
    logger.info("Stitching %d pred/GT frame pairs …", frames_generated)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predicted_vs_gt.mp4"
    stitched = [stitch_pred_gt(p, g) for p, g in zip(all_pred_frames, all_gt_frames)]
    save_mp4(stitched, str(output_path), fps=fps)
    logger.info("Done — video saved to %s (%d frames, %d fps, %.1f s)",
                output_path, frames_generated, fps, frames_generated / fps)


if __name__ == "__main__":
    main()
