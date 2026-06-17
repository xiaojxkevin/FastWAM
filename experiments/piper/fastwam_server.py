"""Websocket inference server for FastWAM cloth folding model.

Serves a trained FastWAM checkpoint over websocket using msgpack_numpy
serialisation.  Clients send observations (images + state + prompt) and
receive denormalised action chunks.

Protocol (msgpack_numpy over websocket)
---------------------------------------
Frame 0   server -> client :  metadata dict  {"action_dim": 14, "action_horizon": 32}
Frame 1+  client -> server :  {"state": ndarray[14], "images": {...}, "prompt": str}
          server -> client :  {"actions": ndarray[T, 14]}

Image preprocessing (robotwin layout)
-------------------------------------
3 cameras at raw 480x640 are resized and concatenated exactly as the
training dataset does (robot_video_dataset.py:154-178):

  fixed_front (cam 0)  ->  resize(320,256)  ->  top    [256,320,3]
  left_arm    (cam 1)  ->  resize(160,128)  ->  bottom-left  [128,160,3]
  right_arm   (cam 2)  ->  resize(160,128)  ->  bottom-right [128,160,3]

  bottom = hstack(left, right)               -> [128,320,3]
  combined = vstack(top, bottom)             -> [384,320,3]

References
----------
- kai0/src/openpi/serving/websocket_policy_server.py  (websocket pattern)
- FastWAM/experiments/robotwin/fastwam_policy/deploy_policy.py  (model loading + norm)
- FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py  (image concat)
"""

import asyncio
import hashlib
import http
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# Project root for resolving relative paths in config values (e.g. cache dir).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
import websockets.asyncio.server as _server
import websockets.frames

from .msgpack_numpy import packb, unpackb

from fastwam.runtime import create_fastwam
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign

logger = logging.getLogger(__name__)


class FastWAMCudaGraphRunner:
    """Reusable CUDA-graph runner for fixed-shape uncond action denoising."""

    def __init__(self, model, action_horizon: int, num_inference_steps: int, device: str):
        self._model = model
        self._action_horizon = int(action_horizon)
        self._num_inference_steps = int(num_inference_steps)
        self._device = torch.device(device)
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_latents_action: Optional[torch.Tensor] = None
        self._static_context: Optional[torch.Tensor] = None
        self._static_context_mask: Optional[torch.Tensor] = None
        self._static_prefix_tokens: Optional[torch.Tensor] = None
        self._static_prefix_mask: Optional[torch.Tensor] = None
        self._static_step_timesteps: Optional[torch.Tensor] = None
        self._static_step_deltas: Optional[torch.Tensor] = None
        self._static_attention_mask: Optional[torch.Tensor] = None
        self._static_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None
        self._captured_signature: Optional[tuple] = None

    def _shape_signature(
        self,
        latents_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        prefix_tokens: Optional[torch.Tensor],
        prefix_mask: Optional[torch.Tensor],
    ) -> tuple:
        cache_sig = tuple((tuple(layer["k"].shape), tuple(layer["v"].shape)) for layer in video_kv_cache)
        return (
            tuple(latents_action.shape),
            tuple(context.shape),
            tuple(context_mask.shape),
            tuple(attention_mask.shape),
            cache_sig,
            None if prefix_tokens is None else tuple(prefix_tokens.shape),
            None if prefix_mask is None else tuple(prefix_mask.shape),
            self._num_inference_steps,
        )

    def _ensure_captured(
        self,
        latents_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        step_timesteps: torch.Tensor,
        step_deltas: torch.Tensor,
        prefix_tokens: Optional[torch.Tensor],
        prefix_mask: Optional[torch.Tensor],
    ) -> None:
        signature = self._shape_signature(
            latents_action=latents_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
        )
        if self._graph is not None and self._captured_signature == signature:
            return

        if not torch.cuda.is_available() or not str(self._device).startswith("cuda"):
            raise RuntimeError("CUDA Graph runner requires a CUDA device.")
        if latents_action.device.type != "cuda":
            raise RuntimeError("CUDA Graph runner requires CUDA tensors.")

        self._static_latents_action = latents_action.clone()
        self._static_context = context.clone()
        self._static_context_mask = context_mask.clone()
        self._static_prefix_tokens = (
            torch.zeros_like(latents_action) if prefix_tokens is None else prefix_tokens.clone()
        )
        self._static_prefix_mask = (
            torch.zeros((1, self._action_horizon, 1), dtype=torch.bool, device=self._device)
            if prefix_mask is None else prefix_mask.clone()
        )
        self._static_step_timesteps = step_timesteps.clone()
        self._static_step_deltas = step_deltas.clone()
        self._static_attention_mask = attention_mask.clone()
        self._static_video_kv_cache = [
            {"k": layer["k"].clone(), "v": layer["v"].clone()} for layer in video_kv_cache
        ]

        stream = torch.cuda.Stream(device=self._device)
        torch.cuda.synchronize(device=self._device)
        with torch.cuda.stream(stream):
            for _ in range(2):
                _ = self._model._predict_action_noise_with_cache(
                    latents_action=self._static_latents_action,
                    timestep_action=self._static_step_timesteps[0].expand(1, self._action_horizon),
                    context=self._static_context,
                    context_mask=self._static_context_mask,
                    video_kv_cache=self._static_video_kv_cache,
                    attention_mask=self._static_attention_mask,
                    video_seq_len=video_seq_len,
                )
        torch.cuda.current_stream(device=self._device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(self._num_inference_steps):
                timestep_step_input = self._static_step_timesteps[i].expand(1, self._action_horizon).clone()
                timestep_step_input = torch.where(
                    self._static_prefix_mask.squeeze(-1),
                    torch.zeros_like(timestep_step_input),
                    timestep_step_input,
                )
                action_step_input = torch.where(
                    self._static_prefix_mask,
                    self._static_prefix_tokens,
                    self._static_latents_action,
                )
                pred_action = self._model._predict_action_noise_with_cache(
                    latents_action=action_step_input,
                    timestep_action=timestep_step_input,
                    context=self._static_context,
                    context_mask=self._static_context_mask,
                    video_kv_cache=self._static_video_kv_cache,
                    attention_mask=self._static_attention_mask,
                    video_seq_len=video_seq_len,
                )
                updated = self._model.infer_action_scheduler.step(
                    pred_action, self._static_step_deltas[i], self._static_latents_action
                )
                self._static_latents_action.copy_(updated)
                self._static_latents_action.copy_(
                    torch.where(self._static_prefix_mask, self._static_prefix_tokens, self._static_latents_action)
                )
        self._graph = graph
        self._captured_signature = signature

    def run(
        self,
        latents_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        step_timesteps: torch.Tensor,
        step_deltas: torch.Tensor,
        prefix_tokens: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._graph is None:
            self._ensure_captured(
                latents_action=latents_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                step_timesteps=step_timesteps,
                step_deltas=step_deltas,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
            )
        else:
            assert self._static_latents_action is not None
            assert self._static_context is not None
            assert self._static_context_mask is not None
            assert self._static_prefix_tokens is not None
            assert self._static_prefix_mask is not None
            assert self._static_step_timesteps is not None
            assert self._static_step_deltas is not None
            assert self._static_attention_mask is not None
            assert self._static_video_kv_cache is not None
            self._static_latents_action.copy_(latents_action)
            self._static_context.copy_(context)
            self._static_context_mask.copy_(context_mask)
            self._static_prefix_tokens.copy_(torch.zeros_like(self._static_prefix_tokens) if prefix_tokens is None else prefix_tokens)
            if prefix_mask is None:
                self._static_prefix_mask.zero_()
            else:
                self._static_prefix_mask.copy_(prefix_mask)
            self._static_step_timesteps.copy_(step_timesteps)
            self._static_step_deltas.copy_(step_deltas)
            self._static_attention_mask.copy_(attention_mask)
            for src_layer, dst_layer in zip(video_kv_cache, self._static_video_kv_cache):
                dst_layer["k"].copy_(src_layer["k"])
                dst_layer["v"].copy_(src_layer["v"])

        assert self._graph is not None
        self._graph.replay()
        assert self._static_latents_action is not None
        return self._static_latents_action


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class FastWAMModelWrapper:
    """Loads the FastWAM model, processor, and normalisation stats once at startup,
    then exposes a synchronous ``infer(observation) -> dict`` method."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        inf_cfg = cfg["inference"]

        # --- resolve dtype & device -----------------------------------------
        dtype_str = cfg.get("model_dtype", "bf16")
        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
        if dtype_str not in _dtype_map:
            raise ValueError(f"Unknown model_dtype: {dtype_str}")
        self._model_dtype = _dtype_map[dtype_str]
        self._device = str(cfg.get("device", "cuda"))

        # --- mode: only uncond is supported --------------------------------
        mode = str(cfg.get("mode", "uncond"))
        if mode in ("joint", "idm"):
            raise NotImplementedError(
                f"Mode {mode!r} is not supported. Only 'uncond' mode is available."
            )
        if mode != "uncond":
            raise ValueError(f"Unknown mode: {mode!r}. Must be 'uncond'.")

        # --- build model ---------------------------------------------------
        logger.info("Creating FastWAM model …")
        self._model = create_fastwam(
            model_id=str(model_cfg["model_id"]),
            tokenizer_model_id=str(model_cfg["tokenizer_model_id"]),
            tokenizer_max_len=int(model_cfg.get("tokenizer_max_len", 128)),
            video_dit_config=dict(model_cfg["video_dit_config"]),
            load_text_encoder=bool(model_cfg.get("load_text_encoder", True)),
            proprio_dim=int(model_cfg.get("proprio_dim", 14)),
            action_dit_config=dict(model_cfg["action_dit_config"]),
            action_dit_pretrained_path=str(model_cfg.get("action_dit_pretrained_path", "")),
            skip_dit_load_from_pretrain=bool(model_cfg.get("skip_dit_load_from_pretrain", False)),
            video_scheduler=dict(model_cfg.get("video_scheduler", {})),
            action_scheduler=dict(model_cfg["action_scheduler"]),
            loss=dict(model_cfg.get("loss", {"lambda_action": 1.0})),
            mot_checkpoint_mixed_attn=bool(model_cfg.get("mot_checkpoint_mixed_attn", True)),
            model_dtype=self._model_dtype,
            device=self._device,
        )

        # --- load finetuned checkpoint -------------------------------------
        ckpt_path = Path(cfg["checkpoint_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info("Loading checkpoint: %s", ckpt_path)
        self._model.load_checkpoint(str(ckpt_path), mmap=True, return_payload=False)
        self._model = self._model.to(self._device).eval()

        # --- build processor & load normalisation stats --------------------
        logger.info("Loading dataset normalisation stats …")
        shape_meta = data_cfg["shape_meta"]
        self._processor = FastWAMProcessor(
            shape_meta=shape_meta,
            num_obs_steps=int(data_cfg.get("num_obs_steps", 33)),
            num_output_cameras=int(data_cfg.get("num_output_cameras", 3)),
            action_output_dim=int(data_cfg.get("action_output_dim", 14)),
            proprio_output_dim=int(data_cfg.get("proprio_output_dim", 14)),
            action_state_transforms=None,
            use_stepwise_action_norm=False,
            norm_default_mode=str(data_cfg.get("norm_default_mode", "z-score")),
            norm_exception_mode=None,
            action_state_merger=ConcatLeftAlign(),
            train_transforms=None,
            val_transforms=None,
        ).eval()

        stats_path = Path(data_cfg["dataset_stats_path"])
        if not stats_path.exists():
            raise FileNotFoundError(f"Dataset stats not found: {stats_path}")
        dataset_stats = load_dataset_stats_from_json(str(stats_path))
        self._processor.set_normalizer_from_stats(dataset_stats)

        # --- text context mode ----------------------------------------------
        self._text_encoder_loaded = bool(model_cfg.get("load_text_encoder", True))
        self._text_cache_dir: Optional[Path] = None
        self._context_len: int = int(model_cfg.get("tokenizer_max_len", 128))
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if not self._text_encoder_loaded:
            cache_dir_str = data_cfg.get("text_embedding_cache_dir", "")
            if not cache_dir_str:
                raise ValueError(
                    "`data.text_embedding_cache_dir` is required when `load_text_encoder=false`. "
                    "Provide the path to precomputed T5 text embedding cache files."
                )
            self._text_cache_dir = Path(cache_dir_str)
            if not self._text_cache_dir.is_absolute():
                self._text_cache_dir = _PROJECT_ROOT / self._text_cache_dir
            if not self._text_cache_dir.is_dir():
                raise FileNotFoundError(
                    f"Text embedding cache directory not found: {self._text_cache_dir}. "
                    "Run `scripts/precompute_text_embeds.py` first, or set `load_text_encoder=true`."
                )
            logger.info("Text context mode: cache (dir=%s)", self._text_cache_dir)
        else:
            logger.info("Text context mode: online encoder")

        # --- inference params ------------------------------------------------
        self._action_horizon = int(inf_cfg.get("action_horizon", 32))
        self._proprio_dim = int(data_cfg.get("proprio_output_dim", 14))
        self._num_inference_steps = int(inf_cfg.get("num_inference_steps", 20))
        self._seed = inf_cfg.get("seed")  # None or int
        self._accel_cfg = dict(inf_cfg.get("acceleration", {}))
        self._accel_enabled = bool(self._accel_cfg.get("enabled", False))
        self._torch_compile_enabled = bool(self._accel_cfg.get("torch_compile", False))
        self._compile_prefill_enabled = bool(self._accel_cfg.get("compile_prefill", True))
        self._compile_action_enabled = bool(self._accel_cfg.get("compile_action", True))
        self._cuda_graph_enabled = bool(self._accel_cfg.get("cuda_graph", False))
        self._compile_mode = str(self._accel_cfg.get("compile_mode", "default"))
        self._compiled_prefill = None
        self._compiled_action_step = None
        self._cuda_graph_runner = None

        if self._accel_enabled and self._torch_compile_enabled and hasattr(torch, "compile"):
            self._maybe_compile_model()
        if self._accel_enabled and self._cuda_graph_enabled:
            if str(self._device).startswith("cuda"):
                self._cuda_graph_runner = FastWAMCudaGraphRunner(
                    model=self._model,
                    action_horizon=self._action_horizon,
                    num_inference_steps=self._num_inference_steps,
                    device=self._device,
                )
                logger.info("Enabled CUDA Graph runner for uncond action denoising.")
            else:
                logger.warning(
                    "CUDA Graph acceleration requires a CUDA device (current=%s); "
                    "falling back to torch.compile/eager.",
                    self._device,
                )

        # ---- init-time warmup + CUDA graph pre-capture -------------------
        self._init_warmup_and_capture()

        logger.info(
            "FastWAM server ready | device=%s dtype=%s action_horizon=%d",
            self._device, dtype_str, self._action_horizon,
        )

    # -------------------------------------------------------------------
    # Inference acceleration
    # -------------------------------------------------------------------

    def _maybe_compile_model(self) -> None:
        compile_kwargs = {} if self._compile_mode == "default" else {"mode": self._compile_mode}
        try:
            if self._compile_prefill_enabled:
                self._compiled_prefill = torch.compile(self._model.mot.prefill_video_cache, **compile_kwargs)
                self._model.mot.prefill_video_cache = self._compiled_prefill
                logger.info("Enabled torch.compile for MoT video KV prefill (mode=%s)", self._compile_mode)
            if self._compile_action_enabled:
                self._compiled_action_step = torch.compile(
                    self._model.mot.forward_action_with_video_cache, **compile_kwargs
                )
                self._model.mot.forward_action_with_video_cache = self._compiled_action_step
                logger.info("Enabled torch.compile for MoT cached action forward (mode=%s)", self._compile_mode)
        except Exception:
            logger.exception("torch.compile setup failed; falling back to eager inference.")
            self._compiled_prefill = None
            self._compiled_action_step = None

    # -------------------------------------------------------------------
    # Init-time warmup & CUDA graph pre-capture
    # -------------------------------------------------------------------
    _WARMUP_ITERS: int = 2  # hard-coded: enough to trigger JIT + kernel warmup

    def _init_warmup_and_capture(self) -> None:
        """Run warmup and pre-capture CUDA graphs during ``__init__``.

        Eliminates cold-start latency on the very first real request:
        1. Dummy inference triggers ``torch.compile`` JIT compilation.
        2. CUDA graph is pre-captured so ``graph.replay()`` is ready immediately.

        If anything fails, acceleration degrades gracefully — the eager /
        ``torch.compile`` path still works for subsequent requests.
        """
        if not self._accel_enabled:
            return

        try:
            # Resolve text context (default prompt via cache, or dummy).
            _warmup_prompt = DEFAULT_PROMPT.format(
                task="take the cloth from the basket and fold the cloth."
            )
            try:
                infer_prompt, context, context_mask = self._get_text_context(_warmup_prompt)
            except Exception:
                logger.warning(
                    "Could not resolve text context for init warmup; using dummy context."
                )
                infer_prompt = None
                context = torch.zeros(
                    (1, self._context_len, self._model.text_dim),
                    device=self._device,
                    dtype=self._model_dtype,
                )
                context_mask = torch.ones(
                    (1, self._context_len), dtype=torch.bool, device=self._device
                )

            image = torch.zeros(
                (1, 3, 384, 320), device=self._device, dtype=self._model_dtype
            )
            proprio = torch.zeros(
                (1, self._proprio_dim), device=self._device, dtype=self._model_dtype
            )

            # Phase 1: warmup — triggers torch.compile JIT + CUDA kernel warmup.
            for _ in range(self._WARMUP_ITERS):
                with torch.no_grad():
                    self._model.infer_action(
                        prompt=infer_prompt,
                        context=context,
                        context_mask=context_mask,
                        input_image=image,
                        action_horizon=self._action_horizon,
                        proprio=proprio,
                        num_inference_steps=self._num_inference_steps,
                        seed=self._seed,
                    )
            if torch.cuda.is_available() and str(self._device).startswith("cuda"):
                torch.cuda.synchronize()
            logger.info(
                "Init warmup completed (iters=%d, compile_mode=%s).",
                self._WARMUP_ITERS,
                self._compile_mode,
            )

            # Phase 2: pre-capture CUDA graph so first real request replays instantly.
            if self._cuda_graph_runner is not None:
                # Capture with a non-zero prefix so both torch.where branches
                # (prefix & suffix) are exercised.  The same graph handles
                # RTC and non-RTC because shapes are identical — only the
                # boolean values in prefix_mask differ across requests.
                dummy_prefix = torch.zeros(
                    (1, self._action_horizon, self._model.action_expert.action_dim),
                    device=self._device,
                    dtype=self._model_dtype,
                )
                with torch.no_grad():
                    self._model.infer_action(
                        prompt=infer_prompt,
                        context=context,
                        context_mask=context_mask,
                        input_image=image,
                        action_horizon=self._action_horizon,
                        proprio=proprio,
                        num_inference_steps=self._num_inference_steps,
                        seed=self._seed,
                        action_prefix=dummy_prefix,
                        prefix_delay=min(5, self._action_horizon),
                        action_denoise_runner=self._cuda_graph_runner.run,
                    )
                if torch.cuda.is_available() and str(self._device).startswith("cuda"):
                    torch.cuda.synchronize()
                logger.info("CUDA graph pre-captured successfully.")

        except Exception:
            logger.exception(
                "Init warmup/capture failed; serving will continue with eager/compile path."
            )

    def _maybe_warmup_uncond(self, prompt: Optional[str], context: Optional[torch.Tensor], context_mask: Optional[torch.Tensor]) -> None:
        """No-op: warmup already happened in ``_init_warmup_and_capture`` during init."""

    # -------------------------------------------------------------------
    # Image preprocessing (robotwin layout — matches training dataset)
    # -------------------------------------------------------------------

    @staticmethod
    def _resize_rgb(image_uint8: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
        pil_img = Image.fromarray(image_uint8, mode="RGB")
        resized = pil_img.resize(size_wh, Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)

    def _preprocess_images(self, images_dict: dict[str, np.ndarray]) -> torch.Tensor:
        """Convert raw camera images to a single robotwin-concatenated tensor.

        ``images_dict`` keys must be ``"fixed_front"``, ``"left_arm"``,
        ``"right_arm"`` (raw uint8 HWC at 480×640).
        Returns ``[1, 3, 384, 320]`` normalised to [-1, 1].
        """
        top = self._resize_rgb(images_dict["fixed_front"], (320, 256))   # [256, 320, 3]
        left = self._resize_rgb(images_dict["left_arm"], (160, 128))     # [128, 160, 3]
        right = self._resize_rgb(images_dict["right_arm"], (160, 128))   # [128, 160, 3]

        bottom = np.concatenate([left, right], axis=1)   # [128, 320, 3]
        combined = np.concatenate([top, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device=self._device, dtype=self._model_dtype)
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor  # [1, 3, 384, 320]

    # -------------------------------------------------------------------
    # State normalisation
    # -------------------------------------------------------------------

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """z-score normalise a raw 14-dim proprio state vector."""
        state_key = self._processor.shape_meta["state"][0]["key"]
        state_batch = {
            "state": {
                state_key: torch.as_tensor(np.array(state, dtype=np.float32, copy=True)).unsqueeze(0),
            }
        }
        state_batch = self._processor.action_state_transform(state_batch)
        state_batch = self._processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]  # [1, 14]

    # -------------------------------------------------------------------
    # Action denormalisation
    # -------------------------------------------------------------------

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """Reverse z-score normalisation on predicted actions."""
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_key = self._processor.shape_meta["action"][0]["key"]
        normalizer = self._processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _normalize_action(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Apply the same z-score normalisation used by the action dataset."""
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(np.array(action, dtype=np.float32, copy=True))
        else:
            action = action.detach().to(dtype=torch.float32, device="cpu")
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_key = self._processor.shape_meta["action"][0]["key"]
        normalizer = self._processor.normalizer.normalizers["action"][action_key]
        norm_action = normalizer.forward(action)
        return norm_action.to(device=self._device, dtype=self._model_dtype)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def shutdown(self) -> None:
        """Graceful shutdown (no-op for uncond mode)."""

    # -------------------------------------------------------------------
    # Text context (cache or online encoder)
    # -------------------------------------------------------------------

    def _get_text_context(self, prompt: str) -> tuple[Optional[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return ``(prompt, context, context_mask)`` for ``infer_action``.

        When ``load_text_encoder=true``: ``prompt`` is returned as-is,
        ``context``/``context_mask`` are ``None``.  ``infer_action`` will
        call ``encode_prompt`` online.

        When ``load_text_encoder=false``: ``prompt`` is ``None``, and
        precomputed ``context``/``context_mask`` tensors are loaded from
        the disk cache (SHA‑256 hashed filename, same naming convention as
        ``scripts/precompute_text_embeds.py``).
        """
        if self._text_encoder_loaded:
            return prompt, None, None

        # Check in‑memory cache first.
        cached = self._text_context_cache.get(prompt)
        if cached is not None:
            return None, cached[0], cached[1]

        # Load from disk.
        assert self._text_cache_dir is not None
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_file = self._text_cache_dir / f"{hashed}.t5_len{self._context_len}.wan22ti2v5b.pt"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Text embedding cache not found for prompt {prompt!r}. "
                f"Expected: {cache_file}. "
                "Run `scripts/precompute_text_embeds.py` for this prompt, "
                "or set `load_text_encoder=true` in the server config."
            )

        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        context = payload["context"]   # [L, D]
        mask = payload["mask"]          # [L]

        if context.ndim != 2:
            raise ValueError(f"Cached context must be 2D [L,D], got {tuple(context.shape)}")
        if mask.ndim != 1:
            raise ValueError(f"Cached mask must be 1D [L], got {tuple(mask.shape)}")

        # Match fourier convention: zero out padded rows, then set full‑True mask.
        context = context.clone()
        context[~mask] = 0.0
        mask_ones = torch.ones_like(mask, dtype=torch.bool)

        # Add batch dim: [1, L, D] and [1, L]
        context = context.unsqueeze(0)
        mask_ones = mask_ones.unsqueeze(0)

        self._text_context_cache[prompt] = (context, mask_ones)
        return None, context, mask_ones

    # -------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------

    def infer(self, observation: dict[str, Any], index: int = 0) -> dict[str, Any]:
        """Run a single synchronous action-only inference step.

        Parameters
        ----------
        observation : dict
            ``{"state": ndarray[14], "images": {...}, "prompt": str}``
            Optional RTC fields:
            ``prev_action_chunk``, ``inference_delay``, ``num_steps``, ``enable_rtc``.
        index : int
            Per-connection step counter.

        Returns
        -------
        dict
            ``{"index": int, "actions": ndarray[T,14], "server_timing": {...}}``.
        """
        image_tensor = self._preprocess_images(observation["images"])
        proprio = self._normalize_state(observation["state"])
        prompt = DEFAULT_PROMPT.format(
            task=observation.get("prompt", "take the cloth from the basket and fold the cloth.")
        )

        infer_prompt, context, context_mask = self._get_text_context(prompt)
        self._maybe_warmup_uncond(infer_prompt, context, context_mask)

        # --- RTC prefix fields (all optional) ------------------------------
        rtc = {
            "inference_delay": int(observation.get("inference_delay", 0)),
            "enable_rtc": bool(observation.get("enable_rtc", False)),
        }
        raw_prev = observation.get("prev_action_chunk")
        rtc["prev_action_chunk"] = (
            self._normalize_action(np.asarray(raw_prev, dtype=np.float32))
            if raw_prev is not None else None
        )
        num_steps_override = observation.get("num_steps")
        if num_steps_override is not None:
            rtc["num_inference_steps"] = int(num_steps_override)

        use_rtc = rtc["enable_rtc"] and rtc["prev_action_chunk"] is not None
        num_steps = rtc.get("num_inference_steps", self._num_inference_steps)

        # --- CUDA graph runner (if shapes match) ---------------------------
        runner = None
        if (
            self._cuda_graph_runner is not None
            and num_steps == self._num_inference_steps
            and str(self._device).startswith("cuda")
        ):
            runner = self._cuda_graph_runner.run

        infer_t0 = time.perf_counter()
        try:
            with torch.no_grad():
                pred = self._model.infer_action(
                    prompt=infer_prompt,
                    context=context,
                    context_mask=context_mask,
                    input_image=image_tensor,
                    action_horizon=self._action_horizon,
                    proprio=proprio,
                    num_inference_steps=num_steps,
                    seed=self._seed,
                    action_prefix=rtc["prev_action_chunk"] if use_rtc else None,
                    prefix_delay=rtc["inference_delay"] if use_rtc else 0,
                    action_denoise_runner=runner,
                )
        except Exception:
            if runner is None:
                raise
            logger.exception(
                "CUDA Graph action runner failed; disabling and retrying eager/compiled path."
            )
            self._cuda_graph_runner = None
            with torch.no_grad():
                pred = self._model.infer_action(
                    prompt=infer_prompt,
                    context=context,
                    context_mask=context_mask,
                    input_image=image_tensor,
                    action_horizon=self._action_horizon,
                    proprio=proprio,
                    num_inference_steps=num_steps,
                    seed=self._seed,
                    action_prefix=rtc["prev_action_chunk"] if use_rtc else None,
                    prefix_delay=rtc["inference_delay"] if use_rtc else 0,
                )

        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        actions = self._denormalize_action(pred["action"])[0]
        result: dict[str, Any] = {
            "index": index,
            "actions": actions.astype(np.float32),
            "server_timing": {"infer_ms": round(infer_ms, 3)},
        }
        if use_rtc:
            result["rtc_applied"] = True
            result["rtc_mode"] = "training_time_prefix"
        
        logger.info(
            "step=%d | rtc=%s delay=%d | steps=%d | infer=%.1fms |runner=%s",
            index,
            use_rtc,
            rtc["inference_delay"],
            num_steps,
            infer_ms,
            "cuda_graph" if runner is not None else "eager/compile",
        )
        
        return result


# ---------------------------------------------------------------------------
# Websocket server
# ---------------------------------------------------------------------------

class FastWAMPolicyServer:
    """Serves a FastWAM model over websocket with msgpack_numpy serialisation.

    Follows the same pattern as ``kai0/src/openpi/serving/websocket_policy_server.py``.
    """

    def __init__(
        self,
        wrapper: FastWAMModelWrapper,
        host: str = "0.0.0.0",
        port: int = 8765,
        metadata: dict | None = None,
    ) -> None:
        self._wrapper = wrapper
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("FastWAM server listening on ws://%s:%s", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        step_index = 0

        # Frame 0: send metadata
        await websocket.send(packb(self._metadata))

        while True:
            try:
                t0 = time.monotonic()
                obs = unpackb(await websocket.recv())

                result = self._wrapper.infer(obs, step_index)
                step_index += 1

                total_ms = (time.monotonic() - t0) * 1000.0
                result.setdefault("server_timing", {})["total_ms"] = round(total_ms, 3)

                await websocket.send(packb(result))

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed (steps=%d)", websocket.remote_address, step_index)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection,
    request: _server.Request,
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
