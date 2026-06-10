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
import queue
import sys
import threading
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

from fastwam.runtime import create_fastwam, create_fastwam_idm
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.utils.video_io import save_mp4
from fastwam.utils.fs import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video saver (background thread)
# ---------------------------------------------------------------------------

class VideoSaverWorker:
    """Background thread that saves PIL frame sequences to MP4 files via a queue.

    The queue decouples GPU inference latency from disk I/O — ``save_mp4``
    uses ``imageio`` / ffmpeg underneath which is a blocking subprocess call.
    """

    def __init__(self, output_dir: Path, fps: int = 15):
        self._queue: queue.Queue = queue.Queue()
        self._output_dir = output_dir
        self._fps = fps
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        ensure_dir(self._output_dir)
        self._thread = threading.Thread(target=self._run, daemon=True, name="video-saver")
        self._thread.start()
        logger.info("VideoSaverWorker started (output_dir=%s, fps=%d)", self._output_dir, self._fps)

    def stop(self, timeout: float = 30.0) -> None:
        self._queue.put(None)  # sentinel
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("VideoSaverWorker did not stop within %s seconds", timeout)

    def enqueue(self, frames: list, filename: str) -> None:
        self._queue.put((frames, filename))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            frames, filename = item
            out_path = str(self._output_dir / filename)
            try:
                save_mp4(frames, out_path, fps=self._fps)
                logger.info("Saved video: %s", out_path)
            except Exception:
                logger.exception("Failed to save video: %s", out_path)


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

        # --- mode -----------------------------------------------------------
        mode = str(cfg.get("mode", "uncond"))
        if mode not in ("uncond", "joint", "idm"):
            raise ValueError(f"Unknown mode: {mode!r}. Must be one of: uncond, joint, idm")
        self._mode = mode

        # --- build model (mode-dependent factory) ---------------------------
        _factory_kwargs = dict(
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
        if self._mode == "idm":
            logger.info("Creating FastWAM model (IDM mode) …")
            self._model = create_fastwam_idm(**_factory_kwargs)
        else:
            logger.info("Creating FastWAM model (mode=%s) …", self._mode)
            self._model = create_fastwam(**_factory_kwargs)

        # --- load finetuned checkpoint ---------------------------------------
        ckpt_path = Path(cfg["checkpoint_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info("Loading checkpoint: %s", ckpt_path)
        self._model.load_checkpoint(str(ckpt_path))
        self._model = self._model.to(self._device).eval()

        # --- build processor & load normalisation stats ----------------------
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

        # --- text context mode ------------------------------------------------
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
        self._num_video_frames = int(inf_cfg.get("num_video_frames", self._action_horizon + 1))
        self._num_inference_steps = int(inf_cfg.get("num_inference_steps", 20))
        self._seed = inf_cfg.get("seed")  # None or int

        # --- video saver (joint / idm modes only) ---------------------------
        self._video_saver: Optional[VideoSaverWorker] = None
        self._video_output_dir: Optional[Path] = None
        if self._mode in ("joint", "idm"):
            video_out_str = str(cfg.get("video_output_dir", "./outputs/videos"))
            self._video_output_dir = Path(video_out_str)
            if not self._video_output_dir.is_absolute():
                self._video_output_dir = _PROJECT_ROOT / self._video_output_dir
            video_fps = int(cfg.get("video_fps", 15))
            self._video_saver = VideoSaverWorker(self._video_output_dir, fps=video_fps)
            self._video_saver.start()

        logger.info(
            "FastWAM server ready | device=%s dtype=%s mode=%s action_horizon=%d",
            self._device, dtype_str, self._mode, self._action_horizon,
        )

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
                state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0),
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

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully stop the video saver thread if running."""
        if self._video_saver is not None:
            self._video_saver.stop()

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
        """Run a single synchronous inference step.

        Parameters
        ----------
        observation : dict
            ``{"state": ndarray[14], "images": {...}, "prompt": str}``
            Optional RTC fields:
            ``prev_action_chunk``, ``inference_delay``, ``execute_horizon``,
            ``num_steps``, ``mask_prefix_delay``, ``prefix_attention_schedule``,
            ``max_guidance_weight``, ``enable_rtc``.
        index : int
            Per-connection step counter; server generates and returns this.

        Returns
        -------
        dict
            ``{"index": int, "actions": ndarray[T,14], "server_timing": {...}}``.
        """

        image_tensor = self._preprocess_images(observation["images"])
        proprio = self._normalize_state(observation["state"])
        prompt = DEFAULT_PROMPT.format(task=observation.get("prompt", "take the cloth from the basket and fold the cloth."))

        infer_prompt, context, context_mask = self._get_text_context(prompt)

        # --- extract RTC fields (all optional) -------------------------------
        rtc_kwargs = {
            "inference_delay": int(observation.get("inference_delay", 0)),
            "execute_horizon": observation.get("execute_horizon"),
            "mask_prefix_delay": bool(observation.get("mask_prefix_delay", True)),
            "prefix_attention_schedule": str(observation.get("prefix_attention_schedule", "linear")),
            "max_guidance_weight": float(observation.get("max_guidance_weight", 1.0)),
            "enable_rtc": bool(observation.get("enable_rtc", False)),
        }
        # convert prev_action_chunk from numpy to torch if present
        raw_prev = observation.get("prev_action_chunk")
        if raw_prev is not None:
            rtc_kwargs["prev_action_chunk"] = torch.from_numpy(
                np.asarray(raw_prev, dtype=np.float32)
            ).unsqueeze(0)  # [1, T, D]
        else:
            rtc_kwargs["prev_action_chunk"] = None

        # num_steps override
        num_steps_override = observation.get("num_steps")
        if num_steps_override is not None:
            rtc_kwargs["num_inference_steps"] = int(num_steps_override)

        if self._mode == "uncond":
            return self._infer_uncond(image_tensor, proprio, infer_prompt,
                                      context, context_mask, index, rtc_kwargs)
        elif self._mode == "joint":
            return self._infer_joint(image_tensor, proprio, infer_prompt,
                                     context, context_mask, index)
        elif self._mode == "idm":
            return self._infer_idm(image_tensor, proprio, infer_prompt,
                                   context, context_mask, index, rtc_kwargs)
        else:
            raise RuntimeError(f"Unsupported mode: {self._mode!r}")

    def _infer_uncond(
        self, image_tensor: torch.Tensor, proprio: torch.Tensor,
        prompt: Optional[str], context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor], index: int,
        rtc_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Action-only inference via ``infer_action`` or ``infer_action_with_rtc``."""
        rtc = rtc_kwargs or {}
        use_rtc = (
            rtc.get("enable_rtc", False)
            and rtc.get("prev_action_chunk") is not None
        )

        infer_t0 = time.perf_counter()
        if use_rtc:
            num_steps = rtc.get("num_inference_steps", self._num_inference_steps)
            with torch.no_grad():
                # no_grad is safe here — RTC guidance uses its own grad context
                # inside _rtc_action_step via requires_grad_ on the action input.
                pred = self._model.infer_action_with_rtc(
                    prompt=prompt,
                    context=context,
                    context_mask=context_mask,
                    input_image=image_tensor,
                    action_horizon=self._action_horizon,
                    proprio=proprio,
                    num_inference_steps=num_steps,
                    seed=self._seed,
                    prev_action_chunk=rtc["prev_action_chunk"],
                    inference_delay=rtc["inference_delay"],
                    execute_horizon=rtc["execute_horizon"],
                    mask_prefix_delay=rtc["mask_prefix_delay"],
                    prefix_attention_schedule=rtc["prefix_attention_schedule"],
                    max_guidance_weight=rtc["max_guidance_weight"],
                    enable_rtc=True,
                )
        else:
            with torch.no_grad():
                pred = self._model.infer_action(
                    prompt=prompt,
                    context=context,
                    context_mask=context_mask,
                    input_image=image_tensor,
                    action_horizon=self._action_horizon,
                    proprio=proprio,
                    num_inference_steps=self._num_inference_steps,
                    seed=self._seed,
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
        return result

    def _infer_joint(
        self, image_tensor: torch.Tensor, proprio: torch.Tensor,
        prompt: Optional[str], context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor], index: int,
    ) -> dict[str, Any]:
        """Joint video+action inference; video saved locally by background thread."""
        infer_t0 = time.perf_counter()
        with torch.no_grad():
            pred = self._model.infer_joint(
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                input_image=image_tensor,
                num_video_frames=self._num_video_frames,
                action_horizon=self._action_horizon,
                proprio=proprio,
                num_inference_steps=self._num_inference_steps,
                seed=self._seed,
                test_action_with_infer_action=False,
            )
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        actions = self._denormalize_action(pred["action"])[0]

        video_filename = f"index_{index:06d}.mp4"
        self._video_saver.enqueue(pred["video"], video_filename)  # type: ignore[union-attr]
        video_path = str(self._video_output_dir / video_filename)  # type: ignore[operator]

        return {
            "index": index,
            "actions": actions.astype(np.float32),
            "video_path": video_path,
            "server_timing": {"infer_ms": round(infer_ms, 3)},
        }

    def _infer_idm(
        self, image_tensor: torch.Tensor, proprio: torch.Tensor,
        prompt: Optional[str], context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor], index: int,
        rtc_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """IDM two-stage inference; video saved locally by background thread.

        .. note::

            RTC prefix guidance for IDM stage 2 is not yet implemented.
            RTC fields are accepted but silently ignored — the standard
            two-stage pipeline is used regardless.
        """
        rtc = rtc_kwargs or {}
        if rtc.get("enable_rtc") and rtc.get("prev_action_chunk") is not None:
            logger.warning(
                "RTC prefix guidance is not yet implemented for IDM mode. "
                "Falling back to standard two-stage inference."
            )
        num_steps = rtc.get("num_inference_steps", self._num_inference_steps)
        infer_t0 = time.perf_counter()
        with torch.no_grad():
            pred = self._model.infer_joint(
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                input_image=image_tensor,
                num_video_frames=self._num_video_frames,
                action_horizon=self._action_horizon,
                proprio=proprio,
                num_inference_steps=num_steps,
                seed=self._seed,
            )
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        actions = self._denormalize_action(pred["action"])[0]

        video_filename = f"index_{index:06d}.mp4"
        self._video_saver.enqueue(pred["video"], video_filename)  # type: ignore[union-attr]
        video_path = str(self._video_output_dir / video_filename)  # type: ignore[operator]

        return {
            "index": index,
            "actions": actions.astype(np.float32),
            "video_path": video_path,
            "server_timing": {"infer_ms": round(infer_ms, 3)},
        }


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
