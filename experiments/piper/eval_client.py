#!/usr/bin/env python3
"""FastWAM Policy Server — Async Evaluation Client.

Connects to a running FastWAM policy server via WebSocket and measures
end-to-end inference latency under a 30 Hz consumption model: the client
sends the next observation as soon as the previous result arrives (fire-and-wait).

Usage::

    python experiments/piper/eval_client.py \
        --uri ws://localhost:6006 \
        --num-steps 50 \
        --rtc

If ``--replay`` is provided, observations are loaded from a replay file
(``.npz`` with arrays ``state`` [N,14], ``fixed_front`` / ``left_arm`` /
``right_arm`` [N,H,W,3]).  Otherwise synthetic random observations are used
for pure latency benchmarking.
"""

import argparse
import asyncio
import time
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from msgpack_numpy import packb, unpackb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_HORIZON = 32       # chunk size
ACTION_DIM = 14           # 7-DoF dual-arm
CONTROL_HZ = 30           # action consumption rate
CHUNK_DURATION_S = ACTION_HORIZON / CONTROL_HZ  # ~1.067 s per chunk
STATE_DIM = 14

DEFAULT_PROMPT = "take the cloth from the basket and fold the cloth."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_obs(index: int, prev_action: Optional[np.ndarray] = None,
                       enable_rtc: bool = False, inference_delay: int = 0,
                       execute_horizon: Optional[int] = None) -> dict:
    """Build a synthetic observation dict matching the server protocol."""
    # Random uint8 images (3 cameras, 480×640 raw)
    rng = np.random.RandomState(index)
    obs: dict = {
        "state": rng.randn(STATE_DIM).astype(np.float32),
        "images": {
            "fixed_front": rng.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            "left_arm":    rng.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            "right_arm":   rng.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        },
        "prompt": DEFAULT_PROMPT,
    }
    if enable_rtc:
        obs["enable_rtc"] = True
        obs["inference_delay"] = inference_delay
        if execute_horizon is not None:
            obs["execute_horizon"] = execute_horizon
        if prev_action is not None:
            obs["prev_action_chunk"] = prev_action.astype(np.float32)
    return obs


class ReplayLoader:
    """Load observations from a ``.npz`` replay file."""

    def __init__(self, path: Path):
        data = np.load(str(path))
        self._states = data["state"]         # [N, 14]
        self._fixed = data["fixed_front"]     # [N, H, W, 3]
        self._left = data["left_arm"]         # [N, H, W, 3]
        self._right = data["right_arm"]       # [N, H, W, 3]
        self._n = len(self._states)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        i = idx % self._n
        return {
            "state": self._states[i].astype(np.float32),
            "images": {
                "fixed_front": self._fixed[i],
                "left_arm":    self._left[i],
                "right_arm":   self._right[i],
            },
            "prompt": DEFAULT_PROMPT,
        }


# ---------------------------------------------------------------------------
# Eval client
# ---------------------------------------------------------------------------

class EvalClient:
    def __init__(self, uri: str, num_steps: int = 50,
                 enable_rtc: bool = False,
                 inference_delay: int = 0,
                 execute_horizon: Optional[int] = None,
                 replay: Optional[ReplayLoader] = None):
        self.uri = uri
        self.num_steps = num_steps
        self.enable_rtc = enable_rtc
        self.inference_delay = inference_delay
        self.execute_horizon = execute_horizon
        self.replay = replay

        # Per-step stats
        self.latencies_ms: list[float] = []      # RTT per request
        self.server_timings: list[dict] = []      # server-reported timing
        self.prev_action: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    async def run(self) -> None:
        import websockets

        async with websockets.connect(
            self.uri, max_size=None,
            ping_interval=30, ping_timeout=60, close_timeout=10,
        ) as ws:
            # --- receive metadata -----------------------------------------
            raw = await ws.recv()
            meta = unpackb(raw)
            print(f"Connected. metadata={meta}")
            if self.enable_rtc and not meta.get("rtc_supported", False):
                print("[warn] Server does not advertise RTC support; "
                      "rtc requests may be ignored.")

            print(f"Starting eval: steps={self.num_steps} rtc={self.enable_rtc} "
                  f"chunk_duration={CHUNK_DURATION_S:.2f}s @ {CONTROL_HZ}Hz")
            print(f"{'step':>5s}  {'lat_ms':>8s}  {'server_ms':>10s}  "
                  f"{'eff_hz':>8s}  {'util%':>7s}")
            print("-" * 55)

            # --- request loop (fire-and-wait) ------------------------------
            for step_idx in range(self.num_steps):
                obs = self._build_obs(step_idx)

                t_send = time.perf_counter()
                await ws.send(packb(obs))
                raw = await ws.recv()
                t_recv = time.perf_counter()

                # Handle server-side errors (traceback sent as str)
                if isinstance(raw, str):
                    print(f"\n[error] Server returned error at step {step_idx}:")
                    print(raw[:500])
                    break

                result = unpackb(raw)
                rtt_ms = (t_recv - t_send) * 1000.0
                self.latencies_ms.append(rtt_ms)

                if "actions" in result:
                    self.prev_action = np.asarray(result["actions"], dtype=np.float32)

                srv_timing = result.get("server_timing", {})
                self.server_timings.append(srv_timing)
                srv_ms = srv_timing.get("total_ms", srv_timing.get("infer_ms", 0))

                # Effective Hz: 1 / RTT
                eff_hz = 1000.0 / max(rtt_ms, 0.001)
                # Chunk utilisation: chunk_duration / RTT (are we keeping up?)
                util_pct = (CHUNK_DURATION_S / max(t_recv - t_send, 0.001)) * 100.0

                print(f"{step_idx:5d}  {rtt_ms:8.1f}  {srv_ms:10.1f}  "
                      f"{eff_hz:8.1f}  {util_pct:7.1f}")

            self._print_summary()

    def _build_obs(self, step_idx: int) -> dict:
        """Construct observation for step *step_idx*."""
        if self.replay is not None:
            obs = self.replay[step_idx]
        else:
            obs = make_synthetic_obs(step_idx, self.prev_action,
                                     self.enable_rtc, self.inference_delay,
                                     self.execute_horizon)

        if self.enable_rtc:
            obs["enable_rtc"] = True
            obs["inference_delay"] = self.inference_delay
            if self.execute_horizon is not None:
                obs["execute_horizon"] = self.execute_horizon
            if self.prev_action is not None:
                obs["prev_action_chunk"] = self.prev_action.astype(np.float32)
        return obs

    def _print_summary(self) -> None:
        arr = np.array(self.latencies_ms)
        srv = np.array([t.get("total_ms", t.get("infer_ms", 0))
                        for t in self.server_timings])

        print()
        print("=" * 55)
        print("Summary")
        print("=" * 55)
        print(f"  Requests:           {len(arr)}")
        print(f"  RTT mean / median:  {arr.mean():.1f} / {np.median(arr):.1f} ms")
        print(f"  RTT min / max:      {arr.min():.1f} / {arr.max():.1f} ms")
        print(f"  RTT std:            {arr.std():.1f} ms")
        if len(srv) > 0:
            print(f"  Server mean / med:  {srv.mean():.1f} / {np.median(srv):.1f} ms")
        print(f"  Effective Hz mean:  {1000.0 / arr.mean():.1f}")
        print(f"  Chunk utilisation:  {CHUNK_DURATION_S / (arr.mean() / 1000.0) * 100:.1f}%")
        print(f"  Chunk duration:     {CHUNK_DURATION_S:.3f}s (@ {CONTROL_HZ}Hz)")
        print()

        if arr.mean() > CHUNK_DURATION_S * 1000:
            lag = arr.mean() / 1000.0 - CHUNK_DURATION_S
            print(f"  ⚠  RTT ({arr.mean():.0f}ms) exceeds chunk duration "
                  f"({CHUNK_DURATION_S*1000:.0f}ms) by {lag*1000:.0f}ms")
            print(f"     → System CANNOT run in real time at {CONTROL_HZ}Hz.")
        else:
            slack = CHUNK_DURATION_S - arr.mean() / 1000.0
            print(f"  ✓  RTT ({arr.mean():.0f}ms) within chunk duration "
                  f"({CHUNK_DURATION_S*1000:.0f}ms), slack={slack*1000:.0f}ms")
            print(f"     → System can operate at {CONTROL_HZ}Hz.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastWAM eval client — measure end-to-end inference latency")
    parser.add_argument("--uri", default="ws://localhost:8765",
                        help="WebSocket URI of the policy server")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of inference requests to send")
    parser.add_argument("--rtc", action="store_true",
                        help="Enable RTC prefix guidance")
    parser.add_argument("--inference-delay", type=int, default=0,
                        help="RTC inference delay (steps already executed)")
    parser.add_argument("--execute-horizon", type=int, default=None,
                        help="RTC execute horizon")
    parser.add_argument("--replay", type=Path, default=None,
                        help="Path to .npz replay file with observations")
    args = parser.parse_args()

    replay = None
    if args.replay is not None:
        if not args.replay.is_file():
            print(f"Error: replay file not found: {args.replay}", file=sys.stderr)
            sys.exit(1)
        replay = ReplayLoader(args.replay)
        if args.num_steps > len(replay):
            print(f"[warn] --num-steps={args.num_steps} > replay length={len(replay)}; "
                  f"observations will wrap around.")

    client = EvalClient(
        uri=args.uri,
        num_steps=args.num_steps,
        enable_rtc=args.rtc,
        inference_delay=args.inference_delay,
        execute_horizon=args.execute_horizon,
        replay=replay,
    )
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
