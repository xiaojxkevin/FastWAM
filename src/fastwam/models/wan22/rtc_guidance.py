"""RTC (Real-Time Chunking) prefix-guidance primitives for FastWAM.

Ported from ``kai0/src/openpi/models/pi0_rtc.py`` (JAX) to PyTorch.
These are pure functions with no learned parameters — they operate on the
diffusion latents and previous-action-chunk tensors during inference.
"""

import torch
from torch import Tensor


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: str,
) -> Tensor:
    """Prefix weights for RTC guidance (adapted from real-time-chunking-kinetix).

    Args:
        start: First index under guidance (``inference_delay`` **d**).
        end: One-past-last index under guidance (``d + execute_horizon``).
        total: Total number of action steps (``action_horizon``).
        schedule: Weight schedule — ``"ones"``, ``"zeros"``, ``"linear"``, or ``"exp"``.

    Returns:
        Float tensor of shape ``[total]``.  Zero outside ``[start, end)``.
    """
    start = min(start, end)

    if schedule == "ones":
        w = torch.ones(total)
    elif schedule == "zeros":
        w = (torch.arange(total) < start).float()
    elif schedule in ("linear", "exp"):
        w = torch.clamp(
            (start - 1 - torch.arange(total, dtype=torch.float32))
            / (end - start + 1)
            + 1,
            0.0,
            1.0,
        )
        if schedule == "exp":
            w = w * torch.expm1(w) / (torch.e - 1)
    else:
        raise ValueError(
            f"Invalid schedule: {schedule!r}. "
            f"Must be one of: ones, zeros, linear, exp."
        )

    return torch.where(torch.arange(total) >= end, torch.zeros_like(w), w)


def compute_rtc_guidance_weight(
    time: Tensor,
    max_guidance_weight: float = 1.0,
) -> Tensor:
    """Compute the RTC guidance weight from the current diffusion timestep.

    Follows the LeRobot RTC formula: invert time so guidance is strongest
    early in denoising (when uncertainty is highest) and weakest near
    convergence.

    Args:
        time: Current diffusion timestep ``t ∈ [0, 1]`` (scalar tensor).
        max_guidance_weight: Hard clamp on the guidance multiplier.

    Returns:
        Scalar guidance weight tensor.
    """
    tau = 1.0 - time
    tau_safe = torch.clamp(tau, min=1e-3, max=1.0)
    squared_one_minus_tau = (1.0 - tau_safe) ** 2
    inv_r2 = (squared_one_minus_tau + tau_safe**2) / squared_one_minus_tau
    c = torch.nan_to_num(
        (1.0 - tau_safe) / tau_safe,
        nan=0.0,
        posinf=max_guidance_weight,
        neginf=0.0,
    )
    return torch.minimum(c * inv_r2, torch.tensor(max_guidance_weight, device=time.device))
