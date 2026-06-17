from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def temporary_default_dtype(dtype: torch.dtype):
    previous_dtype = torch.get_default_dtype()
    if dtype.is_floating_point:
        torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def instantiate_module_on_device(
    module_cls: type[torch.nn.Module],
    kwargs: dict[str, Any],
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Instantiate large modules directly on the target device/dtype.

    This avoids briefly materialising multi-billion-parameter modules as
    float32 CPU tensors before they are moved to CUDA/bf16.
    """
    with torch.device(device), temporary_default_dtype(dtype):
        module = module_cls(**kwargs)
    return module.to(device=device, dtype=dtype)
