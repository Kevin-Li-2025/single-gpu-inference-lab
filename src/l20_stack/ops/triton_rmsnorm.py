"""Triton RMSNorm baseline tuned for L20-style Ada GPUs.

The module is importable without Triton. Calling `rmsnorm_triton` requires
PyTorch and Triton on a CUDA machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


try:  # pragma: no cover - optional GPU dependency
    import torch
except ImportError:  # pragma: no cover - optional GPU dependency
    torch = None

try:  # pragma: no cover - optional GPU dependency
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency
    triton = None
    tl = None


@dataclass(frozen=True)
class RmsNormLaunchConfig:
    hidden_size: int
    block_size: int
    num_warps: int
    num_stages: int
    sm_target: str
    rationale: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def rmsnorm_launch_config(hidden_size: int) -> RmsNormLaunchConfig:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    block_size = next_power_of_2(hidden_size)
    if block_size > 16384:
        raise ValueError("single-pass RMSNorm baseline supports hidden_size <= 16384")

    if block_size <= 1024:
        num_warps = 4
    elif block_size <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    return RmsNormLaunchConfig(
        hidden_size=hidden_size,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=4,
        sm_target="sm_89",
        rationale=(
            "one Triton program per row, FP32 reduction, power-of-two block, "
            "warps chosen to balance occupancy and register pressure on Ada"
        ),
    )


if triton is not None:  # pragma: no cover - requires Triton

    @triton.jit
    def _rmsnorm_kernel(x, weight, y, n_cols: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < n_cols
        x_row = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        weight_row = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x_row * x_row, axis=0) / n_cols
        inv_rms = tl.rsqrt(mean_square + eps)
        y_row = x_row * inv_rms * weight_row
        tl.store(y + row * n_cols + offsets, y_row, mask=mask)


def rmsnorm_reference(x, weight, eps: float = 1e-6):
    """PyTorch reference implementation for correctness checks."""

    if torch is None:
        raise RuntimeError("rmsnorm_reference requires torch")
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def rmsnorm_triton(x, weight, eps: float = 1e-6):
    """Run the Triton RMSNorm baseline.

    `x` must be a 2D CUDA tensor with a contiguous hidden dimension.
    `weight` must be a 1D CUDA tensor with length equal to hidden size.
    """

    if torch is None or triton is None:
        raise RuntimeError("rmsnorm_triton requires torch and triton")
    if x.dim() != 2:
        raise ValueError("x must be 2D [rows, hidden_size]")
    if weight.dim() != 1 or weight.numel() != x.shape[1]:
        raise ValueError("weight must be 1D and match hidden_size")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")
    if not x.is_contiguous():
        x = x.contiguous()

    rows, hidden_size = x.shape
    config = rmsnorm_launch_config(int(hidden_size))
    y = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](
        x,
        weight,
        y,
        hidden_size,
        eps,
        config.block_size,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return y
