"""Triton RMSNorm kernels tuned for L20-style Ada GPUs.

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

    measured_warps = {
        4096: 4,
        5120: 8,
        6144: 4,
        8192: 8,
    }
    if hidden_size in measured_warps:
        num_warps = measured_warps[hidden_size]
    elif block_size <= 512:
        num_warps = 2
    elif block_size <= 1024:
        num_warps = 4
    else:
        num_warps = 8

    return RmsNormLaunchConfig(
        hidden_size=hidden_size,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
        sm_target="sm_89",
        rationale=(
            "one Triton program per row, FP32 reduction, power-of-two block, "
            "and L20-measured warps for common LLM hidden sizes"
        ),
    )


def residual_rmsnorm_launch_config(hidden_size: int) -> RmsNormLaunchConfig:
    base = rmsnorm_launch_config(hidden_size)
    measured_warps = {
        4096: 4,
        5120: 4,
        6144: 8,
        8192: 4,
    }
    num_warps = measured_warps.get(hidden_size, base.num_warps)
    return RmsNormLaunchConfig(
        hidden_size=base.hidden_size,
        block_size=base.block_size,
        num_warps=num_warps,
        num_stages=base.num_stages,
        sm_target=base.sm_target,
        rationale=(
            "fused residual RMSNorm warps selected from three cold-cache L20 runs "
            "for common LLM hidden sizes"
        ),
    )


def rmsnorm_warp_candidates(hidden_size: int):
    """Return the small launch sweep worth measuring on the target L20."""

    config = rmsnorm_launch_config(hidden_size)
    if config.block_size <= 512:
        return (2, 4)
    if config.block_size <= 1024:
        return (4, 8)
    return (4, 8)


if triton is not None:  # pragma: no cover - requires Triton

    @triton.jit
    def _rmsnorm_kernel(x, weight, y, n_cols: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < n_cols
        x_row = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x_row * x_row, axis=0) / n_cols
        inv_rms = tl.rsqrt(mean_square + eps)
        weight_row = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        y_row = x_row * inv_rms * weight_row
        tl.store(y + row * n_cols + offsets, y_row, mask=mask)

    @triton.jit
    def _residual_rmsnorm_kernel(
        x,
        residual,
        weight,
        y,
        residual_out,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < n_cols
        row_offsets = row * n_cols + offsets
        x_row = tl.load(x + row_offsets, mask=mask, other=0.0)
        residual_row = tl.load(residual + row_offsets, mask=mask, other=0.0)
        merged = (x_row + residual_row).to(x_row.dtype)
        tl.store(residual_out + row_offsets, merged, mask=mask)
        merged_float = merged.to(tl.float32)
        mean_square = tl.sum(merged_float * merged_float, axis=0) / n_cols
        inv_rms = tl.rsqrt(mean_square + eps)
        weight_row = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(y + row_offsets, merged_float * inv_rms * weight_row, mask=mask)

def rmsnorm_reference(x, weight, eps: float = 1e-6):
    """PyTorch reference implementation for correctness checks."""

    if torch is None:
        raise RuntimeError("rmsnorm_reference requires torch")
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def residual_rmsnorm_reference(x, residual, weight, eps: float = 1e-6):
    """PyTorch reference for residual add followed by RMSNorm."""

    if torch is None:
        raise RuntimeError("residual_rmsnorm_reference requires torch")
    merged = x + residual
    merged_float = merged.float()
    variance = merged_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = merged_float * torch.rsqrt(variance + eps) * weight.float()
    return normalized.to(x.dtype), merged


def rmsnorm_triton(x, weight, eps: float = 1e-6, num_warps=None):
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
    launch_warps = config.num_warps if num_warps is None else int(num_warps)
    if launch_warps not in rmsnorm_warp_candidates(int(hidden_size)):
        raise ValueError("num_warps is not an L20 launch candidate for this hidden size")
    y = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](
        x,
        weight,
        y,
        hidden_size,
        eps,
        config.block_size,
        num_warps=launch_warps,
        num_stages=config.num_stages,
    )
    return y


def residual_rmsnorm_triton(x, residual, weight, eps: float = 1e-6, num_warps=None):
    """Fuse residual addition with RMSNorm and return ``(normalized, residual_out)``.

    The fused path avoids reading the materialized residual sum back from device
    memory. It is an inference-forward kernel and does not provide autograd.
    """

    if torch is None or triton is None:
        raise RuntimeError("residual_rmsnorm_triton requires torch and triton")
    if x.dim() != 2:
        raise ValueError("x must be 2D [rows, hidden_size]")
    if residual.shape != x.shape:
        raise ValueError("residual must match x shape")
    if weight.dim() != 1 or weight.numel() != x.shape[1]:
        raise ValueError("weight must be 1D and match hidden_size")
    if not x.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise ValueError("x, residual, and weight must be CUDA tensors")
    if x.dtype != residual.dtype:
        raise ValueError("x and residual must have the same dtype")
    if not x.is_contiguous():
        x = x.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()

    rows, hidden_size = x.shape
    config = residual_rmsnorm_launch_config(int(hidden_size))
    launch_warps = config.num_warps if num_warps is None else int(num_warps)
    if launch_warps not in rmsnorm_warp_candidates(int(hidden_size)):
        raise ValueError("num_warps is not an L20 launch candidate for this hidden size")
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    _residual_rmsnorm_kernel[(rows,)](
        x,
        residual,
        weight,
        y,
        residual_out,
        hidden_size,
        eps,
        config.block_size,
        num_warps=launch_warps,
        num_stages=config.num_stages,
    )
    return y, residual_out


def residual_rmsnorm_backend(
    rows: int, hidden_size: int, flashinfer_available: bool = False
) -> str:
    """Return the measured L20 backend for a workload shape."""

    triton_decode_shapes = {
        (8, 4096),
        (8, 6144),
        (32, 4096),
        (32, 6144),
        (128, 4096),
        (512, 4096),
        (512, 5120),
    }
    if flashinfer_available and (rows, hidden_size) in triton_decode_shapes:
        return "triton"
    if flashinfer_available:
        return "flashinfer"
    if rows <= 512 or hidden_size == 8192:
        return "triton"
    return "torch_eager"


def residual_rmsnorm_l20(x, residual, weight, eps: float = 1e-6):
    """Dispatch to the fastest measured residual RMSNorm path on L20."""

    if torch is None:
        raise RuntimeError("residual_rmsnorm_l20 requires torch")
    if x.dim() != 2 or residual.shape != x.shape:
        raise ValueError("x and residual must be matching 2D tensors")
    if weight.dim() != 1 or weight.numel() != x.shape[1]:
        raise ValueError("weight must be 1D and match hidden_size")

    hidden_size = int(x.shape[1])
    if residual_rmsnorm_backend(int(x.shape[0]), hidden_size) == "triton":
        return residual_rmsnorm_triton(x, residual, weight, eps)

    residual_out = x + residual
    normalized = torch.nn.functional.rms_norm(
        residual_out, (hidden_size,), weight, eps
    )
    return normalized, residual_out


def residual_rmsnorm_triton_inplace(x, residual, weight, eps: float = 1e-6):
    """Apply the L20 Triton fused kernel in place."""

    if torch is None or triton is None:
        raise RuntimeError("residual_rmsnorm_triton_inplace requires torch and triton")
    if x.dim() != 2 or residual.shape != x.shape:
        raise ValueError("x and residual must be matching 2D tensors")
    if weight.dim() != 1 or weight.numel() != x.shape[1]:
        raise ValueError("weight must be 1D and match hidden_size")
    if not x.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise ValueError("x, residual, and weight must be CUDA tensors")
    if not x.is_contiguous() or not residual.is_contiguous():
        raise ValueError("in-place tensors must be contiguous")

    rows, hidden_size = x.shape
    config = residual_rmsnorm_launch_config(int(hidden_size))
    measured_warps = {
        (8, 4096): 8,
        (32, 4096): 8,
    }
    num_warps = measured_warps.get((int(rows), int(hidden_size)), config.num_warps)
    _residual_rmsnorm_kernel[(rows,)](
        x,
        residual,
        weight,
        x,
        residual,
        hidden_size,
        eps,
        config.block_size,
        num_warps=num_warps,
        num_stages=config.num_stages,
    )


def residual_rmsnorm_l20_inplace(x, residual, weight, eps: float = 1e-6) -> str:
    """Apply the fastest measured in-place backend and return its name."""

    try:
        import flashinfer
    except ImportError:
        flashinfer = None

    backend = residual_rmsnorm_backend(
        int(x.shape[0]), int(x.shape[1]), flashinfer is not None
    )
    if backend == "flashinfer":
        flashinfer.norm.fused_add_rmsnorm(x, residual, weight, eps)
        return backend
    if backend == "triton":
        residual_rmsnorm_triton_inplace(x, residual, weight, eps)
        return backend

    residual.add_(x)
    normalized = torch.nn.functional.rms_norm(
        residual, (int(x.shape[1]),), weight, eps
    )
    x.copy_(normalized)
    return backend
