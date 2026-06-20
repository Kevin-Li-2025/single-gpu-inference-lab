"""Operator planning utilities for L20 kernel work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from l20_stack.hardware import L20_SPEC, classify_roofline
from l20_stack.ops.triton_rmsnorm import (
    residual_rmsnorm_launch_config,
    rmsnorm_launch_config,
)


@dataclass(frozen=True)
class OperatorShape:
    rows: int
    hidden_size: int
    dtype_bytes: int = 2

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "OperatorShape":
        return cls(
            rows=int(payload["rows"]),
            hidden_size=int(payload["hidden_size"]),
            dtype_bytes=int(payload.get("dtype_bytes", 2)),
        )


@dataclass(frozen=True)
class OperatorTarget:
    name: str
    shape: OperatorShape

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "OperatorTarget":
        return cls(name=str(payload["name"]), shape=OperatorShape.from_dict(payload["shape"]))


@dataclass(frozen=True)
class OperatorPlan:
    name: str
    shape: OperatorShape
    arithmetic_intensity_flops_per_byte: float
    roofline_class: str
    priority: int
    recommendation: str
    launch: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["arithmetic_intensity_flops_per_byte"] = round(
            self.arithmetic_intensity_flops_per_byte, 4
        )
        return payload


def rmsnorm_arithmetic_intensity(shape: OperatorShape) -> float:
    """Approximate RMSNorm intensity using minimum device-memory traffic."""

    if shape.rows <= 0 or shape.hidden_size <= 0 or shape.dtype_bytes <= 0:
        raise ValueError("operator shape dimensions must be positive")
    # Square + reduction add + scale multiply + weight multiply.
    flops = shape.rows * shape.hidden_size * 4
    bytes_moved = rmsnorm_minimum_bytes(shape)
    return flops / bytes_moved


def rmsnorm_minimum_bytes(shape: OperatorShape) -> int:
    if shape.rows <= 0 or shape.hidden_size <= 0 or shape.dtype_bytes <= 0:
        raise ValueError("operator shape dimensions must be positive")
    elements = shape.rows * shape.hidden_size
    return (2 * elements + shape.hidden_size) * shape.dtype_bytes


def residual_rmsnorm_minimum_bytes(shape: OperatorShape, fused: bool) -> int:
    """Return the semantic traffic lower bound, including required residual output."""

    if shape.rows <= 0 or shape.hidden_size <= 0 or shape.dtype_bytes <= 0:
        raise ValueError("operator shape dimensions must be positive")
    elements = shape.rows * shape.hidden_size
    tensor_elements = 4 * elements if fused else 5 * elements
    return (tensor_elements + shape.hidden_size) * shape.dtype_bytes


def residual_rmsnorm_arithmetic_intensity(shape: OperatorShape) -> float:
    # Residual add plus RMSNorm square, reduction, and two scaling multiplies.
    flops = shape.rows * shape.hidden_size * 5
    return flops / residual_rmsnorm_minimum_bytes(shape, fused=True)


def rope_arithmetic_intensity(shape: OperatorShape) -> float:
    if shape.rows <= 0 or shape.hidden_size <= 0 or shape.dtype_bytes <= 0:
        raise ValueError("operator shape dimensions must be positive")
    # Approximate sin/cos application as six scalar ops per element pair.
    flops = shape.rows * shape.hidden_size * 3
    bytes_moved = shape.rows * shape.hidden_size * shape.dtype_bytes * 2
    return flops / bytes_moved


def dequant_matvec_arithmetic_intensity(shape: OperatorShape) -> float:
    if shape.rows <= 0 or shape.hidden_size <= 0 or shape.dtype_bytes <= 0:
        raise ValueError("operator shape dimensions must be positive")
    # Approximate one dequant scale and one multiply-add per element.
    flops = shape.rows * shape.hidden_size * 3
    bytes_moved = shape.rows * shape.hidden_size * shape.dtype_bytes
    return flops / bytes_moved


def plan_operator(target: OperatorTarget) -> OperatorPlan:
    name = target.name.lower()
    shape = target.shape

    if name == "residual_rmsnorm":
        intensity = residual_rmsnorm_arithmetic_intensity(shape)
        fused_bytes = residual_rmsnorm_minimum_bytes(shape, fused=True)
        unfused_bytes = residual_rmsnorm_minimum_bytes(shape, fused=False)
        reduction_pct = 100 * (unfused_bytes - fused_bytes) / unfused_bytes
        launch = residual_rmsnorm_launch_config(shape.hidden_size).to_dict()
        launch.update(
            {
                "fused_minimum_bytes": fused_bytes,
                "unfused_minimum_bytes": unfused_bytes,
                "minimum_traffic_reduction_pct": round(reduction_pct, 2),
            }
        )
        return OperatorPlan(
            name=name,
            shape=shape,
            arithmetic_intensity_flops_per_byte=intensity,
            roofline_class=classify_roofline(intensity, "fp16"),
            priority=1,
            recommendation=(
                "Benchmark fused residual add + RMSNorm against PyTorch eager and "
                "torch.compile on the target L20."
            ),
            launch=launch,
        )

    if name == "rmsnorm":
        intensity = rmsnorm_arithmetic_intensity(shape)
        return OperatorPlan(
            name=name,
            shape=shape,
            arithmetic_intensity_flops_per_byte=intensity,
            roofline_class=classify_roofline(intensity, "fp16"),
            priority=2,
            recommendation=(
                "Keep standalone RMSNorm as a control for the fused residual path and compare "
                "against PyTorch eager and torch.compile."
            ),
            launch=rmsnorm_launch_config(shape.hidden_size).to_dict(),
        )

    if name == "rope":
        intensity = rope_arithmetic_intensity(shape)
        return OperatorPlan(
            name=name,
            shape=shape,
            arithmetic_intensity_flops_per_byte=intensity,
            roofline_class=classify_roofline(intensity, "fp16"),
            priority=3,
            recommendation=(
                "Fuse RoPE with KV-cache layout writes after RMSNorm benchmark data exists."
            ),
            launch={"strategy": "not_implemented_yet"},
        )

    if name == "dequant_matvec":
        intensity = dequant_matvec_arithmetic_intensity(shape)
        return OperatorPlan(
            name=name,
            shape=shape,
            arithmetic_intensity_flops_per_byte=intensity,
            roofline_class=classify_roofline(intensity, "fp16"),
            priority=5,
            recommendation=(
                "Fuse dequantization with the consumer matvec/matmul staging path; do not "
                "benchmark dequant as an isolated final target."
            ),
            launch={"strategy": "not_implemented_yet"},
        )

    raise ValueError("unsupported operator target: " + target.name)


def plan_operators(targets: Iterable[OperatorTarget]) -> List[OperatorPlan]:
    return sorted((plan_operator(target) for target in targets), key=lambda plan: plan.priority)


def l20_operator_summary() -> Dict[str, object]:
    return {
        "hardware": L20_SPEC.to_dict(),
        "compile_target": "sm_89",
        "first_kernel_target": "residual_rmsnorm",
        "baseline_rule": "measure PyTorch eager and torch.compile before claiming custom speedup",
    }
