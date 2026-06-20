"""Operator planning utilities for L20 kernel work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from l20_stack.hardware import L20_SPEC, classify_roofline
from l20_stack.ops.triton_rmsnorm import rmsnorm_launch_config


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
    """Approximate RMSNorm arithmetic intensity for one read, weight read, and output write."""

    if shape.rows <= 0 or shape.hidden_size <= 0 or shape.dtype_bytes <= 0:
        raise ValueError("operator shape dimensions must be positive")
    # Square + reduction add + scale multiply + weight multiply.
    flops = shape.rows * shape.hidden_size * 4
    bytes_moved = shape.rows * shape.hidden_size * shape.dtype_bytes * 3
    return flops / bytes_moved


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

    if name == "rmsnorm":
        intensity = rmsnorm_arithmetic_intensity(shape)
        return OperatorPlan(
            name=name,
            shape=shape,
            arithmetic_intensity_flops_per_byte=intensity,
            roofline_class=classify_roofline(intensity, "fp16"),
            priority=1,
            recommendation=(
                "Use the L20 Triton RMSNorm baseline first; benchmark against PyTorch eager "
                "and torch.compile before adding residual fusion."
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
            priority=2,
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
            priority=3,
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
        "first_kernel_target": "rmsnorm",
        "baseline_rule": "measure PyTorch eager and torch.compile before claiming custom speedup",
    }
