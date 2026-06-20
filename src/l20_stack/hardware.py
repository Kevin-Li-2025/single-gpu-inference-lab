"""Hardware facts and roofline helpers for L20-specific planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class GpuSpec:
    name: str
    architecture: str
    compute_capability: str
    vram_gb: float
    memory_bandwidth_gbps: float
    fp16_tflops: float
    bf16_tflops: float
    fp8_tflops: float
    int8_tops: float
    tdp_w: int
    interconnect: str
    pcie_generation: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


L20_SPEC = GpuSpec(
    name="NVIDIA L20",
    architecture="Ada",
    compute_capability="8.9",
    vram_gb=48.0,
    memory_bandwidth_gbps=864.0,
    fp16_tflops=239.0,
    bf16_tflops=239.0,
    fp8_tflops=478.0,
    int8_tops=478.0,
    tdp_w=275,
    interconnect="PCIe",
    pcie_generation=4,
)


def roofline_balance_flops_per_byte(spec: GpuSpec = L20_SPEC, precision: str = "fp16") -> float:
    """Return the compute/memory balance point for a precision."""

    precision_key = precision.lower()
    if precision_key == "fp16":
        peak_tflops = spec.fp16_tflops
    elif precision_key == "bf16":
        peak_tflops = spec.bf16_tflops
    elif precision_key == "fp8":
        peak_tflops = spec.fp8_tflops
    else:
        raise ValueError("precision must be one of fp16, bf16, fp8")

    return (peak_tflops * 1_000) / spec.memory_bandwidth_gbps


def classify_roofline(arithmetic_intensity: float, precision: str = "fp16") -> str:
    """Classify an operator as memory-bound or compute-bound on L20."""

    if arithmetic_intensity < 0:
        raise ValueError("arithmetic_intensity must be non-negative")
    balance = roofline_balance_flops_per_byte(L20_SPEC, precision)
    return "memory_bound" if arithmetic_intensity < balance else "compute_bound"
