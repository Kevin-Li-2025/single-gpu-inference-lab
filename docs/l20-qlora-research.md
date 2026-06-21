# L20 QLoRA Research

## Distinguishing Target

The model target is not a generic code assistant trained on public benchmark
answers. It is an L20 kernel coding model that proposes Triton or CUDA
implementations and is evaluated by execution:

1. candidate code compiles on `sm_89`
2. outputs match a held-out PyTorch reference
3. latency is measured with CUDA Events after warmup
4. speedup must survive repeated runs and a conservative policy gate

Public KernelBench tasks remain evaluation-only. Training data must come from
disjoint operator implementations, official programming material, and
independently generated transformations whose outputs are compiled and checked.

## Real Training Path

`scripts/train_qlora.py` implements a single-L20 training loop with:

- NF4 double-quantized base weights
- BF16 computation and TF32-enabled FP32 matmul
- assistant-only loss masking
- optional fixed-length example packing
- non-reentrant activation checkpointing
- fused AdamW over trainable adapter parameters
- cosine learning-rate schedule and gradient clipping
- exact train/eval overlap rejection and normalized prompt contamination report
- peak allocated/reserved memory, effective tokens/s, loss history, config hash,
  and dataset fingerprints

The trainer rejects non-L20 GPUs because its defaults and performance claims
are calibrated for Ada `sm_89`.

## Initial L20 Capacity Measurements

These are plumbing and capacity measurements on tiny fixtures, not model-quality
results.

| Base model | LoRA rank | Trainable parameters | Peak allocated | Peak reserved | Effective tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct, packed | 8 | 4.40M | 1.17 GiB | 1.21 GiB | 126.43 |
| Qwen2.5-Coder-14B-Instruct | 8 | 34.41M | 13.91 GiB | 15.74 GiB | 15.88 |

Both runs completed real 4-bit forward, backward, evaluation, and adapter save
on the NVIDIA L20. The next performance experiments must use representative
packed sequences because tiny variable-length fixtures underutilize the GPU.

## KernelBench Pilot

The first executable coding target is a 3-task L20 pilot from KernelBench:
ReLU, RMSNorm, and Matmul + scaling + residual add. The metric follows
KernelBench: `fast_0` is correct, `fast_1` is correct and faster than PyTorch,
and `fast_2` is correct and at least 2x faster. Static and interface checks are
run before treating a generated candidate as a real compile.

| Model / prompt | Compile rate | Interface-gated compile | fast_0 | fast_1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B base | 0/3 | 0/3 | 0/3 | 0/3 |
| Qwen2.5-Coder-1.5B base, prompt v1 | 0/3 | 0/3 | 0/3 | 0/3 |
| Qwen2.5-Coder-1.5B base, prompt v2 | 1/3 | 1/3 | 0/3 | 0/3 |
| Qwen2.5-Coder-1.5B LoRA v3, prompt v2 | 3/3 | 0/3 | 0/3 | 0/3 |
| Qwen2.5-Coder-1.5B LoRA v4, prompt v3 | 3/3 | 1/3 | 0/3 | 0/3 |
| Qwen2.5-Coder-1.5B LoRA v4, prompt v4 | 2/3 | 1/3 | 0/3 | 0/3 |

The apparent LoRA compile improvement is not yet a quality win. The stricter
interface gate shows that the model often emits a loadable class with the wrong
contract: missing constructor state, helper wrappers requiring extra arguments,
or placeholder `NotImplementedError`. Therefore the current verified conclusion
is conservative: the L20 QLoRA stack trains efficiently, but the kernel-coding
model has not yet produced a correct held-out KernelBench solution.

### Elementwise Control

The level-1 ReLU task uses a `[4096, 393216]` FP32 input. On a 48 GB L20, the
default KernelBench `torch.allclose` correctness step runs out of memory after
holding the input, reference output, and candidate output. The evaluator now
has an opt-in `--chunked-allclose-elements` mode that applies the same
`torch.allclose` tolerances to contiguous chunks and records the non-default
comparator in the report.

With that comparator, the handwritten shape-preserving Triton control in
`benchmarks/kernelbench_baselines/relu_l20.py` passes 3/3 correctness trials:

| Candidate | Compile | fast_0 | fast_1 | PyTorch | Triton | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Handwritten ReLU control | 1/1 | 1/1 | 1/1 | 20.1 ms | 19.5 ms | 1.031x |

This is an evaluator and curriculum control, not a model-quality result. The v4
QLoRA model remains `fast_0=0/3`. A v5 elementwise curriculum may use analogous
shape-preserving examples, but the exact KernelBench problem must remain
evaluation-only and category-level results must not be presented as general
KernelBench performance.

Training telemetry for the 1.5B pilot:

| Run | Data contract | Steps | Best eval loss | Selected step | Tokens/s | Peak allocated |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v2 | wrapper-to-ModelNew adapter | 80 | 0.5090 | 60 | 3460.91 | 6.05 GiB |
| v3 | explicit ModelNew adapter in labels | 60 | 0.4998 | 60 | 3477.79 | 6.05 GiB |
| v4 | reference-class adapter + interface-gated labels | 60 | 0.5660 | 60 | 3523.91 | 6.06 GiB |

The v4 run used a stricter dataset transformer:

- selects the reference `nn.Module` class even when it is not named `Model`
- copies the reference `__init__` and `forward` signatures into `ModelNew`
- rewrites `super(OriginalClass, self)` to `super(ModelNew, self)`
- maps wrapper parameters to forward inputs, `self` attributes, or common
  module state such as `self.linear.weight`
- drops labels that fail the same interface gate used for evaluation

The resulting split had 50 train records and 5 eval records; all 55 labels
passed the interface gate and contained no `*args` or `**kwargs` in `ModelNew`.
This fixed the earlier contract-learning failure, but did not yet produce a
correct held-out kernel.

Prompt v4 changed the failure mode rather than solving the benchmark:

- ReLU compiled but returned a flattened tensor instead of the reference shape.
- RMSNorm was rejected by the new static gate for one-argument `tl.arange`.
- Matmul + scaling + residual was rejected by the static gate for dynamic
  block-tensor `.view()` in a Triton JIT function.

Current static/interface gates reject missing `ModelNew`, evaluator-helper
leakage, placeholder implementations, `ModelNew` varargs, executable test
harnesses, wrapper argument mismatches, unsupported `tl.sum(..., keepdims=...)`,
dynamic Triton block tensor `.view()`/`.reshape()`, missing launcher arguments,
and one-argument `tl.arange`.

Raw reports:

- `benchmarks/results/l20-kernelbench-pilot/`
- `benchmarks/results/l20-qlora-kernel-1.5b-v2/`
- `benchmarks/results/l20-qlora-kernel-1.5b-v3/`
- `benchmarks/results/l20-qlora-kernel-1.5b-v4/`

The next training dataset must include negative feedback from
`src/l20_stack/kernel_checks.py` and positive examples that preserve output
shape exactly. Continuing to optimize loss without execution-derived static
feedback is misaligned with the benchmark.

## Quality Gates

No superiority claim is allowed until all of the following are recorded:

- base-model and adapter results from the same evaluator
- held-out task families, not random rows from the same templates
- exact and normalized contamination checks
- at least two training seeds for the selected recipe
- compile rate, correctness rate, and performance rate on the L20
- general coding regression checks on an uncontaminated public benchmark
- loss curves showing train/eval divergence did not select the checkpoint

## Current Constraint

The remote host temporarily cannot reach Hugging Face, so the cached 0.5B and
14B models were used to validate the stack. The intended first quality run is
Qwen2.5-Coder-1.5B-Instruct once the model and evaluation data are available.
