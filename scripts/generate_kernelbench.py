#!/usr/bin/env python3
"""Generate Triton KernelBench candidates with a local Hugging Face model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


L20_CONTEXT = """Target only NVIDIA L20 (Ada sm_89): 92 SMs, 48 GB GDDR6, warp size 32,
1536 resident threads per SM. Optimize measured latency for the exact input shapes.
Prefer coalesced global memory, low launch count, bounded registers, and no unnecessary
intermediate tensors. Do not assume persistent input values or reuse outputs across calls."""

TRITON_ONE_SHOT = r'''import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask), mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out = torch.empty_like(x)
        n = x.numel()
        add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK_SIZE=256)
        return out
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--kernelbench-root", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def find_problem(root: Path, level: int, problem_id: int) -> Path:
    matches = sorted((root / "KernelBench" / f"level{level}").glob(f"{problem_id}_*.py"))
    if len(matches) != 1:
        raise ValueError(f"expected one level {level} problem {problem_id}, found {len(matches)}")
    return matches[0]


def build_prompt(reference_source: str) -> str:
    return f"""You are optimizing a PyTorch program with a custom Triton kernel.
{L20_CONTEXT}

Return complete executable Python code defining `class ModelNew`.

Hard requirements:
- `ModelNew.__init__` must accept exactly the same arguments as `Model.__init__`
  from `get_init_inputs()`. Store constants, weights, and bias needed by forward.
- `ModelNew.forward` must accept exactly the same arguments as `Model.forward`,
  which are the tensors returned by `get_inputs()`.
- Do not use `*args` or `**kwargs` in `ModelNew.__init__` or `ModelNew.forward`.
- Do not define only a helper function. The evaluator imports `ModelNew`.
- Do not call undefined names such as `get_inputs`, `out_features`, `rms_norm`,
  `scaling_factor`, or `weight`; every value must come from `__init__`,
  `forward`, or tensors stored on `self`.
- Do not include a test harness, `if __name__ == "__main__"`, asserts about input
  shapes, or hard-coded demo tensors. Return only importable model code.
- Avoid extra full-size temporary tensors. KernelBench keeps reference and
  candidate outputs for comparison; one unnecessary copy can OOM on L20.
- Use `BLOCK_SIZE` consistently for Triton constexpr names.
- Use `x * x` instead of `x ** 2` inside Triton kernels.
- Inside `@triton.jit` kernels, do not use `tl.sum(..., keepdims=...)`; Triton
  does not accept that keyword. Shape reductions must be written over explicit
  block axes.
- Inside `@triton.jit` kernels, do not call `.view()` or `.reshape()` on Triton
  block tensors with runtime tensor dimensions. Compute flat offsets explicitly.
- Inside `@triton.jit` kernels, always call `tl.arange(0, BLOCK_SIZE)` or another
  explicit `(start, end)` form; never call `tl.arange(dim)` with one argument.
- Every Triton launcher call `kernel[grid](...)` must pass every required
  argument in the `@triton.jit` function signature exactly once, including
  scalar constants such as `dim2` and `eps`.
- Return tensors with exactly the same shape, dtype, and device as the reference
  PyTorch `Model.forward`; do not flatten the final output unless the reference
  does so.
- The result must compile, match PyTorch on randomized inputs, and contain no
  testing code. Output code only.

Valid format example for a different task:
```python
{TRITON_ONE_SHOT}
```

Reference program:
```python
{reference_source}
```
"""


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = max(blocks, key=len) if blocks else text
    candidate = candidate.strip()
    if not candidate:
        raise ValueError("model returned no code")
    return candidate + "\n"


def main() -> int:
    args = parse_args()
    if not args.kernelbench_root.is_dir():
        raise SystemExit("kernelbench root does not exist")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_kwargs = {"device_map": {"": 0}, "dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for task in suite["tasks"]:
        level, problem_id = int(task["level"]), int(task["problem_id"])
        reference_path = find_problem(args.kernelbench_root, level, problem_id)
        reference_source = reference_path.read_text(encoding="utf-8")
        prompt = build_prompt(reference_source)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        candidate = extract_code(response)
        stem = f"level{level}_problem{problem_id}"
        candidate_path = args.output_dir / f"{stem}.py"
        candidate_path.write_text(candidate, encoding="utf-8")
        manifest.append(
            {
                "level": level,
                "problem_id": problem_id,
                "reference": str(reference_path),
                "candidate": str(candidate_path),
                "prompt_characters": len(prompt),
                "response_characters": len(response),
            }
        )
        print(json.dumps(manifest[-1], sort_keys=True), flush=True)

    report = {"model": args.model, "adapter": args.adapter, "tasks": manifest}
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
