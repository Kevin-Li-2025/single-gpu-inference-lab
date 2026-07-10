#!/usr/bin/env python3
"""Install or remove the opt-in Apple M4 affine Q4_K SME2 decode path."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


INCLUDE_ANCHOR = '#include "ggml-common.h"\n'
INCLUDE_BLOCK = (
    "// KEVIN_M4_Q4K_SME2_INCLUDE_BEGIN\n"
    '#include "kevin_m4_q4k_sme2.h"\n'
    "// KEVIN_M4_Q4K_SME2_INCLUDE_END\n"
)
WORK_ANCHOR = (
    "    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {\n"
)
WORK_BLOCK = (
    "// KEVIN_M4_Q4K_SME2_WORK_BEGIN\n"
    "        if (kevin_m4_q4k_sme2_work_size(op, size)) {\n"
    "            return true;\n"
    "        }\n"
    "// KEVIN_M4_Q4K_SME2_WORK_END\n"
)
COMPUTE_ANCHOR = (
    "    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * dst) override {\n"
)
COMPUTE_BLOCK = (
    "// KEVIN_M4_Q4K_SME2_COMPUTE_BEGIN\n"
    "        if (kevin_m4_q4k_sme2_compute(params, dst)) {\n"
    "            return true;\n"
    "        }\n"
    "// KEVIN_M4_Q4K_SME2_COMPUTE_END\n"
)
REPACK_ANCHOR = (
    "    int repack(struct ggml_tensor * tensor, const void * data, size_t data_size) {\n"
)
REPACK_BLOCK = (
    "// KEVIN_M4_Q4K_SME2_REPACK_BEGIN\n"
    "        if (kevin_m4_q4k_sme2_tensor_eligible(tensor) && kevin_m4_q4k_sme2_enabled()) {\n"
    "            return kevin_m4_q4k_sme2_repack(tensor, data, data_size);\n"
    "        }\n"
    "// KEVIN_M4_Q4K_SME2_REPACK_END\n"
)
ALLOC_ANCHOR = (
    "static size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size("
    "ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {\n"
    "    GGML_UNUSED(buft);\n"
)
ALLOC_BLOCK = (
    "// KEVIN_M4_Q4K_SME2_ALLOC_BEGIN\n"
    "    if (kevin_m4_q4k_sme2_tensor_eligible(tensor) && kevin_m4_q4k_sme2_enabled()) {\n"
    "        return kevin_m4_q4k_sme2_alloc_size(tensor);\n"
    "    }\n"
    "// KEVIN_M4_Q4K_SME2_ALLOC_END\n"
)
SUPPORT_ANCHOR = (
    "    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {\n"
)
SUPPORT_BLOCK = (
    "// KEVIN_M4_Q4K_SME2_SUPPORT_BEGIN\n"
    "        if (kevin_m4_q4k_sme2_supports_op(op)) {\n"
    "            return true;\n"
    "        }\n"
    "// KEVIN_M4_Q4K_SME2_SUPPORT_END\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", required=True)
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def remove_block(text: str, begin: str, end: str) -> str:
    start = text.find(begin)
    if start < 0:
        return text
    finish = text.find(end, start)
    if finish < 0:
        raise RuntimeError(f"found {begin.strip()} without {end.strip()}")
    finish += len(end)
    if finish < len(text) and text[finish] == "\n":
        finish += 1
    return text[:start] + text[finish:]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    llama_root = Path(args.llama_root).resolve()
    target = llama_root / "ggml/src/ggml-cpu/kleidiai/kleidiai.cpp"
    destination = target.parent / "kevin_m4_q4k_sme2.h"
    if not target.exists():
        raise FileNotFoundError(f"llama.cpp KleidiAI source not found: {target}")

    text = target.read_text(encoding="utf-8")
    markers = ("INCLUDE", "WORK", "COMPUTE", "REPACK", "ALLOC", "SUPPORT")
    for marker in markers:
        text = remove_block(
            text,
            f"// KEVIN_M4_Q4K_SME2_{marker}_BEGIN",
            f"// KEVIN_M4_Q4K_SME2_{marker}_END",
        )

    if args.uninstall:
        target.write_text(text, encoding="utf-8")
        destination.unlink(missing_ok=True)
        print(f"removed Kevin M4 affine Q4_K SME2 path from {target}")
        return 0

    replacements = (
        (INCLUDE_ANCHOR, INCLUDE_ANCHOR + INCLUDE_BLOCK),
        (WORK_ANCHOR, WORK_ANCHOR + WORK_BLOCK),
        (COMPUTE_ANCHOR, COMPUTE_ANCHOR + COMPUTE_BLOCK),
        (REPACK_ANCHOR, REPACK_ANCHOR + REPACK_BLOCK),
        (ALLOC_ANCHOR, ALLOC_ANCHOR + ALLOC_BLOCK),
        (SUPPORT_ANCHOR, SUPPORT_ANCHOR + SUPPORT_BLOCK),
    )
    for anchor, replacement in replacements:
        if text.count(anchor) != 1:
            raise RuntimeError(f"llama.cpp anchor changed: {anchor.strip()}")
        text = text.replace(anchor, replacement, 1)

    target.write_text(text, encoding="utf-8")
    shutil.copyfile(
        repo_root / "integrations/llama_cpp/kevin_m4_q4k_sme2.h", destination
    )
    print(f"installed Kevin M4 affine Q4_K SME2 path in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
