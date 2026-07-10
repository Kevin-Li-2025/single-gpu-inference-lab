#!/usr/bin/env python3
"""Install or remove the opt-in Apple M4 Q4_K kernel in a llama.cpp tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


INCLUDE_ANCHOR = '#include "../../ggml-cpu-impl.h"\n'
INCLUDE_BLOCK = (
    "// KEVIN_M4_Q4K_INCLUDE_BEGIN\n"
    '#include "kevin_m4_q4k.h"\n'
    "// KEVIN_M4_Q4K_INCLUDE_END\n"
)
FUNCTION_ANCHOR = (
    "void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, "
    "const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, "
    "size_t by, int nrc) {\n"
)
HOOK_BLOCK = (
    "// KEVIN_M4_Q4K_HOOK_BEGIN\n"
    "    if (nrc == 1 && n % QK_K == 0 && kevin_m4_q4k_enabled()) {\n"
    "        kevin_m4_vec_dot_q4_K_q8_K(n, s, vx, vy);\n"
    "        return;\n"
    "    }\n"
    "// KEVIN_M4_Q4K_HOOK_END\n"
)
REPACK_ANCHOR = "    } else if (cur->type == GGML_TYPE_Q4_K) {\n"
REPACK_BLOCK = (
    "// KEVIN_M4_Q4K_REPACK_BEGIN\n"
    "        const char * kevin_m4_q4k = getenv(\"GGML_M4_Q4K_CUSTOM\");\n"
    "        if (kevin_m4_q4k != nullptr && kevin_m4_q4k[0] == '1' && "
    "kevin_m4_q4k[1] == '\\0') {\n"
    "            return nullptr;\n"
    "        }\n"
    "// KEVIN_M4_Q4K_REPACK_END\n"
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
    target = llama_root / "ggml/src/ggml-cpu/arch/arm/quants.c"
    repack_target = llama_root / "ggml/src/ggml-cpu/repack.cpp"
    destination_header = target.parent / "kevin_m4_q4k.h"
    if not target.exists() or not repack_target.exists():
        raise FileNotFoundError(f"llama.cpp ARM quant source not found: {target}")

    text = target.read_text(encoding="utf-8")
    text = remove_block(
        text, "// KEVIN_M4_Q4K_INCLUDE_BEGIN", "// KEVIN_M4_Q4K_INCLUDE_END"
    )
    text = remove_block(
        text, "// KEVIN_M4_Q4K_HOOK_BEGIN", "// KEVIN_M4_Q4K_HOOK_END"
    )
    repack_text = repack_target.read_text(encoding="utf-8")
    repack_text = remove_block(
        repack_text,
        "// KEVIN_M4_Q4K_REPACK_BEGIN",
        "// KEVIN_M4_Q4K_REPACK_END",
    )

    if args.uninstall:
        target.write_text(text, encoding="utf-8")
        repack_target.write_text(repack_text, encoding="utf-8")
        destination_header.unlink(missing_ok=True)
        print(f"removed Kevin M4 Q4_K hook from {target}")
        return 0

    if INCLUDE_ANCHOR not in text:
        raise RuntimeError("llama.cpp include anchor changed")
    if text.count(FUNCTION_ANCHOR) != 1:
        raise RuntimeError("llama.cpp Q4_K function anchor changed")
    if repack_text.count(REPACK_ANCHOR) != 1:
        raise RuntimeError("llama.cpp Q4_K repack anchor changed")
    text = text.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + INCLUDE_BLOCK, 1)
    text = text.replace(FUNCTION_ANCHOR, FUNCTION_ANCHOR + HOOK_BLOCK, 1)
    repack_text = repack_text.replace(REPACK_ANCHOR, REPACK_ANCHOR + REPACK_BLOCK, 1)
    target.write_text(text, encoding="utf-8")
    repack_target.write_text(repack_text, encoding="utf-8")
    shutil.copyfile(repo_root / "integrations/llama_cpp/kevin_m4_q4k.h", destination_header)
    print(f"installed opt-in Kevin M4 Q4_K hook in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
