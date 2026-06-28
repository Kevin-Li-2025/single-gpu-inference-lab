#!/usr/bin/env python3
"""Install the opt-in L20 top-k/top-p sampler hook into a vLLM checkout."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path


IMPORT_LINE = (
    "from vllm.v1.sample.ops.l20_topk_topp_sampling import "
    "maybe_l20_topk_topp_sample\n"
)

TOPK_IMPORT_MARKER = "from vllm.triton_utils import HAS_TRITON\n"
FLASHINFER_PATCH_POINT = """    assert not (k is None and p is None)
    if k is None:
"""
FLASHINFER_PATCHED = """    assert not (k is None and p is None)
    l20_sampled = maybe_l20_topk_topp_sample(logits, k, p, generators)
    if l20_sampled is not None:
        return l20_sampled
    if k is None:
"""

WORKER_IMPORT_MARKER = "from vllm.v1.sample.ops.topk_topp_sampler import (\n"
WORKER_PATCH_POINT = """        if use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
"""
WORKER_PATCHED = """        if use_flashinfer:
            l20_top_k_values = self.sampling_states.top_k.np[idx_mapping_np]
            l20_top_p_values = self.sampling_states.top_p.np[idx_mapping_np]
            l20_top_k_uniform = bool((l20_top_k_values == l20_top_k_values[0]).all())
            l20_top_p_uniform = bool((l20_top_p_values == l20_top_p_values[0]).all())
            l20_sampled = None
            if l20_top_k_uniform and l20_top_p_uniform:
                l20_sampled = maybe_l20_topk_topp_sample(
                    processed_logits,
                    top_k,
                    top_p,
                    expanded_idx_mapping=expanded_idx_mapping,
                    seeds=self.sampling_states.seeds.gpu,
                    positions=pos,
                    top_k_value=int(l20_top_k_values[0]),
                    top_p_value=float(l20_top_p_values[0]),
                )
            if l20_sampled is not None:
                sampled = l20_sampled.to(torch.int64)
            else:
                sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="Path to a vLLM source checkout root. Defaults to imported package.",
    )
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def resolve_package(vllm_source: Path | None) -> Path:
    if vllm_source is not None:
        return vllm_source.expanduser().resolve() / "vllm"
    import vllm

    return Path(inspect.getfile(vllm)).parent


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"cannot find patch point: {label}")
    return source.replace(old, new, 1)


def patch_topk_topp_sampler(source: str) -> str:
    source = replace_once(
        source,
        TOPK_IMPORT_MARKER,
        TOPK_IMPORT_MARKER + IMPORT_LINE,
        "topk_topp_sampler import",
    )
    return replace_once(
        source,
        FLASHINFER_PATCH_POINT,
        FLASHINFER_PATCHED,
        "flashinfer_sample hook",
    )


def patch_worker_sampler(source: str) -> str:
    source = replace_once(
        source,
        WORKER_IMPORT_MARKER,
        WORKER_IMPORT_MARKER + "    maybe_l20_topk_topp_sample,\n",
        "worker sampler import",
    )
    return replace_once(
        source,
        WORKER_PATCH_POINT,
        WORKER_PATCHED,
        "worker native sampler hook",
    )


def _install_target(path: Path, patcher) -> bool:
    if not path.exists():
        return False
    backup = path.with_suffix(".py.l20-topk-topp-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patcher(path.read_text(encoding="utf-8")), encoding="utf-8")
    return True


def _restore_target(path: Path) -> bool:
    backup = path.with_suffix(".py.l20-topk-topp-backup")
    if not backup.exists():
        return False
    shutil.copy2(backup, path)
    return True


def install(package: Path) -> None:
    helper = package / "v1" / "sample" / "ops" / "l20_topk_topp_sampling.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("l20_topk_topp_sampling.py"), helper)
    patched = [
        _install_target(
            package / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
            patch_topk_topp_sampler,
        ),
        _install_target(
            package / "v1" / "worker" / "gpu" / "sample" / "sampler.py",
            patch_worker_sampler,
        ),
    ]
    if not any(patched):
        raise RuntimeError(f"missing supported vLLM sampler under: {package}")


def uninstall(package: Path) -> None:
    paths = [
        package / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        package / "v1" / "worker" / "gpu" / "sample" / "sampler.py",
    ]
    for path in paths:
        _restore_target(path)
    (package / "v1" / "sample" / "ops" / "l20_topk_topp_sampling.py").unlink(
        missing_ok=True
    )


def main() -> int:
    args = parse_args()
    package = resolve_package(args.vllm_source)
    if args.uninstall:
        uninstall(package)
    else:
        install(package)
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
