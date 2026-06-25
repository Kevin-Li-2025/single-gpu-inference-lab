#!/usr/bin/env python3
"""Install the experimental L20 shared-prefix decode ops into vLLM."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import vllm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    package = Path(next(iter(vllm.__path__)))
    op_dir = package / "v1" / "attention" / "ops"
    targets = {
        "l20_decode_attention.py": Path("src/l20_stack/ops/triton_decode_attention.py"),
        "l20_shared_prefix_decode_dispatch.py": Path(
            "integrations/vllm/l20_shared_prefix_decode_dispatch.py"
        ),
    }
    backups = {
        name: (op_dir / name).with_suffix(".py.l20-shared-prefix-backup")
        for name in targets
    }
    if args.uninstall:
        for name, backup in backups.items():
            target = op_dir / name
            if backup.exists():
                shutil.copy2(backup, target)
            elif target.exists():
                target.unlink()
        return 0

    root = Path(__file__).resolve().parents[2]
    install_dirs = [op_dir]
    source_tree = os.getenv("VLLM_SOURCE_TREE")
    if source_tree:
        source_op_dir = Path(source_tree) / "vllm" / "v1" / "attention" / "ops"
        if source_op_dir.exists() and source_op_dir != op_dir:
            install_dirs.append(source_op_dir)
    for install_dir in install_dirs:
        install_dir.mkdir(parents=True, exist_ok=True)
        for name, source in targets.items():
            target = install_dir / name
            backup = target.with_suffix(".py.l20-shared-prefix-backup")
            if target.exists() and not backup.exists():
                shutil.copy2(target, backup)
            shutil.copy2(root / source, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
