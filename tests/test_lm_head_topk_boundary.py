import importlib.util
from pathlib import Path

import pytest

from l20_stack.ops.triton_lm_head_top1 import (
    lm_head_top1_launch_config,
    should_use_l20_lm_head_top1,
)


def test_lm_head_top1_policy_matches_qwen_shape():
    config = lm_head_top1_launch_config(151_936, 1536)
    assert config.block_vocab == 32
    assert config.block_hidden == 64
    assert config.blocks_per_row == 4748
    assert config.reduce_block == 8192
    assert config.num_warps == 4
    assert config.strategy == "two_stage_direct_lm_head_top1"


def test_lm_head_top1_gate_is_conservative():
    assert should_use_l20_lm_head_top1(1, 151_936, 1536, top_k=1)
    assert should_use_l20_lm_head_top1(4, 151_936, 1536, top_k=1)
    assert not should_use_l20_lm_head_top1(5, 151_936, 1536, top_k=1)
    assert not should_use_l20_lm_head_top1(1, 151_936, 1536, top_k=50)
    assert not should_use_l20_lm_head_top1(1, 300_000, 1536, top_k=1)
    assert not should_use_l20_lm_head_top1(1, 151_936, 1537, top_k=1)


def test_lm_head_top1_rejects_bad_blocks():
    with pytest.raises(ValueError):
        lm_head_top1_launch_config(151_936, 1536, block_vocab=128)
    with pytest.raises(ValueError):
        lm_head_top1_launch_config(151_936, 1536, block_hidden=96)
    with pytest.raises(ValueError):
        lm_head_top1_launch_config(151_936, 1537)


def test_lm_head_topk_boundary_script_tracks_all_paths():
    source = Path("scripts/benchmark_lm_head_topk_boundary.py").read_text()
    assert "full_logits_topk" in source
    assert "chunked_lm_head_topk" in source
    assert "include-triton-top1" in source
    assert "materialized_logits_bytes" in source
    assert "chunked_over_full_logits_topk" in source
    assert "triton_top1_over_full_logits_top1" in source


def test_lm_head_top1_source_is_available():
    spec = importlib.util.find_spec("l20_stack.ops.triton_lm_head_top1")
    assert spec is not None
