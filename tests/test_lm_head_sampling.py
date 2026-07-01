from pathlib import Path

import pytest

from l20_stack.ops.triton_lm_head_sampling import (
    lm_head_sampling_launch_config,
    should_use_l20_lm_head_sampling,
)


def test_lm_head_sampling_policy_uses_tensor_core_compatible_batch_tile():
    batch1 = lm_head_sampling_launch_config(1, 151_936, 1536)
    batch4 = lm_head_sampling_launch_config(4, 151_936, 1536)

    assert (batch1.block_batch, batch1.block_vocab, batch1.block_hidden) == (16, 32, 256)
    assert (batch4.block_batch, batch4.block_vocab, batch4.block_hidden) == (16, 64, 256)
    assert batch4.blocks_per_row == 2374
    assert batch4.reduce_block == 4096
    assert batch4.num_warps == 8
    assert batch4.strategy == "two_stage_lm_head_gumbel_max"


def test_lm_head_sampling_explicit_blocks_override_policy():
    config = lm_head_sampling_launch_config(
        4,
        151_936,
        1536,
        block_vocab=16,
        block_hidden=64,
    )
    assert (config.block_vocab, config.block_hidden) == (16, 64)


def test_lm_head_sampling_rejects_bad_launch_shapes():
    with pytest.raises(ValueError):
        lm_head_sampling_launch_config(0, 151_936, 1536)
    with pytest.raises(ValueError):
        lm_head_sampling_launch_config(4, 151_936, 1537)
    with pytest.raises(ValueError):
        lm_head_sampling_launch_config(4, 151_936, 1536, block_hidden=96)


def test_lm_head_sampling_gate_is_intentionally_narrow():
    assert should_use_l20_lm_head_sampling(1, 151_936, 1536)
    assert should_use_l20_lm_head_sampling(4, 151_936, 1536)
    assert should_use_l20_lm_head_sampling(4, 151_936, 1536, top_k=1, top_p=1.0)

    assert not should_use_l20_lm_head_sampling(5, 151_936, 1536)
    assert not should_use_l20_lm_head_sampling(4, 300_000, 1536)
    assert not should_use_l20_lm_head_sampling(4, 151_936, 1537)
    assert not should_use_l20_lm_head_sampling(4, 151_936, 1536, top_k=50)
    assert not should_use_l20_lm_head_sampling(4, 151_936, 1536, top_p=0.9)


def test_lm_head_sampling_source_has_no_full_logits_materialization():
    import l20_stack.ops.triton_lm_head_sampling as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_lm_head_sampling_partial_kernel" in source
    assert "_lm_head_sampling_reduce_kernel" in source
    assert "USE_GUMBEL" in source
    assert "tl.log(-tl.log(uniform))" in source
    assert "tl.dot(w, h" in source
    assert "hidden @ weight.T" not in source


def test_lm_head_sampling_import_is_cpu_safe():
    import l20_stack.ops.triton_lm_head_sampling as module

    assert module.lm_head_sampling_launch_config(1, 8192, 512).blocks_per_row == 256
    assert module.should_use_l20_lm_head_sampling(1, 8192, 512)
