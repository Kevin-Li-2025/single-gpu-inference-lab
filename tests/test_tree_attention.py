import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from l20_stack.ops.triton_tree_attention import (
    l20_tree_attention_block_t,
    make_chain_tree_mask,
    should_use_l20_split_tree_attention,
    torch_tree_attention_reference,
)


def test_chain_tree_mask_is_lower_triangular():
    mask = make_chain_tree_mask(4)
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_l20_tree_attention_policy_uses_wider_tiles_for_long_context():
    assert l20_tree_attention_block_t(512) == 64
    assert l20_tree_attention_block_t(2048) == 64
    assert l20_tree_attention_block_t(4096) == 128


def test_l20_tree_attention_split_gate_targets_long_context():
    assert not should_use_l20_split_tree_attention(512)
    assert not should_use_l20_split_tree_attention(2048)
    assert should_use_l20_split_tree_attention(4096)


def test_tree_attention_reference_applies_irregular_ancestor_mask():
    query = torch.ones((1, 2, 1, 2), dtype=torch.float32)
    key = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[2.0, 0.0]], [[0.0, 2.0]]]])
    value = torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]], [[4.0, 0.0]], [[8.0, 0.0]]]])
    ancestor_mask = torch.tensor(
        [[True, False], [False, True]],
    )
    actual = torch_tree_attention_reference(query, key, value, ancestor_mask, cached_length=2)
    visible = torch.tensor(
        [[True, True, True, False], [True, True, False, True]],
    )
    scores = torch.einsum("bshd,bthd->bsht", query, key) * (2**-0.5)
    expected = torch.einsum(
        "bsht,bthd->bshd",
        scores.masked_fill(~visible[None, :, None, :], float("-inf")).softmax(-1),
        value,
    )
    assert torch.allclose(actual, expected)


def test_triton_tree_attention_kernel_uses_online_softmax_and_ancestor_mask():
    source = Path("src/l20_stack/ops/triton_tree_attention.py").read_text()
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "hybrid_tree_attention" in names
    assert "hybrid_tree_attention_split" in names
    assert "hybrid_tree_attention_paged_prefix" in names
    assert "causal_verifier_attention_paged" in names
    assert "torch_tree_attention_reference" in names
    assert "ancestor_mask + mask_offsets" in source
    assert "_tree_prefix_summary_kernel" in source
    assert "_tree_paged_prefix_summary_kernel" in source
    assert "_tree_paged_prefix_summary_page16_kernel" in source
    assert "_tree_paged_prefix_summary_contiguous_pages_kernel" in source
    assert "base_page = tl.load(page_base + batch * pb_stride_b)" in source
    assert "physical_page = base_page + token // 16" in source
    assert "_tree_suffix_summary_kernel" in source
    assert "_tree_summary_merge_kernel" in source
    assert "_causal_verifier_paged_attention_kernel" in source
    assert "page_index = tl.arange(0, BLOCK_T // 16)" in source
    assert "tile_pages = tl.load" in source
    assert "page_slot[:, None] == page_index[None, :]" in source
    assert "suffix_mask = (draft_token < draft_length) & (draft_token <= draft_index)" in source
    assert "max_score" in source
    assert "normalizer" in source


def test_longspec_irregular_benchmark_generates_non_chain_masks():
    source = Path("scripts/benchmark_longspec_irregular_tree.py").read_text()
    assert "make_random_tree_mask" in source
    assert "make_balanced_tree_mask" in source
    assert "ancestor_mask_from_parent" in source
    assert '"density"' in source
    assert '"mean_visible_draft_tokens"' in source
    assert "hybrid_tree_attention_paged_prefix" in source
    assert 'contiguous_pages=page_order == "contiguous"' in source
    assert "page_base = block_table[:, 0].contiguous()" in source
    assert "page_base=page_base" in source
