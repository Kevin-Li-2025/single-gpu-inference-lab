"""L20-oriented hybrid tree attention for speculative verification."""

from __future__ import annotations

from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:  # pragma: no cover - requires CUDA

    @triton.jit
    def _hybrid_tree_attention_kernel(
        query,
        key,
        value,
        ancestor_mask,
        output,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        k_stride_b,
        k_stride_t,
        k_stride_h,
        v_stride_b,
        v_stride_t,
        v_stride_h,
        mask_stride_q,
        mask_stride_k,
        o_stride_b,
        o_stride_s,
        o_stride_h,
        cached_length: tl.constexpr,
        draft_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        program = tl.program_id(0)
        q_head = program % num_q_heads
        draft_index = (program // num_q_heads) % draft_length
        batch = program // (num_q_heads * draft_length)
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        query_values = tl.load(
            query + batch * q_stride_b + draft_index * q_stride_s + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)
        total_length = cached_length + draft_length

        for start in range(0, total_length, BLOCK_T):
            token = start + tl.arange(0, BLOCK_T)
            in_cached_prefix = token < cached_length
            draft_token = token - cached_length
            in_draft = (token >= cached_length) & (draft_token < draft_length)
            mask_offsets = draft_index * mask_stride_q + draft_token * mask_stride_k
            is_ancestor = (
                tl.load(
                    ancestor_mask + mask_offsets,
                    mask=in_draft,
                    other=0,
                )
                != 0
            )
            token_mask = in_cached_prefix | (in_draft & is_ancestor)
            keys = tl.load(
                key
                + batch * k_stride_b
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query_values[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value
                + batch * v_stride_b
                + token[:, None] * v_stride_t
                + kv_head * v_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * values, axis=0)
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
            max_score = next_max

        tl.store(
            output + batch * o_stride_b + draft_index * o_stride_s + q_head * o_stride_h + dim,
            accumulator / normalizer,
        )

    @triton.jit
    def _tree_prefix_summary_kernel(
        query,
        key,
        value,
        partial_output,
        partial_max,
        partial_sum,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        k_stride_b,
        k_stride_t,
        k_stride_h,
        v_stride_b,
        v_stride_t,
        v_stride_h,
        cached_length: tl.constexpr,
        draft_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        program = tl.program_id(0)
        q_head = program % num_q_heads
        draft_index = (program // num_q_heads) % draft_length
        batch = program // (num_q_heads * draft_length)
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        q = tl.load(
            query + batch * q_stride_b + draft_index * q_stride_s + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)

        for start in range(0, cached_length, BLOCK_T):
            token = start + tl.arange(0, BLOCK_T)
            token_mask = token < cached_length
            keys = tl.load(
                key
                + batch * k_stride_b
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value
                + batch * v_stride_b
                + token[:, None] * v_stride_t
                + kv_head * v_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * values, axis=0)
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
            max_score = next_max

        partial_index = program
        tl.store(partial_output + partial_index * head_dim + dim, accumulator)
        tl.store(partial_max + partial_index, max_score)
        tl.store(partial_sum + partial_index, normalizer)

    @triton.jit
    def _tree_paged_prefix_summary_kernel(
        query,
        key_cache,
        value_cache,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        kc_stride_p,
        kc_stride_t,
        kc_stride_h,
        vc_stride_p,
        vc_stride_t,
        vc_stride_h,
        bt_stride_b,
        cached_length: tl.constexpr,
        draft_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        program = tl.program_id(0)
        q_head = program % num_q_heads
        draft_index = (program // num_q_heads) % draft_length
        batch = program // (num_q_heads * draft_length)
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        q = tl.load(
            query + batch * q_stride_b + draft_index * q_stride_s + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)

        for start in range(0, cached_length, BLOCK_T):
            tile_offset = tl.arange(0, BLOCK_T)
            token = start + tile_offset
            token_mask = token < cached_length
            page_slot = tile_offset // 16
            page_index = tl.arange(0, BLOCK_T // 16)
            logical_page_base = start // 16
            page_mask = logical_page_base + page_index < (cached_length + 15) // 16
            tile_pages = tl.load(
                block_table + batch * bt_stride_b + logical_page_base + page_index,
                mask=page_mask,
                other=0,
            )
            page_offset = token % 16
            physical_page = tl.sum(
                tl.where(page_slot[:, None] == page_index[None, :], tile_pages[None, :], 0),
                axis=1,
            )
            keys = tl.load(
                key_cache
                + physical_page[:, None] * kc_stride_p
                + page_offset[:, None] * kc_stride_t
                + kv_head * kc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value_cache
                + physical_page[:, None] * vc_stride_p
                + page_offset[:, None] * vc_stride_t
                + kv_head * vc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * values, axis=0)
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
            max_score = next_max

        partial_index = program
        tl.store(partial_output + partial_index * head_dim + dim, accumulator)
        tl.store(partial_max + partial_index, max_score)
        tl.store(partial_sum + partial_index, normalizer)

    @triton.jit
    def _tree_paged_prefix_summary_page16_kernel(
        query,
        key_cache,
        value_cache,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        kc_stride_p,
        kc_stride_t,
        kc_stride_h,
        vc_stride_p,
        vc_stride_t,
        vc_stride_h,
        bt_stride_b,
        cached_length: tl.constexpr,
        draft_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
    ):
        program = tl.program_id(0)
        q_head = program % num_q_heads
        draft_index = (program // num_q_heads) % draft_length
        batch = program // (num_q_heads * draft_length)
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        page_offset = tl.arange(0, 16)
        q = tl.load(
            query + batch * q_stride_b + draft_index * q_stride_s + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)

        for token_start in range(0, cached_length, 16):
            token = token_start + page_offset
            token_mask = token < cached_length
            logical_page = token_start // 16
            physical_page = tl.load(block_table + batch * bt_stride_b + logical_page)
            keys = tl.load(
                key_cache
                + physical_page * kc_stride_p
                + page_offset[:, None] * kc_stride_t
                + kv_head * kc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value_cache
                + physical_page * vc_stride_p
                + page_offset[:, None] * vc_stride_t
                + kv_head * vc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * values, axis=0)
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
            max_score = next_max

        partial_index = program
        tl.store(partial_output + partial_index * head_dim + dim, accumulator)
        tl.store(partial_max + partial_index, max_score)
        tl.store(partial_sum + partial_index, normalizer)

    @triton.jit
    def _tree_paged_prefix_summary_contiguous_pages_kernel(
        query,
        key_cache,
        value_cache,
        partial_output,
        partial_max,
        partial_sum,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        kc_stride_p,
        kc_stride_t,
        kc_stride_h,
        vc_stride_p,
        vc_stride_t,
        vc_stride_h,
        cached_length: tl.constexpr,
        draft_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        pages_per_batch: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        program = tl.program_id(0)
        q_head = program % num_q_heads
        draft_index = (program // num_q_heads) % draft_length
        batch = program // (num_q_heads * draft_length)
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        q = tl.load(
            query + batch * q_stride_b + draft_index * q_stride_s + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)

        for start in range(0, cached_length, BLOCK_T):
            token = start + tl.arange(0, BLOCK_T)
            token_mask = token < cached_length
            physical_page = batch * pages_per_batch + token // 16
            page_offset = token % 16
            keys = tl.load(
                key_cache
                + physical_page[:, None] * kc_stride_p
                + page_offset[:, None] * kc_stride_t
                + kv_head * kc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value_cache
                + physical_page[:, None] * vc_stride_p
                + page_offset[:, None] * vc_stride_t
                + kv_head * vc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * values, axis=0)
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
            max_score = next_max

        partial_index = program
        tl.store(partial_output + partial_index * head_dim + dim, accumulator)
        tl.store(partial_max + partial_index, max_score)
        tl.store(partial_sum + partial_index, normalizer)

    @triton.jit
    def _tree_suffix_summary_kernel(
        query,
        key,
        value,
        ancestor_mask,
        partial_output,
        partial_max,
        partial_sum,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        k_stride_b,
        k_stride_t,
        k_stride_h,
        v_stride_b,
        v_stride_t,
        v_stride_h,
        mask_stride_q,
        mask_stride_k,
        cached_length: tl.constexpr,
        draft_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        program = tl.program_id(0)
        q_head = program % num_q_heads
        draft_index = (program // num_q_heads) % draft_length
        batch = program // (num_q_heads * draft_length)
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        draft_token = tl.arange(0, BLOCK_T)
        token_mask = draft_token < draft_length
        is_ancestor = (
            tl.load(
                ancestor_mask + draft_index * mask_stride_q + draft_token * mask_stride_k,
                mask=token_mask,
                other=0,
            )
            != 0
        )
        token_mask = token_mask & is_ancestor
        q = tl.load(
            query + batch * q_stride_b + draft_index * q_stride_s + q_head * q_stride_h + dim
        ).to(tl.float32)
        token = cached_length + draft_token
        keys = tl.load(
            key
            + batch * k_stride_b
            + token[:, None] * k_stride_t
            + kv_head * k_stride_h
            + dim[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1) * (1.0 / tl.sqrt(float(head_dim)))
        scores = tl.where(token_mask, scores, -float("inf"))
        max_score = tl.max(scores, axis=0)
        probabilities = tl.exp(scores - max_score)
        values = tl.load(
            value
            + batch * v_stride_b
            + token[:, None] * v_stride_t
            + kv_head * v_stride_h
            + dim[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.sum(probabilities[:, None] * values, axis=0)
        normalizer = tl.sum(probabilities, axis=0)
        partial_index = program
        tl.store(partial_output + partial_index * head_dim + dim, accumulator)
        tl.store(partial_max + partial_index, max_score)
        tl.store(partial_sum + partial_index, normalizer)

    @triton.jit
    def _tree_summary_merge_kernel(
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
        output,
        o_stride_b,
        o_stride_s,
        o_stride_h,
        draft_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        head_dim: tl.constexpr,
    ):
        program = tl.program_id(0)
        q_head = program % num_q_heads
        draft_index = (program // num_q_heads) % draft_length
        batch = program // (num_q_heads * draft_length)
        dim = tl.arange(0, head_dim)
        prefix_m = tl.load(prefix_max + program)
        suffix_m = tl.load(suffix_max + program)
        global_m = tl.maximum(prefix_m, suffix_m)
        prefix_scale = tl.exp(prefix_m - global_m)
        suffix_scale = tl.exp(suffix_m - global_m)
        prefix_l = tl.load(prefix_sum + program)
        suffix_l = tl.load(suffix_sum + program)
        denominator = prefix_l * prefix_scale + suffix_l * suffix_scale
        prefix_acc = tl.load(prefix_output + program * head_dim + dim).to(tl.float32)
        suffix_acc = tl.load(suffix_output + program * head_dim + dim).to(tl.float32)
        merged = (prefix_acc * prefix_scale + suffix_acc * suffix_scale) / denominator
        tl.store(
            output + batch * o_stride_b + draft_index * o_stride_s + q_head * o_stride_h + dim,
            merged,
        )

    @triton.jit
    def _causal_verifier_paged_attention_kernel(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        output,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        kc_stride_p,
        kc_stride_t,
        kc_stride_h,
        vc_stride_p,
        vc_stride_t,
        vc_stride_h,
        sk_stride_b,
        sk_stride_t,
        sk_stride_h,
        sv_stride_b,
        sv_stride_t,
        sv_stride_h,
        bt_stride_b,
        o_stride_b,
        o_stride_s,
        o_stride_h,
        cached_length: tl.constexpr,
        draft_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_DRAFT: tl.constexpr,
    ):
        program = tl.program_id(0)
        q_head = program % num_q_heads
        draft_index = (program // num_q_heads) % draft_length
        batch = program // (num_q_heads * draft_length)
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        q = tl.load(
            query + batch * q_stride_b + draft_index * q_stride_s + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)

        for start in range(0, cached_length, BLOCK_T):
            tile_offset = tl.arange(0, BLOCK_T)
            token = start + tile_offset
            token_mask = token < cached_length
            page_slot = tile_offset // 16
            page_index = tl.arange(0, BLOCK_T // 16)
            logical_page_base = start // 16
            page_mask = logical_page_base + page_index < (cached_length + 15) // 16
            tile_pages = tl.load(
                block_table + batch * bt_stride_b + logical_page_base + page_index,
                mask=page_mask,
                other=0,
            )
            page_offset = token % 16
            physical_page = tl.sum(
                tl.where(page_slot[:, None] == page_index[None, :], tile_pages[None, :], 0),
                axis=1,
            )
            keys = tl.load(
                key_cache
                + physical_page[:, None] * kc_stride_p
                + page_offset[:, None] * kc_stride_t
                + kv_head * kc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value_cache
                + physical_page[:, None] * vc_stride_p
                + page_offset[:, None] * vc_stride_t
                + kv_head * vc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * values, axis=0)
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
            max_score = next_max

        draft_token = tl.arange(0, BLOCK_DRAFT)
        suffix_mask = (draft_token < draft_length) & (draft_token <= draft_index)
        suffix_keys = tl.load(
            suffix_key
            + batch * sk_stride_b
            + draft_token[:, None] * sk_stride_t
            + kv_head * sk_stride_h
            + dim[None, :],
            mask=suffix_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        suffix_scores = tl.sum(suffix_keys * q[None, :], axis=1) * scale
        suffix_scores = tl.where(suffix_mask, suffix_scores, -float("inf"))
        suffix_max = tl.max(suffix_scores, axis=0)
        next_max = tl.maximum(max_score, suffix_max)
        old_scale = tl.exp(max_score - next_max)
        suffix_probabilities = tl.exp(suffix_scores - next_max)
        suffix_values = tl.load(
            suffix_value
            + batch * sv_stride_b
            + draft_token[:, None] * sv_stride_t
            + kv_head * sv_stride_h
            + dim[None, :],
            mask=suffix_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale + tl.sum(
            suffix_probabilities[:, None] * suffix_values,
            axis=0,
        )
        normalizer = normalizer * old_scale + tl.sum(suffix_probabilities, axis=0)

        tl.store(
            output + batch * o_stride_b + draft_index * o_stride_s + q_head * o_stride_h + dim,
            accumulator / normalizer,
        )


def _validate_tree_attention_inputs(query, key, value, ancestor_mask, cached_length):
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("expected query=[B,S,Hq,D], key/value=[B,T,Hkv,D]")
    if ancestor_mask.ndim != 2:
        raise ValueError("expected ancestor_mask=[S,S]")
    batch, draft_length, num_q_heads, head_dim = query.shape
    key_batch, total_length, num_kv_heads, key_dim = key.shape
    if value.shape != key.shape:
        raise ValueError("key and value shapes must match")
    if key_batch != batch or key_dim != head_dim:
        raise ValueError("query and KV dimensions do not match")
    if total_length != cached_length + draft_length:
        raise ValueError("KV length must equal cached_length + draft_length")
    if ancestor_mask.shape != (draft_length, draft_length):
        raise ValueError("ancestor_mask must match draft_length")
    if head_dim != 128:
        raise ValueError("L20 tree attention currently requires head_dim=128")
    if num_q_heads % num_kv_heads:
        raise ValueError("requires an integral GQA ratio")


def hybrid_tree_attention(
    query,
    key,
    value,
    ancestor_mask,
    cached_length: int,
    *,
    block_t: Optional[int] = None,
):
    """Run contiguous-cache tree attention for speculative verification.

    Cached prefix tokens are visible to every draft query. Draft tokens are
    visible according to ``ancestor_mask[q, k]``. This is the first L20-specific
    building block before paged-cache and vLLM speculative decoding integration.
    """
    _validate_tree_attention_inputs(query, key, value, ancestor_mask, cached_length)
    batch, draft_length, num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    output = torch.empty_like(query)
    if block_t is None:
        block_t = l20_tree_attention_block_t(cached_length)
    if block_t not in (32, 64, 128):
        raise ValueError("block_t must be one of 32, 64, or 128")
    _hybrid_tree_attention_kernel[(batch * draft_length * num_q_heads,)](
        query,
        key,
        value,
        ancestor_mask,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        ancestor_mask.stride(0),
        ancestor_mask.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        cached_length=cached_length,
        draft_length=draft_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=1,
    )
    return output


def allocate_tree_attention_workspace(query):
    """Allocate prefix/suffix summary buffers for split hybrid tree attention."""
    if torch is None:
        raise RuntimeError("requires PyTorch")
    if query.ndim != 4:
        raise ValueError("expected query=[B,S,Hq,D]")
    batch, draft_length, num_q_heads, head_dim = query.shape
    summary_shape = (batch, draft_length, num_q_heads)
    return (
        torch.empty((*summary_shape, head_dim), device=query.device, dtype=torch.float32),
        torch.empty(summary_shape, device=query.device, dtype=torch.float32),
        torch.empty(summary_shape, device=query.device, dtype=torch.float32),
        torch.empty((*summary_shape, head_dim), device=query.device, dtype=torch.float32),
        torch.empty(summary_shape, device=query.device, dtype=torch.float32),
        torch.empty(summary_shape, device=query.device, dtype=torch.float32),
    )


def hybrid_tree_attention_split(
    query,
    key,
    value,
    ancestor_mask,
    cached_length: int,
    *,
    workspace=None,
    block_t: Optional[int] = None,
):
    """Run LongSpec-style prefix/suffix tree attention with log-sum-exp merge.

    The cached prefix is summarized without a mask. The speculative suffix is
    summarized with the irregular ancestor mask. A small merge kernel combines
    both summaries exactly in softmax space.
    """
    _validate_tree_attention_inputs(query, key, value, ancestor_mask, cached_length)
    batch, draft_length, num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    if block_t is None:
        block_t = l20_tree_attention_block_t(cached_length)
    if block_t not in (32, 64, 128):
        raise ValueError("block_t must be one of 32, 64, or 128")
    if draft_length > 64:
        raise ValueError("split suffix path currently supports draft_length <= 64")
    if workspace is None:
        workspace = allocate_tree_attention_workspace(query)
    (
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
    ) = workspace
    output = torch.empty_like(query)
    grid = (batch * draft_length * num_q_heads,)
    _tree_prefix_summary_kernel[grid](
        query,
        key,
        value,
        prefix_output,
        prefix_max,
        prefix_sum,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        cached_length=cached_length,
        draft_length=draft_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=1,
    )
    _tree_suffix_summary_kernel[grid](
        query,
        key,
        value,
        ancestor_mask,
        suffix_output,
        suffix_max,
        suffix_sum,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        ancestor_mask.stride(0),
        ancestor_mask.stride(1),
        cached_length=cached_length,
        draft_length=draft_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=64,
        num_warps=4,
        num_stages=1,
    )
    _tree_summary_merge_kernel[grid](
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
        output,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        draft_length=draft_length,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        num_warps=4,
        num_stages=1,
    )
    return output


def _validate_paged_tree_attention_inputs(
    query,
    key_cache,
    value_cache,
    suffix_key,
    suffix_value,
    block_table,
    ancestor_mask,
    cached_length,
):
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if query.ndim != 4:
        raise ValueError("expected query=[B,S,Hq,D]")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("expected key/value cache=[pages,16,Hkv,D]")
    if suffix_key.ndim != 4 or suffix_value.shape != suffix_key.shape:
        raise ValueError("expected suffix key/value=[B,S,Hkv,D]")
    if block_table.ndim != 2:
        raise ValueError("expected block_table=[B,pages]")
    batch, draft_length, num_q_heads, head_dim = query.shape
    pages, page_size, num_kv_heads, key_dim = key_cache.shape
    if page_size != 16:
        raise ValueError("paged prefix currently requires page_size=16")
    if key_dim != head_dim or suffix_key.shape[-1] != head_dim:
        raise ValueError("query and KV dimensions do not match")
    if suffix_key.shape[:3] != (batch, draft_length, num_kv_heads):
        raise ValueError("suffix KV must match query batch/draft and cache kv heads")
    if block_table.shape[0] != batch or block_table.shape[1] * page_size < cached_length:
        raise ValueError("block_table does not cover cached_length")
    if ancestor_mask.shape != (draft_length, draft_length):
        raise ValueError("ancestor_mask must match draft_length")
    if head_dim != 128:
        raise ValueError("L20 tree attention currently requires head_dim=128")
    if num_q_heads % num_kv_heads:
        raise ValueError("requires an integral GQA ratio")


def _validate_paged_causal_verifier_inputs(
    query,
    key_cache,
    value_cache,
    suffix_key,
    suffix_value,
    block_table,
    cached_length,
):
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if query.ndim != 4:
        raise ValueError("expected query=[B,S,Hq,D]")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("expected key/value cache=[pages,16,Hkv,D]")
    if suffix_key.ndim != 4 or suffix_value.shape != suffix_key.shape:
        raise ValueError("expected suffix key/value=[B,S,Hkv,D]")
    if block_table.ndim != 2:
        raise ValueError("expected block_table=[B,pages]")
    batch, draft_length, num_q_heads, head_dim = query.shape
    page_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    key_dim = key_cache.shape[3]
    if page_size != 16:
        raise ValueError("paged causal verifier currently requires page_size=16")
    if key_dim != head_dim or suffix_key.shape[-1] != head_dim:
        raise ValueError("query and KV dimensions do not match")
    if suffix_key.shape[:3] != (batch, draft_length, num_kv_heads):
        raise ValueError("suffix KV must match query batch/draft and cache kv heads")
    if block_table.shape[0] != batch or block_table.shape[1] * page_size < cached_length:
        raise ValueError("block_table does not cover cached_length")
    if head_dim != 128:
        raise ValueError("L20 tree attention currently requires head_dim=128")
    if num_q_heads % num_kv_heads:
        raise ValueError("requires an integral GQA ratio")
    if draft_length < 1 or draft_length > 64:
        raise ValueError("causal verifier path supports draft_length in [1, 64]")


def causal_verifier_attention_paged(
    query,
    key_cache,
    value_cache,
    suffix_key,
    suffix_value,
    block_table,
    cached_length: int,
    *,
    block_t: Optional[int] = None,
):
    """Run a single-kernel paged causal verifier attention path.

    This is the causal-chain specialization used by vLLM speculative verifier
    prefill. It fuses the long paged prefix scan, short causal suffix scan, and
    log-sum-exp merge into one Triton launch.
    """
    _validate_paged_causal_verifier_inputs(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        cached_length,
    )
    batch, draft_length, num_q_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    if block_t is None:
        block_t = l20_tree_attention_block_t(cached_length)
    if block_t not in (32, 64, 128):
        raise ValueError("block_t must be one of 32, 64, or 128")
    output = torch.empty_like(query)
    grid = (batch * draft_length * num_q_heads,)
    _causal_verifier_paged_attention_kernel[grid](
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        suffix_key.stride(0),
        suffix_key.stride(1),
        suffix_key.stride(2),
        suffix_value.stride(0),
        suffix_value.stride(1),
        suffix_value.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        cached_length=cached_length,
        draft_length=draft_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=block_t,
        BLOCK_DRAFT=64,
        num_warps=4,
        num_stages=1,
    )
    return output


def hybrid_tree_attention_paged_prefix(
    query,
    key_cache,
    value_cache,
    suffix_key,
    suffix_value,
    block_table,
    ancestor_mask,
    cached_length: int,
    *,
    workspace=None,
    block_t: Optional[int] = None,
    contiguous_pages: bool = False,
):
    """Run split tree attention with a page-table cached prefix.

    This is the first vLLM-shaped tree attention interface: the long cached
    prefix comes from a page-16 NHD cache, while the short speculative suffix is
    contiguous and masked by the tree ancestor matrix.
    """
    _validate_paged_tree_attention_inputs(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        ancestor_mask,
        cached_length,
    )
    batch, draft_length, num_q_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    if block_t is None:
        block_t = l20_tree_attention_block_t(cached_length)
    if block_t not in (32, 64, 128):
        raise ValueError("block_t must be one of 32, 64, or 128")
    if draft_length > 64:
        raise ValueError("paged suffix path currently supports draft_length <= 64")
    if workspace is None:
        workspace = allocate_tree_attention_workspace(query)
    (
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
    ) = workspace
    output = torch.empty_like(query)
    grid = (batch * draft_length * num_q_heads,)
    if contiguous_pages:
        _tree_paged_prefix_summary_contiguous_pages_kernel[grid](
            query,
            key_cache,
            value_cache,
            prefix_output,
            prefix_max,
            prefix_sum,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            cached_length=cached_length,
            draft_length=draft_length,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            pages_per_batch=cached_length // 16,
            BLOCK_T=block_t,
            num_warps=4,
            num_stages=1,
        )
    else:
        _tree_paged_prefix_summary_kernel[grid](
            query,
            key_cache,
            value_cache,
            block_table,
            prefix_output,
            prefix_max,
            prefix_sum,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_table.stride(0),
            cached_length=cached_length,
            draft_length=draft_length,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            BLOCK_T=block_t,
            num_warps=4,
            num_stages=1,
        )
    _tree_suffix_summary_kernel[grid](
        query,
        suffix_key,
        suffix_value,
        ancestor_mask,
        suffix_output,
        suffix_max,
        suffix_sum,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        suffix_key.stride(0),
        suffix_key.stride(1),
        suffix_key.stride(2),
        suffix_value.stride(0),
        suffix_value.stride(1),
        suffix_value.stride(2),
        ancestor_mask.stride(0),
        ancestor_mask.stride(1),
        cached_length=0,
        draft_length=draft_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=64,
        num_warps=4,
        num_stages=1,
    )
    _tree_summary_merge_kernel[grid](
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
        output,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        draft_length=draft_length,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        num_warps=4,
        num_stages=1,
    )
    return output


def should_use_l20_split_tree_attention(cached_length: int) -> bool:
    """Use split summary only where L20 measurements beat monolithic tree attention."""
    return cached_length >= 4096


def l20_tree_attention_block_t(cached_length: int) -> int:
    """Measured L20 tile policy for contiguous hybrid tree attention."""
    return 128 if cached_length >= 4096 else 64


def make_chain_tree_mask(draft_length: int, *, device=None):
    """Return a lower-triangular ancestor mask for a linear draft chain."""
    if torch is None:
        raise RuntimeError("requires PyTorch")
    positions = torch.arange(draft_length, device=device)
    return positions[None, :] <= positions[:, None]


def torch_tree_attention_reference(query, key, value, ancestor_mask, cached_length: int):
    """Dense PyTorch reference for correctness checks and CPU-free tests."""
    if torch is None:
        raise RuntimeError("requires PyTorch")
    _validate_reference_inputs(query, key, value, ancestor_mask, cached_length)
    batch, draft_length, num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    group = num_q_heads // num_kv_heads
    expanded_key = key.repeat_interleave(group, dim=2)
    expanded_value = value.repeat_interleave(group, dim=2)
    scores = torch.einsum("bshd,bthd->bsht", query.float(), expanded_key.float()) * (head_dim**-0.5)
    visible = torch.zeros(
        draft_length,
        cached_length + draft_length,
        device=query.device,
        dtype=torch.bool,
    )
    visible[:, :cached_length] = True
    visible[:, cached_length:] = ancestor_mask.to(torch.bool)
    scores = scores.masked_fill(~visible[None, :, None, :], float("-inf"))
    probabilities = scores.softmax(dim=-1)
    output = torch.einsum("bsht,bthd->bshd", probabilities, expanded_value.float())
    return output.to(query.dtype)


def _validate_reference_inputs(query, key, value, ancestor_mask, cached_length):
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("expected query=[B,S,Hq,D], key/value=[B,T,Hkv,D]")
    draft_length = query.shape[1]
    if key.shape[1] != cached_length + draft_length:
        raise ValueError("KV length must equal cached_length + draft_length")
    if ancestor_mask.shape != (draft_length, draft_length):
        raise ValueError("ancestor_mask must match draft_length")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and KV dimensions do not match")
    if query.shape[2] % key.shape[2]:
        raise ValueError("requires an integral GQA ratio")
