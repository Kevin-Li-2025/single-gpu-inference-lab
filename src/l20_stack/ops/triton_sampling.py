"""L20-oriented GPU-side decode sampling primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class SamplingLaunchConfig:
    block_vocab: int
    blocks_per_row: int
    num_warps: int
    num_stages: int
    strategy: str

    def to_dict(self):
        return asdict(self)


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def greedy_sampling_launch_config(
    vocab_size: int,
    *,
    block_vocab_override: Optional[int] = None,
) -> SamplingLaunchConfig:
    """Return the L20 launch policy for greedy sampling.

    The target is decode serving, where transferring `[batch, vocab]` logits to
    CPU just to choose one token is usually worse than a small GPU reduction. A
    single CTA is too serial for Qwen-sized vocabularies, so large vocabularies
    use a two-stage block reduction.
    """

    if vocab_size > 262_144:
        raise ValueError("vocab_size above 262144 requires a multi-stage sampling path")
    if vocab_size > 65_536:
        block_vocab = block_vocab_override or 1024
        if block_vocab not in {512, 1024, 2048, 4096, 8192}:
            raise ValueError("block_vocab_override must be one of 512, 1024, 2048, 4096, 8192")
        blocks_per_row = (vocab_size + block_vocab - 1) // block_vocab
        num_warps = 8 if block_vocab >= 4096 else 4
        strategy = "two_stage_block_argmax"
    else:
        if block_vocab_override is not None:
            raise ValueError("block_vocab_override is only supported for large vocabularies")
        block_vocab = next_power_of_2(vocab_size)
        blocks_per_row = 1
        num_warps = 4 if block_vocab >= 32_768 else 2
        strategy = "single_cta_argmax"
    return SamplingLaunchConfig(
        block_vocab=block_vocab,
        blocks_per_row=blocks_per_row,
        num_warps=num_warps,
        num_stages=1,
        strategy=strategy,
    )


def should_use_l20_gpu_greedy_sampling(batch: int, vocab_size: int, top_k: int = 1) -> bool:
    """Conservative L20 gate for the first GPU-side sampler path."""

    if top_k != 1:
        return False
    if batch <= 0 or vocab_size <= 0:
        return False
    return batch <= 64 and vocab_size <= 262_144


def topk_topp_sampling_launch_config(
    vocab_size: int,
    top_k: int,
    *,
    batch: int | None = None,
    block_vocab_override: Optional[int] = None,
) -> SamplingLaunchConfig:
    """Return the L20 launch policy for stochastic top-k/top-p sampling."""

    if top_k <= 1 or top_k > 64:
        raise ValueError("top_k must be in [2, 64]")
    if vocab_size <= 0 or vocab_size > 262_144:
        raise ValueError("vocab_size must be in [1, 262144]")
    if top_k > vocab_size:
        raise ValueError("top_k cannot exceed vocab_size")
    block_vocab = block_vocab_override or (2048 if batch is not None and batch <= 4 else 1024)
    if block_vocab not in {512, 1024, 2048}:
        raise ValueError("block_vocab_override must be one of 512, 1024, 2048")
    blocks_per_row = (vocab_size + block_vocab - 1) // block_vocab
    return SamplingLaunchConfig(
        block_vocab=block_vocab,
        blocks_per_row=blocks_per_row,
        num_warps=4 if block_vocab <= 1024 else 8,
        num_stages=1,
        strategy="two_stage_topk_topp_from_uniform",
    )


def should_use_l20_topk_topp_sampling(
    batch: int,
    vocab_size: int,
    top_k: int,
    top_p: float,
) -> bool:
    """Conservative L20 gate for the first stochastic sampler kernel."""

    if batch <= 0 or vocab_size <= 0:
        return False
    if batch > 64 or vocab_size > 262_144:
        return False
    if top_k <= 1 or top_k > min(64, vocab_size):
        return False
    return 0.0 < top_p <= 1.0


def should_prefer_l20_topk_topp_sampling(
    batch: int,
    vocab_size: int,
    top_k: int,
    top_p: float,
) -> bool:
    """Measured L20 profitability gate against FlashInfer 0.6.12."""

    return (
        should_use_l20_topk_topp_sampling(batch, vocab_size, top_k, top_p)
        and batch <= 4
    )


def should_use_l20_sparse_topk_topp_penalty_sampling(
    batch: int,
    vocab_size: int,
    top_k: int,
    top_p: float,
    max_history: int,
) -> bool:
    """Conservative gate for sparse token-history penalty sampling.

    This path is meant for vLLM serving states where prior token IDs are sparse
    per request. It deliberately rejects long histories in v1 because the
    scatter kernel scans the fixed history window to deduplicate repeated token
    IDs before applying presence/repetition penalties.
    """

    if not should_prefer_l20_topk_topp_sampling(batch, vocab_size, top_k, top_p):
        return False
    return 0 < max_history <= 256


def logprob_topk_launch_config(
    vocab_size: int,
    top_n: int,
    *,
    batch: int | None = None,
    block_vocab_override: Optional[int] = None,
) -> SamplingLaunchConfig:
    """Return the launch policy for fused top-logprobs selection.

    The kernel computes top-N token IDs and normalized logprobs without
    materializing a full ``[batch, vocab]`` log-softmax tensor.
    """

    if top_n <= 0 or top_n > 32:
        raise ValueError("top_n must be in [1, 32]")
    if vocab_size <= 0 or vocab_size > 262_144:
        raise ValueError("vocab_size must be in [1, 262144]")
    if top_n > vocab_size:
        raise ValueError("top_n cannot exceed vocab_size")
    block_vocab = block_vocab_override or (2048 if batch is not None and batch <= 4 else 1024)
    if block_vocab not in {512, 1024, 2048, 4096}:
        raise ValueError("block_vocab_override must be one of 512, 1024, 2048, 4096")
    blocks_per_row = (vocab_size + block_vocab - 1) // block_vocab
    return SamplingLaunchConfig(
        block_vocab=block_vocab,
        blocks_per_row=blocks_per_row,
        num_warps=4 if block_vocab <= 1024 else 8,
        num_stages=1,
        strategy="two_stage_top_logprobs",
    )


def should_use_l20_logprob_topk(batch: int, vocab_size: int, top_n: int) -> bool:
    """Conservative gate for the first fused logprob-selection primitive."""

    if batch <= 0 or vocab_size <= 0:
        return False
    if batch > 64 or vocab_size > 262_144:
        return False
    return 1 <= top_n <= min(32, vocab_size)


if triton is not None:  # pragma: no cover - requires CUDA

    @triton.jit
    def _greedy_sample_kernel(
        logits,
        output,
        BATCH: tl.constexpr,
        VOCAB: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(logits + row * VOCAB + offsets, mask=mask, other=-float("inf"))
        if TEMPERATURE != 1.0:
            values = values / TEMPERATURE
        max_value = tl.max(values, axis=0)
        is_max = values == max_value
        # Tie-break with the smallest token id to match torch.argmax semantics.
        token_values = tl.where(is_max, offsets, BLOCK_VOCAB)
        token = tl.min(token_values, axis=0)
        tl.store(output + row, token)

    @triton.jit
    def _greedy_sample_partial_kernel(
        logits,
        partial_values,
        partial_tokens,
        VOCAB: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(logits + row * VOCAB + offsets, mask=mask, other=-float("inf"))
        if TEMPERATURE != 1.0:
            values = values / TEMPERATURE
        max_value = tl.max(values, axis=0)
        is_max = values == max_value
        token_values = tl.where(is_max, offsets, VOCAB)
        token = tl.min(token_values, axis=0)
        out_offset = row * BLOCKS_PER_ROW + block
        tl.store(partial_values + out_offset, max_value)
        tl.store(partial_tokens + out_offset, token)

    @triton.jit
    def _greedy_sample_reduce_kernel(
        partial_values,
        partial_tokens,
        output,
        VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        REDUCE_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, REDUCE_BLOCK)
        mask = offsets < BLOCKS_PER_ROW
        values = tl.load(
            partial_values + row * BLOCKS_PER_ROW + offsets,
            mask=mask,
            other=-float("inf"),
        )
        tokens = tl.load(
            partial_tokens + row * BLOCKS_PER_ROW + offsets,
            mask=mask,
            other=VOCAB,
        )
        max_value = tl.max(values, axis=0)
        is_max = values == max_value
        token_values = tl.where(is_max, tokens, VOCAB)
        token = tl.min(token_values, axis=0)
        tl.store(output + row, token)

    @triton.jit
    def _topk_topp_partial_kernel(
        logits,
        partial_values,
        partial_tokens,
        VOCAB: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(logits + row * VOCAB + offsets, mask=mask, other=-float("inf"))
        values = values.to(tl.float32)
        if TEMPERATURE != 1.0:
            values = values / TEMPERATURE

        base = (row * BLOCKS_PER_ROW + block) * TOP_K
        for rank in tl.static_range(0, TOP_K):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & mask
            token_values = tl.where(is_max, offsets, VOCAB)
            token = tl.min(token_values, axis=0)
            tl.store(partial_values + base + rank, max_value)
            tl.store(partial_tokens + base + rank, token)
            values = tl.where(offsets == token, -float("inf"), values)

    @triton.jit
    def _topk_topp_penalty_partial_kernel(
        logits,
        token_counts,
        partial_values,
        partial_tokens,
        VOCAB: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        TEMPERATURE: tl.constexpr,
        FREQUENCY_PENALTY: tl.constexpr,
        PRESENCE_PENALTY: tl.constexpr,
        REPETITION_PENALTY: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(logits + row * VOCAB + offsets, mask=mask, other=-float("inf"))
        values = values.to(tl.float32)
        counts = tl.load(token_counts + row * VOCAB + offsets, mask=mask, other=0)
        counts = counts.to(tl.float32)
        present = counts > 0.0

        if REPETITION_PENALTY != 1.0:
            repeated_values = tl.where(
                values < 0.0,
                values * REPETITION_PENALTY,
                values / REPETITION_PENALTY,
            )
            values = tl.where(present, repeated_values, values)
        if FREQUENCY_PENALTY != 0.0:
            values -= counts * FREQUENCY_PENALTY
        if PRESENCE_PENALTY != 0.0:
            values -= tl.where(present, PRESENCE_PENALTY, 0.0)
        if TEMPERATURE != 1.0:
            values = values / TEMPERATURE

        base = (row * BLOCKS_PER_ROW + block) * TOP_K
        for rank in tl.static_range(0, TOP_K):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & mask
            token_values = tl.where(is_max, offsets, VOCAB)
            token = tl.min(token_values, axis=0)
            tl.store(partial_values + base + rank, max_value)
            tl.store(partial_tokens + base + rank, token)
            values = tl.where(offsets == token, -float("inf"), values)

    @triton.jit
    def _top_logprobs_partial_kernel(
        logits,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        VOCAB: tl.constexpr,
        TOP_N: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(logits + row * VOCAB + offsets, mask=mask, other=-float("inf"))
        values = values.to(tl.float32)
        if TEMPERATURE != 1.0:
            values = values / TEMPERATURE

        block_max = tl.max(values, axis=0)
        block_sum = tl.sum(tl.exp(values - block_max), axis=0)
        block_offset = row * BLOCKS_PER_ROW + block
        tl.store(partial_max + block_offset, block_max)
        tl.store(partial_sum_exp + block_offset, block_sum)

        base = block_offset * TOP_N
        for rank in tl.static_range(0, TOP_N):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & mask
            token_values = tl.where(is_max, offsets, VOCAB)
            token = tl.min(token_values, axis=0)
            tl.store(partial_values + base + rank, max_value)
            tl.store(partial_tokens + base + rank, token)
            values = tl.where(offsets == token, -float("inf"), values)

    @triton.jit
    def _top_logprobs_reduce_kernel(
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        output_values,
        output_tokens,
        VOCAB: tl.constexpr,
        TOP_N: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        CANDIDATE_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, CANDIDATE_BLOCK)
        block_mask = offsets < BLOCKS_PER_ROW
        block_maxes = tl.load(
            partial_max + row * BLOCKS_PER_ROW + offsets,
            mask=block_mask,
            other=-float("inf"),
        ).to(tl.float32)
        global_max = tl.max(block_maxes, axis=0)
        block_sums = tl.load(
            partial_sum_exp + row * BLOCKS_PER_ROW + offsets,
            mask=block_mask,
            other=0.0,
        ).to(tl.float32)
        total_exp = tl.sum(tl.exp(block_maxes - global_max) * block_sums, axis=0)
        log_denom = global_max + tl.log(total_exp)

        candidate_count = BLOCKS_PER_ROW * TOP_N
        candidate_mask = offsets < candidate_count
        values = tl.load(
            partial_values + row * candidate_count + offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + offsets,
            mask=candidate_mask,
            other=VOCAB,
        )

        output_base = row * TOP_N
        for rank in tl.static_range(0, TOP_N):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token_values = tl.where(is_max, tokens, VOCAB)
            token = tl.min(token_values, axis=0)
            tl.store(output_values + output_base + rank, max_value - log_denom)
            tl.store(output_tokens + output_base + rank, token)
            values = tl.where(tokens == token, -float("inf"), values)

    @triton.jit
    def _vllm_top_logprobs_partial_kernel(
        logits,
        token_ids,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        partial_ranks,
        VOCAB: tl.constexpr,
        TOP_N: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(logits + row * VOCAB + offsets, mask=mask, other=-float("inf"))
        values = values.to(tl.float32)
        if TEMPERATURE != 1.0:
            values = values / TEMPERATURE

        selected_token = tl.load(token_ids + row).to(tl.int64)
        selected_value = tl.load(logits + row * VOCAB + selected_token).to(tl.float32)
        if TEMPERATURE != 1.0:
            selected_value = selected_value / TEMPERATURE
        rank_count = tl.sum(tl.where((values >= selected_value) & mask, 1, 0), axis=0)

        block_max = tl.max(values, axis=0)
        block_sum = tl.sum(tl.exp(values - block_max), axis=0)
        block_offset = row * BLOCKS_PER_ROW + block
        tl.store(partial_max + block_offset, block_max)
        tl.store(partial_sum_exp + block_offset, block_sum)
        tl.store(partial_ranks + block_offset, rank_count)

        base = block_offset * TOP_N
        for rank in tl.static_range(0, TOP_N):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & mask
            token_values = tl.where(is_max, offsets, VOCAB)
            token = tl.min(token_values, axis=0)
            tl.store(partial_values + base + rank, max_value)
            tl.store(partial_tokens + base + rank, token)
            values = tl.where(offsets == token, -float("inf"), values)

    @triton.jit
    def _vllm_top_logprobs_reduce_kernel(
        logits,
        token_ids,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        partial_ranks,
        output_token_ids,
        output_logprobs,
        output_ranks,
        VOCAB: tl.constexpr,
        TOP_N: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        CANDIDATE_BLOCK: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, CANDIDATE_BLOCK)
        block_mask = offsets < BLOCKS_PER_ROW
        block_maxes = tl.load(
            partial_max + row * BLOCKS_PER_ROW + offsets,
            mask=block_mask,
            other=-float("inf"),
        ).to(tl.float32)
        global_max = tl.max(block_maxes, axis=0)
        block_sums = tl.load(
            partial_sum_exp + row * BLOCKS_PER_ROW + offsets,
            mask=block_mask,
            other=0.0,
        ).to(tl.float32)
        total_exp = tl.sum(tl.exp(block_maxes - global_max) * block_sums, axis=0)
        log_denom = global_max + tl.log(total_exp)

        selected_token = tl.load(token_ids + row).to(tl.int64)
        selected_value = tl.load(logits + row * VOCAB + selected_token).to(tl.float32)
        if TEMPERATURE != 1.0:
            selected_value = selected_value / TEMPERATURE
        rank_parts = tl.load(
            partial_ranks + row * BLOCKS_PER_ROW + offsets,
            mask=block_mask,
            other=0,
        )
        selected_rank = tl.sum(rank_parts, axis=0)

        output_base = row * (TOP_N + 1)
        tl.store(output_token_ids + output_base, selected_token)
        tl.store(output_logprobs + output_base, selected_value - log_denom)
        tl.store(output_ranks + row, selected_rank)

        candidate_count = BLOCKS_PER_ROW * TOP_N
        candidate_mask = offsets < candidate_count
        values = tl.load(
            partial_values + row * candidate_count + offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + offsets,
            mask=candidate_mask,
            other=VOCAB,
        )

        for rank in tl.static_range(0, TOP_N):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token = tl.min(tl.where(is_max, tokens, VOCAB), axis=0)
            tl.store(output_logprobs + output_base + rank + 1, max_value - log_denom)
            tl.store(output_token_ids + output_base + rank + 1, token)
            values = tl.where(tokens == token, -float("inf"), values)

    @triton.jit
    def _copy_logits_to_fp32_kernel(
        logits,
        adjusted_logits,
        TOTAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        block = tl.program_id(0)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        values = tl.load(logits + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(adjusted_logits + offsets, values, mask=mask)

    @triton.jit
    def _sparse_token_penalty_scatter_kernel(
        adjusted_logits,
        history_tokens,
        history_lengths,
        frequency_penalties,
        presence_penalties,
        repetition_penalties,
        VOCAB: tl.constexpr,
        MAX_HISTORY: tl.constexpr,
    ):
        row = tl.program_id(0)
        hist_idx = tl.program_id(1)
        length = tl.load(history_lengths + row)
        active = hist_idx < length
        token = tl.load(
            history_tokens + row * MAX_HISTORY + hist_idx,
            mask=active,
            other=VOCAB,
        )
        valid = active & (token >= 0) & (token < VOCAB)

        seen_before = tl.full((), False, tl.int1)
        count = tl.full((), 0, tl.int32)
        for other_idx in tl.static_range(0, MAX_HISTORY):
            other_active = other_idx < length
            other_token = tl.load(
                history_tokens + row * MAX_HISTORY + other_idx,
                mask=other_active,
                other=-1,
            )
            same_token = other_active & valid & (other_token == token)
            seen_before = seen_before | (same_token & (other_idx < hist_idx))
            count += tl.where(same_token, 1, 0)

        should_write = valid & (~seen_before)
        offset = row * VOCAB + token
        value = tl.load(adjusted_logits + offset, mask=should_write, other=0.0).to(tl.float32)
        frequency_penalty = tl.load(frequency_penalties + row).to(tl.float32)
        presence_penalty = tl.load(presence_penalties + row).to(tl.float32)
        repetition_penalty = tl.load(repetition_penalties + row).to(tl.float32)

        repeated_value = tl.where(
            value < 0.0,
            value * repetition_penalty,
            value / repetition_penalty,
        )
        value = tl.where(repetition_penalty != 1.0, repeated_value, value)
        value -= count.to(tl.float32) * frequency_penalty
        value -= presence_penalty
        tl.store(adjusted_logits + offset, value, mask=should_write)

    @triton.jit
    def _topk_topp_reduce_sample_kernel(
        partial_values,
        partial_tokens,
        uniforms,
        output,
        VOCAB: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        CANDIDATE_BLOCK: tl.constexpr,
        TOP_P: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidate_offsets = tl.arange(0, CANDIDATE_BLOCK)
        candidate_count = BLOCKS_PER_ROW * TOP_K
        candidate_mask = candidate_offsets < candidate_count

        values = tl.load(
            partial_values + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=VOCAB,
        )

        max_for_softmax = tl.full((), -float("inf"), tl.float32)
        total_exp = tl.full((), 0.0, tl.float32)
        for rank in tl.static_range(0, TOP_K):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token_values = tl.where(is_max, tokens, VOCAB)
            token = tl.min(token_values, axis=0)
            if rank == 0:
                max_for_softmax = max_value
            total_exp += tl.exp(max_value - max_for_softmax)
            values = tl.where(tokens == token, -float("inf"), values)

        values = tl.load(
            partial_values + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=VOCAB,
        )
        cumulative_exp = tl.full((), 0.0, tl.float32)
        kept_exp = tl.full((), 0.0, tl.float32)
        for rank in tl.static_range(0, TOP_K):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token_values = tl.where(is_max, tokens, VOCAB)
            token = tl.min(token_values, axis=0)
            weight = tl.exp(max_value - max_for_softmax)
            keep = cumulative_exp / total_exp < TOP_P
            cumulative_exp += weight
            if rank == 0:
                keep = True
            if TOP_P >= 1.0:
                keep = True
            kept_exp += tl.where(keep, weight, 0.0)
            values = tl.where(tokens == token, -float("inf"), values)

        values = tl.load(
            partial_values + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=VOCAB,
        )
        target = tl.load(uniforms + row).to(tl.float32) * kept_exp
        cumulative_exp = tl.full((), 0.0, tl.float32)
        kept_cumulative = tl.full((), 0.0, tl.float32)
        chosen = tl.full((), 0, tl.int32)
        chosen_token = tl.full((), 0, tl.int64)
        first_token = tl.full((), 0, tl.int64)
        for rank in tl.static_range(0, TOP_K):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token_values = tl.where(is_max, tokens, VOCAB)
            token = tl.min(token_values, axis=0)
            if rank == 0:
                first_token = token
            weight = tl.exp(max_value - max_for_softmax)
            keep = cumulative_exp / total_exp < TOP_P
            cumulative_exp += weight
            if rank == 0:
                keep = True
            if TOP_P >= 1.0:
                keep = True
            kept_weight = tl.where(keep, weight, 0.0)
            next_kept = kept_cumulative + kept_weight
            take = (chosen == 0) & keep & (target <= next_kept)
            chosen_token = tl.where(take, token, chosen_token)
            chosen = tl.where(take, 1, chosen)
            kept_cumulative = next_kept
            values = tl.where(tokens == token, -float("inf"), values)
        tl.store(output + row, tl.where(chosen != 0, chosen_token, first_token))

    @triton.jit
    def _topk_topp_reduce_sample_seed_kernel(
        partial_values,
        partial_tokens,
        expanded_idx_mapping,
        seeds,
        positions,
        output,
        VOCAB: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        CANDIDATE_BLOCK: tl.constexpr,
        TOP_P: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidate_offsets = tl.arange(0, CANDIDATE_BLOCK)
        candidate_count = BLOCKS_PER_ROW * TOP_K
        candidate_mask = candidate_offsets < candidate_count

        values = tl.load(
            partial_values + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=VOCAB,
        )

        max_for_softmax = tl.full((), -float("inf"), tl.float32)
        total_exp = tl.full((), 0.0, tl.float32)
        for rank in tl.static_range(0, TOP_K):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token_values = tl.where(is_max, tokens, VOCAB)
            token = tl.min(token_values, axis=0)
            if rank == 0:
                max_for_softmax = max_value
            total_exp += tl.exp(max_value - max_for_softmax)
            values = tl.where(tokens == token, -float("inf"), values)

        values = tl.load(
            partial_values + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=VOCAB,
        )
        cumulative_exp = tl.full((), 0.0, tl.float32)
        kept_exp = tl.full((), 0.0, tl.float32)
        for rank in tl.static_range(0, TOP_K):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token_values = tl.where(is_max, tokens, VOCAB)
            token = tl.min(token_values, axis=0)
            weight = tl.exp(max_value - max_for_softmax)
            keep = cumulative_exp / total_exp < TOP_P
            cumulative_exp += weight
            if rank == 0:
                keep = True
            if TOP_P >= 1.0:
                keep = True
            kept_exp += tl.where(keep, weight, 0.0)
            values = tl.where(tokens == token, -float("inf"), values)

        req_state_idx = tl.load(expanded_idx_mapping + row).to(tl.int64)
        seed = tl.load(seeds + req_state_idx)
        position = tl.load(positions + row)
        sample_seed = tl.randint(seed, position)
        target = tl.rand(sample_seed, row).to(tl.float32) * kept_exp

        values = tl.load(
            partial_values + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        tokens = tl.load(
            partial_tokens + row * candidate_count + candidate_offsets,
            mask=candidate_mask,
            other=VOCAB,
        )
        cumulative_exp = tl.full((), 0.0, tl.float32)
        kept_cumulative = tl.full((), 0.0, tl.float32)
        chosen = tl.full((), 0, tl.int32)
        chosen_token = tl.full((), 0, tl.int64)
        first_token = tl.full((), 0, tl.int64)
        for rank in tl.static_range(0, TOP_K):
            max_value = tl.max(values, axis=0)
            is_max = (values == max_value) & candidate_mask & (tokens < VOCAB)
            token_values = tl.where(is_max, tokens, VOCAB)
            token = tl.min(token_values, axis=0)
            if rank == 0:
                first_token = token
            weight = tl.exp(max_value - max_for_softmax)
            keep = cumulative_exp / total_exp < TOP_P
            cumulative_exp += weight
            if rank == 0:
                keep = True
            if TOP_P >= 1.0:
                keep = True
            kept_weight = tl.where(keep, weight, 0.0)
            next_kept = kept_cumulative + kept_weight
            take = (chosen == 0) & keep & (target <= next_kept)
            chosen_token = tl.where(take, token, chosen_token)
            chosen = tl.where(take, 1, chosen)
            kept_cumulative = next_kept
            values = tl.where(tokens == token, -float("inf"), values)
        tl.store(output + row, tl.where(chosen != 0, chosen_token, first_token))


def greedy_sample(logits, temperature: float = 1.0):
    """Sample greedily on GPU without materializing logits on CPU.

    This implements the deterministic `top_k=1` serving case. Temperature is
    accepted to preserve the sampler contract; for greedy argmax, positive
    scalar temperature does not change the selected token.
    """

    if torch is None or triton is None:
        raise RuntimeError("greedy_sample requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, vocab = logits.shape
    if not should_use_l20_gpu_greedy_sampling(int(batch), int(vocab), top_k=1):
        raise ValueError("shape is outside the L20 greedy sampling gate")
    config = greedy_sampling_launch_config(int(vocab))
    output = torch.empty((batch,), device=logits.device, dtype=torch.int64)
    if config.strategy == "two_stage_block_argmax":
        partial_values = torch.empty(
            (batch, config.blocks_per_row),
            device=logits.device,
            dtype=torch.float32,
        )
        partial_tokens = torch.empty(
            (batch, config.blocks_per_row),
            device=logits.device,
            dtype=torch.int64,
        )
        greedy_sample_out(
            logits,
            output,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            temperature=temperature,
        )
        return output
    greedy_sample_out(logits, output, temperature=temperature)
    return output


def greedy_sample_out(
    logits,
    output,
    *,
    partial_values=None,
    partial_tokens=None,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Write greedy samples into a caller-owned output tensor.

    vLLM-style serving loops can keep `output`, `partial_values`, and
    `partial_tokens` live across decode steps, avoiding allocator noise in the
    hot path. Large-vocab shapes require both partial workspaces.
    """

    if torch is None or triton is None:
        raise RuntimeError("greedy_sample_out requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if output.shape != (logits.shape[0],) or output.dtype != torch.int64:
        raise ValueError("output must have shape [batch] and dtype int64")
    if not logits.is_cuda or not output.is_cuda:
        raise ValueError("logits and output must be CUDA tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, vocab = logits.shape
    if not should_use_l20_gpu_greedy_sampling(int(batch), int(vocab), top_k=1):
        raise ValueError("shape is outside the L20 greedy sampling gate")
    config = greedy_sampling_launch_config(
        int(vocab),
        block_vocab_override=block_vocab_override,
    )
    if config.strategy == "two_stage_block_argmax":
        expected_workspace = (batch, config.blocks_per_row)
        if partial_values is None or partial_values.shape != expected_workspace:
            raise ValueError("partial_values workspace has the wrong shape")
        if partial_tokens is None or partial_tokens.shape != expected_workspace:
            raise ValueError("partial_tokens workspace has the wrong shape")
        if partial_values.dtype != torch.float32 or partial_tokens.dtype != torch.int64:
            raise ValueError("partial workspaces must be float32 and int64")
        if not partial_values.is_cuda or not partial_tokens.is_cuda:
            raise ValueError("partial workspaces must be CUDA tensors")
        _greedy_sample_partial_kernel[(batch, config.blocks_per_row)](
            logits,
            partial_values,
            partial_tokens,
            VOCAB=int(vocab),
            BLOCK_VOCAB=config.block_vocab,
            BLOCKS_PER_ROW=config.blocks_per_row,
            TEMPERATURE=float(temperature),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _greedy_sample_reduce_kernel[(batch,)](
            partial_values,
            partial_tokens,
            output,
            VOCAB=int(vocab),
            BLOCKS_PER_ROW=config.blocks_per_row,
            REDUCE_BLOCK=next_power_of_2(config.blocks_per_row),
            num_warps=1,
            num_stages=1,
        )
        return None
    _greedy_sample_kernel[(batch,)](
        logits,
        output,
        BATCH=int(batch),
        VOCAB=int(vocab),
        BLOCK_VOCAB=config.block_vocab,
        TEMPERATURE=float(temperature),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return None


def greedy_sample_reference(logits, temperature: float = 1.0):
    if torch is None:
        raise RuntimeError("greedy_sample_reference requires PyTorch")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.argmax(logits / temperature, dim=-1).to(torch.int64)


def topk_topp_sample_from_uniform(
    logits,
    uniforms,
    *,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Sample from top-k/top-p logits on GPU using caller-provided uniforms."""

    if torch is None or triton is None:
        raise RuntimeError("topk_topp_sample_from_uniform requires PyTorch and Triton")
    output = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    config = topk_topp_sampling_launch_config(
        int(logits.shape[1]),
        int(top_k),
        batch=int(logits.shape[0]),
        block_vocab_override=block_vocab_override,
    )
    partial_shape = (int(logits.shape[0]), config.blocks_per_row, int(top_k))
    partial_values = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    partial_tokens = torch.empty(partial_shape, device=logits.device, dtype=torch.int64)
    topk_topp_sample_from_uniform_out(
        logits,
        uniforms,
        output,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        block_vocab_override=block_vocab_override,
    )
    return output


def topk_topp_sample_from_uniform_out(
    logits,
    uniforms,
    output,
    *,
    partial_values,
    partial_tokens,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Write top-k/top-p samples into a caller-owned output tensor."""

    if torch is None or triton is None:
        raise RuntimeError("topk_topp_sample_from_uniform_out requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if uniforms.shape != (logits.shape[0],):
        raise ValueError("uniforms must have shape [batch]")
    if output.shape != (logits.shape[0],) or output.dtype != torch.int64:
        raise ValueError("output must have shape [batch] and dtype int64")
    if not logits.is_cuda or not uniforms.is_cuda or not output.is_cuda:
        raise ValueError("logits, uniforms, and output must be CUDA tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, vocab = logits.shape
    if not should_use_l20_topk_topp_sampling(int(batch), int(vocab), int(top_k), float(top_p)):
        raise ValueError("shape or sampling policy is outside the L20 top-k/top-p gate")
    config = topk_topp_sampling_launch_config(
        int(vocab),
        int(top_k),
        batch=int(batch),
        block_vocab_override=block_vocab_override,
    )
    expected_workspace = (batch, config.blocks_per_row, int(top_k))
    if partial_values.shape != expected_workspace:
        raise ValueError("partial_values workspace has the wrong shape")
    if partial_tokens.shape != expected_workspace:
        raise ValueError("partial_tokens workspace has the wrong shape")
    if partial_values.dtype != torch.float32 or partial_tokens.dtype != torch.int64:
        raise ValueError("partial workspaces must be float32 and int64")
    if not partial_values.is_cuda or not partial_tokens.is_cuda:
        raise ValueError("partial workspaces must be CUDA tensors")

    _topk_topp_partial_kernel[(batch, config.blocks_per_row)](
        logits,
        partial_values,
        partial_tokens,
        VOCAB=int(vocab),
        TOP_K=int(top_k),
        BLOCK_VOCAB=config.block_vocab,
        BLOCKS_PER_ROW=config.blocks_per_row,
        TEMPERATURE=float(temperature),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    candidate_count = config.blocks_per_row * int(top_k)
    _topk_topp_reduce_sample_kernel[(batch,)](
        partial_values,
        partial_tokens,
        uniforms,
        output,
        VOCAB=int(vocab),
        TOP_K=int(top_k),
        BLOCKS_PER_ROW=config.blocks_per_row,
        CANDIDATE_BLOCK=next_power_of_2(candidate_count),
        TOP_P=float(top_p),
        num_warps=8,
        num_stages=1,
    )
    return None


def topk_topp_penalty_sample_from_uniform_out(
    logits,
    token_counts,
    uniforms,
    output,
    *,
    partial_values,
    partial_tokens,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Write penalty-adjusted top-k/top-p samples into caller-owned tensors.

    ``token_counts`` is a dense prototype layout: ``[batch, vocab]`` counts of
    prior prompt/output tokens. Production vLLM integration should replace this
    with a sparse token-history layout, but this primitive validates the fused
    arithmetic and sampling semantics.
    """

    if torch is None or triton is None:
        raise RuntimeError("topk_topp_penalty_sample_from_uniform_out requires PyTorch and Triton")
    if logits.ndim != 2 or token_counts.ndim != 2:
        raise ValueError("expected logits and token_counts with shape [batch, vocab]")
    if token_counts.shape != logits.shape:
        raise ValueError("token_counts must match logits shape")
    if uniforms.shape != (logits.shape[0],):
        raise ValueError("uniforms must have shape [batch]")
    if output.shape != (logits.shape[0],) or output.dtype != torch.int64:
        raise ValueError("output must have shape [batch] and dtype int64")
    for name, tensor in (
        ("logits", logits),
        ("token_counts", token_counts),
        ("uniforms", uniforms),
        ("output", output),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    batch, vocab = logits.shape
    if not should_use_l20_topk_topp_sampling(int(batch), int(vocab), int(top_k), float(top_p)):
        raise ValueError("shape or sampling policy is outside the L20 top-k/top-p gate")
    config = topk_topp_sampling_launch_config(
        int(vocab),
        int(top_k),
        batch=int(batch),
        block_vocab_override=block_vocab_override,
    )
    expected_workspace = (batch, config.blocks_per_row, int(top_k))
    if partial_values.shape != expected_workspace:
        raise ValueError("partial_values workspace has the wrong shape")
    if partial_tokens.shape != expected_workspace:
        raise ValueError("partial_tokens workspace has the wrong shape")
    if partial_values.dtype != torch.float32 or partial_tokens.dtype != torch.int64:
        raise ValueError("partial workspaces must be float32 and int64")
    if not partial_values.is_cuda or not partial_tokens.is_cuda:
        raise ValueError("partial workspaces must be CUDA tensors")

    _topk_topp_penalty_partial_kernel[(batch, config.blocks_per_row)](
        logits,
        token_counts,
        partial_values,
        partial_tokens,
        VOCAB=int(vocab),
        TOP_K=int(top_k),
        BLOCK_VOCAB=config.block_vocab,
        BLOCKS_PER_ROW=config.blocks_per_row,
        TEMPERATURE=float(temperature),
        FREQUENCY_PENALTY=float(frequency_penalty),
        PRESENCE_PENALTY=float(presence_penalty),
        REPETITION_PENALTY=float(repetition_penalty),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    candidate_count = config.blocks_per_row * int(top_k)
    _topk_topp_reduce_sample_kernel[(batch,)](
        partial_values,
        partial_tokens,
        uniforms,
        output,
        VOCAB=int(vocab),
        TOP_K=int(top_k),
        BLOCKS_PER_ROW=config.blocks_per_row,
        CANDIDATE_BLOCK=next_power_of_2(candidate_count),
        TOP_P=float(top_p),
        num_warps=8,
        num_stages=1,
    )
    return None


def topk_topp_penalty_sample_from_uniform(
    logits,
    token_counts,
    uniforms,
    *,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Allocate workspaces and run penalty-adjusted top-k/top-p sampling."""

    if torch is None or triton is None:
        raise RuntimeError("topk_topp_penalty_sample_from_uniform requires PyTorch and Triton")
    output = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    config = topk_topp_sampling_launch_config(
        int(logits.shape[1]),
        int(top_k),
        batch=int(logits.shape[0]),
        block_vocab_override=block_vocab_override,
    )
    partial_shape = (int(logits.shape[0]), config.blocks_per_row, int(top_k))
    partial_values = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    partial_tokens = torch.empty(partial_shape, device=logits.device, dtype=torch.int64)
    topk_topp_penalty_sample_from_uniform_out(
        logits,
        token_counts,
        uniforms,
        output,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        block_vocab_override=block_vocab_override,
    )
    return output


def _copy_and_apply_sparse_token_penalties_out(
    logits,
    history_tokens,
    history_lengths,
    adjusted_logits,
    *,
    frequency_penalties,
    presence_penalties,
    repetition_penalties,
):
    if torch is None or triton is None:
        raise RuntimeError("_copy_and_apply_sparse_token_penalties_out requires PyTorch and Triton")
    if logits.ndim != 2 or adjusted_logits.shape != logits.shape:
        raise ValueError("expected logits and adjusted_logits with shape [batch, vocab]")
    if adjusted_logits.dtype != torch.float32:
        raise ValueError("adjusted_logits must be float32")
    if history_tokens.ndim != 2 or history_tokens.shape[0] != logits.shape[0]:
        raise ValueError("history_tokens must have shape [batch, max_history]")
    if history_lengths.shape != (logits.shape[0],):
        raise ValueError("history_lengths must have shape [batch]")
    for name, tensor in (
        ("logits", logits),
        ("history_tokens", history_tokens),
        ("history_lengths", history_lengths),
        ("adjusted_logits", adjusted_logits),
        ("frequency_penalties", frequency_penalties),
        ("presence_penalties", presence_penalties),
        ("repetition_penalties", repetition_penalties),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    batch, vocab = logits.shape
    max_history = int(history_tokens.shape[1])
    if max_history <= 0 or max_history > 256:
        raise ValueError("max_history must be in [1, 256]")
    if frequency_penalties.shape != (batch,):
        raise ValueError("frequency_penalties must have shape [batch]")
    if presence_penalties.shape != (batch,):
        raise ValueError("presence_penalties must have shape [batch]")
    if repetition_penalties.shape != (batch,):
        raise ValueError("repetition_penalties must have shape [batch]")
    total = int(batch) * int(vocab)
    block = 1024
    _copy_logits_to_fp32_kernel[((total + block - 1) // block,)](
        logits,
        adjusted_logits,
        TOTAL=total,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    _sparse_token_penalty_scatter_kernel[(int(batch), max_history)](
        adjusted_logits,
        history_tokens,
        history_lengths,
        frequency_penalties,
        presence_penalties,
        repetition_penalties,
        VOCAB=int(vocab),
        MAX_HISTORY=max_history,
        num_warps=1,
        num_stages=1,
    )
    return None


def topk_topp_sparse_penalty_sample_from_uniform_out(
    logits,
    history_tokens,
    history_lengths,
    uniforms,
    output,
    *,
    adjusted_logits,
    partial_values,
    partial_tokens,
    frequency_penalties,
    presence_penalties,
    repetition_penalties,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Write sparse-history penalty-adjusted top-k/top-p samples.

    ``history_tokens`` is a padded sparse token-history tensor with shape
    ``[batch, max_history]``. Each row contains prior prompt/output token IDs,
    padded with any value outside ``[0, vocab)``. ``history_lengths`` gives the
    active length for each row.
    """

    batch, vocab = logits.shape
    if not should_use_l20_sparse_topk_topp_penalty_sampling(
        int(batch), int(vocab), int(top_k), float(top_p), int(history_tokens.shape[1])
    ):
        raise ValueError("shape, history, or sampling policy is outside the sparse penalty gate")
    _copy_and_apply_sparse_token_penalties_out(
        logits,
        history_tokens,
        history_lengths,
        adjusted_logits,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
    )
    topk_topp_sample_from_uniform_out(
        adjusted_logits,
        uniforms,
        output,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        block_vocab_override=block_vocab_override,
    )
    return None


def topk_topp_sparse_penalty_sample_with_vllm_rng_out(
    logits,
    history_tokens,
    history_lengths,
    output,
    *,
    adjusted_logits,
    partial_values,
    partial_tokens,
    expanded_idx_mapping,
    seeds,
    positions,
    frequency_penalties,
    presence_penalties,
    repetition_penalties,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Sparse-history penalty sampling with vLLM-style RNG tensors."""

    batch, vocab = logits.shape
    if not should_use_l20_sparse_topk_topp_penalty_sampling(
        int(batch), int(vocab), int(top_k), float(top_p), int(history_tokens.shape[1])
    ):
        raise ValueError("shape, history, or sampling policy is outside the sparse penalty gate")
    _copy_and_apply_sparse_token_penalties_out(
        logits,
        history_tokens,
        history_lengths,
        adjusted_logits,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
    )
    topk_topp_sample_with_vllm_rng_out(
        adjusted_logits,
        output,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        expanded_idx_mapping=expanded_idx_mapping,
        seeds=seeds,
        positions=positions,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        block_vocab_override=block_vocab_override,
    )
    return None


def topk_topp_sample_with_vllm_rng_out(
    logits,
    output,
    *,
    partial_values,
    partial_tokens,
    expanded_idx_mapping,
    seeds,
    positions,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Write top-k/top-p samples using vLLM-style seed and position tensors."""

    if torch is None or triton is None:
        raise RuntimeError("topk_topp_sample_with_vllm_rng_out requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if output.shape != (logits.shape[0],) or output.dtype != torch.int64:
        raise ValueError("output must have shape [batch] and dtype int64")
    for name, tensor in (
        ("logits", logits),
        ("output", output),
        ("expanded_idx_mapping", expanded_idx_mapping),
        ("seeds", seeds),
        ("positions", positions),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    if expanded_idx_mapping.shape != (logits.shape[0],):
        raise ValueError("expanded_idx_mapping must have shape [batch]")
    if positions.shape != (logits.shape[0],):
        raise ValueError("positions must have shape [batch]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, vocab = logits.shape
    if not should_use_l20_topk_topp_sampling(int(batch), int(vocab), int(top_k), float(top_p)):
        raise ValueError("shape or sampling policy is outside the L20 top-k/top-p gate")
    config = topk_topp_sampling_launch_config(
        int(vocab),
        int(top_k),
        batch=int(batch),
        block_vocab_override=block_vocab_override,
    )
    expected_workspace = (batch, config.blocks_per_row, int(top_k))
    if partial_values.shape != expected_workspace:
        raise ValueError("partial_values workspace has the wrong shape")
    if partial_tokens.shape != expected_workspace:
        raise ValueError("partial_tokens workspace has the wrong shape")
    if partial_values.dtype != torch.float32 or partial_tokens.dtype != torch.int64:
        raise ValueError("partial workspaces must be float32 and int64")
    if not partial_values.is_cuda or not partial_tokens.is_cuda:
        raise ValueError("partial workspaces must be CUDA tensors")

    _topk_topp_partial_kernel[(batch, config.blocks_per_row)](
        logits,
        partial_values,
        partial_tokens,
        VOCAB=int(vocab),
        TOP_K=int(top_k),
        BLOCK_VOCAB=config.block_vocab,
        BLOCKS_PER_ROW=config.blocks_per_row,
        TEMPERATURE=float(temperature),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    candidate_count = config.blocks_per_row * int(top_k)
    _topk_topp_reduce_sample_seed_kernel[(batch,)](
        partial_values,
        partial_tokens,
        expanded_idx_mapping,
        seeds,
        positions,
        output,
        VOCAB=int(vocab),
        TOP_K=int(top_k),
        BLOCKS_PER_ROW=config.blocks_per_row,
        CANDIDATE_BLOCK=next_power_of_2(candidate_count),
        TOP_P=float(top_p),
        num_warps=8,
        num_stages=1,
    )
    return None


def top_logprobs(
    logits,
    *,
    top_n: int = 5,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Return top-N token logprobs without materializing full log-softmax."""

    if torch is None or triton is None:
        raise RuntimeError("top_logprobs requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    batch, vocab = logits.shape
    config = logprob_topk_launch_config(
        int(vocab),
        int(top_n),
        batch=int(batch),
        block_vocab_override=block_vocab_override,
    )
    output_values = torch.empty((batch, int(top_n)), device=logits.device, dtype=torch.float32)
    output_tokens = torch.empty((batch, int(top_n)), device=logits.device, dtype=torch.int64)
    partial_shape = (int(batch), config.blocks_per_row, int(top_n))
    partial_values = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    partial_tokens = torch.empty(partial_shape, device=logits.device, dtype=torch.int64)
    partial_max = torch.empty((batch, config.blocks_per_row), device=logits.device, dtype=torch.float32)
    partial_sum_exp = torch.empty(
        (batch, config.blocks_per_row), device=logits.device, dtype=torch.float32
    )
    top_logprobs_out(
        logits,
        output_values,
        output_tokens,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        partial_max=partial_max,
        partial_sum_exp=partial_sum_exp,
        top_n=top_n,
        temperature=temperature,
        block_vocab_override=block_vocab_override,
    )
    return output_values, output_tokens


def top_logprobs_out(
    logits,
    output_values,
    output_tokens,
    *,
    partial_values,
    partial_tokens,
    partial_max,
    partial_sum_exp,
    top_n: int = 5,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Write top-N normalized logprobs and token IDs into caller-owned tensors."""

    if torch is None or triton is None:
        raise RuntimeError("top_logprobs_out requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, vocab = logits.shape
    top_n = int(top_n)
    if not should_use_l20_logprob_topk(int(batch), int(vocab), top_n):
        raise ValueError("shape or top_n is outside the logprob top-k gate")
    expected_output = (batch, top_n)
    if output_values.shape != expected_output or output_values.dtype != torch.float32:
        raise ValueError("output_values must have shape [batch, top_n] and dtype float32")
    if output_tokens.shape != expected_output or output_tokens.dtype != torch.int64:
        raise ValueError("output_tokens must have shape [batch, top_n] and dtype int64")
    for name, tensor in (
        ("logits", logits),
        ("output_values", output_values),
        ("output_tokens", output_tokens),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    config = logprob_topk_launch_config(
        int(vocab),
        top_n,
        batch=int(batch),
        block_vocab_override=block_vocab_override,
    )
    expected_partial = (batch, config.blocks_per_row, top_n)
    if partial_values.shape != expected_partial or partial_values.dtype != torch.float32:
        raise ValueError("partial_values workspace has the wrong shape or dtype")
    if partial_tokens.shape != expected_partial or partial_tokens.dtype != torch.int64:
        raise ValueError("partial_tokens workspace has the wrong shape or dtype")
    expected_block = (batch, config.blocks_per_row)
    if partial_max.shape != expected_block or partial_max.dtype != torch.float32:
        raise ValueError("partial_max workspace has the wrong shape or dtype")
    if partial_sum_exp.shape != expected_block or partial_sum_exp.dtype != torch.float32:
        raise ValueError("partial_sum_exp workspace has the wrong shape or dtype")
    for name, tensor in (
        ("partial_values", partial_values),
        ("partial_tokens", partial_tokens),
        ("partial_max", partial_max),
        ("partial_sum_exp", partial_sum_exp),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    _top_logprobs_partial_kernel[(batch, config.blocks_per_row)](
        logits,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        VOCAB=int(vocab),
        TOP_N=top_n,
        BLOCK_VOCAB=config.block_vocab,
        BLOCKS_PER_ROW=config.blocks_per_row,
        TEMPERATURE=float(temperature),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    candidate_count = config.blocks_per_row * top_n
    _top_logprobs_reduce_kernel[(batch,)](
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        output_values,
        output_tokens,
        VOCAB=int(vocab),
        TOP_N=top_n,
        BLOCKS_PER_ROW=config.blocks_per_row,
        CANDIDATE_BLOCK=next_power_of_2(candidate_count),
        num_warps=8,
        num_stages=1,
    )
    return None


def vllm_top_logprobs_out(
    logits,
    token_ids,
    output_token_ids,
    output_logprobs,
    output_ranks,
    *,
    partial_values,
    partial_tokens,
    partial_max,
    partial_sum_exp,
    partial_ranks,
    top_n: int = 5,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Write vLLM-compatible token logprobs without full log-softmax.

    ``output_token_ids`` and ``output_logprobs`` have shape
    ``[batch, top_n + 1]``. Column zero is the sampled token requested by vLLM;
    columns ``1:`` are the top-N token IDs and normalized logprobs. Ranks match
    vLLM's ``batched_count_greater_than`` convention, which counts logits
    greater than or equal to the sampled-token logit.
    """

    if torch is None or triton is None:
        raise RuntimeError("vllm_top_logprobs_out requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("token_ids must have shape [batch]")
    if token_ids.dtype != torch.int64:
        raise ValueError("token_ids must be int64")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, vocab = logits.shape
    top_n = int(top_n)
    if not should_use_l20_logprob_topk(int(batch), int(vocab), top_n):
        raise ValueError("shape or top_n is outside the logprob top-k gate")
    expected_output = (batch, top_n + 1)
    if output_token_ids.shape != expected_output or output_token_ids.dtype != torch.int32:
        raise ValueError(
            "output_token_ids must have shape [batch, top_n + 1] and dtype int32"
        )
    if output_logprobs.shape != expected_output or output_logprobs.dtype != torch.float32:
        raise ValueError(
            "output_logprobs must have shape [batch, top_n + 1] and dtype float32"
        )
    if output_ranks.shape != (batch,) or output_ranks.dtype != torch.int32:
        raise ValueError("output_ranks must have shape [batch] and dtype int32")
    for name, tensor in (
        ("logits", logits),
        ("token_ids", token_ids),
        ("output_token_ids", output_token_ids),
        ("output_logprobs", output_logprobs),
        ("output_ranks", output_ranks),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    config = logprob_topk_launch_config(
        int(vocab),
        top_n,
        batch=int(batch),
        block_vocab_override=block_vocab_override,
    )
    expected_partial = (batch, config.blocks_per_row, top_n)
    if partial_values.shape != expected_partial or partial_values.dtype != torch.float32:
        raise ValueError("partial_values workspace has the wrong shape or dtype")
    if partial_tokens.shape != expected_partial or partial_tokens.dtype != torch.int64:
        raise ValueError("partial_tokens workspace has the wrong shape or dtype")
    expected_block = (batch, config.blocks_per_row)
    if partial_max.shape != expected_block or partial_max.dtype != torch.float32:
        raise ValueError("partial_max workspace has the wrong shape or dtype")
    if partial_sum_exp.shape != expected_block or partial_sum_exp.dtype != torch.float32:
        raise ValueError("partial_sum_exp workspace has the wrong shape or dtype")
    if partial_ranks.shape != expected_block or partial_ranks.dtype != torch.int32:
        raise ValueError("partial_ranks workspace has the wrong shape or dtype")
    for name, tensor in (
        ("partial_values", partial_values),
        ("partial_tokens", partial_tokens),
        ("partial_max", partial_max),
        ("partial_sum_exp", partial_sum_exp),
        ("partial_ranks", partial_ranks),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    _vllm_top_logprobs_partial_kernel[(batch, config.blocks_per_row)](
        logits,
        token_ids,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        partial_ranks,
        VOCAB=int(vocab),
        TOP_N=top_n,
        BLOCK_VOCAB=config.block_vocab,
        BLOCKS_PER_ROW=config.blocks_per_row,
        TEMPERATURE=float(temperature),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    candidate_count = config.blocks_per_row * top_n
    _vllm_top_logprobs_reduce_kernel[(batch,)](
        logits,
        token_ids,
        partial_values,
        partial_tokens,
        partial_max,
        partial_sum_exp,
        partial_ranks,
        output_token_ids,
        output_logprobs,
        output_ranks,
        VOCAB=int(vocab),
        TOP_N=top_n,
        BLOCKS_PER_ROW=config.blocks_per_row,
        CANDIDATE_BLOCK=next_power_of_2(candidate_count),
        TEMPERATURE=float(temperature),
        num_warps=8,
        num_stages=1,
    )
    return None


def topk_topp_sample_from_uniform_reference(
    logits,
    uniforms,
    *,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
):
    """Deterministic PyTorch reference for top-k/top-p sampling from uniforms."""

    if torch is None:
        raise RuntimeError("topk_topp_sample_from_uniform_reference requires PyTorch")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if uniforms.shape != (logits.shape[0],):
        raise ValueError("uniforms must have shape [batch]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values, indices = torch.topk(logits.float() / temperature, k=top_k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Nucleus sampling keeps the first token that crosses ``top_p``. Equivalently,
    # a token is retained when the probability mass before it is still below the
    # threshold.
    keep = (cumulative - probs) < top_p
    keep[:, 0] = True
    filtered = torch.where(keep, probs, torch.zeros_like(probs))
    target = uniforms.float().to(logits.device) * filtered.sum(dim=-1)
    cumulative_kept = torch.cumsum(filtered, dim=-1)
    choice = torch.argmax((cumulative_kept >= target[:, None]).to(torch.int32), dim=-1)
    return torch.gather(indices, dim=-1, index=choice[:, None]).squeeze(-1).to(torch.int64)


def top_logprobs_reference(
    logits,
    *,
    top_n: int = 5,
    temperature: float = 1.0,
):
    """PyTorch reference for top-N normalized token logprobs."""

    if torch is None:
        raise RuntimeError("top_logprobs_reference requires PyTorch")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_n <= 0 or top_n > logits.shape[1]:
        raise ValueError("top_n must be in [1, vocab]")
    return torch.topk(torch.log_softmax(logits.float() / temperature, dim=-1), top_n, dim=-1)


def vllm_top_logprobs_reference(
    logits,
    token_ids,
    *,
    top_n: int = 5,
    temperature: float = 1.0,
):
    """PyTorch reference for vLLM-compatible generated-token logprobs."""

    if torch is None:
        raise RuntimeError("vllm_top_logprobs_reference requires PyTorch")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("token_ids must have shape [batch]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logprobs = torch.log_softmax(logits.float() / temperature, dim=-1)
    top_values, top_tokens = torch.topk(logprobs, top_n, dim=-1)
    selected_tokens = token_ids.to(logits.device, dtype=torch.int64).unsqueeze(-1)
    selected_values = logprobs.gather(-1, selected_tokens)
    selected_ranks = (logprobs >= selected_values).sum(-1).to(torch.int32)
    return (
        torch.cat((selected_tokens.to(torch.int32), top_tokens.to(torch.int32)), dim=1),
        torch.cat((selected_values, top_values), dim=1),
        selected_ranks,
    )


def apply_dense_token_penalties_reference(
    logits,
    token_counts,
    *,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
):
    """PyTorch reference for dense-count token penalties."""

    if torch is None:
        raise RuntimeError("apply_dense_token_penalties_reference requires PyTorch")
    if logits.shape != token_counts.shape:
        raise ValueError("token_counts must match logits shape")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    values = logits.float()
    counts = token_counts.to(values.device).float()
    present = counts > 0
    if repetition_penalty != 1.0:
        repeated = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
        values = torch.where(present, repeated, values)
    if frequency_penalty != 0.0:
        values = values - counts * frequency_penalty
    if presence_penalty != 0.0:
        values = values - present.to(values.dtype) * presence_penalty
    return values


def _row_penalty_value(value, row: int):
    if torch is not None and hasattr(value, "ndim"):
        if value.ndim == 0:
            return float(value.item())
        return float(value[row].item())
    return float(value)


def apply_sparse_token_penalties_reference(
    logits,
    history_tokens,
    history_lengths,
    *,
    frequency_penalties=0.0,
    presence_penalties=0.0,
    repetition_penalties=1.0,
):
    """PyTorch reference for sparse token-history penalties."""

    if torch is None:
        raise RuntimeError("apply_sparse_token_penalties_reference requires PyTorch")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if history_tokens.ndim != 2 or history_tokens.shape[0] != logits.shape[0]:
        raise ValueError("history_tokens must have shape [batch, max_history]")
    if history_lengths.shape != (logits.shape[0],):
        raise ValueError("history_lengths must have shape [batch]")
    values = logits.float().clone()
    batch, vocab = values.shape
    for row in range(int(batch)):
        length = max(0, min(int(history_lengths[row].item()), int(history_tokens.shape[1])))
        if length == 0:
            continue
        tokens = history_tokens[row, :length].to(device=values.device, dtype=torch.long)
        tokens = tokens[(tokens >= 0) & (tokens < vocab)]
        if tokens.numel() == 0:
            continue
        unique_tokens, counts = torch.unique(tokens, sorted=False, return_counts=True)
        freq = _row_penalty_value(frequency_penalties, row)
        pres = _row_penalty_value(presence_penalties, row)
        rep = _row_penalty_value(repetition_penalties, row)
        if rep <= 0:
            raise ValueError("repetition_penalties must be positive")
        row_values = values[row, unique_tokens]
        if rep != 1.0:
            row_values = torch.where(row_values < 0, row_values * rep, row_values / rep)
        if freq != 0.0:
            row_values = row_values - counts.to(row_values.dtype) * freq
        if pres != 0.0:
            row_values = row_values - pres
        values[row, unique_tokens] = row_values
    return values


def topk_topp_penalty_sample_from_uniform_reference(
    logits,
    token_counts,
    uniforms,
    *,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
):
    """Reference for fused penalties plus top-k/top-p sampling."""

    adjusted = apply_dense_token_penalties_reference(
        logits,
        token_counts,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    return topk_topp_sample_from_uniform_reference(
        adjusted,
        uniforms,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )


def topk_topp_sparse_penalty_sample_from_uniform_reference(
    logits,
    history_tokens,
    history_lengths,
    uniforms,
    *,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
    frequency_penalties=0.0,
    presence_penalties=0.0,
    repetition_penalties=1.0,
):
    """Reference for sparse-history penalties plus top-k/top-p sampling."""

    adjusted = apply_sparse_token_penalties_reference(
        logits,
        history_tokens,
        history_lengths,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
    )
    return topk_topp_sample_from_uniform_reference(
        adjusted,
        uniforms,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
