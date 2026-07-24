import importlib.util
import unittest
from pathlib import Path

from l20_stack.operators import OperatorShape, OperatorTarget, plan_operator
from l20_stack.ops.triton_sampling import (
    apply_dense_token_penalties_reference,
    apply_sparse_token_penalties_reference,
    greedy_sampling_launch_config,
    logprob_topk_launch_config,
    should_prefer_l20_topk_topp_sampling,
    should_use_l20_gpu_greedy_sampling,
    should_use_l20_logprob_topk,
    should_use_l20_sparse_topk_topp_penalty_sampling,
    should_use_l20_topk_topp_sampling,
    top_logprobs_reference,
    topk_topp_penalty_sample_from_uniform_reference,
    topk_topp_sample_from_uniform_reference,
    topk_topp_sampling_launch_config,
    topk_topp_sparse_penalty_sample_from_uniform_reference,
    vllm_top_logprobs_reference,
)


class L20SamplingTest(unittest.TestCase):
    def test_qwen_vocab_uses_single_cta_sampling_policy(self):
        config = greedy_sampling_launch_config(151_936)
        self.assertEqual(config.block_vocab, 1024)
        self.assertEqual(config.blocks_per_row, 149)
        self.assertEqual(config.num_warps, 4)
        self.assertEqual(config.num_stages, 1)
        self.assertEqual(config.strategy, "two_stage_block_argmax")

    def test_small_vocab_keeps_single_cta_sampling_policy(self):
        config = greedy_sampling_launch_config(32_000)
        self.assertEqual(config.block_vocab, 32_768)
        self.assertEqual(config.blocks_per_row, 1)
        self.assertEqual(config.strategy, "single_cta_argmax")

    def test_large_vocab_block_override_changes_cta_count(self):
        config = greedy_sampling_launch_config(151_936, block_vocab_override=4096)
        self.assertEqual(config.block_vocab, 4096)
        self.assertEqual(config.blocks_per_row, 38)
        self.assertEqual(config.num_warps, 8)

    def test_sampling_gate_is_conservative(self):
        self.assertTrue(should_use_l20_gpu_greedy_sampling(1, 151_936, top_k=1))
        self.assertTrue(should_use_l20_gpu_greedy_sampling(64, 151_936, top_k=1))
        self.assertFalse(should_use_l20_gpu_greedy_sampling(65, 151_936, top_k=1))
        self.assertFalse(should_use_l20_gpu_greedy_sampling(1, 151_936, top_k=50))
        self.assertFalse(should_use_l20_gpu_greedy_sampling(1, 300_000, top_k=1))

    def test_topk_topp_policy_uses_two_stage_candidate_reduction(self):
        config = topk_topp_sampling_launch_config(151_936, top_k=50, batch=1)
        self.assertEqual(config.block_vocab, 2048)
        self.assertEqual(config.blocks_per_row, 75)
        self.assertEqual(config.num_warps, 8)
        self.assertEqual(config.num_stages, 1)
        self.assertEqual(config.strategy, "two_stage_topk_topp_from_uniform")

    def test_topk_topp_policy_keeps_1024_tiles_for_batched_decode(self):
        config = topk_topp_sampling_launch_config(151_936, top_k=50, batch=16)
        self.assertEqual(config.block_vocab, 1024)
        self.assertEqual(config.blocks_per_row, 149)
        self.assertEqual(config.num_warps, 4)
        self.assertEqual(config.num_stages, 1)
        self.assertEqual(config.strategy, "two_stage_topk_topp_from_uniform")

    def test_topk_topp_policy_uses_2048_tiles_through_batch_four(self):
        config = topk_topp_sampling_launch_config(151_936, top_k=50, batch=4)
        self.assertEqual(config.block_vocab, 2048)
        self.assertEqual(config.blocks_per_row, 75)

    def test_topk_topp_policy_keeps_l20_gate_narrow(self):
        self.assertTrue(should_use_l20_topk_topp_sampling(1, 151_936, 50, 0.9))
        self.assertTrue(should_use_l20_topk_topp_sampling(64, 151_936, 64, 1.0))
        self.assertFalse(should_use_l20_topk_topp_sampling(65, 151_936, 50, 0.9))
        self.assertFalse(should_use_l20_topk_topp_sampling(1, 151_936, 1, 0.9))
        self.assertFalse(should_use_l20_topk_topp_sampling(1, 151_936, 65, 0.9))
        self.assertFalse(should_use_l20_topk_topp_sampling(1, 300_000, 50, 0.9))
        self.assertFalse(should_use_l20_topk_topp_sampling(1, 151_936, 50, 0.0))
        self.assertFalse(should_use_l20_topk_topp_sampling(1, 151_936, 50, 1.1))

    def test_topk_topp_profitability_gate_matches_l20_flashinfer_matrix(self):
        self.assertTrue(should_prefer_l20_topk_topp_sampling(1, 151_936, 50, 0.9))
        self.assertTrue(should_prefer_l20_topk_topp_sampling(4, 151_936, 50, 0.9))
        self.assertFalse(should_prefer_l20_topk_topp_sampling(8, 151_936, 50, 0.9))
        self.assertFalse(should_prefer_l20_topk_topp_sampling(16, 151_936, 50, 0.9))

    def test_top_p_reference_keeps_threshold_crossing_token(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        probabilities = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32)
        logits = torch.log(probabilities)

        sampled = topk_topp_sample_from_uniform_reference(
            logits,
            torch.tensor([0.8]),
            top_k=3,
            top_p=0.8,
        )

        self.assertEqual(int(sampled.item()), 1)

    def test_triton_top_p_kernels_use_pre_token_cumulative_mass(self):
        spec = importlib.util.find_spec("l20_stack.ops.triton_sampling")
        self.assertIsNotNone(spec)
        source = Path(spec.origin).read_text(encoding="utf-8")

        self.assertNotIn("cumulative_exp / total_exp <= TOP_P", source)
        self.assertEqual(source.count("keep = cumulative_exp / total_exp < TOP_P"), 4)

    def test_sparse_penalty_gate_keeps_history_window_bounded(self):
        self.assertTrue(
            should_use_l20_sparse_topk_topp_penalty_sampling(1, 151_936, 50, 0.9, 128)
        )
        self.assertFalse(
            should_use_l20_sparse_topk_topp_penalty_sampling(1, 151_936, 50, 0.9, 0)
        )
        self.assertFalse(
            should_use_l20_sparse_topk_topp_penalty_sampling(1, 151_936, 50, 0.9, 512)
        )
        self.assertFalse(
            should_use_l20_sparse_topk_topp_penalty_sampling(8, 151_936, 50, 0.9, 128)
        )

    def test_logprob_topk_policy_targets_qwen_logprobs(self):
        config = logprob_topk_launch_config(151_936, top_n=5, batch=1)
        self.assertEqual(config.block_vocab, 2048)
        self.assertEqual(config.blocks_per_row, 75)
        self.assertEqual(config.num_warps, 8)
        self.assertEqual(config.strategy, "two_stage_top_logprobs")
        self.assertTrue(should_use_l20_logprob_topk(1, 151_936, 5))
        self.assertTrue(should_use_l20_logprob_topk(64, 151_936, 32))
        self.assertFalse(should_use_l20_logprob_topk(65, 151_936, 5))
        self.assertFalse(should_use_l20_logprob_topk(1, 151_936, 33))

    def test_operator_planner_prioritizes_gpu_sampling(self):
        plan = plan_operator(
            OperatorTarget(
                name="gpu_sampling",
                shape=OperatorShape(rows=1, hidden_size=151_936, dtype_bytes=2),
            )
        )
        self.assertEqual(plan.priority, 1)
        self.assertEqual(plan.roofline_class, "memory_bound")
        self.assertEqual(plan.launch["block_vocab"], 1024)
        self.assertEqual(plan.launch["blocks_per_row"], 149)
        self.assertEqual(plan.launch["supported_top_k"], 1)
        self.assertEqual(plan.launch["target"], "avoid_cpu_gpu_logits_roundtrip")

    def test_operator_planner_tracks_topk_topp_sampling(self):
        plan = plan_operator(
            OperatorTarget(
                name="gpu_topk_topp_sampling",
                shape=OperatorShape(rows=1, hidden_size=151_936, dtype_bytes=2),
            )
        )
        self.assertEqual(plan.priority, 1)
        self.assertEqual(plan.launch["block_vocab"], 2048)
        self.assertEqual(plan.launch["blocks_per_row"], 75)
        self.assertEqual(plan.launch["default_top_k"], 50)
        self.assertEqual(plan.launch["default_top_p"], 0.9)
        self.assertEqual(plan.launch["supported_top_k_max"], 64)
        self.assertEqual(
            plan.launch["target"],
            "fuse_topk_topp_multinomial_without_cpu_roundtrip",
        )

    def test_cuda_kernel_source_is_available(self):
        spec = importlib.util.find_spec("l20_stack.ops.triton_sampling")
        self.assertIsNotNone(spec)

    def test_vllm_rng_sampler_entrypoint_is_available(self):
        spec = importlib.util.find_spec("l20_stack.ops.triton_sampling")
        self.assertIsNotNone(spec)
        source = Path(spec.origin).read_text(encoding="utf-8")
        self.assertIn("topk_topp_sample_with_vllm_rng_out", source)
        self.assertIn("_topk_topp_reduce_sample_seed_kernel", source)
        self.assertIn("tl.randint(seed, position)", source)

    def test_penalty_fused_sampler_entrypoint_is_available(self):
        spec = importlib.util.find_spec("l20_stack.ops.triton_sampling")
        self.assertIsNotNone(spec)
        source = Path(spec.origin).read_text(encoding="utf-8")
        self.assertIn("topk_topp_penalty_sample_from_uniform_out", source)
        self.assertIn("_topk_topp_penalty_partial_kernel", source)
        self.assertIn("REPETITION_PENALTY", source)
        self.assertIn("FREQUENCY_PENALTY", source)
        self.assertIn("PRESENCE_PENALTY", source)

    def test_sparse_penalty_sampler_entrypoint_is_available(self):
        spec = importlib.util.find_spec("l20_stack.ops.triton_sampling")
        self.assertIsNotNone(spec)
        source = Path(spec.origin).read_text(encoding="utf-8")
        self.assertIn("topk_topp_sparse_penalty_sample_with_vllm_rng_out", source)
        self.assertIn("_sparse_token_penalty_scatter_kernel", source)
        self.assertIn("history_tokens", source)
        self.assertIn("history_lengths", source)

    def test_top_logprobs_entrypoint_is_available(self):
        spec = importlib.util.find_spec("l20_stack.ops.triton_sampling")
        self.assertIsNotNone(spec)
        source = Path(spec.origin).read_text(encoding="utf-8")
        self.assertIn("top_logprobs_out", source)
        self.assertIn("vllm_top_logprobs_out", source)
        self.assertIn("_top_logprobs_partial_kernel", source)
        self.assertIn("_top_logprobs_reduce_kernel", source)
        self.assertIn("_vllm_top_logprobs_partial_kernel", source)
        self.assertIn("_vllm_top_logprobs_reduce_kernel", source)
        self.assertIn("partial_sum_exp", source)

    def test_top_logprobs_reference_returns_normalized_values(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        logits = torch.tensor([[1.0, 3.0, -1.0, 2.0]])

        values, tokens = top_logprobs_reference(logits, top_n=2, temperature=1.0)

        expected = torch.topk(torch.log_softmax(logits.float(), dim=-1), 2, dim=-1)
        self.assertTrue(torch.equal(tokens, expected.indices))
        self.assertTrue(torch.allclose(values, expected.values))

    def test_vllm_top_logprobs_reference_includes_selected_token_and_rank(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        logits = torch.tensor([[1.0, 3.0, -1.0, 2.0], [0.0, 0.5, 4.0, -1.0]])
        sampled = torch.tensor([0, 2], dtype=torch.int64)

        token_ids, logprobs, ranks = vllm_top_logprobs_reference(
            logits,
            sampled,
            top_n=2,
            temperature=1.0,
        )

        expected_logprobs = torch.log_softmax(logits.float(), dim=-1)
        self.assertTrue(torch.equal(token_ids[:, 0], sampled.to(torch.int32)))
        self.assertTrue(torch.allclose(logprobs[:, 0], expected_logprobs.gather(1, sampled[:, None])[:, 0]))
        self.assertEqual(ranks.tolist(), [3, 1])
        self.assertEqual(token_ids.shape, (2, 3))

    def test_dense_penalty_reference_matches_repetition_frequency_presence(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        logits = torch.tensor([[2.0, -2.0, 1.0, 0.5]])
        counts = torch.tensor([[2, 1, 0, 3]])

        adjusted = apply_dense_token_penalties_reference(
            logits,
            counts,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            repetition_penalty=2.0,
        )

        expected = torch.tensor([[0.6, -4.3, 1.0, -0.25]])
        self.assertTrue(torch.allclose(adjusted.cpu(), expected))

    def test_penalty_reference_can_change_sampled_token(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        logits = torch.tensor([[2.0, 1.9, 0.0, -1.0]])
        counts = torch.tensor([[3, 0, 0, 0]])
        uniforms = torch.tensor([0.01])

        sampled = topk_topp_penalty_sample_from_uniform_reference(
            logits,
            counts,
            uniforms,
            top_k=2,
            top_p=1.0,
            temperature=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            repetition_penalty=2.0,
        )

        self.assertEqual(int(sampled.item()), 1)

    def test_sparse_penalty_reference_matches_dense_counts(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        logits = torch.tensor([[2.0, -2.0, 1.0, 0.5], [0.5, 1.5, -1.0, 2.0]])
        history_tokens = torch.tensor([[0, 0, 1, 3, 99], [2, 2, 2, 0, -1]])
        history_lengths = torch.tensor([4, 4])
        dense_counts = torch.tensor([[2, 1, 0, 1], [1, 0, 3, 0]])
        frequency = torch.tensor([0.1, 0.2])
        presence = torch.tensor([0.2, 0.3])
        repetition = torch.tensor([2.0, 1.5])

        dense = apply_dense_token_penalties_reference(
            logits,
            dense_counts,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            repetition_penalty=2.0,
        )
        sparse_row0 = apply_sparse_token_penalties_reference(
            logits[:1],
            history_tokens[:1],
            history_lengths[:1],
            frequency_penalties=frequency[:1],
            presence_penalties=presence[:1],
            repetition_penalties=repetition[:1],
        )
        sparse = apply_sparse_token_penalties_reference(
            logits,
            history_tokens,
            history_lengths,
            frequency_penalties=frequency,
            presence_penalties=presence,
            repetition_penalties=repetition,
        )

        self.assertTrue(torch.allclose(sparse_row0, dense[:1]))
        expected_row1 = torch.tensor([[(0.5 / 1.5) - 0.5, 1.5, (-1.0 * 1.5) - 0.9, 2.0]])
        self.assertTrue(torch.allclose(sparse[1:], expected_row1))

    def test_sparse_penalty_reference_can_change_sampled_token(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        logits = torch.tensor([[2.0, 1.9, 0.0, -1.0]])
        history_tokens = torch.tensor([[0, 0, 0, 4]])
        history_lengths = torch.tensor([3])
        uniforms = torch.tensor([0.01])

        sampled = topk_topp_sparse_penalty_sample_from_uniform_reference(
            logits,
            history_tokens,
            history_lengths,
            uniforms,
            top_k=2,
            top_p=1.0,
            temperature=1.0,
            frequency_penalties=torch.tensor([0.2]),
            presence_penalties=torch.tensor([0.1]),
            repetition_penalties=torch.tensor([2.0]),
        )

        self.assertEqual(int(sampled.item()), 1)


if __name__ == "__main__":
    unittest.main()
