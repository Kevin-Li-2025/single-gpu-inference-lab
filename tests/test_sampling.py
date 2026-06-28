import importlib.util
import unittest

from l20_stack.operators import OperatorShape, OperatorTarget, plan_operator
from l20_stack.ops.triton_sampling import (
    greedy_sampling_launch_config,
    should_prefer_l20_topk_topp_sampling,
    should_use_l20_gpu_greedy_sampling,
    should_use_l20_topk_topp_sampling,
    topk_topp_sampling_launch_config,
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


if __name__ == "__main__":
    unittest.main()
