import unittest

from l20_stack.operators import (
    OperatorShape,
    OperatorTarget,
    plan_operator,
    rope_kv_minimum_bytes,
)
from l20_stack.ops.triton_rope_kv import paged_rope_kv_reference, rope_kv_launch_config
from l20_stack.ops.triton_rope_kv import paged_rope_kv_launch_heads, paged_rope_kv_launch_warps


class RopeKvPlanTest(unittest.TestCase):
    def test_rope_kv_launch_targets_l20_decode(self):
        config = rope_kv_launch_config(128)

        self.assertEqual(config.sm_target, "sm_89")
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.num_warps, 4)

    def test_paged_rope_kv_launch_uses_l20_warp_policy(self):
        self.assertEqual(paged_rope_kv_launch_warps(128, 128), 1)
        self.assertEqual(paged_rope_kv_launch_warps(2048, 128), 1)
        self.assertEqual(paged_rope_kv_launch_warps(4096, 128), 2)
        self.assertEqual(paged_rope_kv_launch_warps(2048, 128, 4), 4)
        self.assertEqual(paged_rope_kv_launch_warps(512, 256), 4)

    def test_paged_rope_kv_launch_groups_heads_only_above_measured_boundary(self):
        self.assertEqual(paged_rope_kv_launch_heads(512, 8, 128), 1)
        self.assertEqual(paged_rope_kv_launch_heads(768, 8, 128), 4)
        self.assertEqual(paged_rope_kv_launch_heads(4096, 8, 128), 4)
        self.assertEqual(paged_rope_kv_launch_heads(4096, 6, 128), 1)
        self.assertEqual(paged_rope_kv_launch_heads(4096, 8, 256), 1)

    def test_rope_kv_fusion_reduces_minimum_traffic(self):
        shape = OperatorShape(rows=32 * 8, hidden_size=128, dtype_bytes=2)
        fused = rope_kv_minimum_bytes(shape, fused=True)
        unfused = rope_kv_minimum_bytes(shape, fused=False)
        plan = plan_operator(OperatorTarget(name="rope_kv_cache_write", shape=shape))

        self.assertLess(fused, unfused)
        self.assertEqual(plan.priority, 2)
        self.assertEqual(plan.roofline_class, "memory_bound")
        self.assertAlmostEqual(plan.launch["minimum_traffic_reduction_pct"], 33.33, places=2)

    def test_invalid_rotary_dim_is_rejected(self):
        with self.assertRaises(ValueError):
            rope_kv_launch_config(128, rotary_dim=127)

    def test_grouped_paged_kernel_source_is_available(self):
        import l20_stack.ops.triton_rope_kv as rope_kv

        if rope_kv.triton is None:
            self.skipTest("triton is optional")
        self.assertTrue(hasattr(rope_kv, "_paged_rope_kv_cache_write_grouped_kernel"))

    def test_paged_reference_uses_block_table(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional")
        k = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        v = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
        cos = torch.ones(2, 1)
        sin = torch.zeros(2, 1)
        sequence_ids = torch.tensor([0, 1])
        positions = torch.tensor([0, 0])
        block_table = torch.tensor([[1], [0]])
        k_cache = torch.zeros(2, 2, 1, 2)
        v_cache = torch.zeros_like(k_cache)
        paged_rope_kv_reference(
            k, v, cos, sin, sequence_ids, positions, block_table, k_cache, v_cache
        )
        self.assertTrue(torch.equal(k_cache[1, 0], k[0]))
        self.assertTrue(torch.equal(k_cache[0, 0], k[1]))
        self.assertTrue(torch.equal(v_cache[1, 0], v[0]))


if __name__ == "__main__":
    unittest.main()
