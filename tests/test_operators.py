import unittest

from l20_stack.operators import (
    OperatorShape,
    OperatorTarget,
    plan_operator,
    residual_rmsnorm_minimum_bytes,
)
from l20_stack.ops.triton_rmsnorm import (
    next_power_of_2,
    residual_rmsnorm_backend,
    residual_rmsnorm_launch_config,
    rmsnorm_launch_config,
    rmsnorm_warp_candidates,
)


class OperatorPlanTest(unittest.TestCase):
    def test_rmsnorm_plan_targets_sm89(self):
        plan = plan_operator(
            OperatorTarget(
                name="rmsnorm",
                shape=OperatorShape(rows=4096, hidden_size=4096, dtype_bytes=2),
            )
        )

        self.assertEqual(plan.roofline_class, "memory_bound")
        self.assertEqual(plan.priority, 2)
        self.assertEqual(plan.launch["sm_target"], "sm_89")
        self.assertEqual(plan.launch["num_warps"], 4)

    def test_launch_config_bounds(self):
        self.assertEqual(next_power_of_2(4097), 8192)
        config = rmsnorm_launch_config(8192)
        self.assertEqual(config.block_size, 8192)
        self.assertEqual(config.num_warps, 8)
        self.assertEqual(rmsnorm_warp_candidates(8192), (4, 8))
        self.assertEqual(rmsnorm_launch_config(4096).num_warps, 4)
        self.assertEqual(residual_rmsnorm_launch_config(8192).num_warps, 4)

    def test_fused_residual_rmsnorm_reduces_minimum_traffic(self):
        shape = OperatorShape(rows=4096, hidden_size=4096, dtype_bytes=2)
        fused = residual_rmsnorm_minimum_bytes(shape, fused=True)
        unfused = residual_rmsnorm_minimum_bytes(shape, fused=False)
        plan = plan_operator(OperatorTarget(name="residual_rmsnorm", shape=shape))

        self.assertLess(fused, unfused)
        self.assertEqual(plan.priority, 1)
        self.assertEqual(plan.roofline_class, "memory_bound")
        self.assertAlmostEqual(plan.launch["minimum_traffic_reduction_pct"], 20.0, places=2)

    def test_l20_residual_rmsnorm_dispatch_uses_measured_crossover(self):
        self.assertEqual(residual_rmsnorm_backend(4096, 4096), "torch_eager")
        self.assertEqual(residual_rmsnorm_backend(1, 4096), "triton")
        self.assertEqual(residual_rmsnorm_backend(4096, 8192), "triton")
        self.assertEqual(residual_rmsnorm_backend(128, 5120, True), "flashinfer")
        self.assertEqual(residual_rmsnorm_backend(512, 4096, True), "flashinfer")
        self.assertEqual(residual_rmsnorm_backend(4096, 6144, True), "flashinfer")
        self.assertEqual(residual_rmsnorm_backend(32, 4096, True), "triton")

    def test_oversized_single_pass_rmsnorm_is_rejected(self):
        with self.assertRaises(ValueError):
            rmsnorm_launch_config(32768)


if __name__ == "__main__":
    unittest.main()
