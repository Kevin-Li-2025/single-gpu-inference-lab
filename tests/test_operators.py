import unittest

from l20_stack.operators import OperatorShape, OperatorTarget, plan_operator
from l20_stack.ops.triton_rmsnorm import next_power_of_2, rmsnorm_launch_config


class OperatorPlanTest(unittest.TestCase):
    def test_rmsnorm_plan_targets_sm89(self):
        plan = plan_operator(
            OperatorTarget(
                name="rmsnorm",
                shape=OperatorShape(rows=4096, hidden_size=4096, dtype_bytes=2),
            )
        )

        self.assertEqual(plan.roofline_class, "memory_bound")
        self.assertEqual(plan.priority, 1)
        self.assertEqual(plan.launch["sm_target"], "sm_89")
        self.assertEqual(plan.launch["num_warps"], 8)

    def test_launch_config_bounds(self):
        self.assertEqual(next_power_of_2(4097), 8192)
        config = rmsnorm_launch_config(8192)
        self.assertEqual(config.block_size, 8192)
        self.assertEqual(config.num_warps, 8)

    def test_oversized_single_pass_rmsnorm_is_rejected(self):
        with self.assertRaises(ValueError):
            rmsnorm_launch_config(32768)


if __name__ == "__main__":
    unittest.main()
