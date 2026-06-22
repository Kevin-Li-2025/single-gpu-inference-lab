import importlib.util
import unittest
from pathlib import Path


def load_profile_script():
    path = Path("scripts/profile_vllm_l20_rope_kv.py")
    spec = importlib.util.spec_from_file_location("profile_vllm_l20_rope_kv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class L20KernelProfileTest(unittest.TestCase):
    def test_register_limited_occupancy(self):
        module = load_profile_script()
        result = module.theoretical_occupancy(
            num_warps=4, num_regs=128, shared_bytes=0
        )
        self.assertEqual(result["limiting_resource"], "register_blocks")
        self.assertEqual(result["resident_blocks_per_sm"], 4)
        self.assertAlmostEqual(result["theoretical_occupancy_pct"], 33.33)

    def test_measured_resource_shape_is_warp_limited(self):
        module = load_profile_script()
        result = module.theoretical_occupancy(
            num_warps=4, num_regs=24, shared_bytes=0
        )
        self.assertEqual(result["limiting_resource"], "warp_blocks")
        self.assertEqual(result["resident_warps_per_sm"], 48)


if __name__ == "__main__":
    unittest.main()
