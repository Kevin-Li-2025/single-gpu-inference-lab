import importlib.util
import unittest
from pathlib import Path


def load_profile_script():
    path = Path("scripts/profile_vllm_l20_rope_kv.py")
    spec = importlib.util.spec_from_file_location("profile_vllm_l20_rope_kv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ncu_summary_script():
    path = Path("scripts/summarize_ncu_profile.py")
    spec = importlib.util.spec_from_file_location("summarize_ncu_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class L20KernelProfileTest(unittest.TestCase):
    def test_register_limited_occupancy(self):
        module = load_profile_script()
        result = module.theoretical_occupancy(num_warps=4, num_regs=128, shared_bytes=0)
        self.assertEqual(result["limiting_resource"], "register_blocks")
        self.assertEqual(result["resident_blocks_per_sm"], 4)
        self.assertAlmostEqual(result["theoretical_occupancy_pct"], 33.33)

    def test_measured_resource_shape_is_warp_limited(self):
        module = load_profile_script()
        result = module.theoretical_occupancy(num_warps=4, num_regs=24, shared_bytes=0)
        self.assertEqual(result["limiting_resource"], "warp_blocks")
        self.assertEqual(result["resident_warps_per_sm"], 48)

    def test_ncu_summary_extracts_roofline_metrics(self):
        module = load_ncu_summary_script()
        rows = module.read_ncu_csv(Path("tests/fixtures/ncu_raw_sample.csv"))
        summary = module.summarize_kernel("_l20_test_kernel", rows["_l20_test_kernel"])
        self.assertEqual(summary["kernel_name"], "_l20_test_kernel")
        self.assertAlmostEqual(summary["achieved_memory_bandwidth_gbps"], 864.0)
        self.assertAlmostEqual(summary["arithmetic_intensity_flops_per_byte"], 1.0)
        self.assertEqual(summary["roofline_class"], "memory_bound")
        self.assertEqual(summary["memory_bandwidth_utilization_pct"], 100.0)
        self.assertEqual(summary["active_warps_pct"], 65.0)
        self.assertEqual(summary["stall_long_scoreboard_pct"], 31.0)
        self.assertAlmostEqual(summary["sector_excess_ratio_l1_over_l2"], 2.0)

    def test_profile_kernel_wrapper_exports_dashboard_artifacts(self):
        source = Path("scripts/profile_kernel.sh").read_text()
        self.assertIn("--section SpeedOfLight", source)
        self.assertIn("--section MemoryWorkloadAnalysis", source)
        self.assertIn("--section WarpStateStats", source)
        self.assertIn("scripts/summarize_ncu_profile.py", source)
        self.assertIn("--markdown-output", source)
        rope_source = Path("scripts/profile_vllm_l20_rope_kv_ncu.sh").read_text()
        self.assertIn("scripts/profile_kernel.sh", rope_source)
        self.assertIn("regex:_l20_.*rope_kv_kernel", rope_source)


if __name__ == "__main__":
    unittest.main()
