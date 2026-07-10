import json
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class M4Q4MatvecCppTest(unittest.TestCase):
    def test_kernel_compiles_and_matches_scalar(self):
        if platform.machine() != "arm64":
            self.skipTest("M4 kernel requires arm64")
        cxx = shutil.which("clang++")
        if cxx is None:
            self.skipTest("clang++ is unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            binary = Path(tmpdir) / "m4_q4_matvec"
            subprocess.check_call(
                [
                    cxx,
                    "-O3",
                    "-std=c++20",
                    "-mcpu=apple-m4",
                    "cpp/m4_q4_matvec.cpp",
                    "-o",
                    str(binary),
                ]
            )
            output = subprocess.check_output(
                [
                    str(binary),
                    "--rows",
                    "96",
                    "--cols",
                    "128",
                    "--threads",
                    "4",
                    "--warmup",
                    "2",
                    "--iterations",
                    "5",
                ],
                text=True,
            )

        payload = json.loads(output)
        self.assertEqual(payload["implementation"], "cpp/m4_q4_matvec.cpp")
        self.assertEqual(payload["mode"], "model_shaped_q4_q8_matvec_microbenchmark")
        self.assertTrue(payload["neon_dotprod_compiled"])
        self.assertTrue(payload["correct"])
        self.assertLessEqual(payload["max_abs_diff"], 1e-5)
        self.assertEqual(payload["requested_threads"], 4)
        self.assertEqual(payload["selected_threads"], 1)
        self.assertGreater(payload["speedup_vs_scalar"], 0.0)
        self.assertGreater(payload["effective_weight_bandwidth_gib_s"], 0.0)

    def test_benchmark_script_uses_m4_flags(self):
        source = Path("scripts/bench_m4_q4_matvec.sh").read_text(encoding="utf-8")
        self.assertIn("cpp/m4_q4_matvec.cpp", source)
        self.assertIn("-mcpu=apple-m4", source)
        self.assertIn("cpu-m4-q4-matvec/local-smoke/summary.json", source)


if __name__ == "__main__":
    unittest.main()
