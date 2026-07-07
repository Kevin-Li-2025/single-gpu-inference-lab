import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class CpuTinyTransformerCppTest(unittest.TestCase):
    def test_my_cpp_compiles_and_emits_json(self):
        cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if cxx is None:
            self.skipTest("no C++ compiler available")

        with tempfile.TemporaryDirectory() as tmpdir:
            binary = Path(tmpdir) / "my_tiny_transformer"
            subprocess.check_call(
                [
                    cxx,
                    "-O2",
                    "-std=c++17",
                    "cpp/my.cpp",
                    "-o",
                    str(binary),
                ]
            )
            output = subprocess.check_output(
                [
                    str(binary),
                    "--layers",
                    "1",
                    "--dim",
                    "16",
                    "--heads",
                    "4",
                    "--vocab",
                    "64",
                    "--prompt",
                    "4",
                    "--decode",
                    "3",
                    "--matmul",
                    "naive",
                    "--seed",
                    "3",
                ],
                text=True,
            )

        payload = json.loads(output)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["implementation"], "cpp/my.cpp")
        self.assertEqual(payload["mode"], "synthetic_fp32_tiny_transformer")
        self.assertEqual(payload["matmul"], "naive")
        self.assertEqual(payload["layers"], 1)
        self.assertEqual(payload["dim"], 16)
        self.assertEqual(payload["heads"], 4)
        self.assertEqual(payload["vocab"], 64)
        self.assertEqual(payload["prompt_tokens"], 4)
        self.assertEqual(payload["decode_tokens"], 3)
        self.assertGreater(payload["prefill_ms"], 0.0)
        self.assertGreater(payload["decode_ms"], 0.0)
        self.assertGreater(payload["weight_bytes"], 0)
        self.assertGreater(payload["kv_cache_bytes"], 0)
        self.assertGreaterEqual(payload["final_token"], 0)
        self.assertLess(payload["final_token"], 64)

    def test_bench_script_targets_my_cpp_and_artifact_output(self):
        source = Path("scripts/bench_cpu_tiny_transformer.sh").read_text(encoding="utf-8")
        self.assertIn("cpp/my.cpp", source)
        self.assertIn("benchmarks/results/cpu-tiny-transformer/local-smoke/summary.json", source)
        self.assertIn("-std=c++17", source)


if __name__ == "__main__":
    unittest.main()
