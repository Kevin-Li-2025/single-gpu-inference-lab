import subprocess
import sys
import unittest
from pathlib import Path


class CpuRealModelBenchmarkTest(unittest.TestCase):
    def test_script_declares_real_gguf_model_path(self):
        source = Path("scripts/benchmark_cpu_real_model.py").read_text(encoding="utf-8")
        self.assertIn("bartowski/SmolLM2-135M-Instruct-GGUF", source)
        self.assertIn("SmolLM2-135M-Instruct-Q4_K_M.gguf", source)
        self.assertIn("real_gguf_cpu_decode", source)
        self.assertIn("hf_hub_download", source)
        self.assertIn("llama_cpp", source)
        self.assertNotIn("synthetic_fp32_tiny_transformer", source)

    def test_help_does_not_require_model_download(self):
        output = subprocess.check_output(
            [sys.executable, "scripts/benchmark_cpu_real_model.py", "--help"],
            text=True,
        )
        self.assertIn("Benchmark a real GGUF model on CPU", output)
        self.assertIn("--repo-id", output)
        self.assertIn("--model-path", output)

    def test_wrapper_writes_cpu_real_model_artifact(self):
        source = Path("scripts/bench_cpu_real_model.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/benchmark_cpu_real_model.py", source)
        self.assertIn("benchmarks/results/cpu-real-model", source)
        self.assertIn("summary.json", source)


if __name__ == "__main__":
    unittest.main()
