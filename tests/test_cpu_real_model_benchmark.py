import subprocess
import sys
import unittest
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


class CpuRealModelBenchmarkTest(unittest.TestCase):
    def load_real_model_script(self):
        spec = importlib.util.spec_from_file_location(
            "benchmark_cpu_real_model", "scripts/benchmark_cpu_real_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        return module

    def load_m4_inference_script(self):
        spec = importlib.util.spec_from_file_location(
            "run_m4_cpu_qwen_inference", "scripts/run_m4_cpu_qwen_inference.py"
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        return module

    def test_script_declares_real_gguf_model_path(self):
        source = Path("scripts/benchmark_cpu_real_model.py").read_text(encoding="utf-8")
        self.assertIn("bartowski/SmolLM2-135M-Instruct-GGUF", source)
        self.assertIn("SmolLM2-135M-Instruct-Q4_K_M.gguf", source)
        self.assertIn("real_gguf_cpu_decode", source)
        self.assertIn("hf_hub_download", source)
        self.assertIn("llama_cpp", source)
        self.assertIn("GGUF", source)
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

    def test_rejects_non_gguf_cache_file(self):
        module = self.load_real_model_script()
        with TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.gguf"
            bad.write_bytes(b"\x00\x00\x00\x00")
            with self.assertRaisesRegex(ValueError, "missing the GGUF magic"):
                module.validate_gguf(bad)

    def test_llama_bench_wrapper_and_summary_contract(self):
        wrapper = Path("scripts/bench_cpu_llama_bench.sh").read_text(encoding="utf-8")
        self.assertIn("llama-bench", wrapper)
        self.assertIn("summarize_cpu_llama_bench.py", wrapper)
        self.assertIn("-pg", wrapper)

    def test_m4_inference_script_uses_cpp_completion_path(self):
        source = Path("scripts/run_m4_cpu_qwen_inference.py").read_text(encoding="utf-8")
        self.assertIn("llama-completion", source)
        self.assertIn("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", source)
        self.assertIn('"threads": args.threads', source)
        self.assertIn('"threads_batch": args.threads_batch', source)
        self.assertIn("--mlock", source)
        self.assertIn("--log-file", source)

    def test_m4_inference_parses_perf_and_sanitizes_paths(self):
        module = self.load_m4_inference_script()
        log_text = """
0.00 I common_perf_print: prompt eval time =      58.33 ms /    16 tokens (    3.65 ms per token,   274.30 tokens per second)
0.00 I common_perf_print:        eval time =     399.69 ms /    63 runs   (    6.34 ms per token,   157.62 tokens per second)
0.00 I common_perf_print:       total time =     466.28 ms /    79 tokens
0.00 I common_perf_print:    graphs reused =         62
"""
        perf = module.parse_common_perf(log_text)
        self.assertEqual(perf["prompt_eval"]["tokens_per_s"], 274.30)
        self.assertEqual(perf["decode_eval"]["count_unit"], "runs")
        self.assertEqual(perf["decode_eval"]["tokens_per_s"], 157.62)
        self.assertEqual(perf["total"]["tokens"], 79)
        self.assertEqual(perf["graphs_reused"], 62)

        root = Path.cwd().resolve()
        command = [
            str(root / "build/llama.cpp/build-cpu/bin/llama-completion"),
            "-m",
            "/private/cache/qwen.gguf",
            "--log-file",
            str(root / "benchmarks/results/cpu-real-model/run/runtime.log"),
        ]
        sanitized = module.sanitize_command(
            command,
            Path("/private/cache/qwen.gguf"),
            root / "benchmarks/results/cpu-real-model/run/runtime.log",
        )
        self.assertEqual(sanitized[0], "build/llama.cpp/build-cpu/bin/llama-completion")
        self.assertEqual(sanitized[2], "qwen.gguf")
        self.assertEqual(sanitized[-1], "runtime.log")

    def test_llama_bench_summary_sanitizes_model_path(self):
        raw = [
            {
                "build_commit": "abc1234",
                "build_number": 1,
                "cpu_info": "CPU",
                "gpu_info": "",
                "backends": "BLAS",
                "model_filename": "/private/cache/model.gguf",
                "model_type": "llama 256M Q4_K - Medium",
                "model_size": 10,
                "model_n_params": 20,
                "n_batch": 128,
                "n_ubatch": 128,
                "n_threads": 4,
                "n_gpu_layers": 0,
                "n_prompt": 17,
                "n_gen": 0,
                "avg_ns": 2_000_000,
                "stddev_ns": 100_000,
                "avg_ts": 8500.0,
                "stddev_ts": 12.0,
                "samples_ts": [8490.0, 8510.0],
            },
            {
                "build_commit": "abc1234",
                "build_number": 1,
                "cpu_info": "CPU",
                "gpu_info": "",
                "backends": "BLAS",
                "model_filename": "/private/cache/model.gguf",
                "model_type": "llama 256M Q4_K - Medium",
                "model_size": 10,
                "model_n_params": 20,
                "n_batch": 128,
                "n_ubatch": 128,
                "n_threads": 4,
                "n_gpu_layers": 0,
                "n_prompt": 0,
                "n_gen": 16,
                "avg_ns": 4_000_000,
                "stddev_ns": 200_000,
                "avg_ts": 4000.0,
                "stddev_ts": 20.0,
                "samples_ts": [3990.0, 4010.0],
            },
            {
                "build_commit": "abc1234",
                "build_number": 1,
                "cpu_info": "CPU",
                "gpu_info": "",
                "backends": "BLAS",
                "model_filename": "/private/cache/model.gguf",
                "model_type": "llama 256M Q4_K - Medium",
                "model_size": 10,
                "model_n_params": 20,
                "n_batch": 128,
                "n_ubatch": 128,
                "n_threads": 8,
                "n_gpu_layers": 0,
                "n_prompt": 0,
                "n_gen": 16,
                "avg_ns": 5_000_000,
                "stddev_ns": 200_000,
                "avg_ts": 3000.0,
                "stddev_ts": 20.0,
                "samples_ts": [2990.0, 3010.0],
            },
        ]
        with TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.json"
            summary_path = Path(tmpdir) / "summary.json"
            raw_path.write_text(__import__("json").dumps(raw), encoding="utf-8")
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/summarize_cpu_llama_bench.py",
                    str(raw_path),
                    str(summary_path),
                ]
            )
            summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["model_filename"], "model.gguf")
        self.assertEqual(summary["tests"]["pp17"]["avg_ms"], 2.0)
        self.assertEqual(summary["tests"]["tg16"]["avg_tokens_per_s"], 4000.0)
        self.assertEqual(summary["tests"]["tg16"]["n_threads"], 4)
        self.assertEqual(summary["recommended_generation_threads"], 4)
        self.assertEqual(len(summary["thread_sweep"]["tg16"]), 2)


if __name__ == "__main__":
    unittest.main()
