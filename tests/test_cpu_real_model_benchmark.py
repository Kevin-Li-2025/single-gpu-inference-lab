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

    def load_break_even_script(self):
        spec = importlib.util.spec_from_file_location(
            "build_cpu_l20_break_even", "scripts/build_cpu_l20_break_even.py"
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

    def test_cpu_l20_break_even_computes_m4_equivalent(self):
        module = self.load_break_even_script()
        cpu_summary = {
            "model_filename": "qwen.gguf",
            "tests": {
                "pp512": {
                    "avg_ms": 1000.0,
                    "avg_tokens_per_s": 512.0,
                    "n_threads": 8,
                },
                "tg32": {
                    "avg_ms": 200.0,
                    "avg_tokens_per_s": 160.0,
                    "n_threads": 6,
                },
                "tg128": {
                    "avg_ms": 800.0,
                    "avg_tokens_per_s": 160.0,
                    "n_threads": 6,
                },
                "pp512+tg32": {
                    "avg_ms": 1250.0,
                    "avg_tokens_per_s": 435.2,
                    "n_threads": 6,
                },
                "pp512+tg128": {
                    "avg_ms": 1800.0,
                    "avg_tokens_per_s": 355.6,
                    "n_threads": 6,
                },
            },
        }
        l20_summary = {
            "pairs": [
                {
                    "shapes": {
                        "c2-i512": {
                            "flashinfer": {
                                "runs": 3,
                                "median_itl_ms": 3.0,
                                "median_ttft_ms": 30.0,
                                "p99_itl_ms": 5.0,
                                "output_throughput": 320.0,
                            }
                        }
                    }
                }
            ]
        }

        cpu = module.cpu_shape(cpu_summary, prompt_tokens=512, output_tokens=32)
        rows = module.attach_break_even(
            cpu,
            module.iter_l20_flashinfer_shapes(l20_summary, output_tokens=32),
        )

        self.assertEqual(cpu["serial_requests_per_s"], 0.8)
        self.assertEqual(rows[0]["estimated_request_throughput"], 10.0)
        self.assertEqual(rows[0]["vs_cpu_serial_request_throughput"], 12.5)
        self.assertEqual(rows[0]["vs_cpu_decode_throughput"], 2.0)

    def test_cpu_l20_break_even_supports_same_model_mode(self):
        module = self.load_break_even_script()
        cpu_summary = {
            "model_filename": "qwen.gguf",
            "tests": {
                "pp512": {
                    "avg_ms": 1000.0,
                    "avg_tokens_per_s": 512.0,
                    "n_threads": 8,
                },
                "tg32": {
                    "avg_ms": 200.0,
                    "avg_tokens_per_s": 160.0,
                    "n_threads": 6,
                },
                "tg128": {
                    "avg_ms": 800.0,
                    "avg_tokens_per_s": 160.0,
                    "n_threads": 6,
                },
                "pp512+tg32": {
                    "avg_ms": 1250.0,
                    "avg_tokens_per_s": 435.2,
                    "n_threads": 6,
                },
                "pp512+tg128": {
                    "avg_ms": 1800.0,
                    "avg_tokens_per_s": 355.6,
                    "n_threads": 6,
                },
            },
        }
        l20_summary = {
            "pairs": [
                {
                    "shapes": {
                        "c1-i512": {
                            "flashinfer": {
                                "runs": 5,
                                "median_itl_ms": 4.0,
                                "median_ttft_ms": 20.0,
                                "p99_itl_ms": 6.0,
                                "output_throughput": 160.0,
                            }
                        }
                    }
                }
            ]
        }

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cpu_o32 = root / "cpu-o32.json"
            cpu_o128 = root / "cpu-o128.json"
            l20_o32 = root / "l20-o32.json"
            l20_o128 = root / "l20-o128.json"
            cpu_o32.write_text(__import__("json").dumps(cpu_summary), encoding="utf-8")
            cpu_o128.write_text(__import__("json").dumps(cpu_summary), encoding="utf-8")
            l20_o32.write_text(__import__("json").dumps(l20_summary), encoding="utf-8")
            l20_o128.write_text(__import__("json").dumps(l20_summary), encoding="utf-8")

            summary = module.build_summary(
                cpu_o32,
                cpu_o128,
                l20_o32,
                l20_o128,
                mode="cpu_l20_same_model_break_even",
                title="CPU vs L20 Break-Even: Qwen2.5-Coder-0.5B",
                l20_model="Qwen2.5-Coder-0.5B-Instruct",
                l20_source="vLLM FlashInfer serving, NVIDIA L20",
            )

        self.assertEqual(summary["mode"], "cpu_l20_same_model_break_even")
        self.assertEqual(
            summary["l20"]["p512_o32"][0]["model"],
            "Qwen2.5-Coder-0.5B-Instruct",
        )
        self.assertIn("same Qwen2.5-Coder", summary["claim_boundary"][0])
        markdown = module.render_markdown(summary)
        self.assertIn("same-model L20 vLLM FlashInfer", markdown)
        self.assertIn("Qwen2.5-Coder-0.5B vLLM FlashInfer", markdown)

    def test_l20_qwen25_coder_break_even_runner_contract(self):
        source = Path(
            "scripts/run_vllm_l20_qwen25_coder_0p5b_break_even.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("Qwen2.5-Coder-0.5B-Instruct", source)
        self.assertIn("qwen25-coder-0p5b", source)
        self.assertIn("OUTPUT_TOKENS=\"$output_tokens\"", source)
        self.assertIn("run_vllm_l20_sampling_winner_matrix.sh", source)
        self.assertIn("p512-o32", source)
        self.assertIn("p512-o128", source)
        self.assertIn("runner_ready_measurement_pending", source)

        pending_config = __import__("json").loads(
            Path(
                "benchmarks/results/cpu-l20-break-even/"
                "qwen25-coder-0p5b-identical-model-pending/run-config.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            pending_config["mode"],
            "l20_qwen25_coder_0p5b_same_model_break_even_runner",
        )
        self.assertEqual(
            pending_config["expected_l20_outputs"],
            [
                "p512-o32/summary.json",
                "p512-o32/README.md",
                "p512-o128/summary.json",
                "p512-o128/README.md",
            ],
        )
        self.assertEqual(pending_config["status"], "runner_ready_measurement_pending")

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
