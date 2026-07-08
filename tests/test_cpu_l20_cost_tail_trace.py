import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class CpuL20CostTailTraceTest(unittest.TestCase):
    def load_cost_tail_script(self):
        spec = importlib.util.spec_from_file_location(
            "build_cpu_l20_cost_tail", "scripts/build_cpu_l20_cost_tail.py"
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        return module

    def load_prompt_trace_client(self):
        spec = importlib.util.spec_from_file_location(
            "run_real_prompt_trace_client", "scripts/run_real_prompt_trace_client.py"
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        return module

    def test_cost_tail_summary_computes_cost_and_p99(self):
        module = self.load_cost_tail_script()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "artifact"
            run_dir = root / "p512-o32" / "qwen-flashinfer-c1-i512-o32-r2"
            run_dir.mkdir(parents=True)
            for idx, itl in enumerate((2.0, 4.0), start=1):
                report = {
                    "request_throughput": 10.0 + idx,
                    "output_throughput": 320.0 + idx * 32.0,
                    "total_token_throughput": 6000.0 + idx * 100.0,
                    "median_ttft_ms": 20.0 + idx,
                    "p95_ttft_ms": 30.0 + idx,
                    "p99_ttft_ms": 40.0 + idx,
                    "median_itl_ms": itl,
                    "p95_itl_ms": itl + 1.0,
                    "p99_itl_ms": itl + 2.0,
                    "median_e2el_ms": 90.0 + idx,
                    "p95_e2el_ms": 110.0 + idx,
                    "p99_e2el_ms": 130.0 + idx,
                }
                (run_dir / f"c1-i512-r{idx}.json").write_text(
                    json.dumps(report),
                    encoding="utf-8",
                )

            summary = module.build_summary(
                root,
                l20_hourly_usd=0.72,
                price_source="test price",
            )

        self.assertEqual(summary["rows"][0]["runs"], 2)
        self.assertEqual(summary["rows"][0]["median_itl_ms"], 3.0)
        self.assertEqual(summary["rows"][0]["p99_itl_ms"], 5.0)
        self.assertAlmostEqual(
            summary["rows"][0]["cost_per_1m_output_tokens_usd"],
            0.72 * 1_000_000.0 / (368.0 * 3600.0),
        )
        markdown = module.render_markdown(summary)
        self.assertIn("$/1M output tok", markdown)
        self.assertIn("p99 ITL", markdown)

    def test_prompt_trace_client_contract_and_prompt_fixture(self):
        client_source = Path("scripts/run_real_prompt_trace_client.py").read_text(
            encoding="utf-8"
        )
        runner_source = Path("scripts/run_vllm_l20_real_prompt_trace.sh").read_text(
            encoding="utf-8"
        )
        prompts_path = Path("benchmarks/prompt_traces/qwen25_coder_real_prompts_v1.jsonl")
        prompts = [
            json.loads(line)
            for line in prompts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertIn("/v1/completions", client_source)
        self.assertIn('"stream": True', client_source)
        self.assertIn("ThreadPoolExecutor", client_source)
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=1", runner_source)
        self.assertIn("run_real_prompt_trace_client.py", runner_source)
        self.assertGreaterEqual(len(prompts), 12)
        self.assertTrue(all("prompt" in prompt for prompt in prompts))
        self.assertTrue(all("max_tokens" in prompt for prompt in prompts))

    def test_percentile_interpolates(self):
        module = self.load_prompt_trace_client()
        self.assertEqual(module.percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(module.percentile([1.0, 2.0, 3.0], 95), 2.9)


if __name__ == "__main__":
    unittest.main()
