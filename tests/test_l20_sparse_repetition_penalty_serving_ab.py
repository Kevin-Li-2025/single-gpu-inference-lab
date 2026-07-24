import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class L20SparseRepetitionPenaltyServingAbTest(unittest.TestCase):
    def test_compilation_config_preserves_sparse_penalty_op(self):
        output = subprocess.check_output(
            [
                sys.executable,
                "scripts/build_l20_sparse_repetition_penalty_compilation_config.py",
                "--no-fuse-rope-kvcache",
            ],
            text=True,
        )
        payload = json.loads(output)
        op = "l20_stack::sparse_repetition_penalty_out"
        self.assertIn(op, payload["splitting_ops"])
        self.assertIn("none", payload["custom_ops"])
        self.assertIn(f"+{op}", payload["custom_ops"])
        self.assertFalse(payload["pass_config"]["fuse_rope_kvcache"])

    def test_serving_ab_runner_carries_custom_processor_payload(self):
        source = Path(
            "scripts/run_vllm_l20_sparse_repetition_penalty_serving_ab.sh"
        ).read_text()
        self.assertIn("LOGITS_PROCESSORS_FLAG:---logits-processors", source)
        self.assertIn("build_l20_sparse_repetition_penalty_compilation_config.py", source)
        self.assertIn("VLLM_L20_SPARSE_REPETITION_PENALTY_LIBRARY", source)
        self.assertIn("VLLM_L20_SPARSE_REPETITION_PENALTY_TRACE", source)
        self.assertIn("--variant", source)
        self.assertIn("baseline", source)
        self.assertIn("candidate", source)

    def test_http_probe_sends_native_baseline_and_vllm_xargs_candidate(self):
        source = Path("scripts/probe_vllm_sparse_repetition_penalty_serving.py").read_text()
        self.assertIn('"standalone"', source)
        self.assertIn('"fused"', source)
        self.assertIn('args.variant in {"baseline", "fused"}', source)
        self.assertIn('"repetition_penalty"', source)
        self.assertIn('"logits_processors"', source)
        self.assertIn('"vllm_xargs"', source)
        self.assertIn('"l20_sparse_repetition_penalty": True', source)
        self.assertIn('"l20_penalty_include_prompt": True', source)
        self.assertIn("median_ttft_ms", source)
        self.assertIn("median_itl_ms", source)
        self.assertIn("output_throughput", source)
        self.assertIn("provider_counts", source)

    def test_sparse_serving_summary_compares_latency_and_trace(self):
        source = Path("scripts/summarize_vllm_sparse_repetition_penalty_ab.py").read_text()
        self.assertIn("candidate_trace", source)
        self.assertIn("request_throughput", source)
        self.assertIn("median_itl_ms", source)
        self.assertIn("change_pct", source)

    def test_sparse_penalty_triangle_runner_separates_latency_and_trace_paths(self):
        source = Path("scripts/run_vllm_l20_sparse_penalty_triangle.sh").read_text()

        self.assertIn("install_l20_topk_topp_sampler.py", source)
        self.assertIn("run_variant baseline", source)
        self.assertIn("run_variant standalone", source)
        self.assertIn("run_variant fused", source)
        self.assertIn("standalone-trace/sparse-rp-trace.jsonl", source)
        self.assertIn("fused-trace/l20-topk-topp-trace.jsonl", source)
        self.assertNotIn("VLLM_L20_TOPK_TOPP_DEFER_PENALTIES=1", source)
        self.assertIn("summarize_vllm_sparse_penalty_triangle.py", source)

    def test_sparse_penalty_triangle_matrix_runner_has_formal_rows(self):
        source = Path("scripts/run_vllm_l20_sparse_penalty_triangle_matrix.sh").read_text()

        self.assertIn("MATRIX_ROWS", source)
        self.assertIn("c2_i512_o32_r64", source)
        self.assertIn("c8_i512_o32_r64", source)
        self.assertIn("run_vllm_l20_sparse_penalty_triangle.sh", source)
        self.assertIn("summarize_vllm_sparse_penalty_triangle_matrix.py", source)

    def test_sparse_penalty_triangle_summary_checks_workload_and_traces(self):
        spec = importlib.util.spec_from_file_location(
            "summarize_vllm_sparse_penalty_triangle",
            "scripts/summarize_vllm_sparse_penalty_triangle.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def summary(variant, itl, throughput):
            return {
                "variant": variant,
                "model": "qwen",
                "input_tokens_requested": 512,
                "output_tokens_requested": 32,
                "num_prompts": 8,
                "max_concurrency": 4,
                "request_throughput": throughput,
                "output_throughput": throughput * 32,
                "median_ttft_ms": 10.0,
                "p95_ttft_ms": 12.0,
                "median_itl_ms": itl,
                "p95_itl_ms": itl + 1.0,
                "median_e2el_ms": 80.0,
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant, itl, throughput in [
                ("baseline", 5.0, 1.0),
                ("standalone", 6.0, 0.9),
                ("fused", 4.0, 1.1),
            ]:
                run_dir = root / variant
                run_dir.mkdir()
                (run_dir / f"{variant}_summary.json").write_text(
                    json.dumps(summary(variant, itl, throughput)),
                    encoding="utf-8",
                )
            standalone_trace = root / "standalone-trace"
            standalone_trace.mkdir()
            (standalone_trace / "sparse-rp-trace.jsonl").write_text(
                json.dumps(
                    {
                        "provider": "sparse_op",
                        "reason": "inside_sparse_gate",
                        "max_unique_tokens": 32,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            fused_trace = root / "fused-trace"
            fused_trace.mkdir()
            (fused_trace / "l20-topk-topp-trace.jsonl").write_text(
                json.dumps(
                    {
                        "eligible": True,
                        "reasons": [],
                        "metadata": {"logits_shape": [1, 151936]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = module.build_summary(root)

        self.assertTrue(result["workload_signature_matches"])
        self.assertFalse(result["comparable_workload"])
        self.assertFalse(result["performance_comparable"])
        self.assertEqual(
            result["evidence_status"], "requires_semantic_validation"
        )
        self.assertEqual(
            result["trace_proof"]["standalone"]["provider_counts"]["sparse_op"],
            1,
        )
        self.assertEqual(result["trace_proof"]["fused"]["eligible_events"], 1)
        fused_itl = next(
            row
            for row in result["historical_delta_vs_baseline"]["fused"]
            if row["metric"] == "median_itl_ms"
        )
        self.assertEqual(fused_itl["improvement_pct"], 25.0)

    def test_sparse_penalty_triangle_matrix_summary_counts_positive_rows(self):
        spec = importlib.util.spec_from_file_location(
            "summarize_vllm_sparse_penalty_triangle_matrix",
            "scripts/summarize_vllm_sparse_penalty_triangle_matrix.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def metric(metric, baseline, candidate, improvement):
            return {
                "metric": metric,
                "baseline": baseline,
                "candidate": candidate,
                "improvement_pct": improvement,
                "speedup": 1.0,
                "higher_is_better": metric in {"request_throughput", "output_throughput"},
            }

        def row(path, fused_itl):
            payload = {
                "evidence_status": "requires_semantic_validation",
                "performance_comparable": False,
                "workload_signature_matches": True,
                "workloads": {
                    name: {
                        "model": "qwen",
                        "input_tokens_requested": 512,
                        "output_tokens_requested": 32,
                        "num_prompts": 64,
                        "max_concurrency": 4,
                    }
                    for name in ("baseline", "standalone", "fused")
                },
                "historical_delta_vs_baseline": {
                    "standalone": [
                        metric("median_itl_ms", 10.0, 10.5, -4.0),
                        metric("median_e2el_ms", 100.0, 101.0, -1.0),
                        metric("output_throughput", 100.0, 99.0, -1.0),
                    ],
                    "fused": [
                        metric("median_itl_ms", 10.0, 9.0, fused_itl),
                        metric("median_e2el_ms", 100.0, 98.0, 2.0),
                        metric("output_throughput", 100.0, 102.0, 2.0),
                    ],
                },
                "trace_proof": {
                    "fused": {
                        "eligible_events": 8,
                        "total_events": 10,
                        "eligible_fraction": 0.8,
                    },
                    "standalone": {"provider_counts": {"sparse_op": 4}},
                },
            }
            path.mkdir()
            (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row(root / "c4_i512_o32_r64", 11.0)
            row(root / "c8_i512_o32_r64", -2.0)
            result = module.build_summary(root)

        self.assertEqual(result["row_count"], 2)
        self.assertFalse(result["performance_comparable"])
        self.assertEqual(result["workload_signature_match_row_count"], 2)
        self.assertEqual(result["historical_fused_itl_positive_rows"], 1)
        self.assertEqual(result["historical_fused_e2e_positive_rows"], 2)
