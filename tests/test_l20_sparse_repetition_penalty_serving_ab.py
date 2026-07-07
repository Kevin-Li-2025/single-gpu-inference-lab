import json
import subprocess
import sys
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
        self.assertIn('"repetition_penalty"', source)
        self.assertIn('"logits_processors"', source)
        self.assertIn('"vllm_xargs"', source)
        self.assertIn('"l20_sparse_repetition_penalty": True', source)
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
