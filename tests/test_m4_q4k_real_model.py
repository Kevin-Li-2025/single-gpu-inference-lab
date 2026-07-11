import json
import os
import platform
import runpy
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


class M4Q4KRealModelTest(unittest.TestCase):
    def test_parser_lists_q4k_tensor_from_minimal_gguf(self):
        if platform.machine() != "arm64":
            self.skipTest("M4 GGUF parser requires arm64")
        compiler = shutil.which("clang++")
        if compiler is None:
            self.skipTest("clang++ is unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            binary = root / "m4_q4k_gguf"
            model = root / "fixture.gguf"
            subprocess.check_call(
                [
                    compiler,
                    "-O2",
                    "-std=c++20",
                    "-mcpu=apple-m4",
                    "cpp/m4_q4k_gguf.cpp",
                    "-o",
                    str(binary),
                ]
            )
            content = bytearray(struct.pack("<IIQQ", 0x46554747, 3, 1, 1))
            content += gguf_string("general.alignment")
            content += struct.pack("<II", 4, 32)
            content += gguf_string("blk.0.ffn_down.weight")
            content += struct.pack("<IQQIQ", 2, 256, 1, 12, 0)
            content += bytes((-len(content)) % 32)
            content += bytes(144)
            model.write_bytes(content)
            output = subprocess.check_output(
                [str(binary), "--model", str(model), "--list"], text=True
            )

        payload = json.loads(output)
        self.assertEqual(payload["version"], 3)
        self.assertEqual(payload["tensor_count"], 1)
        self.assertEqual(
            payload["q4_k_tensors"],
            [{"name": "blk.0.ffn_down.weight", "dims": [256, 1]}],
        )

    def test_installer_has_reversible_markers(self):
        source = Path("integrations/llama_cpp/install_kevin_m4_q4k.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("KEVIN_M4_Q4K_HOOK_BEGIN", source)
        self.assertIn("KEVIN_M4_Q4K_REPACK_BEGIN", source)
        self.assertIn("--uninstall", source)

    def test_real_model_runner_keeps_quantization_boundary(self):
        source = Path("scripts/run_m4_q4k_real_model_ab.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("all_outputs_exact", source)
        self.assertIn("all_candidate_traces_hit", source)
        self.assertIn("different 4-bit format", source)

    def test_large_model_matrix_keeps_real_model_boundaries(self):
        source = Path("scripts/run_m4_large_model_matrix.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('magic != b"GGUF"', source)
        self.assertIn("cpu_metal_outputs_exact", source)
        self.assertIn("different 4-bit format", source)
        self.assertIn("sha256_file", source)
        self.assertIn("--threads must match the winner", source)

    def test_mlx_bootstrap_pins_compatible_stack(self):
        source = Path("scripts/bootstrap_mlx_m4.sh").read_text(encoding="utf-8")
        self.assertIn('"mlx==0.32.0"', source)
        self.assertIn('"mlx-lm==0.31.3"', source)
        self.assertIn('"transformers==5.0.0"', source)

    def test_sme2_probe_is_labeled_as_external_kernel_evidence(self):
        source = Path("scripts/benchmark_m4_sme2_qwen3b.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("external KleidiAI kernel probe", source)
        self.assertIn("not integrated model inference", source)
        self.assertIn("sme2_speedup", source)

    def test_affine_sme2_path_keeps_q4k_math_and_opt_in_boundary(self):
        source = Path("cpp/m4_q4k_sme2.cpp").read_text(encoding="utf-8")
        integration = Path(
            "integrations/llama_cpp/kevin_m4_q4k_sme2.h"
        ).read_text(encoding="utf-8")
        installer = Path(
            "integrations/llama_cpp/install_kevin_m4_q4k_sme2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("8.0f * rounded_scale", source)
        self.assertIn("qsi8_reference_normalized_rmse", source)
        self.assertIn("full_decode_integration_gate_pass", source)
        self.assertIn('getenv("GGML_M4_Q4K_SME2")', integration)
        self.assertIn("kevin_m4_q4k_repack_x8", integration)
        self.assertIn("GGML_M4_Q4K_SME2_SHARE_PERCENT", integration)
        self.assertIn("GGML_M4_Q4K_SME2_TENSORS", integration)
        self.assertIn("GGML_M4_Q4K_SME2_SHARED_Q8", integration)
        self.assertIn("GGML_M4_Q4K_SME2_PARALLEL_CORRECTION", integration)
        self.assertIn("correction_values_offset", integration)
        self.assertIn("parallel_correction", integration)
        self.assertIn("ggml_barrier(params->threadpool)", integration)
        self.assertIn("vcvtnq_s32_f32", integration)
        self.assertIn("vaddlvq_s8(quantized)", integration)
        self.assertIn("kevin_m4_q4k_sme2_correction_rows", integration)
        self.assertIn("--uninstall", installer)
        self.assertIn("KEVIN_M4_Q4K_SME2_COMPUTE_BEGIN", installer)

    def test_affine_sme2_runner_rejects_unqualified_power_state(self):
        source = Path("scripts/run_m4_q4k_sme2_ab.py").read_text(encoding="utf-8")
        self.assertIn('power["source"] == "AC Power"', source)
        self.assertIn('power["low_power_mode"] == 0', source)
        self.assertIn("load_per_cpu <= args.max_load_per_cpu", source)
        self.assertIn("outputs_byte_identical", source)
        self.assertIn('modes = ("baseline", "candidate")', source)
        self.assertIn('else ("candidate", "baseline")', source)
        self.assertIn(
            '"parallel_correction": args.candidate_correction == "parallel"',
            source,
        )
        self.assertIn("--include-serial-control", source)
        self.assertIn('("baseline", "serial", "candidate")', source)
        self.assertIn('env["GGML_M4_Q4K_SME2_PARALLEL_CORRECTION"] = "0"', source)
        self.assertIn("--candidate-correction", source)
        self.assertIn('"correction_schedule": args.candidate_correction', source)
        self.assertIn('"serial_control": serial_control', source)
        self.assertIn("serial_control_gate", source)
        self.assertIn('"minimum_pair_speedup": min(serial_pair_speedups)', source)

    def test_affine_sme2_runner_isolates_triangle_mode_environment(self):
        namespace = runpy.run_path("scripts/run_m4_q4k_sme2_ab.py")
        mode_env = namespace["mode_env"]
        contaminated = {
            "GGML_M4_Q4K_SME2": "stale",
            "GGML_M4_Q4K_SME2_PARALLEL_CORRECTION": "stale",
        }
        with mock.patch.dict(os.environ, contaminated, clear=False):
            baseline = mode_env("baseline")
            serial = mode_env("serial")
            candidate = mode_env("candidate")
            serial_candidate = mode_env("candidate", "serial")

        self.assertNotIn("GGML_M4_Q4K_SME2", baseline)
        self.assertNotIn("GGML_M4_Q4K_SME2_PARALLEL_CORRECTION", baseline)
        self.assertEqual(serial["GGML_M4_Q4K_SME2"], "1")
        self.assertEqual(serial["GGML_M4_Q4K_SME2_PARALLEL_CORRECTION"], "0")
        self.assertEqual(candidate["GGML_M4_Q4K_SME2"], "1")
        self.assertEqual(candidate["GGML_M4_Q4K_SME2_PARALLEL_CORRECTION"], "1")
        self.assertEqual(
            serial_candidate["GGML_M4_Q4K_SME2_PARALLEL_CORRECTION"], "0"
        )

    def test_affine_sme2_artifact_keeps_negative_e2e_decision(self):
        artifact = json.loads(
            Path(
                "benchmarks/results/cpu-m4-q4k-sme2/"
                "qwen25-coder-3b-affine-v1/summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(all(row["gate_pass"] for row in artifact["real_tensor_gates"]))
        self.assertFalse(artifact["decode_tg128"]["gate_pass"])
        self.assertTrue(artifact["real_prompt"]["outputs_byte_identical"])
        self.assertFalse(artifact["implementation"]["default_enabled"])
        self.assertFalse(
            artifact["follow_up_candidate"][
                "parallel_affine_correction_default_enabled"
            ]
        )
        self.assertFalse(artifact["follow_up_candidate"]["qualified_gate_pass"])

        triangle = json.loads(
            Path(
                "benchmarks/results/cpu-m4-q4k-sme2/"
                "qwen25-coder-3b-affine-v1/qualified-triangle.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(triangle["hardware"]["host_qualified"])
        self.assertFalse(triangle["candidate_vs_llama"]["gate_pass"])
        self.assertFalse(triangle["parallel_vs_serial"]["gate_pass"])
        self.assertTrue(triangle["correctness"]["outputs_byte_identical"])


if __name__ == "__main__":
    unittest.main()
