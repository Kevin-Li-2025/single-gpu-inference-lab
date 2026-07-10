import json
import platform
import shutil
import struct
import subprocess
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
