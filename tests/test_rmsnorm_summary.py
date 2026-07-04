import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from l20_stack.cli import main
from l20_stack.rmsnorm_summary import summarize_rmsnorm_report


REPORT = Path("benchmarks/results/l20-residual-rmsnorm-v3/full-matrix-cacheflush64.json")


class RmsNormSummaryTest(unittest.TestCase):
    def test_summarizes_l20_rmsnorm_matrix(self):
        summary = summarize_rmsnorm_report(REPORT).to_dict()

        self.assertTrue(summary["all_correct"])
        self.assertEqual(summary["shape_count"], 24)
        self.assertEqual(summary["gpu_name"], "NVIDIA L20")
        self.assertEqual(summary["dtype"], "float16")
        self.assertEqual(summary["cache_flush_mb"], 64)
        self.assertEqual(
            summary["operators"]["residual_rmsnorm"]["fastest_counts"],
            {"flashinfer": 8, "l20_dispatch": 1, "l20_inplace": 14, "torch_eager": 1},
        )
        self.assertEqual(
            summary["operators"]["rmsnorm"]["fastest_counts"],
            {"torch_eager": 20, "triton_w4": 2, "triton_w8": 2},
        )
        self.assertEqual(len(summary["large_prefill_rows_4096"]), 4)

    def test_cli_rmsnorm_summary_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["rmsnorm-summary", str(REPORT), "--output", str(output_path)]
                )

            emitted = json.loads(stdout.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(emitted, written)
            self.assertEqual(written["artifact"], REPORT.name)
            self.assertIn("residual_rmsnorm", written["operators"])


if __name__ == "__main__":
    unittest.main()
