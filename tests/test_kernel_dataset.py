import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_script():
    path = Path("scripts/prepare_kernelbench_sft.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("prepare_kernelbench_sft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelDatasetTest(unittest.TestCase):
    def test_problem_bucket_is_stable(self):
        module = load_script()
        self.assertEqual(module.problem_bucket(1, 19), module.problem_bucket(1, 19))
        self.assertNotEqual(module.problem_bucket(1, 19), module.problem_bucket(2, 19))

    def test_holdout_and_non_triton_samples_are_removed(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples"
            benchmark = root / "benchmark"
            samples.mkdir()
            for level, problem_id in ((1, 19), (1, 20), (1, 21)):
                problem_dir = benchmark / "KernelBench" / "level1"
                problem_dir.mkdir(parents=True, exist_ok=True)
                (problem_dir / f"{problem_id}_task.py").write_text("class Model: pass\n")
            payloads = [
                {"level": 1, "problem_id": 19, "correct": True, "kernel": "import triton\nA=1"},
                {"level": 1, "problem_id": 20, "correct": True, "kernel": "import torch\nA=2"},
                {"level": 1, "problem_id": 21, "correct": True, "kernel": "import triton\nA=3"},
            ]
            for index, payload in enumerate(payloads):
                path = samples / str(index)
                path.mkdir()
                (path / "kernel.json").write_text(json.dumps(payload))
            records, counts = module.prepare_records(
                samples,
                benchmark,
                {"tasks": [{"level": 1, "problem_id": 19}]},
                0.5,
                True,
            )
            self.assertEqual(counts["holdout"], 1)
            self.assertEqual(counts["non_triton"], 1)
            self.assertEqual(len(records["train"]) + len(records["eval"]), 1)


if __name__ == "__main__":
    unittest.main()
