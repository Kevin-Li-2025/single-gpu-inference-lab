import importlib.util
import sys
import unittest
from pathlib import Path


def load_script():
    path = Path("scripts/prepare_triton_traces.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("prepare_triton_traces", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TritonTraceTest(unittest.TestCase):
    def test_source_fingerprint_ignores_whitespace_and_case(self):
        module = load_script()
        self.assertEqual(module.source_fingerprint("A = 1\n"), module.source_fingerprint(" a=1 "))

    def test_holdout_and_slow_rows_are_removed(self):
        module = load_script()
        reference = (
            "import torch\n"
            "class Model(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n"
        )
        rows = [
            {"source": "kernelbench", "level": 1, "problem_id": 19, "pytorch_code": reference, "triton_code": "def triton_kernel_wrapper(x): return x", "result_correctness": True, "result_fast_1": True},
            {"source": "kernelbook", "level": 1, "problem_id": 2, "pytorch_code": reference + '# slow', "triton_code": "def triton_kernel_wrapper(x): return x", "result_correctness": True, "result_fast_1": False},
            {"source": "kernelbook", "level": 1, "problem_id": 3, "pytorch_code": reference + '# fast', "triton_code": "def triton_kernel_wrapper(x): return x", "result_correctness": True, "result_fast_1": True},
        ]
        records, counts = module.build_records(rows, set(), {(1, 19)}, 0.5, True)
        self.assertEqual(counts["holdout"], 1)
        self.assertEqual(counts["not_fast_1"], 1)
        self.assertEqual(len(records["train"]) + len(records["eval"]), 1)

    def test_wrapper_without_reference_model_is_rejected(self):
        module = load_script()
        with self.assertRaises(ValueError):
            module.ensure_model_new("import torch\ndef triton_kernel_wrapper(x): return x")

    def test_wrapper_gets_reference_model_signature_adapter(self):
        module = load_script()
        reference = (
            "import torch\n"
            "class Model(torch.nn.Module):\n"
            "    def __init__(self, in_features, out_features, scaling_factor):\n"
            "        super().__init__()\n"
            "        self.linear = torch.nn.Linear(in_features, out_features)\n"
            "        self.scaling_factor = scaling_factor\n"
            "    def forward(self, x):\n"
            "        return self.linear(x) * self.scaling_factor\n"
        )
        kernel = "import torch\ndef triton_kernel_wrapper(x, weight, bias, scaling_factor): return x"
        code = module.ensure_model_new(kernel, reference)
        self.assertIn("def __init__(self, in_features, out_features, scaling_factor):", code)
        self.assertIn("def forward(self, x):", code)
        self.assertIn(
            "return triton_kernel_wrapper(x, self.linear.weight, self.linear.bias, self.scaling_factor)",
            code,
        )
        self.assertNotIn("*args", code)

    def test_rewrites_model_super_call_for_modelnew(self):
        module = load_script()
        reference = (
            "import torch\n"
            "class GetMask(torch.nn.Module):\n"
            "    def __init__(self):\n"
            "        super(GetMask, self).__init__()\n"
            "    def forward(self, x):\n"
            "        return x\n"
        )
        kernel = "import torch\ndef triton_kernel_wrapper(x): return x"
        code = module.ensure_model_new(kernel, reference)
        self.assertIn("super(ModelNew, self).__init__()", code)

    def test_finds_single_non_model_module_class(self):
        module = load_script()
        reference = (
            "from torch.nn import Module\n"
            "class Mish(Module):\n"
            "    def __init__(self, inplace=False):\n"
            "        super(Mish, self).__init__()\n"
            "        self.inplace = inplace\n"
            "    def forward(self, x):\n"
            "        return x\n"
        )
        kernel = "import torch\ndef triton_kernel_wrapper(x, inplace): return x"
        code = module.ensure_model_new(kernel, reference)
        self.assertIn("class ModelNew(torch.nn.Module):", code)
        self.assertIn("def __init__(self, inplace=False):", code)
        self.assertIn("return triton_kernel_wrapper(x, self.inplace)", code)

    def test_build_records_drops_interface_invalid_labels(self):
        module = load_script()
        reference = (
            "import torch\n"
            "class Model(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n"
        )
        rows = [
            {
                "source": "kernelbook",
                "level": 1,
                "problem_id": 3,
                "pytorch_code": reference,
                "triton_code": (
                    "import torch\n"
                    "def triton_kernel_wrapper(x): return x\n"
                    "if __name__ == '__main__': print('bad')\n"
                ),
                "result_correctness": True,
                "result_fast_1": True,
            }
        ]
        records, counts = module.build_records(rows, set(), set(), 0.5, True)
        self.assertEqual(counts["interface_invalid"], 1)
        self.assertEqual(len(records["train"]) + len(records["eval"]), 0)


if __name__ == "__main__":
    unittest.main()
