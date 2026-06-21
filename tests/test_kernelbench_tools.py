import importlib.util
import unittest
from pathlib import Path


def load_script(name):
    path = Path("scripts") / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelBenchToolTest(unittest.TestCase):
    def test_extracts_longest_code_block(self):
        module = load_script("generate_kernelbench.py")
        text = "before\n```python\nx = 1\n```\n```python\nclass ModelNew:\n    pass\n```"
        self.assertIn("ModelNew", module.extract_code(text))

    def test_l20_context_is_specific(self):
        module = load_script("generate_kernelbench.py")
        prompt = module.build_prompt("class Model: pass")
        self.assertIn("92 SMs", prompt)
        self.assertIn("sm_89", prompt)
        self.assertIn("randomized inputs", prompt)
        self.assertIn("class ModelNew", prompt)
        self.assertIn("same arguments as `Model.__init__`", prompt)
        self.assertIn("Avoid extra full-size temporary tensors", prompt)

    def test_chunked_allclose_matches_torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional")
        module = load_script("evaluate_kernelbench_l20.py")
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        close = expected + torch.tensor([0.0, 1e-6, 0.0, 0.0])
        far = expected + torch.tensor([0.0, 0.0, 0.1, 0.0])
        original = torch.allclose
        try:
            module.install_chunked_allclose(torch, 2)
            self.assertTrue(torch.allclose(expected, close))
            self.assertFalse(torch.allclose(expected, far))
        finally:
            torch.allclose = original


if __name__ == "__main__":
    unittest.main()
