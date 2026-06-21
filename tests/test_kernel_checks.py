import unittest

from l20_stack.kernel_checks import validate_kernelbench_interface


class KernelInterfaceCheckTest(unittest.TestCase):
    def test_rejects_missing_modelnew(self):
        report = validate_kernelbench_interface("def triton_kernel_wrapper(x): return x")
        self.assertFalse(report.valid)
        self.assertIn("missing class ModelNew", report.errors)

    def test_rejects_evaluator_helper_reference(self):
        report = validate_kernelbench_interface(
            "import torch\nclass ModelNew(torch.nn.Module):\n"
            "    def forward(self, x): return get_inputs()[0]\n"
        )
        self.assertFalse(report.valid)
        self.assertIn("references evaluator helper get_inputs", report.errors)

    def test_rejects_wrapper_with_extra_required_args(self):
        report = validate_kernelbench_interface(
            "import torch\n"
            "def triton_kernel_wrapper(x, w): return x\n"
            "class ModelNew(torch.nn.Module):\n"
            "    def forward(self, *args, **kwargs):\n"
            "        return triton_kernel_wrapper(*args, **kwargs)\n"
        )
        self.assertFalse(report.valid)
        self.assertIn(
            "triton_kernel_wrapper requires more positional args than ModelNew.forward",
            report.errors,
        )

    def test_rejects_modelnew_varargs(self):
        report = validate_kernelbench_interface(
            "import torch\nclass ModelNew(torch.nn.Module):\n"
            "    def __init__(self, *args, **kwargs): super().__init__()\n"
            "    def forward(self, *args, **kwargs): return args[0]\n"
        )
        self.assertFalse(report.valid)
        self.assertIn("ModelNew.__init__ uses varargs", report.errors)
        self.assertIn("ModelNew.forward uses varargs", report.errors)

    def test_rejects_main_harness(self):
        report = validate_kernelbench_interface(
            "import torch\nclass ModelNew(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n"
            "if __name__ == '__main__':\n"
            "    print('demo')\n"
        )
        self.assertFalse(report.valid)
        self.assertIn("contains executable test harness", report.errors)

    def test_rejects_triton_sum_keepdims(self):
        report = validate_kernelbench_interface(
            "import torch, triton\nimport triton.language as tl\n"
            "@triton.jit\n"
            "def kernel(x):\n"
            "    return tl.sum(x, axis=0, keepdims=True)\n"
            "class ModelNew(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n"
        )
        self.assertFalse(report.valid)
        self.assertIn("Triton tl.sum uses unsupported keepdims keyword", report.errors)

    def test_rejects_triton_block_tensor_view(self):
        report = validate_kernelbench_interface(
            "import torch, triton\nimport triton.language as tl\n"
            "@triton.jit\n"
            "def kernel(x):\n"
            "    y = x.view(4, 4)\n"
            "    return y\n"
            "class ModelNew(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n"
        )
        self.assertFalse(report.valid)
        self.assertIn("Triton kernel uses dynamic block tensor view/reshape", report.errors)

    def test_rejects_triton_launcher_missing_required_arg(self):
        report = validate_kernelbench_interface(
            "import torch, triton\nimport triton.language as tl\n"
            "@triton.jit\n"
            "def kernel(x, y, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n"
            "    return\n"
            "class ModelNew(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x):\n"
            "        kernel[(1,)](x, n=4, BLOCK_SIZE=16)\n"
            "        return x\n"
        )
        self.assertFalse(report.valid)
        self.assertIn(
            "Triton launcher for kernel supplies 3 args for 4 required args",
            report.errors,
        )

    def test_rejects_triton_arange_without_end(self):
        report = validate_kernelbench_interface(
            "import torch, triton\nimport triton.language as tl\n"
            "@triton.jit\n"
            "def kernel(n: tl.constexpr):\n"
            "    return tl.arange(n)\n"
            "class ModelNew(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n"
        )
        self.assertFalse(report.valid)
        self.assertIn("Triton tl.arange must use explicit start and end", report.errors)

    def test_accepts_basic_modelnew(self):
        report = validate_kernelbench_interface(
            "import torch\nclass ModelNew(torch.nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return x\n"
        )
        self.assertTrue(report.valid)


if __name__ == "__main__":
    unittest.main()
