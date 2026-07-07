import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_processor():
    path = Path("integrations/vllm/l20_sparse_repetition_penalty_logits_processor.py")
    spec = importlib.util.spec_from_file_location(
        "l20_sparse_repetition_penalty_logits_processor",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class L20SparseRepetitionPenaltyVllmTest(unittest.TestCase):
    def test_sparse_repetition_penalty_gate_matches_l20_benchmark_policy(self):
        module = load_processor()

        self.assertFalse(module.should_use_sparse_repetition_penalty(1, 151936, 128))
        self.assertFalse(module.should_use_sparse_repetition_penalty(8, 32768, 128))
        self.assertTrue(module.should_use_sparse_repetition_penalty(8, 65536, 128))
        self.assertTrue(module.should_use_sparse_repetition_penalty(4, 151936, 1024))
        self.assertFalse(module.should_use_sparse_repetition_penalty(4, 151936, 2048))

        decision = module.select_sparse_repetition_penalty_provider(
            8,
            151936,
            512,
            op_available=True,
        )
        self.assertEqual(decision.provider, "sparse_op")
        self.assertEqual(decision.reason, "inside_sparse_gate")
        self.assertTrue(decision.use_sparse_op)

        fallback = module.select_sparse_repetition_penalty_provider(
            8,
            151936,
            512,
            op_available=False,
        )
        self.assertEqual(fallback.provider, "torch_fallback")
        self.assertEqual(fallback.reason, "op_unavailable")

    def test_vllm_logits_processor_uses_official_processor_shape_and_dispatcher_op(self):
        source = Path(
            "integrations/vllm/l20_sparse_repetition_penalty_logits_processor.py"
        ).read_text()

        self.assertIn("from vllm.v1.sample.logits_processor.interface import", source)
        self.assertIn("from vllm.v1.sample.logits_processor import", source)
        self.assertIn(
            "class L20SparseRepetitionPenaltyProcessor(LogitsProcessor)",
            source,
        )
        self.assertIn("def validate_params", source)
        self.assertIn("def update_state", source)
        self.assertIn("def apply(self, logits", source)
        self.assertIn("torch.ops.l20_stack.sparse_repetition_penalty_out", source)
        self.assertIn('register_fake("l20_stack::sparse_repetition_penalty_out")', source)
        self.assertIn("l20_sparse_repetition_penalty", source)
        self.assertIn("l20_repetition_penalty", source)
        self.assertIn("VLLM_L20_SPARSE_REPETITION_PENALTY_TRACE", source)

    def test_processor_state_handles_added_before_swap_moves(self):
        module = load_processor()
        processor = module.L20SparseRepetitionPenaltyProcessor()
        params = SimpleNamespace(
            extra_args={
                "l20_sparse_repetition_penalty": True,
                "l20_repetition_penalty": 1.1,
            }
        )
        processor.update_state(
            SimpleNamespace(
                removed=[],
                added=[
                    (0, params, [10], [100]),
                    (1, params, [20], [200]),
                ],
                moved=[],
            )
        )
        self.assertEqual(processor.states[0].output_token_ids, [100])
        self.assertEqual(processor.states[1].output_token_ids, [200])

        processor.update_state(
            SimpleNamespace(
                removed=[],
                added=[],
                moved=[(0, 1, module.MoveDirectionality.SWAP)],
            )
        )
        self.assertEqual(processor.states[0].row, 0)
        self.assertEqual(processor.states[0].output_token_ids, [200])
        self.assertEqual(processor.states[1].row, 1)
        self.assertEqual(processor.states[1].output_token_ids, [100])

    def test_sparse_repetition_penalty_op_uses_pytorch_dispatcher_registration(self):
        binding = Path(
            "integrations/vllm/cuda/l20_sparse_repetition_penalty.cpp"
        ).read_text()
        kernel = Path(
            "integrations/vllm/cuda/l20_sparse_repetition_penalty.cu"
        ).read_text()
        smoke = Path("scripts/smoke_cuda_sparse_repetition_penalty_op.py").read_text()

        self.assertIn("TORCH_LIBRARY_FRAGMENT(l20_stack", binding)
        self.assertIn("TORCH_LIBRARY_IMPL(l20_stack, CUDA", binding)
        self.assertIn("sparse_repetition_penalty_out", binding)
        self.assertIn("Tensor(a!) logits", binding)
        self.assertIn("C10_CUDA_KERNEL_LAUNCH_CHECK", kernel)
        self.assertIn("logits.scalar_type() == at::kFloat", kernel)
        self.assertIn("token_ids.scalar_type() == at::kLong", kernel)
        self.assertIn("torch.ops.l20_stack.sparse_repetition_penalty_out", smoke)
        self.assertIn("torch.testing.assert_close", smoke)
