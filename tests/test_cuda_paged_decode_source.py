from pathlib import Path


def test_cuda_prototype_is_l20_specialized_and_checked():
    source = Path("integrations/vllm/cuda/l20_paged_decode.cu").read_text()
    benchmark = Path("scripts/benchmark_cuda_paged_decode.py").read_text()
    assert "threadIdx.x" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    assert "code=sm_89" in benchmark
    assert "torch.allclose" in benchmark
