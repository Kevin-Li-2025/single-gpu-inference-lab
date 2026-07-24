from pathlib import Path


def test_cuda_prototype_is_l20_specialized_and_checked():
    source = Path("integrations/vllm/cuda/l20_paged_decode.cu").read_text()
    benchmark = Path("scripts/benchmark_cuda_paged_decode.py").read_text()
    assert "threadIdx.x" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    assert "code=sm_89" in benchmark
    assert "torch.allclose" in benchmark
    assert "paged_decode_partial_kernel" in source
    assert "paged_decode_fp8_e4m3_partial_kernel" in source
    assert "fp8_e4m3fn_to_float" in source
    assert "paged_decode_merge_kernel" in source
    assert "split_size must be a multiple of 16 from 64 through 1024" in source
    assert "num_splits must be in [1, 64]" in source
    assert "must be on the same CUDA device as query" in source
    assert "K/V cache must have shape [num_pages, 16, num_kv_heads, 128]" in source
    assert "max_seq_len exceeds block_table capacity" in source
    assert "partial_output.numel() >= partial_rows * kHeadDim" in source
    assert "partial_max.numel() >= partial_rows" in source
    assert "partial_output.size(0) == query.size(0)" not in source
    assert "min(max_seq_len, sequence_capacity)" in source
    assert "min(seq_lens[batch], max_pages * page_size)" in source
    assert "if (seq_len == 0)" in source
    assert "int num_pages" in source
    assert "physical_page >= 0 && physical_page < num_pages" in source
    assert "static_cast<int64_t>(physical_page_shared)" in source
    assert "const int64_t cache_base" in source
    assert "const int64_t cache_offset" in source
    assert "cache page count must fit the CUDA kernel" in source
    assert "k_scale and v_scale must be finite and positive" in source
    indices_wrapper = source[
        source.index("void l20_paged_decode_split_indices_out_cuda") :
        source.index("void l20_paged_decode_fp8_e4m3_split_out_cuda")
    ]
    assert "torch::empty(" not in indices_wrapper
    installer = Path("integrations/vllm/install_l20_paged_decode.py").read_text()
    assert "l20_shape = (4, decode_query.shape[1], 64)" in installer
    smoke = Path("scripts/smoke_cuda_paged_decode_op.py").read_text()
    assert "torch.ops.l20_stack.paged_decode_split_out" in smoke
    assert "torch.testing.assert_close" in smoke
    fp8_smoke = Path("scripts/smoke_cuda_paged_fp8_decode_op.py").read_text()
    fp8_benchmark = Path("scripts/benchmark_cuda_paged_fp8_decode.py").read_text()
    assert "torch.float8_e4m3fn" in fp8_smoke
    assert "torch.ops.l20_stack.paged_decode_fp8_e4m3_split_out" in fp8_smoke
    assert "torch.testing.assert_close" in fp8_smoke
    assert "paged_decode_fp8_e4m3_split_out" in fp8_benchmark
    assert "cuda_fp8_fused_dequant" in fp8_benchmark
    assert "fp8_fused_vs_materialized" in fp8_benchmark
    assert 'parser.add_argument("--split-size", type=int, default=64)' in fp8_benchmark


def test_cuda_fp8_binding_is_registered():
    binding = Path("integrations/vllm/cuda/l20_paged_decode.cpp").read_text()
    wrapper = Path("integrations/vllm/l20_paged_decode.py").read_text()

    assert "paged_decode_fp8_e4m3_split_out" in binding
    assert "l20_paged_decode_fp8_e4m3_split_out_cuda" in binding
    assert "def paged_decode_fp8_e4m3_split_out" in wrapper
    assert 'register_fake("l20_stack::paged_decode_fp8_e4m3_split_out")' in wrapper
