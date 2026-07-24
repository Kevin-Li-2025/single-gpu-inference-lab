from pathlib import Path


def test_installer_has_conservative_service_gate():
    source = Path("integrations/vllm/install_l20_paged_decode.py").read_text()
    assert "l20_batch == 1 and l20_max_seq <= 2304" in source
    assert "l20_batch <= 4 and l20_max_seq <= 640" in source
    assert "paged_decode_split_out" in source
    assert "decode_wrapper.run" in source
    assert "decode_query.shape[1] in (12, 16)" in source
    assert "not torch.cuda.is_current_stream_capturing()" in source
    assert "from vllm.v1.attention.ops.l20_paged_decode import" in source
    assert "import l20_paged_decode_cuda" not in source


def test_operator_uses_pytorch_dispatcher_and_fake_registration():
    binding = Path("integrations/vllm/cuda/l20_paged_decode.cpp").read_text()
    wrapper = Path("integrations/vllm/l20_paged_decode.py").read_text()
    assert "TORCH_LIBRARY(l20_stack" in binding
    assert "TORCH_LIBRARY_IMPL(l20_stack, CUDA" in binding
    assert "torch.ops.l20_stack.paged_decode_split_out" in wrapper
    assert 'register_fake("l20_stack::paged_decode_split_out")' in wrapper
    assert '"l20_paged_decode_ops.so"' in wrapper


def test_upstream_patch_uses_vllm_native_extension():
    patch = Path(
        "integrations/vllm/vllm-v0.23.0-l20-paged-decode.patch"
    ).read_text()
    assert "_C::l20_paged_decode_split_out" in patch
    assert "csrc/attention/l20_paged_decode.cu" in patch
    assert "vllm/_custom_ops.py" in patch
    assert "tests/v1/attention/test_l20_paged_decode.py" in patch
    assert "DeviceCapability(8, 9)" in patch


def test_current_main_upstream_patch_uses_stable_libtorch_extension():
    patch = Path(
        "integrations/vllm/vllm-main-l20-paged-decode-rfc.patch"
    ).read_text()
    doc = Path("docs/vllm-upstream-patch.md").read_text()
    assert "bb1ae10f04f1a80e8389df2b38fdbc7acf66f38e" in doc
    assert "9fd00ee006ccd4996bbc756397b039343d2fde94" in doc
    assert "kevin/l20-sm89-paged-decode-rfc" in doc
    assert "csrc/libtorch_stable/attention/l20_paged_decode.cu" in patch
    assert "STABLE_TORCH_LIBRARY" in patch
    assert "_C_stable_libtorch" in doc
    assert "csrc/attention/l20_paged_decode.cu" not in patch
    assert "torch::stable::Tensor" in patch
    assert "tests/v1/attention/test_l20_paged_decode.py" in patch
    assert "DeviceCapability(8, 9)" in patch
    assert "Superseded patch snapshots" in doc
    assert "Do not apply them as current integration" in doc
    assert "authoritative implementation" in doc


def test_cuda13_upstream_build_is_reproducible():
    source = Path("scripts/build_vllm_cuda13_l20.sh").read_text()
    assert "CUDAToolkit_ROOT" in source
    assert "VLLM_CUTLASS_SRC_DIR" in source
    assert "VLLM_FLASH_ATTN_SRC_DIR" in source
    assert "TORCH_CUDA_ARCH_LIST" in source
