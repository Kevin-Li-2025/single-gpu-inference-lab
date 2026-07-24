# vLLM Upstream Patch

> **Superseded patch snapshots:** the two checked-in `.patch` files predate the
> current CUDA boundary checks for device, shape, packed workspace capacity,
> split count, and sequence bounds. Do not apply them as current integration
> patches. The authoritative implementation is
> `integrations/vllm/cuda/l20_paged_decode.cu`, built by
> `integrations/vllm/install_l20_paged_decode.py`. Regenerate and revalidate an
> upstream patch before proposing it.

The historical upstream-facing branch snapshot is:

```text
Kevin-Li-2025/vllm:kevin/l20-sm89-paged-decode-rfc
commit bb1ae10f04f1a80e8389df2b38fdbc7acf66f38e
base vllm-project/vllm main 9fd00ee006ccd4996bbc756397b039343d2fde94
```

The corresponding historical current-main patch is:

```text
integrations/vllm/vllm-main-l20-paged-decode-rfc.patch
```

The original `v0.23.0` patch is still checked in as historical validation
evidence, but it no longer applies cleanly to current `main` because CUDA ops
have moved from the legacy `_C` extension into `_C_stable_libtorch`:

```text
integrations/vllm/vllm-v0.23.0-l20-paged-decode.patch
```

That older fork commit was
`6efb66d4eedf6b410abc8e74db027ee8dca2d8ff`, based on tag `v0.23.0` /
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.

## Patch Scope

- add the SM89 paged-decode CUDA source to vLLM's stable libtorch extension;
- register `_C::l20_paged_decode_split_out` through
  `STABLE_TORCH_LIBRARY` / `_C_stable_libtorch`;
- expose the op through `vllm._custom_ops` with FakeTensor support;
- extend native FlashInfer decode metadata with block tables and sequence
  lengths;
- dispatch only on SM89, FP16, head dimension 128, page size 16, measured
  12Q/2KV or 16Q/8KV shapes, eager execution, and the existing conservative
  batch/context gate;
- preserve FlashInfer for every unsupported shape and CUDA Graph capture;
- add four randomized correctness cases and one FakeTensor test.

## Current Main Rebase Status

Plain `git apply --check` of the `v0.23.0` patch against current `main` failed
only at the native registration layer:

```text
CMakeLists.txt
csrc/ops.h
csrc/torch_bindings.cpp
```

The CUDA source, test, `vllm/_custom_ops.py`, and FlashInfer backend hunks still
applied cleanly or with offset. The new RFC branch adapts those registration
points to the current stable ABI layout:

```text
csrc/libtorch_stable/attention/l20_paged_decode.cu
csrc/libtorch_stable/ops.h
csrc/libtorch_stable/torch_bindings.cpp
```

Local verification completed:

```text
git diff --check
PYTHONPYCACHEPREFIX=/tmp/vllm-l20-pycache \
  /usr/bin/python3 -m py_compile \
  vllm/_custom_ops.py tests/v1/attention/test_l20_paged_decode.py
```

The branch was staged for an RFC, but this snapshot is no longer current or
merge-ready. A replacement must first incorporate the local hardening, build
on the L20 host, and rerun GPU correctness, boundary, Compute Sanitizer, and
serving smoke tests.

## Historical L20 Validation

The `_C` namespace fragment compiled and registered on the L20. The upstream
test file passes `5/5` cases, and Qwen2.5-Coder-1.5B completes a real
eight-token request through the source-tree FlashInfer backend.

The full editable vLLM wheel now builds with the CUDA 13.0 components installed
inside the isolated vLLM environment. The system CUDA 12 compiler must not be
used. NVIDIA wheel components for NVCC, CRT, NVVM, runtime, and CCCL must all
remain on the CUDA 13.0 release line. The reproducible environment setup is in
`scripts/build_vllm_cuda13_l20.sh`.

The CUDA 13 Compute Sanitizer package reports zero memcheck errors across all
four numerical GPU cases. The fully built wheel also completes the
Qwen2.5-Coder-1.5B eight-token FlashInfer eager service request.

The remote host has intermittent GitHub connectivity. CUTLASS and
vLLM FlashAttention are therefore supplied through `VLLM_CUTLASS_SRC_DIR` and
`VLLM_FLASH_ATTN_SRC_DIR` instead of being fetched during CMake configuration.

The checked-in patches are retained only to document the earlier upstream
shape and validation history. Do not apply either snapshot to a current
checkout; regenerate from the authoritative local CUDA source and review the
resulting diff against the exact target vLLM revision.
