import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel: computes the scaled residual addition
# ------------------------------------------------------------------
@triton.jit
def scaled_residual_add_kernel(
    x_ptr,          # *const float*   input tensor (contiguous)
    w_ptr,          # *const float*   weight tensor (contiguous)
    b_ptr,          # *const float*   bias tensor (contiguous)
    out_ptr,        # *float*         output tensor (contiguous)
    N,              # int64           total number of elements
    BLOCK_SIZE: tl.constexpr,  # compile-time block size
):
    pid = tl.program_id(0)
    # Compute linear offsets for this program instance
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load input element
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Load weight and bias
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)

    # Scale and add
    out = x * w + b

    # Store result
    tl.store(out_ptr + offs, out, mask=mask)


# ------------------------------------------------------------------
# Python wrapper: mimics the original PyTorch implementation
# ------------------------------------------------------------------
def triton_kernel_wrapper(x, w, b):
    """
    Wrapper that mimics the original `forward` method but uses a Triton kernel.
    Arguments:
        x (torch.Tensor): input tensor of shape (N, C_in)
        w (torch.Tensor): weight tensor of shape (C_out, C_in)
        b (torch.Tensor): bias tensor of shape (C_out)
    Returns:
        torch.Tensor: output tensor of shape (N, C_out)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert w.is_contiguous(), "Weight tensor must be contiguous"
    assert b.is_contiguous(), "Bias tensor must be contiguous"

    # Allocate output tensor
    out = torch.empty_like(x)

    # Total number of elements
    N = x.numel()

    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Launch kernel
    scaled_residual_add_kernel[grid](
        x,
        w,
        b,
        out,
        N,
        BLOCK_SIZE=1024,  # arbitrary choice, should fit most cases
    )

    return out


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return triton_kernel_wrapper(*args, **kwargs)
