import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel implementing element-wise ReLU
# ------------------------------------------------------------------
@triton.jit
def relu_kernel(
    x_ptr,          # *const float*   input pointer
    y_ptr,          # *float*         output pointer
    N,              # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply ReLU
    y = tl.maximum(x, 0.0)

    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)


# ------------------------------------------------------------------
# Wrapper that mimics the original `forward` method
# ------------------------------------------------------------------
def triton_kernel_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of the ReLU activation function.
    Arguments:
        x (torch.Tensor): input tensor of arbitrary shape
    Returns:
        torch.Tensor: output tensor with ReLU applied
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    out = torch.empty_like(x)
    N = x.numel()

    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    relu_kernel[grid](
        x,
        out,
        N,
        BLOCK_SIZE=1024,  # typical choice for small kernels
    )
    return out


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return triton_kernel_wrapper(*args, **kwargs)
