import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel implementing element-wise ReLU
# ------------------------------------------------------------------
@triton.jit
def relu_kernel(
    x_ptr,          # *const float*   input pointer
    out_ptr,        # *float*         output pointer
    N,              # int64           total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)               # program id in the grid
    block_start = pid * BLOCK_SIZE        # start index of this program's slice
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # linear offset within the block
    mask = offsets < N                     # mask for out-of-bounds accesses

    # Load the input value
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply ReLU
    relu_val = tl.maximum(x_val, 0.0)

    # Store the result
    tl.store(out_ptr + offsets, relu_val, mask=mask)


# ------------------------------------------------------------------
# Wrapper that mimics the original Module.forward signature
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
    N = x.numel()  # total number of elements
    BLOCK_SIZE = 1024  # power of two for efficient vectorization

    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    relu_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return triton_kernel_wrapper(*args, **kwargs)
