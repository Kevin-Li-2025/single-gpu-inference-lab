import torch
import triton
import triton.language as tl

# ------------------------------------------------------------
# Triton kernel implementing the ReLU activation
# ------------------------------------------------------------
@triton.jit
def relu_kernel(
    x_ptr,          # *const float*   input pointer
    out_ptr,        # *float*         output pointer
    N,              # int64           total number of elements
    BLOCK_SIZE: tl.constexpr  # compile-time constant
):
    pid = tl.program_id(0)                # program index
    block_start = pid * BLOCK_SIZE         # offset within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # linear offsets for this program

    # Mask for the end of the block
    mask = offsets < N

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # default value if out-of-bounds

    # Apply ReLU
    out = tl.maximum(x, 0.0)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


# ------------------------------------------------------------
# Wrapper that mimics Model.forward
# ------------------------------------------------------------
def triton_kernel_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of Model.forward.
    Arguments:
        x (torch.Tensor): Input tensor of arbitrary shape.
    Returns:
        torch.Tensor: Output tensor with ReLU applied, same shape as input.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    # Determine output shape and allocate output buffer
    N = x.numel()
    out = torch.empty_like(x)

    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    relu_kernel[grid](
        x,
        out,
        N,
        BLOCK_SIZE=1024,  # power of two, fits most GPUs
    )

    return out


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_kernel_wrapper(x)
