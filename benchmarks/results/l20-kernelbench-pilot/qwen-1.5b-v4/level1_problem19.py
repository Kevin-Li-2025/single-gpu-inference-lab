import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel implementing the ReLU activation
# ------------------------------------------------------------------
@triton.jit
def relu_kernel(
    x_ptr,          # *const float*   input pointer
    out_ptr,        # *float*         output pointer
    N,              # i64             total number of elements
    BLOCK_SIZE: tl.constexpr  # compile-time block size
):
    pid = tl.program_id(0)                # program index
    block_start = pid * BLOCK_SIZE         # offset within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # linear indices for this program

    # Mask for the end of the tensor
    mask = offsets < N

    # Load the input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # default value if out-of-bounds

    # Compute ReLU
    relu_x = tl.maximum(x, 0.0)

    # Store the result
    tl.store(out_ptr + offsets, relu_x, mask=mask)


# ------------------------------------------------------------------
# Wrapper that mimics Model.forward
# ------------------------------------------------------------------
def triton_kernel_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of Model.forward.
    Arguments:
        x (torch.Tensor): Input tensor of arbitrary shape.
    Returns:
        torch.Tensor: Tensor with ReLU applied, same shape as input.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    # Determine the output shape
    out_shape = list(x.shape)
    # Allocate output tensor
    out = torch.empty_like(x)
    # Flatten the tensor into a 1D view
    flat_out = out.view(-1)
    # Total number of elements
    N = flat_out.numel()

    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Launch the kernel
    relu_kernel[grid](
        x,
        flat_out,
        N,
        BLOCK_SIZE=1024,  # compile-time constant
    )

    # Convert back to original shape
    return out.view(*out_shape)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_kernel_wrapper(x)
