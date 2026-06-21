import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel implementing the RMS normalization operation
# ------------------------------------------------------------------
@triton.jit
def rms_norm_kernel(
    x_ptr,          # *const float*   input tensor
    out_ptr,        # *float*         output tensor
    N,              # i32             total number of elements
    F,              # i32             number of features
    EPS,            # float32          epsilon term
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load the feature slice corresponding to this program
    x_slice = tl.load(x_ptr + offs * F, mask=mask, other=0.0)

    # Compute the mean over the batch dimension (sum over first dim)
    mean = tl.sum(x_slice, axis=0) / N

    # Compute the variance (mean squared minus mean squared)
    var = tl.sum((x_slice - mean) ** 2, axis=0) / N

    # Add epsilon to stabilize the square root
    var += EPS

    # Take the square root of the variance
    sqrt_var = tl.sqrt(var)

    # Store the normalized value back into the output tensor
    out_idx = offs * F
    tl.store(out_ptr + out_idx, x_slice / sqrt_var)


# ------------------------------------------------------------------
# Wrapper that mimics the original `forward` method
# ------------------------------------------------------------------
def triton_kernel_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper that implements the same functionality as the original `forward` method.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    N = x.numel()  # total number of elements
    F = x.shape[1]  # number of features
    BLOCK_SIZE = 1024  # block size for the CUDA kernel

    # Allocate output tensor
    y = torch.empty_like(x)

    # Grid configuration: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Launch the kernel
    rms_norm_kernel[grid](
        x,
        y,
        N,
        F,
        1e-5,  # default value for eps
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return triton_kernel_wrapper(*args, **kwargs)
