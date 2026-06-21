import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel implementing RMS normalization
# ------------------------------------------------------------------
@triton.jit
def rms_norm_kernel(
    x_ptr,          # *const float*   input tensor
    out_ptr,        # *float*         output tensor
    N,              # i64               total number of elements
    EPS,            # float32           epsilon term
    BLOCK_SIZE: tl.constexpr,  # compile-time block size
):
    pid = tl.program_id(0)                # program id within grid
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear offset
    mask = offs < N                        # mask for out-of-bounds accesses

    # Load the input element
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute the square of the element
    x_sq = x * x

    # Reduce over the feature dimension (sum over last two dimensions)
    sum_sq = tl.sum(x_sq, axis=(-1, -2))

    # Compute the mean (average over the feature dimension)
    mean = sum_sq / (N - 1.0)  # subtract 1 because we use EPS

    # Add epsilon to stabilize the denominator
    denom = mean + EPS

    # Compute the reciprocal of the square root of the mean
    inv_sqrt_mean = 1.0 / tl.sqrt(denom)

    # Multiply the original element by the inverse square root
    out = x * inv_sqrt_mean

    # Store the result
    tl.store(out_ptr + offs, out, mask=mask)


# ------------------------------------------------------------------
# Wrapper that mimics the original module's forward pass
# ------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return rms_norm(*args, **kwargs)


# ------------------------------------------------------------------
# Triton kernel launcher
# ------------------------------------------------------------------
def triton_kernel_wrapper(x):
    """
    Launches the triton kernel to perform RMS normalization on the input tensor `x`.
    The input tensor `x` should have shape `(batch_size, num_features, *)` where
    `num_features` is the dimension over which RMS normalization is performed.

    Args:
        x (torch.Tensor): Input tensor of any shape.

    Returns:
        torch.Tensor: Output tensor with RMS normalization applied, same shape as input.
    """
    assert x.dim() >= 2, "Input tensor must have at least two dimensions"
    batch_size, num_features = x.shape[:2]
    out_shape = list(x.shape)
    out_shape[1] = num_features  # keep the same number of features

    # Total number of elements
    N = batch_size * num_features * torch.prod(torch.tensor(out_shape[2:]))

    # Allocate output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Launch kernel
    rms_norm_kernel[grid](
        x,
        out,
        N,
        1e-5,  # default epsilon value
        BLOCK_SIZE=1024,  # arbitrary choice, can be tuned
    )

    return out


# ------------------------------------------------------------------
# Test harness
# ------------------------------------------------------------------
def test_rms_norm():
    # Create a dummy input tensor
    x = torch.randn(112, 64, 512, 512, dtype=torch.float32, requires_grad=False)

    # Original implementation using torch.nn.RMSNorm
    orig_out = Model()(x)

    # Triton implementation
    triton_out = triton_kernel_wrapper(x)

    # Check if the results are close (within a tolerance)
    assert torch.allclose(orig_out, triton_out, atol=1e-5, rtol=1e-5), "Results differ"


if __name__ == "__main__":
    test_rms_norm()
