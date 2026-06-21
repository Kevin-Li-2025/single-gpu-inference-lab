import torch
import triton
import triton.language as tl

# ------------------------------------------------------------
# Triton kernel implementing the forward pass of ModelNew
# ------------------------------------------------------------
@triton.jit
def rmsnorm_kernel(
    x_ptr,          # *const float*   input tensor
    out_ptr,        # *float*         output tensor
    batch_size,     # i32              total number of elements
    num_features,   # i32              number of features
    dim1,           # i32              first spatial dimension
    dim2,           # i32              second spatial dimension
    eps,            # float32           epsilon constant
    BLOCK_SIZE: tl.constexpr  # compile-time block size
):
    pid = tl.program_id(0)
    # Compute linear offset for this program instance
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size

    # Load input data
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Flatten the spatial dimensions
    flat_shape = (num_features,) + tuple(dim1) + tuple(dim2)
    flat_idx = tl.index_from_block(offs, block=flat_shape, grid=flat_shape)
    flat_x = tl.load(x + flat_idx, mask=mask, other=0.0)

    # Compute mean squared error (MSE) along the flattened spatial dimensions
    mse = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for d1 in range(dim1):
        for d2 in range(dim2):
            idx = flat_idx + (d1, d2)
            mse += flat_x[idx] * flat_x[idx]

    # Add epsilon to avoid division by zero
    mse += eps

    # Compute square root of MSE
    sqrt_mse = tl.sqrt(mse)

    # Divide by the square root of the mean squared error
    out = flat_x / sqrt_mse

    # Store the result back into the original layout
    for d1 in range(dim1):
        for d2 in range(dim2):
            idx = flat_idx + (d1, d2)
            tl.store(out_ptr + idx, out[idx], mask=mask)


# ------------------------------------------------------------
# Wrapper that mimics ModelNew.forward
# ------------------------------------------------------------
def triton_kernel_wrapper(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of ModelNew.forward.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, dim1, dim2).
        eps (float, optional): Epsilon constant used in the RMS normalization. Defaults to 1e-5.

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert len(x.shape) == 4, "Input tensor must have four dimensions"

    # Get the total number of elements
    N = x.numel()

    # Determine the number of blocks
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch the kernel
    rmsnorm_kernel[grid](
        x,
        out,
        N,
        x.shape[1],
        x.shape[2],
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, dim1, dim2).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_kernel_wrapper(x, self.eps)
