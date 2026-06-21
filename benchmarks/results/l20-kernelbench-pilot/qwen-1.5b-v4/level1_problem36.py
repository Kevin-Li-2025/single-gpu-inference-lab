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
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute linear offsets for this program instance
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size

    # Load input values
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute mean squared over the feature dimension
    mean_sq = tl.sum(x * x, axis=1, keepdims=True)  # (batch_size, 1)

    # Add epsilon to avoid division by zero
    mean_sq += eps

    # Compute square root of the mean squared
    sqrt_mean_sq = tl.sqrt(mean_sq)

    # Normalize the input
    out = x / sqrt_mean_sq

    # Store the result
    tl.store(out_ptr + offs, out, mask=mask)


# ------------------------------------------------------------
# Wrapper that mimics ModelNew.forward
# ------------------------------------------------------------
def triton_kernel_wrapper(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of ModelNew.forward.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, dim1, dim2).
        eps (float, optional): Epsilon constant for numerical stability. Defaults to 1e-5.

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert len(x.shape) == 4, "Input tensor must have four dimensions"

    # Get the total number of elements
    N = x.numel()

    # Determine the grid configuration
    BLOCK_SIZE = 1024  # power of two, fits most GPUs
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch the kernel
    rmsnorm_kernel[grid](
        x,
        out,
        N,
        x.shape[1],  # num_features
        x.shape[2],  # dim1
        x.shape[3],  # dim2
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
