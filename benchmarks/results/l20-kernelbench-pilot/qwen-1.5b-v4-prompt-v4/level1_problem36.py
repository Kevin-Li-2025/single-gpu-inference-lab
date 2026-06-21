import torch
import triton
import triton.language as tl

# ------------------------------------------------------------
# Triton kernel implementing the forward pass of ModelNew
# ------------------------------------------------------------
@triton.jit
def rmsnorm_kernel(
    x_ptr,          # *const float*
    out_ptr,        # *float*
    batch_size,     # i64
    num_features,   # i64
    dim1,           # i64
    dim2,           # i64
    eps,            # float32
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute linear offset for this program instance
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size * num_features * dim1 * dim2

    # Load input elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Flatten the spatial dimensions
    flat_idx = tl.sum(tl.arange(dim1) * dim2 + tl.arange(dim2), axis=0) + offs % dim2

    # Compute mean squared error (MSE) along the flattened spatial dimensions
    mse = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for d in range(dim1):
        start = d * dim2
        end = start + dim2
        mse += tl.sum((x[start:end] - x[flat_idx]) ** 2)

    # Add epsilon to prevent division by zero
    mse += eps

    # Compute square root of MSE
    rms = tl.sqrt(mse)

    # Normalize the original input
    out = x / rms

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
        eps (float, optional): Same as in ModelNew.__init__. Defaults to 1e-5.

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert len(x.shape) == 4, "Input tensor must have four dimensions"

    # Get the total number of elements
    N = x.numel()

    # Total number of elements per program
    BLOCK_SIZE = 1024

    # Allocate output tensor
    out = torch.empty_like(x)

    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Launch kernel
    rmsnorm_kernel[grid](
        x,
        out,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
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
