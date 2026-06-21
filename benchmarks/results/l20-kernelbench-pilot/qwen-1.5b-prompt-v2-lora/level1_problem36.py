import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel: computes the mean squared error loss between two tensors
# ------------------------------------------------------------------
@triton.jit
def mse_loss_kernel(
    pred_ptr,         # *const float* pointer to predictions
    target_ptr,       # *const float* pointer to targets
    loss_ptr,         # *float* pointer to store the loss
    N,                # total number of elements (batch_size * num_features)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Compute linear offsets for this program instance
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for out-of-bounds access
    mask = offs < N

    # Load predictions and targets
    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)

    # Squared difference
    diff = pred - target
    # Mean squared error
    loss = tl.sum(diff * diff) / N

    # Write back the scalar loss
    tl.store(loss_ptr, loss)


# ------------------------------------------------------------------
# Wrapper Python class implementing the same interface as ModelNew
# ------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract base class.")

# ------------------------------------------------------------------
# Triton kernel launcher
# ------------------------------------------------------------------
def triton_kernel_wrapper(*args, **kwargs):
    """
    Launches the Triton kernel to compute the MSE loss.
    """
    # Allocate output buffer
    loss = torch.zeros((), dtype=torch.float32, device=args[0].device)
    # Total number of elements (batch_size * num_features)
    N = args[0].numel()
    # Grid configuration: one program per block
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    # Launch the kernel
    mse_loss_kernel[grid](
        *args,
        loss=loss,
        N=N,
        BLOCK_SIZE=1024,  # Power of two for efficient scheduling
        num_warps=4,     # Default warps per program
    )
    return loss


class ModelNewWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return triton_kernel_wrapper(*args, **kwargs)
