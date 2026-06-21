import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel implementing the core operation: matmul -> scale -> add
# ------------------------------------------------------------------
@triton.jit
def matmul_scale_add_kernel(
    x_ptr,          # *const float*   input tensor (contiguous)
    w_ptr,          # *const float*   weight tensor (contiguous)
    out_ptr,        # *float*         output tensor (contiguous)
    N,              # int64           batch size
    C,              # int64           input channels
    H,              # int64           height of the feature map
    W,              # int64           width of the feature map
    K,              # int64           output channels
    BLOCK_SIZE: tl.constexpr,     # block size (power of two)
    OUT_PER_BLOCK: tl.constexpr, # total elements per block
    SCALE: tl.constexpr,         # scalar multiplier
    TOTAL_OUT: tl.constexpr      # total number of output elements
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL_OUT

    # Compute linear offsets within the flattened output tensor
    out_idx = tl.where(mask, offs, tl.zeros([BLOCK_SIZE], dtype=tl.int64))

    # Calculate the linear index into the flattened input tensor
    i = out_idx // OUT_PER_BLOCK
    j = (out_idx % OUT_PER_BLOCK) // K
    k = (out_idx % OUT_PER_BLOCK) % K

    # Offsets for the three dimensions
    off_i = i * (C * H * W) + j * (H * W) + k
    off_w = j * W + k

    # Load the input and weight values
    x_val = tl.load(x_ptr + off_i, mask=mask, other=0.0)
    w_val = tl.load(w_ptr + off_w, mask=mask, other=0.0)

    # Perform the element-wise multiplication and addition
    out_val = x_val * w_val + out_val

    # Store the result back to the output tensor
    tl.store(out_ptr + out_idx, out_val, mask=mask)


# ------------------------------------------------------------------
# Wrapper that mimics the original module's forward pass
# ------------------------------------------------------------------
def triton_kernel_wrapper(x, w, scaling_factor):
    """
    Wrapper that mimics the forward pass of the original `Model` class.
    
    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).
        w (torch.Tensor): Weight tensor of shape (K, C, H, W).
        scaling_factor (float): Scalar multiplier applied after the addition.
        
    Returns:
        torch.Tensor: Output tensor of shape (N, K, H, W).
    """
    assert x.shape == w.shape, "Input and weight shapes must be identical"
    N, C, H, W = x.shape
    K = w.shape[0]  # output channels
    
    # Flatten the input and weight tensors
    x_flat = x.view(-1)
    w_flat = w.view(K, -1).transpose(0, 1)  # KxC
    
    # Total number of output elements
    total_out = N * K * H * W
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Grid configuration: one program per block
    grid = lambda meta: (triton.cdiv(total_out, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    matmul_scale_add_kernel[grid](
        x_flat,
        w_flat,
        out.flatten(),
        N,
        C,
        H,
        W,
        K,
        BLOCK_SIZE=1024,  # power of two, fits most GPUs
        OUT_PER_BLOCK=C * H * W,  # total elements per block
        SCALE=scaling_factor,
        TOTAL_OUT=total_out,
        num_warps=4,
    )
    
    return out


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return triton_kernel_wrapper(*args, **kwargs)
