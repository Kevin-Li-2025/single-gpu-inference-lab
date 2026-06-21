import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel: element-wise square followed by ReLU
# ------------------------------------------------------------------
@triton.jit
def relu_square_kernel(
    x_ptr,          # *const float*   input tensor
    out_ptr,        # *float*         output tensor
    N,              # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Compute linear offset for this program instance
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for out-of-bounds access
    mask = offs < N
    
    # Load input, compute square, apply ReLU
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.maximum(x * x, 0.0)
    
    # Store result
    tl.store(out_ptr + offs, out, mask=mask)


# ------------------------------------------------------------------
# Wrapper that mimics the original PyTorch implementation
# ------------------------------------------------------------------
def triton_kernel_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper that computes the same output as `relu_square_forward`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    out = torch.empty_like(x)
    N = x.numel()
    
    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    relu_square_kernel[grid](x, out, N, BLOCK_SIZE=1024)
    return out


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return triton_kernel_wrapper(*args, **kwargs)
