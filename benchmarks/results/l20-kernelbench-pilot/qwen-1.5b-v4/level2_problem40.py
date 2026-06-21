import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel implementing ModelNew.forward
# ------------------------------------------------------------------
@triton.jit
def model_new_forward(
    x_ptr,          # *const float*
    out_ptr,        # *float*
    batch_size,     # i32
    in_features,    # i32
    out_features,   # i32
    scale,          # float
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_features

    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Matrix multiply
    matmul_out = x.view(batch_size, out_features).sum(dim=0)

    # Scale and add
    scaled = matmul_out * scale
    out = tl.full([batch_size, out_features], -1e30, dtype=tl.float32)
    out += matmul_out

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


# ------------------------------------------------------------------
# Wrapper that mimics Model.forward signature
# ------------------------------------------------------------------
def triton_kernel_wrapper(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Triton implementation of ModelNew.forward.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        scale (float): Scaling factor applied after matrix multiplication.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features).
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have two dimensions"

    # Allocate output tensor
    batch_size, in_features = x.shape
    out_features = in_features
    total_elements = batch_size * out_features
    out = torch.empty(total_elements, dtype=x.dtype, device=x.device)

    # Grid: one program per BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    model_new_forward[grid](
        x,
        out,
        batch_size,
        in_features,
        out_features,
        scale,
        BLOCK_SIZE=1024,  # power of two, fits most GPUs
    )

    # Gather results into a single contiguous tensor
    out = out.view(batch_size, out_features)

    return out


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scaling_factor

    def forward(self, x):
        return triton_kernel_wrapper(x, self.scale)
