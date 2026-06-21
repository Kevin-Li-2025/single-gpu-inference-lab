import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel implementing the core computation
# ------------------------------------------------------------------
@triton.jit
def matmul_scale_add_kernel(
    x_ptr,          # *const float*   input tensor (contiguous)
    y_ptr,          # *const float*   weight tensor (contiguous)
    out_ptr,        # *float*         output tensor (contiguous)
    N,              # int64           batch size
    C,              # int64           input features
    H,              # int64           output features
    BLOCK_SIZE: tl.constexpr,  # compile-time block size
):
    pid = tl.program_id(0)                # program index along the batch dimension
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear offset within the block
    mask = offs < N * C * H               # mask for elements inside the block

    # Compute offsets for the current thread
    c = tl.load(offs + tl.zeros([BLOCK_SIZE], dtype=tl.int64), mask=mask, other=0.0)
    h = tl.load(offs + C * tl.zeros([BLOCK_SIZE], dtype=tl.int64), mask=mask, other=0.0)
    n = tl.load(offs // (C * H), mask=mask, other=0)

    # Linear index into the flattened input tensor
    idx = n * C * H + c + h * C

    # Load input and weight values
    x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
    w_val = tl.load(y_ptr + idx, mask=mask, other=0.0)

    # Scale and add
    out_val = x_val * scaling_factor + x_val
    tl.store(out_ptr + idx, out_val)


# ------------------------------------------------------------------
# Wrapper that mimics the original `Model.forward` signature
# ------------------------------------------------------------------
@triton.jit
def matmul_scale_add_wrapper(
    x_ptr,                  # *const float*   input tensor (contiguous)
    out_ptr,                # *float*         output tensor (contiguous)
    in_features,            # int64           input feature dimension
    out_features,           # int64           output feature dimension
    scaling_factor,         # float32         scalar multiplier
    BLOCK_SIZE: tl.constexpr,  # compile-time block size
):
    # Program ID is ignored since we have one program per batch element
    pid = tl.program_id(0)
    N = tl.shape(x_ptr)[0]       # batch size

    # Compute the total number of elements processed by this program
    num_elements = N * in_features * out_features

    # Loop over all elements
    for i in range(num_elements):
        # Calculate linear indices
        n = i // (in_features * out_features)
        c = (i % (in_features * out_features)) // out_features
        h = i % out_features

        # Linear index into the flattened input tensor
        idx = n * in_features * out_features + c + h * in_features

        # Load input value
        x_val = tl.load(x_ptr + idx, mask=i < num_elements, other=0.0)

        # Scale and add
        out_val = x_val * scaling_factor + x_val

        # Store result
        tl.store(out_ptr + idx, out_val)


# ------------------------------------------------------------------
# Python wrapper matching the original `Model.forward` signature
# ------------------------------------------------------------------
def triton_kernel_wrapper(x):
    """
    Triton implementation of the same computation as `Model.forward`.
    Arguments:
        x (torch.Tensor): input tensor of shape (N, in_features)
    Returns:
        torch.Tensor: output tensor of shape (N, out_features)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    out = torch.empty_like(x)
    N, C, H = x.shape
    BLOCK_SIZE = 1024  # power of two, fits most GPUs

    # Allocate output buffer
    out_ptr = out.contiguous().ptr

    # Launch grid: one program per batch element
    grid = lambda meta: (N,)

    # Launch the kernel
    matmul_scale_add_wrapper[grid](
        x.contiguous().ptr,
        out_ptr,
        C,
        H,
        scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return triton_kernel_wrapper(*args, **kwargs)
