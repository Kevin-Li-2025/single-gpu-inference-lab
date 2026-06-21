import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(x, y, out, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(out + offsets, tl.dot(x + offsets, y + offsets, mask=mask), mask=mask)

@triton.jit
def scale_kernel(x, out, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(out + offsets, x + offsets * scaling_factor, mask=mask)

@triton.jit
def residual_add_kernel(x, y, out, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(out + offsets, x + offsets + y + offsets, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = matmul_kernel[triton.config.compute_capability()]
        self.scale = scale_kernel[triton.config.compute_capability()]
        self.residual_add = residual_add_kernel[triton.config.compute_capability()]

    def forward(self, x, y):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size = x.shape[0]
        n = x.numel()

        # Matmul
        x = self.matmul(x, y, n, BLOCK_SIZE=256)

        # Scale
        x = self.scale(x, n, BLOCK_SIZE=256)

        # Residual Add
        x = self.residual_add(x, y, n, BLOCK_SIZE=256)

        return x

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
