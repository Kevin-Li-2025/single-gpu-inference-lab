import torch
import torch.nn as nn
from triton import jit, cuda, utils
from triton.language import dtype, device_type

@jit
def rms_norm_kernel(
    x: cuda.device_array[device_type.float32],
    rms_out: cuda.device_array[device_type.float32],
    batch_size: int,
    num_features: int,
    dim1: int,
    dim2: int,
    block_size: int,
    grid_size: tuple[int, int],
):
    start_x = cuda.grid(0)[0] * block_size
    end_x = min(start_x + block_size, batch_size)
    for i in range(start_x, end_x):
        for j in range(dim1):
            for k in range(dim2):
                x_i = x[i, :, j, k]
                rms_out[i, :, j, k] = torch.sqrt(torch.mean(x_i ** 2, dim=0) + 1e-5)

@jit
def forward_rms_norm(x: cuda.device_array[device_type.float32], out: cuda.device_array[device_type.float32]):
    batch_size, num_features, dim1, dim2 = x.shape
    block_size = 32
    grid_size = (batch_size // block_size, dim1 * dim2)
    rms_norm_kernel[x, out, batch_size, num_features, dim1, dim2, block_size, grid_size]

class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
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
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        batch_size, num_features, dim1, dim2 = x.shape
        out = torch.zeros_like(x)
        forward_rms_norm(x, out)
        return out

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
