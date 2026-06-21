import torch
import torch.nn as nn
from triton import jit, cuda, utils
from triton.language import dtype, device_type

@jit
def relu_kernel(
    x: cuda.device_array[utils.dtype.float32],
    y: cuda.device_array[utils.dtype.float32],
    batch_size: int,
    dim: int,
):
    for i in range(batch_size * dim):
        y[i] = max(0.0, x[i])

class ModelNew(nn.Module):
    """
    Custom Triton kernel-based model that performs a ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    @cuda.jit
    def forward(self, x: cuda.device_array[utils.dtype.float32]) -> cuda.device_array[utils.dtype.float32]:
        batch_size = x.shape[0]
        dim = x.shape[1]
        y = cuda.device_array_like(x)
        relu_kernel[x, y, batch_size, dim]()
        return y

# Test the model
model_new = ModelNew()
inputs = get_inputs()
output = model_new(*inputs)
print(output)
