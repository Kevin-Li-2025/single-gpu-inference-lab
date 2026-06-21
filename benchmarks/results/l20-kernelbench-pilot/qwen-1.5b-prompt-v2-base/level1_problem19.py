import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x, out, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(out + offsets, tl.max(tl.load(x + offsets, mask=mask), 0), mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)
        return out

# Example usage
batch_size = 4096
dim = 393216

inputs = get_inputs()
model = ModelNew()

output = model(*inputs)
print(output.shape)  # Should print torch.Size([4096, 393216])
