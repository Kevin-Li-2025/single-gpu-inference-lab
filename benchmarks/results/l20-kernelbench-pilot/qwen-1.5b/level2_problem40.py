import torch
import torch.nn as nn
from triton import jit, triton_config
from triton.language import cuda

@jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_SIZE=32):
    # Define block dimensions
    BLOCK_M = BLOCK_SIZE
    BLOCK_N = BLOCK_SIZE
    BLOCK_K = BLOCK_SIZE

    # Get grid dimensions
    grid_m = M // BLOCK_M
    if M % BLOCK_M != 0:
        grid_m += 1
    grid_n = N // BLOCK_N
    if N % BLOCK_N != 0:
        grid_n += 1

    # Iterate over blocks
    for i in range(grid_m):
        for j in range(grid_n):
            start_i = i * BLOCK_M
            end_i = min((i + 1) * BLOCK_M, M)
            start_j = j * BLOCK_N
            end_j = min((j + 1) * BLOCK_N, N)

            # Load data into shared memory
            A_block = cuda.shared_array(shape=(BLOCK_M, BLOCK_K), dtype=torch.float32)
            B_block = cuda.shared_array(shape=(BLOCK_K, BLOCK_N), dtype=torch.float32)
            C_block = cuda.shared_array(shape=(BLOCK_M, BLOCK_N), dtype=torch.float32)

            for k in range(BLOCK_K):
                A_block[k] = A[start_i:k*BLOCK_M+start_i, :]
                B_block[k] = B[k:start_k+K, start_j:end_j]

            # Perform matrix multiplication
            for i in range(BLOCK_M):
                for j in range(BLOCK_N):
                    sum = 0.0
                    for k in range(BLOCK_K):
                        sum += A_block[i][k] * B_block[k][j]
                    C_block[i][j] = sum

            # Store results back to global memory
            for i in range(BLOCK_M):
                for j in range(BLOCK_N):
                    C[start_i+i][start_j+j] = C_block[i][j]

@jit
def scale_and_add_kernel(C, D, M, N, S, BLOCK_SIZE=32):
    # Define block dimensions
    BLOCK_M = BLOCK_SIZE
    BLOCK_N = BLOCK_SIZE

    # Get grid dimensions
    grid_m = M // BLOCK_M
    if M % BLOCK_M != 0:
        grid_m += 1
    grid_n = N // BLOCK_N
    if N % BLOCK_N != 0:
        grid_n += 1

    # Iterate over blocks
    for i in range(grid_m):
        for j in range(grid_n):
            start_i = i * BLOCK_M
            end_i = min((i + 1) * BLOCK_M, M)
            start_j = j * BLOCK_N
            end_j = min((j + 1) * BLOCK_N, N)

            # Load data into shared memory
            C_block = cuda.shared_array(shape=(BLOCK_M, BLOCK_N), dtype=torch.float32)
            D_block = cuda.shared_array(shape=(BLOCK_M, BLOCK_N), dtype=torch.float32)

            for k in range(BLOCK_M):
                C_block[k] = C[start_i:k*BLOCK_M+start_i, :]
                D_block[k] = D[start_i:k*BLOCK_M+start_i, :]

            # Perform scaling and addition
            for i in range(BLOCK_M):
                for j in range(BLOCK_N):
                    C_block[i][j] = C_block[i][j] * S + D_block[i][j]

            # Store results back to global memory
            for i in range(BLOCK_M):
                for j in range(BLOCK_N):
                    C[start_i+i][start_j+j] = C_block[i][j]

class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size, in_features = x.shape
        out_features = self.matmul.out_features

        # Allocate memory for intermediate tensors
        C = torch.empty((batch_size, out_features), device=x.device, dtype=torch.float32)
        D = torch.empty((batch_size, out_features), device=x.device, dtype=torch.float32)

        # Perform matrix multiplication
        matmul_kernel[x.device](x, self.matmul.weight.t(), C, batch_size, out_features, in_features, BLOCK_SIZE=32)

        # Scale and add
        scale_and_add_kernel[C.device](C, D, batch_size, out_features, self.scaling_factor, BLOCK_SIZE=32)

        return D

# Example usage
batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

model = ModelNew(in_features, out_features, scaling_factor)
inputs = get_inputs()

output = model(*inputs)
print(output.shape)
