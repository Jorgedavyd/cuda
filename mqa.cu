#include <cuda.h>
#include <torch/extension.h>

__host__ __forceinline__ parameters (
    torch::Tensor query,
    torch::Tensor key
) {
    const struct params{
        const 
    }
}

template <typename scalar_t>
__global__ void kernel (
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> keys,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> values,
    float *out,
) {
    extern __shared__ float key[], value[];
    // Defining the thread idx
    const unsigned int batch_idx = blockIdx.x;
    const unsigned int num_kv = blockIdx.z;
    const unsigned int y = threadIdx.y + blockDim.x * blockIdx.x;
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;

    // putting all into shared memory
    if (x < queries.size(-1) && y < queries.size(-2) && num_kv < keys.size(-3) && batch_idx < queries.size(0)) {
        key[batch_idx][num_kv][y][x] = keys[batch_idx][num_kv][y][x];
    }

    __syncthreads(); // Wait untill all executions finish.

    // making the matrix multiplication Q@K^T
    
    
    
}