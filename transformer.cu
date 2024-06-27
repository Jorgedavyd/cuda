#include <torch/extension.h>
#include <cuda.h>
#include <iostream>

__global__ void kernel (void) {

}

__host__ __forceinline__ void ASSERTIONS (
    torch::Tensor * input,
    std::vector<std::vector<torch::Tensor>> * self_weights,
    std::vector<std::vector<torch::Tensor>> * cross_weights,
    std::vector<std::vector<torch::Tensor>> * ffn_weights
) {
    
}

torch::Tensor self_cross_ffn (
    torch::Tensor input,
    std::vector<std::vector<torch::Tensor>> self_weights,
    std::vector<std::vector<torch::Tensor>> self_weights,
) {
    ASSERTIONS();

}

torch::Tensor transformer_fwd (
    torch::Tensor input,
    torch::
) {
    
    // Serializing all into a stream and creating a graph

    
}

torch::Tensor transformer_bwd (
    torch::Tensor out
)