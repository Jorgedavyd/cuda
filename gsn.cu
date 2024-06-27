#include <torch/extension.h>
#include <cuda.h>
#include <iostream>

template <typename scalar_t>
__global__ void GramSchmidtKernel (void) {
    // Define parameters

    // 
}

template <typename scalar_t>
torch::Tensor gsn (
    torch::Tensor input,
    std::vector<torch::Tensor> weights,
    torch::Tensor fc,
) {
    ASSERTIONS(input, weights, fc);
    // Defining the output
    torch::Tensor out = torch::empty()// define this
    // Defining the weights
    std::vector<auto> weights_addresses[weights.length()];
    for (i = 0, i < weight.length(), i++) {
        weights_addresses[i] = &weigths[i];
    }
    // Defining the stream and using CUDA graphs
    cudaStream_t stream[weights.length()];
    for (auto stream_i: stream) {
        AT_DISPATCH_FLOATING_TYPES(
            scalar_t.type(), 
            /*put all accessors*/
        )
    }

    // Culminate the graph 

    // return the output
    return out;

}   

/*
Frobenius inner product, serializing gram schmidt process,
forward and backward method.
*/