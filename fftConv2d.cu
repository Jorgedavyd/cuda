#include <cufft.h>
#include <torch/extension.h>
#include <cuda.h>
#include <iostream>

__global__ void main_kernel (void) {
    
}

__host__ __forceinline__ std::vector<auto> CREATE_ITERABLE (
    torch::Tensor * input, 
    std::vector<std::vector<int>> * parameters
)

template <typename scalar_t>
torch::Tensor seqConv2d(
    std::vector<std::vector<padding>> parameters,
    std::vector<torch::Tensor> weights /*Already in the Fourier space*/

) {
    // Creates a kernel that automates several convolution in a serie;
    
    // weights -> constant memory;

    // Other parameters -> constant 
    
    // If the amount of convolutions excede the capability of the machine

    // then it will be split into serialized kernels.
    AT_DISPATCH_FLOATING_TYPES(
        scalar_t.type(), 
    )

    std::vector<auto> iterable = CREATE_ITERABLE(&input, &parameters);
    main_kernel<<<>>><scalar_t>(&input);
    
    // Deallocate memory
    
    // Return the output
    return out;

}