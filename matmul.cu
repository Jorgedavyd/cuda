#include <torch/extension.h>

template <typename T>
__global__ __cluster_dim__(1,1,1)/*cluster for d shared memory between blocks*/ void multKernel (
    torch::Tensor<T> input,
    torch::Tensor<T> other,
    torch::Tensor<T> * out
) {
    // I'll define blocks of (input.size(-2), input.size(-1)) size
    const unsigned int y_block = blockIdx.y;
    const unsigned int x_block = blockIdx.x;

    const unsigned int x_thread = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;

    if (x_block < input.size(-2) && y_block < input.size(-1)) {
        *out[x_block][y_block] += input[x_block][x_thread] + other[x_thread][y_block]
    }
};

torch::Tensor forward (torch::Tensor input, torch::Tensor other) {
    CHECK_INPUT(input);
    CHECK_INPUT(other);
    // Defining the output in the share memory
    __shared__ torch::Tensor out = torch::zeros({input.size(0), other.size(1)})
    // Defining the parameters
    dim3 gridSize(input.size(0), other.size(1), 1); // Like the output size of the matrix
    const unsigned int prods = input.size(1); // Number of linear operations per location
    // Running the kernel
    multKernel<float><<<gridSize, prods>>>(input, other, &out);
    // returning the output
    return out;
}

int main (void) {
    torch::Tensor input = torch::randn({10, 10}, torch::device(torch::kCUDA));
    torch::Tensor other = torch::randn({10, 10}, torch::device(torch::kCUDA));
    torch::Tensor out = forward(input, other);
    return 0;
};
