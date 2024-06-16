#include <cuda.h>
#include <torch/extensions.h>


__host__ __forceinline__ unsigned int product (torch::Tensor& input) {
    unsigned int out = 1;
    for (int i = 0, i <input.size(0), i++) {
        out *= input[i].item<int>();
    }
    return out;
}

struct KernelParams {
    unsigned int n_threads;
    unsigned int n_blocks;
};

__host__ __forceinline__ KernelParams getKernelParams (
    torch::Tensor& input
) {
    const unsigned int n_threads = product(input.slice(1, 1, input.size(0)));
    const unsigned int n_blocks = input.size({0});
    return {n_threads, n_blocks};
}

__global__ void kernel (torch::Tensor input, torch::Tensor other, torch::Tensor out) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < input.size(0)) {
        out[x] = input[x] + other[x];
    }
}

__host__ torch::Tensor forward (torch::Tensor& input, torch::Tensor& other) {
    KernelParams kernelParams = getKernelParams(input);
    torch::Tensor out = torch::empty_like(input);
    kernel<<<kernelParams.n_blocks, kernelParams.n_threads>>>(input, other, out);
}

int main (void) {
    torch::Tensor input = torch::randn({10,10}, torch::device(torch::kCUDA))
    torch::Tensor other = torch::randn({10,10}, torch::device(torch::kCUDA))

    torch::Tensor result_add = forward(input, other);

    return 0;
}
