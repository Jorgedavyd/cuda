#include <torch/extension.h>
#include <math.h>

/*
query: (batch_size, num_heads, sequence, head_dim)
values: (batch_size, num_heads, sequence, head_dim)
keys: (batch_size, num_heads, sequence, head_dim)

out = single_query@single_key.T/sqrt(d_model).apply(softmax)@v

threads(num_heads) -> blocks(num_batches) -> clusters (1)-> grid (1)
*/

namespace F = torch::nn::functional;

__global__ void kernel (
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    const float d_model,
    torch::Tensor& out
) {
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int batch = blockIdx.x;
    if (batch <= query.size(0)) {
        
    }
}

torch::Tensor forward (torch::Tensor& input, torch::Tensor& other) {
    const unsigned int batch_size = input.size(0);
    const unsigned int num_heads = input.size(1);
    torch::Tensor out = torch::empty_like(input);
    kernel<<<batch_size, num_heads>>>(input, other, out);
    return out;
}

int main (void) {
    torch::Tensor input = torch::randn({10, 10}, torch::device(torch::kCUDA));
    torch::Tensor other = torch::randn({10, 10}, torch::device(torch::kCUDA));
    
    auto out = forward(input, other);

    return 0;
}
