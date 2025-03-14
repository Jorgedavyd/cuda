#include <cuda_runtime.h>
#include <stdio.h>
#include <mma.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void tensor_core_matmul(half* W, half* X, float* Y, int m, int n, int k) {
    // Warp-level WMMA
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    // One warp per 16x16 tile
    if (warpId == 0) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_W;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_X;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_Y;

        // Load matrices into fragments
        wmma::load_matrix_sync(frag_W, W, k);  // W: 16x16
        wmma::load_matrix_sync(frag_X, X, k);  // X: 16x8 (padded internally)
        wmma::fill_fragment(frag_Y, 0.0f);

        // Compute Y = W * X
        wmma::mma_sync(frag_Y, frag_W, frag_X, frag_Y);

        // Store result
        wmma::store_matrix_sync(Y, frag_Y, n, wmma::mem_row_major);
    }
}

int main() {
    const int M = 32, N = 16, K = 32;  // WMMA tile sizes
    half *d_W, *d_X;
    float *d_Y;

    // Allocate
    cudaMalloc(&d_W, M * K * sizeof(half));
    cudaMalloc(&d_X, K * N * sizeof(half));
    cudaMalloc(&d_Y, M * N * sizeof(float));

    // Host data
    half h_W[M * K], h_X[K * N];
    float h_Y[M * N];
    for (int i = 0; i < M * K; i++) h_W[i] = __float2half(1.0f);  // W = 1s
    for (int i = 0; i < K * N; i++) h_X[i] = __float2half(2.0f);  // X = 2s

    // Copy to device
    cudaMemcpy(d_W, h_W, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Launch (1 block, 32 threads = 1 warp)
    tensor_core_matmul<<<1, WARP_SIZE>>>(d_W, d_X, d_Y, M, N, K);

    // Copy back
    cudaMemcpy(h_Y, d_Y, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
    return 0;
}
