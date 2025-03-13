#include <cuda_runtime.h>

__global__ void wave_residual(float* u, float* r, float dx, float dt, float c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1 && idx < N-1) {
        // Finite differences
        float d2u_dx2 = (u[idx-1] - 2*u[idx] + u[idx+1]) / (dx * dx);
        float d2u_dt2 = (u[idx-N] - 2*u[idx] + u[idx+N]) / (dt * dt);
        r[idx] = d2u_dt2 - c*c * d2u_dx2;
    }
}

__global__ void wave_residual_shared(float* u, float* r, float dx, float c, int N) {
    __shared__ float tile[258];  // 256 + 2 halo
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load tile
    if (idx < N) tile[tid + 1] = u[idx];
    if (tid == 0 && idx > 0) tile[0] = u[idx - 1];
    if (tid == blockDim.x-1 && idx < N-1) tile[257] = u[idx + 1];
    __syncthreads();

    // Compute
    if (idx >= 1 && idx < N-1) {
        float d2u_dx2 = (tile[tid] - 2*tile[tid+1] + tile[tid+2]) / (dx * dx);
        r[idx] = -c * c * d2u_dx2;
    }
}

int main() {
    int N = 104857600;
    float dx = 0.01, dt = 0.01, c = 1.0;
    float *d_u, *d_r;

    // Allocate GPU memory
    cudaMalloc(&d_u, N * sizeof(float));
    cudaMalloc(&d_r, N * sizeof(float));

    // Dummy input (fill u with something simple, e.g., sin(x))
    float* h_u = new float[N];
    for (int i = 0; i < N; i++) h_u[i] = sin(i * dx);
    cudaMemcpy(d_u, h_u, N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(destiny, source, size of allocation, method/enum);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    wave_residual<<<gridSize, blockSize>>>(d_u, d_r, dx, dt, c, N);

    // Clean up
    cudaDeviceSynchronize();
    cudaFree(d_u); cudaFree(d_r);
    delete[] h_u;
    return 0;
}
