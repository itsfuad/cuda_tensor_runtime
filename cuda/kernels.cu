#include "tensor_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>

static char g_last_error[1024];

static int set_error_from_cuda(cudaError_t err) {
    const char* s = cudaGetErrorString(err);
    snprintf(g_last_error, sizeof(g_last_error), "%s", s ? s : "unknown cuda error");
    return (int)err;
}

const char* tensor_cuda_last_error(void) {
    return g_last_error;
}

int tensor_cuda_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        set_error_from_cuda(err);
        return 0;
    }
    return count > 0 ? 1 : 0;
}

int tensor_cuda_malloc(void** ptr, size_t bytes) {
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        return set_error_from_cuda(err);
    }
    return 0;
}

int tensor_cuda_free(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        return set_error_from_cuda(err);
    }
    return 0;
}

int tensor_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return set_error_from_cuda(err);
    }
    return 0;
}

int tensor_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return set_error_from_cuda(err);
    }
    return 0;
}

__global__ void add_f32_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__global__ void relu_f32_kernel(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

#define TILE 16
__global__ void matmul_f32_kernel(const float* a, const float* b, float* out, int m, int k, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out[row * n + col] = acc;
    }
}

static int synchronize_after_launch(void) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return set_error_from_cuda(err);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return set_error_from_cuda(err);
    }
    return 0;
}

int tensor_cuda_add_f32(const void* a, const void* b, void* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_f32_kernel<<<grid, block>>>((const float*)a, (const float*)b, (float*)out, n);
    return synchronize_after_launch();
}

int tensor_cuda_relu_f32(const void* a, void* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_f32_kernel<<<grid, block>>>((const float*)a, (float*)out, n);
    return synchronize_after_launch();
}

int tensor_cuda_matmul_f32(const void* a, const void* b, void* out, int m, int k, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    matmul_f32_kernel<<<grid, block>>>((const float*)a, (const float*)b, (float*)out, m, k, n);
    return synchronize_after_launch();
}
