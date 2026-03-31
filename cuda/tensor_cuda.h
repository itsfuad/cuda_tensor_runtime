#ifndef TENSOR_CUDA_H
#define TENSOR_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int tensor_cuda_available(void);
int tensor_cuda_malloc(void** ptr, size_t bytes);
int tensor_cuda_free(void* ptr);
int tensor_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes);
int tensor_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);
int tensor_cuda_add_f32(const void* a, const void* b, void* out, int n);
int tensor_cuda_relu_f32(const void* a, void* out, int n);
int tensor_cuda_matmul_f32(const void* a, const void* b, void* out, int m, int k, int n);
const char* tensor_cuda_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
