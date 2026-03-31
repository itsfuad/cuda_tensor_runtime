#ifndef TENSOR_CUDA_H
#define TENSOR_CUDA_H

#include <stddef.h>

#if defined(_WIN32)
#if defined(TENSOR_CUDA_BUILD_DLL)
#define TENSOR_CUDA_API __declspec(dllexport)
#else
#define TENSOR_CUDA_API
#endif
#else
#define TENSOR_CUDA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

TENSOR_CUDA_API int tensor_cuda_available(void);
TENSOR_CUDA_API int tensor_cuda_malloc(void** ptr, size_t bytes);
TENSOR_CUDA_API int tensor_cuda_free(void* ptr);
TENSOR_CUDA_API int tensor_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes);
TENSOR_CUDA_API int tensor_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);
TENSOR_CUDA_API int tensor_cuda_add_f32(const void* a, const void* b, void* out, int n);
TENSOR_CUDA_API int tensor_cuda_relu_f32(const void* a, void* out, int n);
TENSOR_CUDA_API int tensor_cuda_matmul_f32(const void* a, const void* b, void* out, int m, int k, int n);
TENSOR_CUDA_API const char* tensor_cuda_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
