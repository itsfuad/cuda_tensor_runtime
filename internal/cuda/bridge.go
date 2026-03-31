//go:build cgo && cuda

package cuda

/*
#cgo CFLAGS: -I${SRCDIR}/../../cuda
#cgo LDFLAGS: -L${SRCDIR}/../../cuda -lcudatensor -lcudart
#include "tensor_cuda.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func Available() bool {
	return int(C.tensor_cuda_available()) == 1
}

func MallocFloat32(count int) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	code := int(C.tensor_cuda_malloc((*unsafe.Pointer)(&ptr), C.size_t(count*4)))
	if code != 0 {
		return nil, errorFromCode("cuda malloc", code)
	}
	return ptr, nil
}

func Free(ptr unsafe.Pointer) error {
	code := int(C.tensor_cuda_free(ptr))
	if code != 0 {
		return errorFromCode("cuda free", code)
	}
	return nil
}

func CopyFloat32HostToDevice(dst unsafe.Pointer, src []float32) error {
	if len(src) == 0 {
		return nil
	}
	code := int(C.tensor_cuda_memcpy_h2d(dst, unsafe.Pointer(&src[0]), C.size_t(len(src)*4)))
	if code != 0 {
		return errorFromCode("cuda memcpy h2d", code)
	}
	return nil
}

func CopyFloat32DeviceToHost(src unsafe.Pointer, dst []float32) error {
	if len(dst) == 0 {
		return nil
	}
	code := int(C.tensor_cuda_memcpy_d2h(unsafe.Pointer(&dst[0]), src, C.size_t(len(dst)*4)))
	if code != 0 {
		return errorFromCode("cuda memcpy d2h", code)
	}
	return nil
}

func LaunchAddFloat32(a, b, out unsafe.Pointer, n int) error {
	code := int(C.tensor_cuda_add_f32(a, b, out, C.int(n)))
	if code != 0 {
		return errorFromCode("cuda add", code)
	}
	return nil
}

func LaunchReLUFloat32(a, out unsafe.Pointer, n int) error {
	code := int(C.tensor_cuda_relu_f32(a, out, C.int(n)))
	if code != 0 {
		return errorFromCode("cuda relu", code)
	}
	return nil
}

func LaunchMatMulFloat32(a, b, out unsafe.Pointer, m, k, n int) error {
	code := int(C.tensor_cuda_matmul_f32(a, b, out, C.int(m), C.int(k), C.int(n)))
	if code != 0 {
		return errorFromCode("cuda matmul", code)
	}
	return nil
}

func errorFromCode(op string, code int) error {
	msg := C.GoString(C.tensor_cuda_last_error())
	if msg == "" {
		msg = fmt.Sprintf("code=%d", code)
	}
	return fmt.Errorf("%s failed: %s", op, msg)
}
