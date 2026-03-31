//go:build !cuda || !cgo

package cuda

import (
	"fmt"
	"unsafe"
)

func Available() bool {
	return false
}

func MallocFloat32(count int) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("cuda unavailable")
}

func Free(ptr unsafe.Pointer) error {
	return fmt.Errorf("cuda unavailable")
}

func CopyFloat32HostToDevice(dst unsafe.Pointer, src []float32) error {
	return fmt.Errorf("cuda unavailable")
}

func CopyFloat32DeviceToHost(src unsafe.Pointer, dst []float32) error {
	return fmt.Errorf("cuda unavailable")
}

func LaunchAddFloat32(a, b, out unsafe.Pointer, n int) error {
	return fmt.Errorf("cuda unavailable")
}

func LaunchReLUFloat32(a, out unsafe.Pointer, n int) error {
	return fmt.Errorf("cuda unavailable")
}

func LaunchMatMulFloat32(a, b, out unsafe.Pointer, m, k, n int) error {
	return fmt.Errorf("cuda unavailable")
}
