package tensor

import "unsafe"

type CUDABuffer struct {
	ptr  unsafe.Pointer
	size int
}
