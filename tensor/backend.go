package tensor

import (
	"fmt"

	cuda "github.com/itsfuad/cuda_tensor_runtime/internal/cuda"
)

func CUDAAvailable() bool { return cuda.Available() }

func (t *Tensor) ToCUDA() error {
	if t.Device == DeviceCUDA { return nil }
	if !cuda.Available() { return fmt.Errorf("cuda unavailable") }
	buf, err := cuda.MallocFloat32(len(t.Data))
	if err != nil { return err }
	if err := cuda.CopyFloat32HostToDevice(buf, t.Data); err != nil {
		_ = cuda.Free(buf)
		return err
	}
	t.cudaBuf = &CUDABuffer{ptr: buf, size: len(t.Data)}
	t.Device = DeviceCUDA
	return nil
}

func (t *Tensor) ToCPU() error {
	if t.Device == DeviceCPU { return nil }
	if t.cudaBuf == nil { return fmt.Errorf("missing cuda buffer") }
	host := make([]float32, t.cudaBuf.size)
	if err := cuda.CopyFloat32DeviceToHost(t.cudaBuf.ptr, host); err != nil { return err }
	t.Data = host
	t.Device = DeviceCPU
	return nil
}

func (t *Tensor) Free() error {
	if t.cudaBuf != nil {
		if err := cuda.Free(t.cudaBuf.ptr); err != nil { return err }
		t.cudaBuf = nil
	}
	return nil
}
