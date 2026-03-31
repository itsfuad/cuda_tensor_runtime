package tensor

import (
	"fmt"

	cuda "github.com/itsfuad/cuda_tensor_runtime/internal/cuda"
)

func AddCUDA(a, b *Tensor) (*Tensor, error) {
	if err := a.ValidateSameShape(b); err != nil {
		return nil, err
	}
	if !cuda.Available() {
		return nil, fmt.Errorf("cuda unavailable")
	}
	ac, bc := a.Clone(), b.Clone()
	if err := ac.ToCUDA(); err != nil {
		return nil, err
	}
	defer ac.Free()
	if err := bc.ToCUDA(); err != nil {
		return nil, err
	}
	defer bc.Free()
	out, _ := New(a.Shape, DeviceCPU)
	outBuf, err := cuda.MallocFloat32(out.Numel())
	if err != nil {
		return nil, err
	}
	defer cuda.Free(outBuf)
	if err := cuda.LaunchAddFloat32(ac.cudaBuf.ptr, bc.cudaBuf.ptr, outBuf, out.Numel()); err != nil {
		return nil, err
	}
	if err := cuda.CopyFloat32DeviceToHost(outBuf, out.Data); err != nil {
		return nil, err
	}
	return out, nil
}

func ReLUCUDA(a *Tensor) (*Tensor, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("cuda unavailable")
	}
	ac := a.Clone()
	if err := ac.ToCUDA(); err != nil {
		return nil, err
	}
	defer ac.Free()
	out, _ := New(a.Shape, DeviceCPU)
	outBuf, err := cuda.MallocFloat32(out.Numel())
	if err != nil {
		return nil, err
	}
	defer cuda.Free(outBuf)
	if err := cuda.LaunchReLUFloat32(ac.cudaBuf.ptr, outBuf, out.Numel()); err != nil {
		return nil, err
	}
	if err := cuda.CopyFloat32DeviceToHost(outBuf, out.Data); err != nil {
		return nil, err
	}
	return out, nil
}

func AddReLUCUDA(a, b *Tensor) (*Tensor, error) {
	sum, err := AddCUDA(a, b)
	if err != nil {
		return nil, err
	}
	return ReLUCUDA(sum)
}

func MatMulCUDA(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("matmul requires 2D tensors")
	}
	m, k1 := a.Shape[0], a.Shape[1]
	k2, n := b.Shape[0], b.Shape[1]
	if k1 != k2 {
		return nil, fmt.Errorf("matmul inner dimension mismatch: %d != %d", k1, k2)
	}
	if !cuda.Available() {
		return nil, fmt.Errorf("cuda unavailable")
	}
	ac, bc := a.Clone(), b.Clone()
	if err := ac.ToCUDA(); err != nil {
		return nil, err
	}
	defer ac.Free()
	if err := bc.ToCUDA(); err != nil {
		return nil, err
	}
	defer bc.Free()
	out, _ := New([]int{m, n}, DeviceCPU)
	outBuf, err := cuda.MallocFloat32(out.Numel())
	if err != nil {
		return nil, err
	}
	defer cuda.Free(outBuf)
	if err := cuda.LaunchMatMulFloat32(ac.cudaBuf.ptr, bc.cudaBuf.ptr, outBuf, m, k1, n); err != nil {
		return nil, err
	}
	if err := cuda.CopyFloat32DeviceToHost(outBuf, out.Data); err != nil {
		return nil, err
	}
	return out, nil
}
