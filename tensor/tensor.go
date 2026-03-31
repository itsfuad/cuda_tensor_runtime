package tensor

import (
	"errors"
	"fmt"
)

type Tensor struct {
	Shape   []int
	Strides []int
	Device  Device
	Data    []float32
	cudaBuf *CUDABuffer
}

func New(shape []int, device Device) (*Tensor, error) {
	n, err := numel(shape)
	if err != nil {
		return nil, err
	}
	return &Tensor{
		Shape:   cloneInts(shape),
		Strides: computeStrides(shape),
		Device:  device,
		Data:    make([]float32, n),
	}, nil
}

func FromSlice(shape []int, data []float32, device Device) (*Tensor, error) {
	n, err := numel(shape)
	if err != nil {
		return nil, err
	}
	if len(data) != n {
		return nil, fmt.Errorf("data length mismatch: got=%d want=%d", len(data), n)
	}
	return &Tensor{
		Shape:   cloneInts(shape),
		Strides: computeStrides(shape),
		Device:  device,
		Data:    append([]float32(nil), data...),
	}, nil
}

func (t *Tensor) Numel() int { return len(t.Data) }

func (t *Tensor) Clone() *Tensor {
	return &Tensor{
		Shape:   cloneInts(t.Shape),
		Strides: cloneInts(t.Strides),
		Device:  t.Device,
		Data:    append([]float32(nil), t.Data...),
	}
}

func (t *Tensor) ValidateSameShape(other *Tensor) error {
	if len(t.Shape) != len(other.Shape) {
		return errors.New("rank mismatch")
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return fmt.Errorf("shape mismatch at dim %d: %d != %d", i, t.Shape[i], other.Shape[i])
		}
	}
	return nil
}

func cloneInts(v []int) []int { return append([]int(nil), v...) }

func numel(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, errors.New("shape cannot be empty")
	}
	n := 1
	for _, d := range shape {
		if d <= 0 {
			return 0, fmt.Errorf("invalid dimension: %d", d)
		}
		n *= d
	}
	return n, nil
}

func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}
