package tensor

import "fmt"

func AddCPU(a, b *Tensor) (*Tensor, error) {
	if err := a.ValidateSameShape(b); err != nil { return nil, err }
	out, _ := New(a.Shape, DeviceCPU)
	for i := range a.Data { out.Data[i] = a.Data[i] + b.Data[i] }
	return out, nil
}

func ReLUCPU(a *Tensor) (*Tensor, error) {
	out, _ := New(a.Shape, DeviceCPU)
	for i, v := range a.Data {
		if v > 0 { out.Data[i] = v }
	}
	return out, nil
}

func MatMulCPU(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 { return nil, fmt.Errorf("matmul requires 2D tensors") }
	m, k1 := a.Shape[0], a.Shape[1]
	k2, n := b.Shape[0], b.Shape[1]
	if k1 != k2 { return nil, fmt.Errorf("matmul inner dimension mismatch: %d != %d", k1, k2) }
	out, _ := New([]int{m, n}, DeviceCPU)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for k := 0; k < k1; k++ {
				sum += a.Data[i*k1+k] * b.Data[k*n+j]
			}
			out.Data[i*n+j] = sum
		}
	}
	return out, nil
}
