package tensor

func Add(a, b *Tensor) (*Tensor, error) {
	if ShouldUseCUDA(a.Numel()) && CUDAAvailable() { return AddCUDA(a, b) }
	return AddCPU(a, b)
}

func ReLU(a *Tensor) (*Tensor, error) {
	if ShouldUseCUDA(a.Numel()) && CUDAAvailable() { return ReLUCUDA(a) }
	return ReLUCPU(a)
}

func MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) == 2 && len(b.Shape) == 2 {
		work := a.Shape[0] * a.Shape[1] * b.Shape[1]
		if ShouldUseCUDA(work) && CUDAAvailable() { return MatMulCUDA(a, b) }
	}
	return MatMulCPU(a, b)
}
