package tensor

func Add(a, b *Tensor) (*Tensor, error) {
	if ShouldUseCUDA(OpAdd, a.Numel()) && CUDAAvailable() {
		return AddCUDA(a, b)
	}
	return AddCPU(a, b)
}

func ReLU(a *Tensor) (*Tensor, error) {
	if ShouldUseCUDA(OpReLU, a.Numel()) && CUDAAvailable() {
		return ReLUCUDA(a)
	}
	return ReLUCPU(a)
}

func AddReLU(a, b *Tensor) (*Tensor, error) {
	if ShouldUseCUDA(OpAddReLU, a.Numel()) && CUDAAvailable() {
		return AddReLUCUDA(a, b)
	}
	return AddReLUCPU(a, b)
}

func MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) == 2 && len(b.Shape) == 2 {
		work := a.Shape[0] * a.Shape[1] * b.Shape[1]
		if ShouldUseCUDA(OpMatMul, work) && CUDAAvailable() {
			return MatMulCUDA(a, b)
		}
	}
	return MatMulCPU(a, b)
}
