package tensor

import "time"

func Add(a, b *Tensor) (*Tensor, error) {
	return dispatch(OpAdd, a.Numel(), func() (*Tensor, error) {
		return AddCPU(a, b)
	}, func() (*Tensor, error) {
		return AddCUDA(a, b)
	})
}

func ReLU(a *Tensor) (*Tensor, error) {
	return dispatch(OpReLU, a.Numel(), func() (*Tensor, error) {
		return ReLUCPU(a)
	}, func() (*Tensor, error) {
		return ReLUCUDA(a)
	})
}

func AddReLU(a, b *Tensor) (*Tensor, error) {
	return dispatch(OpAddReLU, a.Numel(), func() (*Tensor, error) {
		return AddReLUCPU(a, b)
	}, func() (*Tensor, error) {
		return AddReLUCUDA(a, b)
	})
}

func MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) == 2 && len(b.Shape) == 2 {
		work := a.Shape[0] * a.Shape[1] * b.Shape[1]
		return dispatch(OpMatMul, work, func() (*Tensor, error) {
			return MatMulCPU(a, b)
		}, func() (*Tensor, error) {
			return MatMulCUDA(a, b)
		})
	}
	return MatMulCPU(a, b)
}

func dispatch(op OpKind, work int, cpu func() (*Tensor, error), cuda func() (*Tensor, error)) (*Tensor, error) {
	backend := BackendCPU
	run := cpu
	useCUDA := ShouldUseCUDA(op, work)
	if CUDAAvailable() && useCUDA {
		backend = BackendCUDA
		run = cuda
	}

	start := time.Now()
	out, err := run()
	if err == nil {
		elapsed := time.Since(start)
		Observe(op, work, backend, elapsed)
		emitExecutionObservation(ExecutionObservation{
			Op:      op,
			Work:    work,
			Backend: backend,
			Elapsed: elapsed,
		})
	}
	return out, err
}
