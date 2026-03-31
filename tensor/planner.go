package tensor

const defaultCUDAThreshold = 1 << 12

func ShouldUseCUDA(work int) bool {
	return work >= defaultCUDAThreshold
}
