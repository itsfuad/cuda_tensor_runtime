package tensor

import "sync"

const defaultCUDAThreshold = 1 << 12

type OpKind string

const (
	OpAdd     OpKind = "add"
	OpReLU    OpKind = "relu"
	OpAddReLU OpKind = "add_relu"
	OpMatMul  OpKind = "matmul"
)

type CostModel interface {
	ShouldUseCUDA(op OpKind, work int) bool
}

type ThresholdCostModel struct {
	CUDAThreshold int
}

func (m ThresholdCostModel) ShouldUseCUDA(op OpKind, work int) bool {
	threshold := m.CUDAThreshold
	if threshold <= 0 {
		threshold = defaultCUDAThreshold
	}
	return work >= threshold
}

var (
	plannerMu       sync.RWMutex
	activeCostModel CostModel = ThresholdCostModel{CUDAThreshold: defaultCUDAThreshold}
)

func SetCostModel(model CostModel) {
	plannerMu.Lock()
	defer plannerMu.Unlock()
	if model == nil {
		activeCostModel = ThresholdCostModel{CUDAThreshold: defaultCUDAThreshold}
		return
	}
	activeCostModel = model
}

func CurrentCostModel() CostModel {
	plannerMu.RLock()
	defer plannerMu.RUnlock()
	return activeCostModel
}

func ShouldUseCUDA(op OpKind, work int) bool {
	return CurrentCostModel().ShouldUseCUDA(op, work)
}

func withCostModel(model CostModel, fn func()) {
	prev := CurrentCostModel()
	SetCostModel(model)
	defer SetCostModel(prev)
	fn()
}
