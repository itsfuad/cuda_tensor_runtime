package tensor

import (
	"math/bits"
	"sync"
	"time"
)

const defaultCUDAThreshold = 1 << 12

type OpKind string

const (
	OpAdd     OpKind = "add"
	OpReLU    OpKind = "relu"
	OpAddReLU OpKind = "add_relu"
	OpMatMul  OpKind = "matmul"
)

type ExecBackend string

const (
	BackendCPU  ExecBackend = "cpu"
	BackendCUDA ExecBackend = "cuda"
)

type CostModel interface {
	ShouldUseCUDA(op OpKind, work int) bool
	Observe(op OpKind, work int, backend ExecBackend, elapsed time.Duration)
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

func (m ThresholdCostModel) Observe(op OpKind, work int, backend ExecBackend, elapsed time.Duration) {
}

type AdaptiveCostModel struct {
	Fallback     CostModel
	MinSamples   int
	ExploreEvery int
	ExploreRatio float64

	mu      sync.Mutex
	buckets map[OpKind]map[int]*adaptiveBucketStats
}

type adaptiveBackendStats struct {
	count int
	avgNs float64
}

type adaptiveBucketStats struct {
	decisions int
	cpu       adaptiveBackendStats
	cuda      adaptiveBackendStats
}

func NewAdaptiveCostModel(fallback CostModel, minSamples, exploreEvery int) *AdaptiveCostModel {
	if fallback == nil {
		fallback = ThresholdCostModel{CUDAThreshold: defaultCUDAThreshold}
	}
	if minSamples <= 0 {
		minSamples = 2
	}
	if exploreEvery < 0 {
		exploreEvery = 0
	}
	return &AdaptiveCostModel{
		Fallback:     fallback,
		MinSamples:   minSamples,
		ExploreEvery: exploreEvery,
		ExploreRatio: 2.0,
		buckets:      make(map[OpKind]map[int]*adaptiveBucketStats),
	}
}

func (m *AdaptiveCostModel) ShouldUseCUDA(op OpKind, work int) bool {
	if m == nil {
		return false
	}
	bucket := bucketWork(work)
	m.mu.Lock()
	defer m.mu.Unlock()
	stats := m.bucketStatsLocked(op, bucket)
	stats.decisions++

	switch {
	case stats.cpu.count == 0 && stats.cuda.count == 0:
		return m.Fallback.ShouldUseCUDA(op, work)
	case stats.cpu.count < m.MinSamples:
		return false
	case stats.cuda.count < m.MinSamples:
		return true
	}

	preferCUDA := stats.cuda.avgNs < stats.cpu.avgNs
	if m.shouldExplore(stats, preferCUDA) {
		return !preferCUDA
	}
	return preferCUDA
}

func (m *AdaptiveCostModel) Observe(op OpKind, work int, backend ExecBackend, elapsed time.Duration) {
	if m == nil {
		return
	}
	bucket := bucketWork(work)
	m.mu.Lock()
	defer m.mu.Unlock()
	stats := m.bucketStatsLocked(op, bucket)
	switch backend {
	case BackendCPU:
		updateAdaptiveStats(&stats.cpu, elapsed)
	case BackendCUDA:
		updateAdaptiveStats(&stats.cuda, elapsed)
	}
}

func (m *AdaptiveCostModel) bucketStatsLocked(op OpKind, bucket int) *adaptiveBucketStats {
	opBuckets, ok := m.buckets[op]
	if !ok {
		opBuckets = make(map[int]*adaptiveBucketStats)
		m.buckets[op] = opBuckets
	}
	stats, ok := opBuckets[bucket]
	if !ok {
		stats = &adaptiveBucketStats{}
		opBuckets[bucket] = stats
	}
	return stats
}

func updateAdaptiveStats(stats *adaptiveBackendStats, elapsed time.Duration) {
	stats.count++
	ns := float64(elapsed.Nanoseconds())
	if stats.count == 1 {
		stats.avgNs = ns
		return
	}
	stats.avgNs += (ns - stats.avgNs) / float64(stats.count)
}

func (m *AdaptiveCostModel) shouldExplore(stats *adaptiveBucketStats, preferCUDA bool) bool {
	if m.ExploreEvery <= 0 || stats.decisions%m.ExploreEvery != 0 {
		return false
	}
	faster := stats.cpu.avgNs
	slower := stats.cuda.avgNs
	if preferCUDA {
		faster = stats.cuda.avgNs
		slower = stats.cpu.avgNs
	}
	if faster <= 0 {
		return false
	}
	ratio := slower / faster
	return ratio < m.ExploreRatio
}

func bucketWork(work int) int {
	if work <= 1 {
		return 1
	}
	return 1 << bits.Len(uint(work-1))
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

func Observe(op OpKind, work int, backend ExecBackend, elapsed time.Duration) {
	CurrentCostModel().Observe(op, work, backend, elapsed)
}

func withCostModel(model CostModel, fn func()) {
	prev := CurrentCostModel()
	SetCostModel(model)
	defer SetCostModel(prev)
	fn()
}
