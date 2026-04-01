package tensor

import (
	"fmt"
	"sort"
	"time"

	"github.com/itsfuad/cuda_tensor_runtime/internal/benchfmt"
)

type MeasuredCostModel struct {
	Fallback    CostModel
	Thresholds  map[OpKind]int
	DefaultCUDA map[OpKind]bool
}

func (m MeasuredCostModel) ShouldUseCUDA(op OpKind, work int) bool {
	if threshold, ok := m.Thresholds[op]; ok {
		return work >= threshold
	}
	if useCUDA, ok := m.DefaultCUDA[op]; ok {
		return useCUDA
	}
	if m.Fallback != nil {
		return m.Fallback.ShouldUseCUDA(op, work)
	}
	return ThresholdCostModel{CUDAThreshold: defaultCUDAThreshold}.ShouldUseCUDA(op, work)
}

func (m MeasuredCostModel) Observe(op OpKind, work int, backend ExecBackend, elapsed time.Duration) {}

func LoadMeasuredCostModel(cpuResultsPath, cudaResultsPath string) (MeasuredCostModel, error) {
	cpuResults, err := benchfmt.ReadFile(cpuResultsPath)
	if err != nil {
		return MeasuredCostModel{}, err
	}
	cudaResults, err := benchfmt.ReadFile(cudaResultsPath)
	if err != nil {
		return MeasuredCostModel{}, err
	}
	thresholds, defaults, err := deriveThresholds(cpuResults, cudaResults)
	if err != nil {
		return MeasuredCostModel{}, err
	}
	return MeasuredCostModel{
		Fallback:    ThresholdCostModel{CUDAThreshold: defaultCUDAThreshold},
		Thresholds:  thresholds,
		DefaultCUDA: defaults,
	}, nil
}

func deriveThresholds(cpuResults, cudaResults []benchfmt.Result) (map[OpKind]int, map[OpKind]bool, error) {
	type sample struct {
		work   int
		cpuMs  float64
		cudaMs float64
	}

	cpuIndex := make(map[string]benchfmt.Result, len(cpuResults))
	for _, result := range cpuResults {
		cpuIndex[fmt.Sprintf("%s|%d", result.Name, result.Size)] = result
	}

	grouped := make(map[OpKind][]sample)
	for _, result := range cudaResults {
		cpu, ok := cpuIndex[fmt.Sprintf("%s|%d", result.Name, result.Size)]
		if !ok {
			continue
		}
		op, work, err := workloadWork(result.Name, result.Size)
		if err != nil {
			continue
		}
		grouped[op] = append(grouped[op], sample{
			work:   work,
			cpuMs:  cpu.AvgMs,
			cudaMs: result.AvgMs,
		})
	}

	if len(grouped) == 0 {
		return nil, nil, fmt.Errorf("no overlapping cpu/cuda benchmark results found")
	}

	thresholds := make(map[OpKind]int, len(grouped))
	defaults := make(map[OpKind]bool, len(grouped))
	for op, samples := range grouped {
		sort.Slice(samples, func(i, j int) bool { return samples[i].work < samples[j].work })
		defaults[op] = false
		for _, s := range samples {
			if s.cudaMs < s.cpuMs {
				thresholds[op] = s.work
				defaults[op] = true
				break
			}
		}
	}
	return thresholds, defaults, nil
}

func workloadWork(name string, size int) (OpKind, int, error) {
	switch name {
	case string(OpAdd):
		return OpAdd, size, nil
	case string(OpReLU):
		return OpReLU, size, nil
	case string(OpAddReLU):
		return OpAddReLU, size, nil
	case string(OpMatMul):
		return OpMatMul, size * size * size, nil
	case "compiled_graph":
		return OpMatMul, size * size * size, nil
	default:
		return "", 0, fmt.Errorf("unsupported workload %q", name)
	}
}
