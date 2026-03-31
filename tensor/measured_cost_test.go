package tensor

import (
	"path/filepath"
	"testing"

	"github.com/itsfuad/cuda_tensor_runtime/internal/benchfmt"
)

func TestMeasuredCostModelUsesDerivedThresholds(t *testing.T) {
	dir := t.TempDir()
	cpuPath := filepath.Join(dir, "cpu.json")
	cudaPath := filepath.Join(dir, "cuda.json")

	cpuResults := []benchfmt.Result{
		{Name: "add", Size: 64, Device: "cpu", AvgMs: 2.0},
		{Name: "add", Size: 256, Device: "cpu", AvgMs: 4.0},
		{Name: "matmul", Size: 8, Device: "cpu", AvgMs: 5.0},
		{Name: "matmul", Size: 16, Device: "cpu", AvgMs: 20.0},
	}
	cudaResults := []benchfmt.Result{
		{Name: "add", Size: 64, Device: "cuda", AvgMs: 3.0},
		{Name: "add", Size: 256, Device: "cuda", AvgMs: 1.0},
		{Name: "matmul", Size: 8, Device: "cuda", AvgMs: 7.0},
		{Name: "matmul", Size: 16, Device: "cuda", AvgMs: 10.0},
	}
	if err := benchfmt.WriteFile(cpuPath, cpuResults); err != nil {
		t.Fatal(err)
	}
	if err := benchfmt.WriteFile(cudaPath, cudaResults); err != nil {
		t.Fatal(err)
	}

	model, err := LoadMeasuredCostModel(cpuPath, cudaPath)
	if err != nil {
		t.Fatal(err)
	}

	if !model.ShouldUseCUDA(OpAdd, 256) {
		t.Fatal("ShouldUseCUDA(add, 256) = false, want true")
	}
	if model.ShouldUseCUDA(OpAdd, 64) {
		t.Fatal("ShouldUseCUDA(add, 64) = true, want false")
	}
	if !model.ShouldUseCUDA(OpMatMul, 16*16*16) {
		t.Fatal("ShouldUseCUDA(matmul, 4096) = false, want true")
	}
	if model.ShouldUseCUDA(OpMatMul, 8*8*8) {
		t.Fatal("ShouldUseCUDA(matmul, 512) = true, want false")
	}
}
