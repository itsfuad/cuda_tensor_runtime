package tensor

import "testing"

type recordingCostModel struct {
	lastOp   OpKind
	lastWork int
	useCUDA  bool
}

func (m *recordingCostModel) ShouldUseCUDA(op OpKind, work int) bool {
	m.lastOp = op
	m.lastWork = work
	return m.useCUDA
}

func TestThresholdCostModelPreservesThresholdBehavior(t *testing.T) {
	model := ThresholdCostModel{CUDAThreshold: 16}
	if model.ShouldUseCUDA(OpAdd, 15) {
		t.Fatal("ShouldUseCUDA(15) = true, want false")
	}
	if !model.ShouldUseCUDA(OpAdd, 16) {
		t.Fatal("ShouldUseCUDA(16) = false, want true")
	}
}

func TestSetCostModelNilRestoresDefault(t *testing.T) {
	SetCostModel(ThresholdCostModel{CUDAThreshold: 32})
	SetCostModel(nil)

	if !ShouldUseCUDA(OpAdd, defaultCUDAThreshold) {
		t.Fatalf("ShouldUseCUDA(default threshold) = false, want true")
	}
	if ShouldUseCUDA(OpAdd, defaultCUDAThreshold-1) {
		t.Fatalf("ShouldUseCUDA(default threshold-1) = true, want false")
	}
}

func TestAddUsesConfiguredCostModel(t *testing.T) {
	model := &recordingCostModel{}
	a, err := FromSlice([]int{4}, []float32{1, 2, 3, 4}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}
	b, err := FromSlice([]int{4}, []float32{5, 6, 7, 8}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}

	withCostModel(model, func() {
		out, err := Add(a, b)
		if err != nil {
			t.Fatalf("Add() error = %v", err)
		}
		if out.Data[0] != 6 {
			t.Fatalf("out.Data[0] = %v, want 6", out.Data[0])
		}
	})

	if model.lastOp != OpAdd {
		t.Fatalf("lastOp = %s, want %s", model.lastOp, OpAdd)
	}
	if model.lastWork != 4 {
		t.Fatalf("lastWork = %d, want 4", model.lastWork)
	}
}

func TestMatMulUsesConfiguredCostModel(t *testing.T) {
	model := &recordingCostModel{}
	a, err := FromSlice([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}
	b, err := FromSlice([]int{3, 2}, []float32{7, 8, 9, 10, 11, 12}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}

	withCostModel(model, func() {
		out, err := MatMul(a, b)
		if err != nil {
			t.Fatalf("MatMul() error = %v", err)
		}
		if out.Data[0] != 58 {
			t.Fatalf("out.Data[0] = %v, want 58", out.Data[0])
		}
	})

	if model.lastOp != OpMatMul {
		t.Fatalf("lastOp = %s, want %s", model.lastOp, OpMatMul)
	}
	if model.lastWork != 12 {
		t.Fatalf("lastWork = %d, want 12", model.lastWork)
	}
}
