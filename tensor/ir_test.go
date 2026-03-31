package tensor

import (
	"strings"
	"testing"
)

func TestCompileInfersOutputShape(t *testing.T) {
	x := NewInput("x", []int{2, 3}, DeviceCPU)
	w := NewInput("w", []int{3, 2}, DeviceCPU)
	biasTensor, err := FromSlice([]int{2, 2}, []float32{1, 1, 1, 1}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}

	prog, err := Compile(ReLUNode(AddNode(MatMulNode(x, w), NewConst(biasTensor))))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	out := prog.OutputType()
	if out.DType != DTypeFloat32 {
		t.Fatalf("dtype = %s, want %s", out.DType, DTypeFloat32)
	}
	if !sameShape(out.Shape, []int{2, 2}) {
		t.Fatalf("shape = %v, want [2 2]", out.Shape)
	}
}

func TestCompileFusesAddReLU(t *testing.T) {
	x := NewInput("x", []int{4}, DeviceCPU)
	y := NewInput("y", []int{4}, DeviceCPU)

	prog, err := Compile(ReLUNode(AddNode(x, y)))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := prog.root.(*AddReLUExpr); !ok {
		t.Fatalf("compiled root = %T, want *AddReLUExpr", prog.root)
	}
}

func TestCompileRejectsShapeMismatch(t *testing.T) {
	x := NewInput("x", []int{2, 3}, DeviceCPU)
	y := NewInput("y", []int{4, 3}, DeviceCPU)

	_, err := Compile(AddNode(x, y))
	if err == nil {
		t.Fatal("Compile() error = nil, want shape mismatch")
	}
	if !strings.Contains(err.Error(), "add shape mismatch") {
		t.Fatalf("Compile() error = %v, want add shape mismatch", err)
	}
}

func TestProgramRunExecutesGraph(t *testing.T) {
	x := NewInput("x", []int{2, 3}, DeviceCPU)
	w := NewInput("w", []int{3, 2}, DeviceCPU)
	bias, err := FromSlice([]int{2, 2}, []float32{1, -20, 2, 3}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}

	prog, err := Compile(ReLUNode(AddNode(MatMulNode(x, w), NewConst(bias))))
	if err != nil {
		t.Fatal(err)
	}

	inputX, err := FromSlice([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}
	inputW, err := FromSlice([]int{3, 2}, []float32{7, 8, 9, 10, 11, 12}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}

	out, err := prog.Run(map[string]*Tensor{
		"x": inputX,
		"w": inputW,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	want := []float32{59, 44, 141, 157}
	if len(out.Data) != len(want) {
		t.Fatalf("len(out.Data) = %d, want %d", len(out.Data), len(want))
	}
	for i, got := range out.Data {
		if got != want[i] {
			t.Fatalf("out.Data[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestProgramRunRejectsWrongInputShape(t *testing.T) {
	x := NewInput("x", []int{2, 3}, DeviceCPU)
	prog, err := Compile(ReLUNode(x))
	if err != nil {
		t.Fatal(err)
	}

	input, err := FromSlice([]int{6}, []float32{1, 2, 3, 4, 5, 6}, DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}

	_, err = prog.Run(map[string]*Tensor{"x": input})
	if err == nil {
		t.Fatal("Run() error = nil, want shape mismatch")
	}
	if !strings.Contains(err.Error(), "shape mismatch") {
		t.Fatalf("Run() error = %v, want shape mismatch", err)
	}
}
