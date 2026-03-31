package tensor

import "fmt"

type DType string

const (
	DTypeFloat32 DType = "float32"
)

type TensorType struct {
	Shape  []int
	DType  DType
	Device Device
}

type Expr interface {
	infer(*inferState) (TensorType, error)
	eval(map[string]*Tensor) (*Tensor, error)
}

type inferState struct {
	types map[Expr]TensorType
}

type InputExpr struct {
	Name string
	Type TensorType
}

type ConstExpr struct {
	Value *Tensor
}

type AddExpr struct {
	Left  Expr
	Right Expr
}

type AddReLUExpr struct {
	Left  Expr
	Right Expr
}

type ReLUExpr struct {
	Input Expr
}

type MatMulExpr struct {
	Left  Expr
	Right Expr
}

type Program struct {
	root  Expr
	types map[Expr]TensorType
}

func NewInput(name string, shape []int, device Device) *InputExpr {
	return &InputExpr{
		Name: name,
		Type: TensorType{
			Shape:  cloneInts(shape),
			DType:  DTypeFloat32,
			Device: device,
		},
	}
}

func NewConst(value *Tensor) *ConstExpr {
	return &ConstExpr{Value: value.Clone()}
}

func AddNode(left, right Expr) *AddExpr {
	return &AddExpr{Left: left, Right: right}
}

func ReLUNode(input Expr) *ReLUExpr {
	return &ReLUExpr{Input: input}
}

func MatMulNode(left, right Expr) *MatMulExpr {
	return &MatMulExpr{Left: left, Right: right}
}

func Compile(root Expr) (*Program, error) {
	if root == nil {
		return nil, fmt.Errorf("root expression cannot be nil")
	}
	root = optimizeExpr(root)
	state := &inferState{types: make(map[Expr]TensorType)}
	if _, err := root.infer(state); err != nil {
		return nil, err
	}
	return &Program{root: root, types: state.types}, nil
}

func (p *Program) OutputType() TensorType {
	return cloneTensorType(p.types[p.root])
}

func (p *Program) TypeOf(expr Expr) (TensorType, bool) {
	typ, ok := p.types[expr]
	if !ok {
		return TensorType{}, false
	}
	return cloneTensorType(typ), true
}

func (p *Program) Run(inputs map[string]*Tensor) (*Tensor, error) {
	if p == nil {
		return nil, fmt.Errorf("program is nil")
	}
	return p.root.eval(inputs)
}

func (e *InputExpr) infer(state *inferState) (TensorType, error) {
	if e == nil {
		return TensorType{}, fmt.Errorf("input expression is nil")
	}
	if e.Name == "" {
		return TensorType{}, fmt.Errorf("input name cannot be empty")
	}
	if e.Type.DType == "" {
		e.Type.DType = DTypeFloat32
	}
	if _, err := numel(e.Type.Shape); err != nil {
		return TensorType{}, fmt.Errorf("input %q: %w", e.Name, err)
	}
	typ := cloneTensorType(e.Type)
	state.types[e] = typ
	return typ, nil
}

func (e *InputExpr) eval(inputs map[string]*Tensor) (*Tensor, error) {
	if inputs == nil {
		return nil, fmt.Errorf("missing inputs")
	}
	t, ok := inputs[e.Name]
	if !ok {
		return nil, fmt.Errorf("missing input %q", e.Name)
	}
	if err := validateTensorMatchesType(t, e.Type); err != nil {
		return nil, fmt.Errorf("input %q: %w", e.Name, err)
	}
	return t.Clone(), nil
}

func (e *ConstExpr) infer(state *inferState) (TensorType, error) {
	if e == nil || e.Value == nil {
		return TensorType{}, fmt.Errorf("const value cannot be nil")
	}
	typ := TensorType{
		Shape:  cloneInts(e.Value.Shape),
		DType:  DTypeFloat32,
		Device: e.Value.Device,
	}
	state.types[e] = typ
	return typ, nil
}

func (e *ConstExpr) eval(map[string]*Tensor) (*Tensor, error) {
	if e == nil || e.Value == nil {
		return nil, fmt.Errorf("const value cannot be nil")
	}
	return e.Value.Clone(), nil
}

func (e *AddExpr) infer(state *inferState) (TensorType, error) {
	left, err := e.Left.infer(state)
	if err != nil {
		return TensorType{}, err
	}
	right, err := e.Right.infer(state)
	if err != nil {
		return TensorType{}, err
	}
	if !sameShape(left.Shape, right.Shape) {
		return TensorType{}, fmt.Errorf("add shape mismatch: %v vs %v", left.Shape, right.Shape)
	}
	if left.DType != right.DType {
		return TensorType{}, fmt.Errorf("add dtype mismatch: %s vs %s", left.DType, right.DType)
	}
	typ := TensorType{Shape: cloneInts(left.Shape), DType: left.DType, Device: left.Device}
	state.types[e] = typ
	return typ, nil
}

func (e *AddExpr) eval(inputs map[string]*Tensor) (*Tensor, error) {
	left, err := e.Left.eval(inputs)
	if err != nil {
		return nil, err
	}
	right, err := e.Right.eval(inputs)
	if err != nil {
		return nil, err
	}
	return Add(left, right)
}

func (e *AddReLUExpr) infer(state *inferState) (TensorType, error) {
	left, err := e.Left.infer(state)
	if err != nil {
		return TensorType{}, err
	}
	right, err := e.Right.infer(state)
	if err != nil {
		return TensorType{}, err
	}
	if !sameShape(left.Shape, right.Shape) {
		return TensorType{}, fmt.Errorf("addrelu shape mismatch: %v vs %v", left.Shape, right.Shape)
	}
	if left.DType != right.DType {
		return TensorType{}, fmt.Errorf("addrelu dtype mismatch: %s vs %s", left.DType, right.DType)
	}
	typ := TensorType{Shape: cloneInts(left.Shape), DType: left.DType, Device: left.Device}
	state.types[e] = typ
	return typ, nil
}

func (e *AddReLUExpr) eval(inputs map[string]*Tensor) (*Tensor, error) {
	left, err := e.Left.eval(inputs)
	if err != nil {
		return nil, err
	}
	right, err := e.Right.eval(inputs)
	if err != nil {
		return nil, err
	}
	return AddReLU(left, right)
}

func (e *ReLUExpr) infer(state *inferState) (TensorType, error) {
	input, err := e.Input.infer(state)
	if err != nil {
		return TensorType{}, err
	}
	typ := cloneTensorType(input)
	state.types[e] = typ
	return typ, nil
}

func (e *ReLUExpr) eval(inputs map[string]*Tensor) (*Tensor, error) {
	input, err := e.Input.eval(inputs)
	if err != nil {
		return nil, err
	}
	return ReLU(input)
}

func (e *MatMulExpr) infer(state *inferState) (TensorType, error) {
	left, err := e.Left.infer(state)
	if err != nil {
		return TensorType{}, err
	}
	right, err := e.Right.infer(state)
	if err != nil {
		return TensorType{}, err
	}
	if len(left.Shape) != 2 || len(right.Shape) != 2 {
		return TensorType{}, fmt.Errorf("matmul requires 2D tensors")
	}
	if left.Shape[1] != right.Shape[0] {
		return TensorType{}, fmt.Errorf("matmul inner dimension mismatch: %d != %d", left.Shape[1], right.Shape[0])
	}
	if left.DType != right.DType {
		return TensorType{}, fmt.Errorf("matmul dtype mismatch: %s vs %s", left.DType, right.DType)
	}
	typ := TensorType{
		Shape:  []int{left.Shape[0], right.Shape[1]},
		DType:  left.DType,
		Device: left.Device,
	}
	state.types[e] = typ
	return typ, nil
}

func (e *MatMulExpr) eval(inputs map[string]*Tensor) (*Tensor, error) {
	left, err := e.Left.eval(inputs)
	if err != nil {
		return nil, err
	}
	right, err := e.Right.eval(inputs)
	if err != nil {
		return nil, err
	}
	return MatMul(left, right)
}

func cloneTensorType(typ TensorType) TensorType {
	typ.Shape = cloneInts(typ.Shape)
	return typ
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func validateTensorMatchesType(t *Tensor, typ TensorType) error {
	if t == nil {
		return fmt.Errorf("tensor is nil")
	}
	if !sameShape(t.Shape, typ.Shape) {
		return fmt.Errorf("shape mismatch: got=%v want=%v", t.Shape, typ.Shape)
	}
	if typ.Device != "" && t.Device != typ.Device {
		return fmt.Errorf("device mismatch: got=%s want=%s", t.Device, typ.Device)
	}
	return nil
}

func optimizeExpr(expr Expr) Expr {
	switch e := expr.(type) {
	case *AddExpr:
		return &AddExpr{
			Left:  optimizeExpr(e.Left),
			Right: optimizeExpr(e.Right),
		}
	case *ReLUExpr:
		input := optimizeExpr(e.Input)
		if add, ok := input.(*AddExpr); ok {
			return &AddReLUExpr{
				Left:  add.Left,
				Right: add.Right,
			}
		}
		return &ReLUExpr{Input: input}
	case *MatMulExpr:
		return &MatMulExpr{
			Left:  optimizeExpr(e.Left),
			Right: optimizeExpr(e.Right),
		}
	case *ConstExpr, *InputExpr, *AddReLUExpr:
		return expr
	default:
		return expr
	}
}
