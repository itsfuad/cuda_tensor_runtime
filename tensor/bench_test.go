package tensor

import (
	"fmt"
	"testing"
)

type fixedCostModel struct {
	useCUDA bool
}

func (m fixedCostModel) ShouldUseCUDA(op OpKind, work int) bool {
	return m.useCUDA
}

func benchmarkTensor(shape []int) *Tensor {
	n, err := numel(shape)
	if err != nil {
		panic(err)
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32((i % 17) - 8)
	}
	t, err := FromSlice(shape, data, DeviceCPU)
	if err != nil {
		panic(err)
	}
	return t
}

func BenchmarkAddCPU(b *testing.B) {
	for _, size := range []int{256, 4096, 65536} {
		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			a := benchmarkTensor([]int{size})
			c := benchmarkTensor([]int{size})
			withCostModel(fixedCostModel{useCUDA: false}, func() {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					out, err := Add(a, c)
					if err != nil {
						b.Fatal(err)
					}
					if out.Data[0] == 1e9 {
						b.Fatal("unreachable")
					}
				}
			})
		})
	}
}

func BenchmarkMatMulCPU(b *testing.B) {
	for _, size := range []int{16, 64, 128} {
		b.Run(fmt.Sprintf("m=n=k=%d", size), func(b *testing.B) {
			a := benchmarkTensor([]int{size, size})
			c := benchmarkTensor([]int{size, size})
			withCostModel(fixedCostModel{useCUDA: false}, func() {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					out, err := MatMul(a, c)
					if err != nil {
						b.Fatal(err)
					}
					if out.Data[0] == 1e9 {
						b.Fatal("unreachable")
					}
				}
			})
		})
	}
}

func BenchmarkCompiledGraphCPU(b *testing.B) {
	for _, size := range []int{16, 64, 128} {
		b.Run(fmt.Sprintf("m=n=k=%d", size), func(b *testing.B) {
			x := NewInput("x", []int{size, size}, DeviceCPU)
			w := NewInput("w", []int{size, size}, DeviceCPU)
			bias := NewConst(benchmarkTensor([]int{size, size}))
			prog, err := Compile(ReLUNode(AddNode(MatMulNode(x, w), bias)))
			if err != nil {
				b.Fatal(err)
			}

			inputs := map[string]*Tensor{
				"x": benchmarkTensor([]int{size, size}),
				"w": benchmarkTensor([]int{size, size}),
			}

			withCostModel(fixedCostModel{useCUDA: false}, func() {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					out, err := prog.Run(inputs)
					if err != nil {
						b.Fatal(err)
					}
					if out.Data[0] == 1e9 {
						b.Fatal("unreachable")
					}
				}
			})
		})
	}
}

func BenchmarkAddCUDA(b *testing.B) {
	if !CUDAAvailable() {
		b.Skip("cuda unavailable")
	}
	for _, size := range []int{4096, 65536, 1 << 20} {
		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			a := benchmarkTensor([]int{size})
			c := benchmarkTensor([]int{size})
			withCostModel(fixedCostModel{useCUDA: true}, func() {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					out, err := Add(a, c)
					if err != nil {
						b.Fatal(err)
					}
					if out.Data[0] == 1e9 {
						b.Fatal("unreachable")
					}
				}
			})
		})
	}
}

func BenchmarkMatMulCUDA(b *testing.B) {
	if !CUDAAvailable() {
		b.Skip("cuda unavailable")
	}
	for _, size := range []int{64, 128, 256} {
		b.Run(fmt.Sprintf("m=n=k=%d", size), func(b *testing.B) {
			a := benchmarkTensor([]int{size, size})
			c := benchmarkTensor([]int{size, size})
			withCostModel(fixedCostModel{useCUDA: true}, func() {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					out, err := MatMul(a, c)
					if err != nil {
						b.Fatal(err)
					}
					if out.Data[0] == 1e9 {
						b.Fatal("unreachable")
					}
				}
			})
		})
	}
}
