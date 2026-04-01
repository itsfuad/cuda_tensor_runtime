package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/itsfuad/cuda_tensor_runtime/internal/benchfmt"
	"github.com/itsfuad/cuda_tensor_runtime/tensor"
)

type fixedCostModel struct {
	useCUDA bool
}

func (m fixedCostModel) ShouldUseCUDA(op tensor.OpKind, work int) bool {
	return m.useCUDA
}

func (m fixedCostModel) Observe(op tensor.OpKind, work int, backend tensor.ExecBackend, elapsed time.Duration) {
}

func main() {
	workloadsFlag := flag.String("workloads", "add,matmul,compiled_graph", "comma-separated workloads: add,matmul,compiled_graph")
	sizesFlag := flag.String("sizes", "64,128,256", "comma-separated sizes")
	itersFlag := flag.Int("iters", 10, "iterations per workload")
	formatFlag := flag.String("format", "text", "output format: text or json")
	outputFlag := flag.String("output", "", "optional file path for JSON results")
	cudaFlag := flag.Bool("cuda", false, "force CUDA dispatch")
	flag.Parse()

	if *itersFlag <= 0 {
		log.Fatal("iters must be > 0")
	}
	if *cudaFlag && !tensor.CUDAAvailable() {
		log.Fatal("cuda requested but unavailable")
	}

	sizes, err := parseIntList(*sizesFlag)
	if err != nil {
		log.Fatal(err)
	}
	workloads := parseList(*workloadsFlag)
	if len(workloads) == 0 {
		log.Fatal("no workloads selected")
	}

	prev := tensor.CurrentCostModel()
	tensor.SetCostModel(fixedCostModel{useCUDA: *cudaFlag})
	defer tensor.SetCostModel(prev)

	device := "cpu"
	if *cudaFlag {
		device = "cuda"
	}

	var results []benchfmt.Result
	for _, workload := range workloads {
		for _, size := range sizes {
			result, err := runWorkload(workload, size, *itersFlag, device)
			if err != nil {
				log.Fatal(err)
			}
			results = append(results, result)
		}
	}

	switch *formatFlag {
	case "text":
		printText(results)
	case "json":
		printJSON(results)
	default:
		log.Fatalf("unsupported format %q", *formatFlag)
	}

	if *outputFlag != "" {
		if err := benchfmt.WriteFile(*outputFlag, results); err != nil {
			log.Fatal(err)
		}
	}
}

func runWorkload(workload string, size, iterations int, device string) (benchfmt.Result, error) {
	switch workload {
	case "add":
		a := benchmarkTensor([]int{size})
		b := benchmarkTensor([]int{size})
		return measure(workload, size, device, iterations, func() error {
			_, err := tensor.Add(a, b)
			return err
		})
	case "matmul":
		a := benchmarkTensor([]int{size, size})
		b := benchmarkTensor([]int{size, size})
		return measure(workload, size, device, iterations, func() error {
			_, err := tensor.MatMul(a, b)
			return err
		})
	case "compiled_graph":
		x := tensor.NewInput("x", []int{size, size}, tensor.DeviceCPU)
		w := tensor.NewInput("w", []int{size, size}, tensor.DeviceCPU)
		bias := tensor.NewConst(benchmarkTensor([]int{size, size}))
		prog, err := tensor.Compile(tensor.ReLUNode(tensor.AddNode(tensor.MatMulNode(x, w), bias)))
		if err != nil {
			return benchfmt.Result{}, err
		}
		inputs := map[string]*tensor.Tensor{
			"x": benchmarkTensor([]int{size, size}),
			"w": benchmarkTensor([]int{size, size}),
		}
		return measure(workload, size, device, iterations, func() error {
			_, err := prog.Run(inputs)
			return err
		})
	default:
		return benchfmt.Result{}, fmt.Errorf("unsupported workload %q", workload)
	}
}

func measure(name string, size int, device string, iterations int, fn func() error) (benchfmt.Result, error) {
	start := time.Now()
	for i := 0; i < iterations; i++ {
		if err := fn(); err != nil {
			return benchfmt.Result{}, err
		}
	}
	total := time.Since(start)
	return benchfmt.Result{
		Name:       name,
		Size:       size,
		Device:     device,
		Iterations: iterations,
		TotalMs:    float64(total) / float64(time.Millisecond),
		AvgMs:      float64(total) / float64(time.Millisecond) / float64(iterations),
	}, nil
}

func printText(results []benchfmt.Result) {
	fmt.Printf("%-16s %-8s %-8s %-12s %-12s %-12s\n", "workload", "size", "device", "iterations", "total_ms", "avg_ms")
	for _, result := range results {
		fmt.Printf("%-16s %-8d %-8s %-12d %-12.3f %-12.3f\n",
			result.Name, result.Size, result.Device, result.Iterations, result.TotalMs, result.AvgMs)
	}
}

func printJSON(results []benchfmt.Result) {
	if err := json.NewEncoder(os.Stdout).Encode(results); err != nil {
		log.Fatal(err)
	}
}

func parseList(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseIntList(raw string) ([]int, error) {
	parts := parseList(raw)
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		var value int
		if _, err := fmt.Sscanf(part, "%d", &value); err != nil {
			return nil, fmt.Errorf("invalid integer %q", part)
		}
		if value <= 0 {
			return nil, fmt.Errorf("sizes must be > 0")
		}
		out = append(out, value)
	}
	return out, nil
}

func benchmarkTensor(shape []int) *tensor.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32((i % 17) - 8)
	}
	t, err := tensor.FromSlice(shape, data, tensor.DeviceCPU)
	if err != nil {
		panic(err)
	}
	return t
}
