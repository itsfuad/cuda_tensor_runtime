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

type traceRecorder struct {
	planner   string
	workload  string
	size      int
	iteration int
	samples   []benchfmt.Sample
}

func main() {
	workloadsFlag := flag.String("workloads", "add,matmul,compiled_graph", "comma-separated workloads: add,matmul,compiled_graph")
	sizesFlag := flag.String("sizes", "64,128,256", "comma-separated sizes")
	itersFlag := flag.Int("iters", 10, "iterations per workload")
	formatFlag := flag.String("format", "text", "output format: text or json")
	outputFlag := flag.String("output", "", "optional file path for JSON results")
	traceOutputFlag := flag.String("trace-output", "", "optional file path for per-iteration planner samples")
	plannerFlag := flag.String("planner", "threshold", "planner mode: cpu, cuda, threshold, measured, adaptive")
	plannerLabelFlag := flag.String("planner-label", "", "optional label to store in output instead of planner mode")
	cpuResultsFlag := flag.String("cpu-results", "", "CPU benchmark JSON for measured planner mode")
	cudaResultsFlag := flag.String("cuda-results", "", "CUDA benchmark JSON for measured planner mode")
	adaptiveMinSamplesFlag := flag.Int("adaptive-min-samples", 2, "minimum samples per backend before adaptive planner exploits")
	adaptiveExploreEveryFlag := flag.Int("adaptive-explore-every", 8, "adaptive exploration interval; 0 disables periodic exploration")
	warmupItersFlag := flag.Int("warmup-iters", 0, "iterations to run before timing; useful for adaptive steady-state evaluation")
	cudaFlag := flag.Bool("cuda", false, "force CUDA dispatch")
	flag.Parse()

	if *itersFlag <= 0 {
		log.Fatal("iters must be > 0")
	}
	if *warmupItersFlag < 0 {
		log.Fatal("warmup-iters must be >= 0")
	}
	plannerMode := *plannerFlag
	if *cudaFlag {
		plannerMode = "cuda"
	}
	if plannerMode != "cpu" && plannerMode != "threshold" && plannerMode != "measured" && plannerMode != "adaptive" && plannerMode != "cuda" {
		log.Fatalf("unsupported planner mode %q", plannerMode)
	}
	if (plannerMode == "cuda" || plannerMode == "threshold" || plannerMode == "measured" || plannerMode == "adaptive") && !tensor.CUDAAvailable() && plannerMode == "cuda" {
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
	model, err := buildCostModel(plannerMode, *cpuResultsFlag, *cudaResultsFlag, *adaptiveMinSamplesFlag, *adaptiveExploreEveryFlag)
	if err != nil {
		log.Fatal(err)
	}
	tensor.SetCostModel(model)
	defer tensor.SetCostModel(prev)
	tensor.SetExecutionObserver(nil)

	device := plannerDevice(plannerMode)
	plannerLabel := plannerMode
	if *plannerLabelFlag != "" {
		plannerLabel = *plannerLabelFlag
	}

	var results []benchfmt.Result
	var samples []benchfmt.Sample
	for _, workload := range workloads {
		for _, size := range sizes {
			recorder := &traceRecorder{planner: plannerLabel, workload: workload, size: size}
			if *traceOutputFlag != "" {
				tensor.SetExecutionObserver(recorder.observe)
			}
			result, err := runWorkload(workload, size, *itersFlag, *warmupItersFlag, device, plannerLabel, recorder)
			if err != nil {
				log.Fatal(err)
			}
			results = append(results, result)
			if *traceOutputFlag != "" {
				samples = append(samples, recorder.samples...)
				tensor.SetExecutionObserver(nil)
			}
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
	if *traceOutputFlag != "" {
		if err := benchfmt.WriteSamples(*traceOutputFlag, samples); err != nil {
			log.Fatal(err)
		}
	}
}

func runWorkload(workload string, size, iterations, warmupIters int, device string, planner string, recorder *traceRecorder) (benchfmt.Result, error) {
	switch workload {
	case "add":
		a := benchmarkTensor([]int{size})
		b := benchmarkTensor([]int{size})
		return measure(workload, size, device, planner, iterations, warmupIters, recorder, func() error {
			_, err := tensor.Add(a, b)
			return err
		})
	case "matmul":
		a := benchmarkTensor([]int{size, size})
		b := benchmarkTensor([]int{size, size})
		return measure(workload, size, device, planner, iterations, warmupIters, recorder, func() error {
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
		return measure(workload, size, device, planner, iterations, warmupIters, recorder, func() error {
			_, err := prog.Run(inputs)
			return err
		})
	default:
		return benchfmt.Result{}, fmt.Errorf("unsupported workload %q", workload)
	}
}

func measure(name string, size int, device string, planner string, iterations int, warmupIters int, recorder *traceRecorder, fn func() error) (benchfmt.Result, error) {
	for i := 0; i < warmupIters; i++ {
		if err := fn(); err != nil {
			return benchfmt.Result{}, err
		}
	}
	start := time.Now()
	for i := 0; i < iterations; i++ {
		if recorder != nil {
			recorder.iteration = i + 1
		}
		if err := fn(); err != nil {
			return benchfmt.Result{}, err
		}
	}
	total := time.Since(start)
	return benchfmt.Result{
		Name:       name,
		Size:       size,
		Device:     device,
		Planner:    planner,
		Iterations: iterations,
		TotalMs:    float64(total) / float64(time.Millisecond),
		AvgMs:      float64(total) / float64(time.Millisecond) / float64(iterations),
	}, nil
}

func printText(results []benchfmt.Result) {
	fmt.Printf("%-16s %-8s %-10s %-8s %-12s %-12s %-12s\n", "workload", "size", "planner", "device", "iterations", "total_ms", "avg_ms")
	for _, result := range results {
		fmt.Printf("%-16s %-8d %-10s %-8s %-12d %-12.3f %-12.3f\n",
			result.Name, result.Size, result.Planner, result.Device, result.Iterations, result.TotalMs, result.AvgMs)
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

func buildCostModel(plannerMode, cpuResultsPath, cudaResultsPath string, adaptiveMinSamples, adaptiveExploreEvery int) (tensor.CostModel, error) {
	switch plannerMode {
	case "cpu":
		return fixedCostModel{useCUDA: false}, nil
	case "cuda":
		return fixedCostModel{useCUDA: true}, nil
	case "threshold":
		return tensor.ThresholdCostModel{}, nil
	case "measured":
		if cpuResultsPath == "" || cudaResultsPath == "" {
			return nil, fmt.Errorf("measured planner requires -cpu-results and -cuda-results")
		}
		model, err := tensor.LoadMeasuredCostModel(cpuResultsPath, cudaResultsPath)
		if err != nil {
			return nil, err
		}
		return model, nil
	case "adaptive":
		return tensor.NewAdaptiveCostModel(tensor.ThresholdCostModel{}, adaptiveMinSamples, adaptiveExploreEvery), nil
	default:
		return nil, fmt.Errorf("unsupported planner mode %q", plannerMode)
	}
}

func plannerDevice(plannerMode string) string {
	switch plannerMode {
	case "cpu":
		return "cpu"
	case "cuda":
		return "cuda"
	default:
		return "hybrid"
	}
}

func (r *traceRecorder) observe(observation tensor.ExecutionObservation) {
	r.samples = append(r.samples, benchfmt.Sample{
		Name:      r.workload,
		Size:      r.size,
		Planner:   r.planner,
		Iteration: r.iteration,
		Backend:   string(observation.Backend),
		ElapsedMs: float64(observation.Elapsed) / float64(time.Millisecond),
	})
}
