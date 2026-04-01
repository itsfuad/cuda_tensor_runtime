package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/itsfuad/cuda_tensor_runtime/tensor"
)

func main() {
	costModelFlag := flag.String("cost-model", "default", "cost model: default, measured, or adaptive")
	cpuResultsFlag := flag.String("cpu-results", "", "CPU benchmark JSON for measured cost model")
	cudaResultsFlag := flag.String("cuda-results", "", "CUDA benchmark JSON for measured cost model")
	flag.Parse()

	switch *costModelFlag {
	case "measured":
		if *cpuResultsFlag == "" || *cudaResultsFlag == "" {
			log.Fatal("measured cost model requires -cpu-results and -cuda-results")
		}
		model, err := tensor.LoadMeasuredCostModel(*cpuResultsFlag, *cudaResultsFlag)
		if err != nil {
			log.Fatal(err)
		}
		tensor.SetCostModel(model)
		fmt.Println("cost model: measured")
	case "adaptive":
		tensor.SetCostModel(tensor.NewAdaptiveCostModel(tensor.CurrentCostModel(), 2, 8))
		fmt.Println("cost model: adaptive")
	case "default":
		fmt.Println("cost model: default")
	default:
		log.Fatalf("unsupported cost model %q", *costModelFlag)
	}

	fmt.Println("CUDA available:", tensor.CUDAAvailable())

	a, err := tensor.FromSlice([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}, tensor.DeviceCPU)
	if err != nil {
		log.Fatal(err)
	}
	b, err := tensor.FromSlice([]int{3, 2}, []float32{7, 8, 9, 10, 11, 12}, tensor.DeviceCPU)
	if err != nil {
		log.Fatal(err)
	}
	c, err := tensor.MatMul(a, b)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("matmul [2x3] x [3x2] =>", c.Data)

	x, err := tensor.FromSlice([]int{8}, []float32{-4, -3, -2, -1, 0, 1, 2, 3}, tensor.DeviceCPU)
	if err != nil {
		log.Fatal(err)
	}
	y, err := tensor.ReLU(x)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("relu =>", y.Data)

	p, err := tensor.FromSlice([]int{8}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, tensor.DeviceCPU)
	if err != nil {
		log.Fatal(err)
	}
	q, err := tensor.FromSlice([]int{8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, tensor.DeviceCPU)
	if err != nil {
		log.Fatal(err)
	}
	r, err := tensor.Add(p, q)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("add =>", r.Data)
}
