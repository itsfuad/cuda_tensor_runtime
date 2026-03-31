# cuda_tensor_runtime

A small Go tensor runtime with a CUDA backend via cgo.

## Implemented

- float32 dense tensors
- CPU ops: add, relu, matmul
- CUDA ops: add, relu, tiled matmul
- configurable CPU/GPU execution planner with default threshold model
- demo program

## Build CUDA static library

```bash
cd cuda
make
```

This builds `libcudatensor.a`, which the Go bridge links through cgo.

## Run

```bash
go run ./cmd/demo
```

This runs with CPU fallback by default. To enable the CUDA bridge, first build the static library and then use the `cuda` build tag:

```bash
cd cuda
make
cd ..
go run -tags cuda ./cmd/demo
```

## Benchmarks

Run the CPU benchmark suite:

```bash
go test -run ^$ -bench . -benchmem ./tensor
```

Run the optional CUDA benchmarks after building the CUDA library:

```bash
cd cuda
make
cd ..
go test -tags cuda -run ^$ -bench 'CUDA|CompiledGraph|CPU' -benchmem ./tensor
```

For repeatable latency summaries outside `go test`, use the benchmark runner:

```bash
go run ./cmd/bench -workloads add,matmul,compiled_graph -sizes 64,128 -iters 20
go run ./cmd/bench -format json -workloads compiled_graph -sizes 128
go run -tags cuda ./cmd/bench -cuda -workloads matmul -sizes 128,256
go run ./cmd/bench -format json -output results/runtime_cpu.json -sizes 64,128 -iters 20
```

Compare this runtime against an external baseline JSON file that uses the same schema:

```bash
go run ./cmd/compare -base results/pytorch_cpu.json -candidate results/runtime_cpu.json -base-name pytorch -candidate-name runtime
go run ./cmd/compare -format json -base results/onnx_cpu.json -candidate results/runtime_cpu.json
```

## Notes

- The CUDA matmul kernel is a shared-memory tiled kernel.
- It is a baseline kernel, not a cuBLAS replacement.
- The default planner uses a threshold cost model; it can now be swapped for a measured cost model for research use.
- CUDA runtime libraries and NVIDIA drivers must be installed on the target machine.
