# cuda_tensor_runtime

A small Go tensor runtime with a CUDA backend via cgo.

## Dependencies

- Go `1.22`
- Python `3.x`
- NVIDIA driver compatible with your installed CUDA toolkit
- CUDA Toolkit with `nvcc` available on `PATH`, or `CUDA_PATH` set on Windows so the runner can find `CUDA_PATH\bin\nvcc.exe`
- On Windows, Microsoft C++ Build Tools with `cl.exe` on `PATH`
- Static archiver: `ar` or `llvm-ar` on Unix-like systems
- `make` for the Unix shell runner and manual `cuda/Makefile` builds
- PyTorch (`torch`) for `scripts/pytorch_bench.py`
- NumPy (`numpy`), ONNX (`onnx`), and ONNX Runtime (`onnxruntime`) for `scripts/onnx_bench.py`
- cuDNN version compatible with your CUDA-enabled PyTorch and ONNX Runtime builds when running the CUDA Python benchmark helpers; this repo does not pin a cuDNN version

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

This builds `libcudatensor.a`, which the non-Windows Go bridge links through cgo.

The preferred cross-platform evaluation entry point is the Python runner:

```bash
python scripts/gpu_smoke.py
```

On Windows, that runner builds `cuda\cudatensor.dll` for the installed GPU by default before running the CUDA-tagged tests.

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

You can also load a measured cost model derived from benchmark JSON:

```bash
go run -tags cuda ./cmd/demo -cost-model measured -cpu-results results/runtime_cpu.json -cuda-results results/runtime_cuda.json
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

To generate export files, plots, and a summary in one pass, use:

```bash
python scripts/gpu_smoke.py
python scripts/plot_results.py --cpu-results results/runtime_cpu.json --cuda-results results/runtime_cuda.json --output-dir results/plots
```

Compare this runtime against an external baseline JSON file that uses the same schema:

```bash
go run ./cmd/compare -base results/pytorch_cpu.json -candidate results/runtime_cpu.json -base-name pytorch -candidate-name runtime
go run ./cmd/compare -format json -base results/onnx_cpu.json -candidate results/runtime_cpu.json
```

Generate external baseline files with the helper scripts:

```bash
python3 scripts/pytorch_bench.py --device cpu --output results/pytorch_cpu.json
python3 scripts/onnx_bench.py --device cpu --output results/onnx_cpu.json
python3 scripts/pytorch_bench.py --device cuda --output results/pytorch_cuda.json
python3 scripts/onnx_bench.py --device cuda --output results/onnx_cuda.json
```

## GPU Machine Workflow

On a CUDA machine, the preferred smoke path is:

```bash
python scripts/gpu_smoke.py
```

That runner builds the required CUDA artifact for the current platform, runs tagged tests, runs the demo, exports `results/runtime_cpu.json` and `results/runtime_cuda.json`, writes `results/runtime_metadata.json`, and generates plots under `results/plots/`.

## Notes

- The CUDA matmul kernel is a shared-memory tiled kernel.
- It is a baseline kernel, not a cuBLAS replacement.
- The default planner uses a threshold cost model; it can also load a measured threshold model derived from CPU and CUDA benchmark JSON.
- CUDA runtime libraries and NVIDIA drivers must be installed on the target machine.
