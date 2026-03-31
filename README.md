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

## Notes

- The CUDA matmul kernel is a shared-memory tiled kernel.
- It is a baseline kernel, not a cuBLAS replacement.
- The default planner uses a threshold cost model; it can now be swapped for a measured cost model for research use.
- CUDA runtime libraries and NVIDIA drivers must be installed on the target machine.
