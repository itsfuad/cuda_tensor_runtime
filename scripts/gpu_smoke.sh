#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

echo "Building CUDA static library"
( cd cuda && make )

echo "Running tests with CUDA tag"
go test -tags cuda ./...

echo "Running demo with CUDA tag"
go run -tags cuda ./cmd/demo

echo "Running tensor benchmarks with CUDA tag"
go test -tags cuda -run '^$' -bench . -benchmem ./tensor

echo "Writing CPU and CUDA benchmark JSON results"
mkdir -p results
go run ./cmd/bench -format json -output results/runtime_cpu.json -workloads add,matmul,compiled_graph -sizes 64,128,256 -iters 20
go run -tags cuda ./cmd/bench -cuda -format json -output results/runtime_cuda.json -workloads add,matmul,compiled_graph -sizes 64,128,256 -iters 20

echo "Done"
