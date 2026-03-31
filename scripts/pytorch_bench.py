#!/usr/bin/env python3
import argparse
import json
import sys
import time
from importlib.util import find_spec


def parse_csv(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_sizes(raw):
    return [int(item) for item in parse_csv(raw)]


def tensor_data(count):
    return [float((i % 17) - 8) for i in range(count)]


def benchmark_add(torch, size, device, iterations):
    a = torch.tensor(tensor_data(size), dtype=torch.float32, device=device)
    b = torch.tensor(tensor_data(size), dtype=torch.float32, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        out = a + b
        if device == "cuda":
            torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return result("add", size, device, iterations, total_ms)


def benchmark_matmul(torch, size, device, iterations):
    values = tensor_data(size * size)
    a = torch.tensor(values, dtype=torch.float32, device=device).reshape(size, size)
    b = torch.tensor(values, dtype=torch.float32, device=device).reshape(size, size)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        out = torch.matmul(a, b)
        if device == "cuda":
            torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return result("matmul", size, device, iterations, total_ms)


def benchmark_compiled_graph(torch, size, device, iterations):
    values = tensor_data(size * size)
    x = torch.tensor(values, dtype=torch.float32, device=device).reshape(size, size)
    w = torch.tensor(values, dtype=torch.float32, device=device).reshape(size, size)
    bias = torch.tensor(values, dtype=torch.float32, device=device).reshape(size, size)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        out = torch.relu(torch.matmul(x, w) + bias)
        if device == "cuda":
            torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return result("compiled_graph", size, device, iterations, total_ms)


def result(name, size, device, iterations, total_ms):
    return {
        "name": name,
        "size": size,
        "device": device,
        "iterations": iterations,
        "total_ms": total_ms,
        "avg_ms": total_ms / iterations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Emit PyTorch benchmark results in the runtime JSON schema."
    )
    parser.add_argument("--workloads", default="add,matmul,compiled_graph")
    parser.add_argument("--sizes", default="64,128,256")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    missing = [name for name in ("torch",) if find_spec(name) is None]
    if missing:
        print(
            "Missing required Python packages: {}. Install them in the active environment first.".format(
                ", ".join(missing)
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        device_count = (
            torch.cuda.device_count() if hasattr(torch.cuda, "device_count") else 0
        )
        print(
            f"CUDA requested but PyTorch cannot see a CUDA device. device_count={device_count}",
            file=sys.stderr,
        )
        sys.exit(1)

    workloads = parse_csv(args.workloads)
    sizes = parse_sizes(args.sizes)
    if args.iters <= 0:
        print("--iters must be > 0", file=sys.stderr)
        sys.exit(1)

    results = []
    for workload in workloads:
        for size in sizes:
            if workload == "add":
                results.append(benchmark_add(torch, size, args.device, args.iters))
            elif workload == "matmul":
                results.append(benchmark_matmul(torch, size, args.device, args.iters))
            elif workload == "compiled_graph":
                results.append(
                    benchmark_compiled_graph(torch, size, args.device, args.iters)
                )
            else:
                print(f"unsupported workload: {workload}", file=sys.stderr)
                sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
