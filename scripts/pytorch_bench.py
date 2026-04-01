#!/usr/bin/env python3
import argparse
import json
import sys
import time
from importlib.util import find_spec
from pathlib import Path


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


def is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def workload_config(args):
    return {
        "add": {"sizes": parse_sizes(args.add_sizes), "iters": args.add_iters},
        "matmul": {
            "sizes": parse_sizes(args.matmul_sizes),
            "iters": args.matmul_iters,
        },
        "compiled_graph": {
            "sizes": parse_sizes(args.compiled_graph_sizes),
            "iters": args.compiled_graph_iters,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Emit PyTorch benchmark results in the runtime JSON schema."
    )
    parser.add_argument("--workloads", default="add,matmul,compiled_graph")
    parser.add_argument("--sizes", default="64,128,256")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--add-sizes", default="")
    parser.add_argument("--matmul-sizes", default="")
    parser.add_argument("--compiled-graph-sizes", default="")
    parser.add_argument("--add-iters", type=int, default=0)
    parser.add_argument("--matmul-iters", type=int, default=0)
    parser.add_argument("--compiled-graph-iters", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.add_sizes:
        args.add_sizes = args.sizes
    if not args.matmul_sizes:
        args.matmul_sizes = args.sizes
    if not args.compiled_graph_sizes:
        args.compiled_graph_sizes = args.sizes
    if args.add_iters <= 0:
        args.add_iters = args.iters
    if args.matmul_iters <= 0:
        args.matmul_iters = args.iters
    if args.compiled_graph_iters <= 0:
        args.compiled_graph_iters = args.iters

    missing = [name for name in ("torch",) if find_spec(name) is None]
    if missing:
        print(
            "Missing required Python packages: {}. Install them in the active environment first.".format(
                ", ".join(missing)
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    torch_spec = find_spec("torch")
    active_prefix = Path(sys.prefix).resolve()
    torch_origin = (
        Path(torch_spec.origin).resolve() if torch_spec and torch_spec.origin else None
    )
    if torch_origin is not None and not is_within(torch_origin, active_prefix):
        print(
            "The active interpreter is loading torch from a different Python environment.\n"
            f"interpreter: {sys.executable}\n"
            f"torch: {torch_origin}\n"
            "This commonly triggers duplicate OpenMP runtime errors on Windows.\n"
            "Use the Python interpreter from the environment that owns torch, or run with `python -s` to ignore user-site packages.",
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
    if args.iters <= 0:
        print("--iters must be > 0", file=sys.stderr)
        sys.exit(1)
    config = workload_config(args)
    for workload, entry in config.items():
        if entry["iters"] <= 0:
            print(f"{workload} iterations must be > 0", file=sys.stderr)
            sys.exit(1)

    results = []
    for workload in workloads:
        entry = config.get(workload)
        if entry is None:
            print(f"unsupported workload: {workload}", file=sys.stderr)
            sys.exit(1)
        for size in entry["sizes"]:
            if workload == "add":
                results.append(benchmark_add(torch, size, args.device, entry["iters"]))
            elif workload == "matmul":
                results.append(
                    benchmark_matmul(torch, size, args.device, entry["iters"])
                )
            elif workload == "compiled_graph":
                results.append(
                    benchmark_compiled_graph(torch, size, args.device, entry["iters"])
                )
            else:
                print(f"unsupported workload: {workload}", file=sys.stderr)
                sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
