#!/usr/bin/env python3
import argparse
import json
import sys
import time


def parse_csv(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_sizes(raw):
    return [int(item) for item in parse_csv(raw)]


def tensor_data(size):
    total = 1
    for dim in size:
        total *= dim
    return [float((i % 17) - 8) for i in range(total)]


def result(name, size, device, iterations, total_ms):
    return {
        "name": name,
        "size": size,
        "device": device,
        "iterations": iterations,
        "total_ms": total_ms,
        "avg_ms": total_ms / iterations,
    }


def make_session(onnx, ort, workload, size, provider):
    from onnx import TensorProto, helper

    if workload == "add":
        inputs = [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [size]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [size]),
        ]
        outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [size])]
        nodes = [helper.make_node("Add", ["a", "b"], ["out"])]
    elif workload == "matmul":
        inputs = [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [size, size]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [size, size]),
        ]
        outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [size, size])]
        nodes = [helper.make_node("MatMul", ["a", "b"], ["out"])]
    elif workload == "compiled_graph":
        inputs = [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [size, size]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [size, size]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [size, size]),
        ]
        outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [size, size])]
        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["mm"]),
            helper.make_node("Add", ["mm", "bias"], ["sum"]),
            helper.make_node("Relu", ["sum"], ["out"]),
        ]
    else:
        raise ValueError(f"unsupported workload: {workload}")

    graph = helper.make_graph(nodes, f"{workload}_graph", inputs, outputs)
    model = helper.make_model(graph, producer_name="cuda_tensor_runtime_bench")
    return ort.InferenceSession(
        model.SerializeToString(),
        providers=[provider],
    )


def benchmark_workload(np, session, workload, size, device, iterations):
    if workload == "add":
        feeds = {
            "a": np.array(tensor_data([size]), dtype=np.float32),
            "b": np.array(tensor_data([size]), dtype=np.float32),
        }
    elif workload == "matmul":
        feeds = {
            "a": np.array(tensor_data([size, size]), dtype=np.float32).reshape(size, size),
            "b": np.array(tensor_data([size, size]), dtype=np.float32).reshape(size, size),
        }
    else:
        feeds = {
            "x": np.array(tensor_data([size, size]), dtype=np.float32).reshape(size, size),
            "w": np.array(tensor_data([size, size]), dtype=np.float32).reshape(size, size),
            "bias": np.array(tensor_data([size, size]), dtype=np.float32).reshape(size, size),
        }

    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, feeds)
    total_ms = (time.perf_counter() - start) * 1000.0
    return result(workload, size, device, iterations, total_ms)


def main():
    parser = argparse.ArgumentParser(description="Emit ONNX Runtime benchmark results in the runtime JSON schema.")
    parser.add_argument("--workloads", default="add,matmul,compiled_graph")
    parser.add_argument("--sizes", default="64,128,256")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("ONNX, ONNX Runtime, and NumPy must be installed first.", file=sys.stderr)
        sys.exit(1)

    provider = "CPUExecutionProvider"
    if args.device == "cuda":
        provider = "CUDAExecutionProvider"
        if provider not in ort.get_available_providers():
            print("CUDAExecutionProvider is unavailable in this ONNX Runtime install.", file=sys.stderr)
            sys.exit(1)

    if args.iters <= 0:
        print("--iters must be > 0", file=sys.stderr)
        sys.exit(1)

    workloads = parse_csv(args.workloads)
    sizes = parse_sizes(args.sizes)
    results = []
    for workload in workloads:
        for size in sizes:
            session = make_session(onnx, ort, workload, size, provider)
            results.append(benchmark_workload(np, session, workload, size, args.device, args.iters))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
