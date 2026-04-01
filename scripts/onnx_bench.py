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
        outputs = [
            helper.make_tensor_value_info("out", TensorProto.FLOAT, [size, size])
        ]
        nodes = [helper.make_node("MatMul", ["a", "b"], ["out"])]
    elif workload == "compiled_graph":
        inputs = [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [size, size]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [size, size]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [size, size]),
        ]
        outputs = [
            helper.make_tensor_value_info("out", TensorProto.FLOAT, [size, size])
        ]
        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["mm"]),
            helper.make_node("Add", ["mm", "bias"], ["sum"]),
            helper.make_node("Relu", ["sum"], ["out"]),
        ]
    else:
        raise ValueError(f"unsupported workload: {workload}")

    graph = helper.make_graph(nodes, f"{workload}_graph", inputs, outputs)
    model = helper.make_model(
        graph,
        producer_name="cuda_tensor_runtime_bench",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=[provider],
    )
    if provider == "CUDAExecutionProvider" and provider not in session.get_providers():
        raise RuntimeError(
            "CUDAExecutionProvider requested, but the session fell back to {}".format(
                ", ".join(session.get_providers()) or "<none>"
            )
        )
    return session


def benchmark_workload(np, session, workload, size, device, iterations):
    if workload == "add":
        feeds = {
            "a": np.array(tensor_data([size]), dtype=np.float32),
            "b": np.array(tensor_data([size]), dtype=np.float32),
        }
    elif workload == "matmul":
        feeds = {
            "a": np.array(tensor_data([size, size]), dtype=np.float32).reshape(
                size, size
            ),
            "b": np.array(tensor_data([size, size]), dtype=np.float32).reshape(
                size, size
            ),
        }
    else:
        feeds = {
            "x": np.array(tensor_data([size, size]), dtype=np.float32).reshape(
                size, size
            ),
            "w": np.array(tensor_data([size, size]), dtype=np.float32).reshape(
                size, size
            ),
            "bias": np.array(tensor_data([size, size]), dtype=np.float32).reshape(
                size, size
            ),
        }

    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, feeds)
    total_ms = (time.perf_counter() - start) * 1000.0
    return result(workload, size, device, iterations, total_ms)


def main():
    parser = argparse.ArgumentParser(
        description="Emit ONNX Runtime benchmark results in the runtime JSON schema."
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

    missing = [
        name for name in ("numpy", "onnx", "onnxruntime") if find_spec(name) is None
    ]
    if missing:
        print(
            "Missing required Python packages: {}. Install them in the active environment first.".format(
                ", ".join(missing)
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    import numpy as np
    import onnx
    import onnxruntime as ort

    provider = "CPUExecutionProvider"
    if args.device == "cuda":
        ort.preload_dlls()
        provider = "CUDAExecutionProvider"
        available_providers = ort.get_available_providers()
        if provider not in available_providers:
            print(
                "CUDAExecutionProvider is unavailable. Available providers: {}".format(
                    ", ".join(available_providers) if available_providers else "<none>"
                ),
                file=sys.stderr,
            )
            sys.exit(1)

    if args.iters <= 0:
        print("--iters must be > 0", file=sys.stderr)
        sys.exit(1)

    workloads = parse_csv(args.workloads)
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
            session = make_session(onnx, ort, workload, size, provider)
            results.append(
                benchmark_workload(
                    np, session, workload, size, args.device, entry["iters"]
                )
            )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
