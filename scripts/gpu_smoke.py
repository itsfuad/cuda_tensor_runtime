#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CUDA_DIR = ROOT / "cuda"
RESULTS_DIR = ROOT / "results"


def parse_int_list(raw):
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def run_checked(cmd, *, env=None, cwd=None, capture=False):
    print("$", " ".join(shlex.quote(str(part)) for part in cmd), flush=True)
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd or ROOT,
        env=env,
        text=True,
        capture_output=capture,
        check=True,
    )


def find_nvcc():
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate = Path(cuda_path) / "bin" / "nvcc.exe"
        if candidate.exists():
            return str(candidate)
    return shutil.which("nvcc.exe") or shutil.which("nvcc")


def find_vcvars():
    candidates = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def go_env(cuda_enabled):
    env = os.environ.copy()
    if platform.system() == "Windows":
        env["CGO_ENABLED"] = "0"
        cuda_path = env.get("CUDA_PATH")
        if cuda_path:
            env["PATH"] = os.path.join(cuda_path, "bin") + os.pathsep + env["PATH"]
    elif cuda_enabled:
        env.setdefault("CGO_ENABLED", "1")
    return env


def run_windows_nvcc(nvcc, arch):
    cuda_dll = CUDA_DIR / "cudatensor.dll"
    if shutil.which("cl.exe"):
        run_checked(
            [
                nvcc,
                "-shared",
                "-O3",
                "-Xcompiler",
                "/MD",
                "-DTENSOR_CUDA_BUILD_DLL",
                f"-arch={arch}",
                CUDA_DIR / "kernels.cu",
                "-o",
                cuda_dll,
            ]
        )
        return

    vcvars = find_vcvars()
    if not vcvars:
        raise RuntimeError("Unable to find cl.exe or vcvars64.bat for the Windows CUDA build.")

    command = (
        f'call "{vcvars}" >nul && '
        f'"{nvcc}" -shared -O3 -Xcompiler /MD -DTENSOR_CUDA_BUILD_DLL '
        f'-arch={arch} "{CUDA_DIR / "kernels.cu"}" -o "{cuda_dll}"'
    )
    run_checked(["cmd", "/d", "/s", "/c", command])


def build_cuda_artifact(arch):
    if platform.system() == "Windows":
        nvcc = find_nvcc()
        if not nvcc:
            raise RuntimeError("Unable to find nvcc. Set CUDA_PATH or add nvcc to PATH.")
        run_windows_nvcc(nvcc, arch)
        return

    run_checked(["make", f"ARCH={arch}"], cwd=CUDA_DIR)


def merge_results(chunks):
    merged = [item for chunk in chunks for item in chunk]
    merged.sort(key=lambda item: (item["name"], item["size"], item["device"]))
    return merged


def run_bench_export(device, config):
    chunks = []
    for workload, entry in config.items():
        cmd = ["go", "run"]
        if device == "cuda":
            cmd += ["-tags", "cuda"]
        cmd += [
            "./cmd/bench",
            "-format",
            "json",
            "-workloads",
            workload,
            "-sizes",
            ",".join(str(size) for size in entry["sizes"]),
            "-iters",
            str(entry["iters"]),
        ]
        if device == "cuda":
            cmd.append("-cuda")
        proc = run_checked(cmd, env=go_env(cuda_enabled=device == "cuda"), capture=True)
        chunks.append(json.loads(proc.stdout))
    return merge_results(chunks)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def capture_command_output(cmd):
    try:
        return run_checked(cmd, capture=True).stdout.strip()
    except Exception:
        return ""


def write_metadata(path, config, arch):
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "toolchain": {
            "go_version": capture_command_output(["go", "version"]),
            "nvcc_version": capture_command_output([find_nvcc() or "nvcc", "--version"]),
            "nvidia_smi": capture_command_output(["nvidia-smi"]),
        },
        "bench_config": {
            "arch": arch,
            "workloads": config,
        },
    }
    write_json(path, metadata)


def run_plot_script(cpu_output, cuda_output, output_dir):
    run_checked(
        [
            sys.executable,
            ROOT / "scripts" / "plot_results.py",
            "--cpu-results",
            cpu_output,
            "--cuda-results",
            cuda_output,
            "--output-dir",
            output_dir,
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Cross-platform CUDA smoke runner with JSON export and plotting.")
    parser.add_argument("--arch", default="native" if platform.system() == "Windows" else "sm_75")
    parser.add_argument("--cpu-output", default=str(RESULTS_DIR / "runtime_cpu.json"))
    parser.add_argument("--cuda-output", default=str(RESULTS_DIR / "runtime_cuda.json"))
    parser.add_argument("--metadata-output", default=str(RESULTS_DIR / "runtime_metadata.json"))
    parser.add_argument("--plots-dir", default=str(RESULTS_DIR / "plots"))
    parser.add_argument("--add-sizes", default="256,1024,4096,16384,65536")
    parser.add_argument("--matmul-sizes", default="64,128,256,512")
    parser.add_argument("--compiled-graph-sizes", default="64,128,256,512")
    parser.add_argument("--add-iters", type=int, default=5000)
    parser.add_argument("--matmul-iters", type=int, default=50)
    parser.add_argument("--compiled-graph-iters", type=int, default=50)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    config = {
        "add": {"sizes": parse_int_list(args.add_sizes), "iters": args.add_iters},
        "matmul": {"sizes": parse_int_list(args.matmul_sizes), "iters": args.matmul_iters},
        "compiled_graph": {
            "sizes": parse_int_list(args.compiled_graph_sizes),
            "iters": args.compiled_graph_iters,
        },
    }

    build_cuda_artifact(args.arch)

    if not args.skip_tests:
        run_checked(["go", "test", "-tags", "cuda", "./..."], env=go_env(cuda_enabled=True))

    if not args.skip_demo:
        run_checked(["go", "run", "-tags", "cuda", "./cmd/demo"], env=go_env(cuda_enabled=True))

    if args.skip_bench:
        return

    cpu_results = run_bench_export("cpu", config)
    cuda_results = run_bench_export("cuda", config)
    cpu_output = Path(args.cpu_output)
    cuda_output = Path(args.cuda_output)
    write_json(cpu_output, cpu_results)
    write_json(cuda_output, cuda_results)
    write_metadata(Path(args.metadata_output), config, args.arch)

    if not args.skip_plots:
        run_plot_script(cpu_output, cuda_output, Path(args.plots_dir))


if __name__ == "__main__":
    main()
