#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle))


def build_merged(cpu_path, cuda_path):
    cpu = load_results(cpu_path).rename(
        columns={
            "avg_ms": "cpu_avg_ms",
            "total_ms": "cpu_total_ms",
            "iterations": "cpu_iterations",
        }
    )
    cuda = load_results(cuda_path).rename(
        columns={
            "avg_ms": "cuda_avg_ms",
            "total_ms": "cuda_total_ms",
            "iterations": "cuda_iterations",
        }
    )
    merged = cpu.merge(cuda, on=["name", "size"], suffixes=("_cpu", "_cuda"))
    merged["cuda_speedup"] = merged["cpu_avg_ms"] / merged["cuda_avg_ms"]
    return merged.sort_values(["name", "size"])


def save_latency_plot(merged, output_dir):
    workloads = list(merged["name"].unique())
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(6 * len(workloads), 4.5), constrained_layout=True
    )
    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads):
        subset = merged[merged["name"] == workload]
        ax.plot(
            subset["size"], subset["cpu_avg_ms"], marker="o", linewidth=2, label="CPU"
        )
        ax.plot(
            subset["size"], subset["cuda_avg_ms"], marker="s", linewidth=2, label="CUDA"
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(workload.replace("_", " ").title())
        ax.set_xlabel("Problem Size")
        ax.set_ylabel("Average Latency (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Runtime Latency by Workload and Backend", fontsize=14)
    fig.savefig(output_dir / "runtime_latency.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "runtime_latency.svg", bbox_inches="tight")
    plt.close(fig)


def save_speedup_plot(merged, output_dir):
    workloads = list(merged["name"].unique())
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(6 * len(workloads), 4.5), constrained_layout=True
    )
    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads):
        subset = merged[merged["name"] == workload]
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.plot(
            subset["size"],
            subset["cuda_speedup"],
            marker="o",
            linewidth=2,
            color="#0c7c59",
        )
        ax.set_xscale("log", base=2)
        ax.set_title(workload.replace("_", " ").title())
        ax.set_xlabel("Problem Size")
        ax.set_ylabel("CPU / CUDA Speedup")
        ax.grid(True, alpha=0.3)

    fig.suptitle("CUDA Speedup Relative to CPU", fontsize=14)
    fig.savefig(output_dir / "runtime_speedup.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "runtime_speedup.svg", bbox_inches="tight")
    plt.close(fig)


def save_heatmap(merged, output_dir):
    pivot = merged.pivot(index="name", columns="size", values="cuda_speedup")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", center=1.0, ax=ax)
    ax.set_title("CUDA Speedup Heatmap")
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Workload")
    fig.savefig(
        output_dir / "runtime_speedup_heatmap.png", dpi=220, bbox_inches="tight"
    )
    fig.savefig(output_dir / "runtime_speedup_heatmap.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(merged, output_dir):
    lines = [
        "# Runtime Benchmark Summary",
        "",
        "| workload | size | cpu_avg_ms | cuda_avg_ms | cuda_speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.name} | {row.size} | {row.cpu_avg_ms:.6f} | {row.cuda_avg_ms:.6f} | {row.cuda_speedup:.3f} |"
        )

    top_rows = merged.sort_values("cuda_speedup", ascending=False).head(3)
    lines.extend(["", "## Highlights", ""])
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"- `{row.name}` at size `{row.size}`: CUDA is `{row.cuda_speedup:.2f}x` faster than CPU."
        )

    (output_dir / "runtime_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-style figures from runtime benchmark JSON files."
    )
    parser.add_argument("--cpu-results", required=True)
    parser.add_argument("--cuda-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = build_merged(args.cpu_results, args.cuda_results)
    save_latency_plot(merged, output_dir)
    save_speedup_plot(merged, output_dir)
    save_heatmap(merged, output_dir)
    write_summary(merged, output_dir)


if __name__ == "__main__":
    main()
