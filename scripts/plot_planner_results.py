#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(path):
    return pd.read_json(path)


def load_samples(path):
    return pd.read_json(path)


def save_planner_latency_plot(results, output_dir):
    workloads = list(results["name"].unique())
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(6 * len(workloads), 4.5), constrained_layout=True
    )
    if len(workloads) == 1:
        axes = [axes]

    palette = {"threshold": "#3366cc", "measured": "#dc3912", "adaptive": "#109618"}
    for ax, workload in zip(axes, workloads):
        subset = results[results["name"] == workload]
        for planner, planner_subset in subset.groupby("planner"):
            planner_subset = planner_subset.sort_values("size")
            ax.plot(
                planner_subset["size"],
                planner_subset["avg_ms"],
                marker="o",
                linewidth=2,
                label=planner,
                color=palette.get(planner),
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(workload.replace("_", " ").title())
        ax.set_xlabel("Problem Size")
        ax.set_ylabel("Average Latency (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Planner Latency Comparison", fontsize=14)
    fig.savefig(output_dir / "planner_latency.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "planner_latency.svg", bbox_inches="tight")
    plt.close(fig)


def save_planner_speedup_plot(results, output_dir):
    threshold = results[results["planner"] == "threshold"][
        ["name", "size", "avg_ms"]
    ].rename(columns={"avg_ms": "threshold_avg_ms"})
    adaptive = results[results["planner"] == "adaptive"][
        ["name", "size", "avg_ms"]
    ].rename(columns={"avg_ms": "adaptive_avg_ms"})
    measured = results[results["planner"] == "measured"][
        ["name", "size", "avg_ms"]
    ].rename(columns={"avg_ms": "measured_avg_ms"})

    merged = threshold.merge(adaptive, on=["name", "size"]).merge(
        measured, on=["name", "size"], how="left"
    )
    merged["adaptive_speedup"] = merged["threshold_avg_ms"] / merged["adaptive_avg_ms"]
    if "measured_avg_ms" in merged:
        merged["measured_speedup"] = (
            merged["threshold_avg_ms"] / merged["measured_avg_ms"]
        )

    workloads = list(merged["name"].unique())
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(6 * len(workloads), 4.5), constrained_layout=True
    )
    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads):
        subset = merged[merged["name"] == workload].sort_values("size")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.plot(
            subset["size"],
            subset["adaptive_speedup"],
            marker="o",
            linewidth=2,
            label="threshold/adaptive",
            color="#109618",
        )
        if "measured_speedup" in subset.columns:
            ax.plot(
                subset["size"],
                subset["measured_speedup"],
                marker="s",
                linewidth=2,
                label="threshold/measured",
                color="#dc3912",
            )
        ax.set_xscale("log", base=2)
        ax.set_title(workload.replace("_", " ").title())
        ax.set_xlabel("Problem Size")
        ax.set_ylabel("Speedup vs Threshold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Planner Speedup Relative to Threshold", fontsize=14)
    fig.savefig(output_dir / "planner_speedup.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "planner_speedup.svg", bbox_inches="tight")
    plt.close(fig)


def save_adaptive_convergence_plot(samples, output_dir):
    adaptive = samples[samples["planner"] == "adaptive"].copy()
    if adaptive.empty:
        return

    adaptive["mean_elapsed_ms"] = adaptive.groupby(["name", "size"])[
        "elapsed_ms"
    ].transform(lambda s: s.expanding().mean())
    adaptive["cuda_choice"] = (adaptive["backend"] == "cuda").astype(int)
    adaptive["cuda_ratio"] = adaptive.groupby(["name", "size"])[
        "cuda_choice"
    ].transform(lambda s: s.expanding().mean())

    workloads = list(adaptive["name"].unique())
    fig, axes = plt.subplots(
        2, len(workloads), figsize=(6 * len(workloads), 8), constrained_layout=True
    )
    if len(workloads) == 1:
        axes = [[axes[0]], [axes[1]]]

    for idx, workload in enumerate(workloads):
        subset = adaptive[adaptive["name"] == workload]
        top = axes[0][idx]
        bottom = axes[1][idx]
        for size, size_subset in subset.groupby("size"):
            top.plot(
                size_subset["iteration"],
                size_subset["mean_elapsed_ms"],
                linewidth=2,
                label=f"size {size}",
            )
            bottom.plot(
                size_subset["iteration"],
                size_subset["cuda_ratio"],
                linewidth=2,
                label=f"size {size}",
            )
        top.set_title(workload.replace("_", " ").title())
        top.set_xlabel("Iteration")
        top.set_ylabel("Running Mean Latency (ms)")
        top.grid(True, alpha=0.3)
        bottom.set_xlabel("Iteration")
        bottom.set_ylabel("CUDA Selection Ratio")
        bottom.set_ylim(-0.05, 1.05)
        bottom.grid(True, alpha=0.3)
        top.legend()

    fig.suptitle("Adaptive Planner Convergence", fontsize=14)
    fig.savefig(output_dir / "adaptive_convergence.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "adaptive_convergence.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(results, samples, output_dir):
    lines = [
        "# Planner Benchmark Summary",
        "",
        "| workload | size | planner | avg_ms |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in results.sort_values(["name", "size", "planner"]).itertuples(index=False):
        lines.append(f"| {row.name} | {row.size} | {row.planner} | {row.avg_ms:.6f} |")

    if not samples.empty:
        lines.extend(
            [
                "",
                "## Adaptive Backend Mix",
                "",
                "| workload | size | cuda_ratio |",
                "| --- | ---: | ---: |",
            ]
        )
        adaptive = samples[samples["planner"] == "adaptive"]
        if not adaptive.empty:
            ratios = (
                adaptive.assign(cuda_choice=(adaptive["backend"] == "cuda").astype(int))
                .groupby(["name", "size"])["cuda_choice"]
                .mean()
                .reset_index()
            )
            for row in ratios.itertuples(index=False):
                lines.append(f"| {row.name} | {row.size} | {row.cuda_choice:.3f} |")

    (output_dir / "planner_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate planner-comparison figures from benchmark outputs."
    )
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--trace", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.concat(
        [load_results(path) for path in args.results], ignore_index=True
    )
    if "planner" not in results.columns:
        raise SystemExit(
            "planner results require benchmark JSON files produced by the updated cmd/bench"
        )
    save_planner_latency_plot(results, output_dir)
    save_planner_speedup_plot(results, output_dir)

    samples = pd.DataFrame()
    if args.trace:
        samples = load_samples(args.trace)
        save_adaptive_convergence_plot(samples, output_dir)

    write_summary(results, samples, output_dir)


if __name__ == "__main__":
    main()
