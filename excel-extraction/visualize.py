"""Produce benchmark charts from `results/benchmark_results.json`.

Outputs:
  charts/load_time.png            - median load time per library (log scale)
  charts/recall_bar.png           - grouped bar of aspect recalls
  charts/recall_heatmap.png       - heatmap of (library x aspect) recall
  charts/coverage_scatter.png     - speed (load_time) vs feature coverage
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results" / "benchmark_results.json"
CHARTS = HERE / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)


def _recall(metric: dict) -> float:
    return float(metric.get("recall", 0.0))


def _acc(metric: dict) -> float:
    return float(metric.get("accuracy", 0.0))


def _recall_composite(metrics: dict) -> float:
    """Composite feature recall across charts/tables/merges/hyperlinks/comments/formulas."""
    aspects = [
        "formula_recall",
        "merged_recall",
        "table_recall",
        "chart_recall",
        "hyperlink_recall",
        "comment_recall",
        "defined_names_recall",
    ]
    return float(np.mean([_recall(metrics.get(a, {})) for a in aspects]))


def main():
    data = json.loads(RESULTS.read_text())

    libs = list(data.keys())
    times_ms = [data[l].get("load_time_ms", float("inf")) for l in libs]
    cell_acc = [_acc(data[l]["metrics"]["cell_value_accuracy"]) if ("metrics" in data[l] and "cell_value_accuracy" in data[l]["metrics"]) else 0.0 for l in libs]
    composite = [_recall_composite(data[l]["metrics"]) if ("metrics" in data[l]) else 0.0 for l in libs]

    # 1. Load time bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = sorted(range(len(libs)), key=lambda i: times_ms[i])
    libs_o = [libs[i] for i in order]
    times_o = [times_ms[i] for i in order]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(libs_o)))
    bars = ax.barh(libs_o, times_o, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Median load+extract time (ms, log scale)")
    ax.set_title("Excel extraction library speed")
    for bar, t in zip(bars, times_o):
        ax.text(t * 1.1, bar.get_y() + bar.get_height() / 2, f"{t:.1f} ms", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "load_time.png", dpi=120)
    plt.close(fig)

    # 2. Per-aspect recall bar
    aspects = [
        ("Cell value\naccuracy", "cell_value_accuracy", "accuracy"),
        ("Formulas", "formula_recall", "recall"),
        ("Merged cells", "merged_recall", "recall"),
        ("Tables", "table_recall", "recall"),
        ("Charts", "chart_recall", "recall"),
        ("Hyperlinks", "hyperlink_recall", "recall"),
        ("Comments", "comment_recall", "recall"),
        ("Defined names", "defined_names_recall", "recall"),
    ]
    arr = np.zeros((len(libs), len(aspects)))
    for i, lib in enumerate(libs):
        m = data[lib].get("metrics")
        if not m:
            continue
        for j, (_, key, kind) in enumerate(aspects):
            v = m.get(key, {}).get(kind, 0.0) if (m and key in m) else 0.0
            arr[i, j] = v
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(aspects))
    width = 0.12
    for i, lib in enumerate(libs):
        ax.bar(x + (i - len(libs) / 2) * width + width / 2, arr[i], width, label=lib)
    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in aspects], rotation=0, fontsize=9)
    ax.set_ylabel("Recall / Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-aspect extraction accuracy")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "recall_bar.png", dpi=120)
    plt.close(fig)

    # 3. Heatmap
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels([a[0] for a in aspects], fontsize=9)
    ax.set_yticks(range(len(libs)))
    ax.set_yticklabels(libs)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "black" if arr[i, j] > 0.5 else "white"
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_title("Library × Aspect extraction accuracy (green=good, red=poor)")
    fig.colorbar(im, ax=ax, label="recall/accuracy")
    fig.tight_layout()
    fig.savefig(CHARTS / "recall_heatmap.png", dpi=120)
    plt.close(fig)

    # 4. Speed vs coverage scatter
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, lib in enumerate(libs):
        ax.scatter(times_ms[i], composite[i], s=180, alpha=0.85,
                   label=f"{lib} (cell={cell_acc[i]:.2f})")
        ax.annotate(lib, (times_ms[i], composite[i]), xytext=(8, 8),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Median load+extract time (ms, log scale)")
    ax.set_ylabel("Mean feature-composite recall (formulas, charts, tables, ...)")
    ax.set_title("Speed vs feature coverage (top-right is best)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "coverage_scatter.png", dpi=120)
    plt.close(fig)

    print(f"Wrote charts to {CHARTS}/")
    for png in CHARTS.glob("*.png"):
        print(f"  {png.name}: {png.stat().st_size} bytes")


if __name__ == "__main__":
    main()
