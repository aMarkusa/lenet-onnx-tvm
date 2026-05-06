"""Generate report-ready tables, plots, and a summary from the ONNX vs TVM results JSON.

Reads ``artifacts/results/onnx_vs_tvm_verification.json`` and writes:

  artifacts/results/comparison_accuracy.csv
  artifacts/results/comparison_performance.csv
  artifacts/results/comparison_model_size.csv
  artifacts/results/comparison_graph_structure.csv
  artifacts/results/comparison_tables.md
  artifacts/results/summary.txt
  artifacts/figures/accuracy_comparison.png
  artifacts/figures/latency_comparison.png
  artifacts/figures/model_size_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "artifacts" / "results" / "onnx_vs_tvm_verification.json"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
ONNX_DIR = PROJECT_ROOT / "artifacts" / "onnx"


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bytes_to_kb(num_bytes: int) -> float:
    return round(num_bytes / 1024.0, 2)


def onnx_total_bytes(results: dict) -> int:
    """Total on-disk ONNX size including any external-data file next to the .onnx."""
    summary = results["model_summary"]
    total = int(summary["onnx_file_bytes"])
    onnx_path_str = results.get("onnx_model_path", "")
    onnx_path = Path(onnx_path_str)
    if not onnx_path.is_absolute():
        onnx_path = PROJECT_ROOT / onnx_path
    if onnx_path.is_file():
        external_data = onnx_path.with_suffix(onnx_path.suffix + ".data")
        if external_data.is_file():
            total += external_data.stat().st_size
    return total


def df_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([header_row, separator, *data_rows])


def build_accuracy_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "ONNX accuracy", "Value": round(results["onnx_accuracy"], 4)},
            {"Metric": "TVM accuracy", "Value": round(results["tvm_accuracy"], 4)},
            {"Metric": "Prediction match rate", "Value": round(results["prediction_match_rate"], 4)},
        ]
    )


def build_performance_table(results: dict) -> pd.DataFrame:
    perf = results["performance"]
    rows = []
    for label, key in (("ONNX", "onnx_runtime"), ("TVM", "tvm")):
        backend = perf[key]
        rows.append(
            {
                "Backend": label,
                "Mean latency (ms)": round(backend["mean_latency_ms"], 3),
                "Median latency (ms)": round(backend["median_latency_ms"], 3),
                "P95 latency (ms)": round(backend["p95_latency_ms"], 3),
                "Throughput (samples/s)": round(backend["throughput_samples_per_sec"], 1),
            }
        )
    return pd.DataFrame(rows)


def build_model_size_table(results: dict) -> pd.DataFrame:
    summary = results["model_summary"]
    onnx_kb = bytes_to_kb(onnx_total_bytes(results))
    tvm_kb = bytes_to_kb(summary["tvm_artifact_sizes"]["total_bytes"])
    return pd.DataFrame(
        [
            {"Representation": "ONNX (.onnx + external data)", "Size (KB)": onnx_kb},
            {"Representation": "TVM (lib + graph + params + relay)", "Size (KB)": tvm_kb},
        ]
    )


def build_graph_structure_table(results: dict) -> pd.DataFrame:
    summary = results["model_summary"]
    op_counts = summary["onnx_op_type_counts"]
    op_summary = ", ".join(f"{op}={n}" for op, n in sorted(op_counts.items()))
    return pd.DataFrame(
        [
            {"Metric": "ONNX graph nodes", "Value": summary["onnx_num_graph_nodes"]},
            {"Metric": "ONNX op-type counts", "Value": op_summary},
            {"Metric": "Relay calls (post import)", "Value": summary["relay_call_count"]},
            {"Metric": "Relay text lines", "Value": summary["relay_text_num_lines"]},
            {"Metric": "TVM compiled graph nodes", "Value": summary["tvm_graph_num_nodes"]},
            {"Metric": "TVM graph arg nodes", "Value": summary["tvm_graph_num_arg_nodes"]},
        ]
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_markdown_tables(tables: dict[str, pd.DataFrame], path: Path) -> None:
    sections = []
    for title, df in tables.items():
        sections.append(f"## {title}\n\n{df_to_markdown(df)}\n")
    path.write_text("# ONNX vs TVM Comparison\n\n" + "\n".join(sections), encoding="utf-8")


def plot_accuracy(results: dict, path: Path) -> None:
    labels = ["ONNX", "TVM"]
    values = [results["onnx_accuracy"], results["tvm_accuracy"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0.95, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("MNIST Test Accuracy")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency(results: dict, path: Path) -> None:
    perf = results["performance"]
    labels = ["ONNX", "TVM"]
    means = [perf["onnx_runtime"]["mean_latency_ms"], perf["tvm"]["mean_latency_ms"]]
    medians = [perf["onnx_runtime"]["median_latency_ms"], perf["tvm"]["median_latency_ms"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], means, width, label="Mean", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], medians, width, label="Median", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-sample Inference Latency (batch=1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_size(results: dict, path: Path) -> None:
    summary = results["model_summary"]
    labels = ["ONNX (graph + data)", "TVM (lib + graph + params + relay)"]
    sizes_kb = [
        bytes_to_kb(onnx_total_bytes(results)),
        bytes_to_kb(summary["tvm_artifact_sizes"]["total_bytes"]),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, sizes_kb, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Size (KB)")
    ax.set_title("Serialized Model / Artifact Size")
    for bar, value in zip(bars, sizes_kb):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_summary_lines(results: dict) -> list[str]:
    summary = results["model_summary"]
    perf = results["performance"]
    onnx_kb = bytes_to_kb(onnx_total_bytes(results))
    tvm_kb = bytes_to_kb(summary["tvm_artifact_sizes"]["total_bytes"])
    onnx_mean = perf["onnx_runtime"]["mean_latency_ms"]
    tvm_mean = perf["tvm"]["mean_latency_ms"]
    onnx_median = perf["onnx_runtime"]["median_latency_ms"]
    tvm_median = perf["tvm"]["median_latency_ms"]
    onnx_tput = perf["onnx_runtime"]["throughput_samples_per_sec"]
    tvm_tput = perf["tvm"]["throughput_samples_per_sec"]

    size_ratio = tvm_kb / onnx_kb if onnx_kb > 0 else float("nan")
    mean_ratio = tvm_mean / onnx_mean if onnx_mean > 0 else float("nan")
    median_ratio = tvm_median / onnx_median if onnx_median > 0 else float("nan")

    return [
        f"- Accuracy is unchanged: ONNX = {results['onnx_accuracy']:.4f}, "
        f"TVM = {results['tvm_accuracy']:.4f} on {results['num_samples']} test samples; "
        f"prediction match rate = {results['prediction_match_rate']:.4f} "
        f"({results['num_prediction_mismatches']} mismatches).",
        f"- Logit outputs are numerically equivalent: max abs diff = "
        f"{results['max_abs_logit_difference']:.2e}, "
        f"mean abs diff = {results['mean_abs_logit_difference']:.2e}.",
        f"- Serialized size grew with TVM: ONNX = {onnx_kb:.1f} KB, "
        f"TVM total artifacts = {tvm_kb:.1f} KB ({size_ratio:.1f}x larger). "
        f"The TVM total includes the compiled kernel library "
        f"({bytes_to_kb(summary['tvm_artifact_sizes']['library_bytes']):.1f} KB) "
        f"in addition to graph JSON, params, and Relay text.",
        f"- Latency at batch size {results['batch_size_eval']} is higher with TVM: "
        f"mean = {tvm_mean:.3f} ms vs {onnx_mean:.3f} ms ({mean_ratio:.1f}x), "
        f"median = {tvm_median:.3f} ms vs {onnx_median:.3f} ms ({median_ratio:.1f}x).",
        f"- Throughput follows the same direction: ONNX = {onnx_tput:.1f} samples/s, "
        f"TVM = {tvm_tput:.1f} samples/s.",
        f"- Graph node count did not shrink after compilation: ONNX has "
        f"{summary['onnx_num_graph_nodes']} graph nodes, the imported Relay program has "
        f"{summary['relay_call_count']} call nodes, and the compiled TVM graph has "
        f"{summary['tvm_graph_num_nodes']} executor nodes.",
        f"- ONNX operator mix: "
        f"{', '.join(f'{op}={n}' for op, n in sorted(summary['onnx_op_type_counts'].items()))}.",
        f"- TVM compile time (opt_level={results['opt_level']}, target={results['target']}): "
        f"{summary['tvm_compile_time_s']:.2f} s.",
        f"- For this small CNN at batch size {results['batch_size_eval']}, "
        f"TVM did not produce a smaller model or lower latency than ONNX Runtime.",
    ]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results(RESULTS_PATH)

    accuracy_df = build_accuracy_table(results)
    performance_df = build_performance_table(results)
    model_size_df = build_model_size_table(results)
    graph_df = build_graph_structure_table(results)

    write_csv(accuracy_df, RESULTS_DIR / "comparison_accuracy.csv")
    write_csv(performance_df, RESULTS_DIR / "comparison_performance.csv")
    write_csv(model_size_df, RESULTS_DIR / "comparison_model_size.csv")
    write_csv(graph_df, RESULTS_DIR / "comparison_graph_structure.csv")

    write_markdown_tables(
        {
            "Accuracy": accuracy_df,
            "Performance": performance_df,
            "Model Size": model_size_df,
            "Graph Structure": graph_df,
        },
        RESULTS_DIR / "comparison_tables.md",
    )

    plot_accuracy(results, FIGURES_DIR / "accuracy_comparison.png")
    plot_latency(results, FIGURES_DIR / "latency_comparison.png")
    plot_model_size(results, FIGURES_DIR / "model_size_comparison.png")

    summary_lines = build_summary_lines(results)
    (RESULTS_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Wrote tables, plots, and summary:")
    for path in [
        RESULTS_DIR / "comparison_accuracy.csv",
        RESULTS_DIR / "comparison_performance.csv",
        RESULTS_DIR / "comparison_model_size.csv",
        RESULTS_DIR / "comparison_graph_structure.csv",
        RESULTS_DIR / "comparison_tables.md",
        RESULTS_DIR / "summary.txt",
        FIGURES_DIR / "accuracy_comparison.png",
        FIGURES_DIR / "latency_comparison.png",
        FIGURES_DIR / "model_size_comparison.png",
    ]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
