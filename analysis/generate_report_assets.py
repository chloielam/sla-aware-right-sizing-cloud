#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

MAIN_POLICIES = ["static", "reactive", "util_base", "forecast_only", "sla_aware"]
FOCUSED_POLICIES = ["util_base", "forecast_only", "sla_aware"]
ABLATION_POLICIES = [
    "sla_aware",
    "sla_aware_no_prediction",
    "sla_aware_no_queue",
    "sla_aware_no_latency",
    "sla_aware_low_aggr",
    "sla_aware_high_aggr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready figures and tables from benchmark outputs.")
    parser.add_argument("--results-dir", default="data/processed", help="directory containing benchmark_results.csv")
    parser.add_argument("--output-dir", default="analysis/output", help="directory for figures and derived tables")
    parser.add_argument("--window-size", type=int, default=30, help="window size for representative time-series plots")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def discover_result_rows(results_dir: Path) -> list[dict]:
    preferred_reports = [
        "report_main_controlled",
        "report_harder_qps",
        "report_harder_boot",
        "report_harder_capacity",
        "report_ablation",
    ]
    preferred_paths = [results_dir / name / "benchmark_results.csv" for name in preferred_reports]
    existing_preferred = [path for path in preferred_paths if path.exists()]
    if existing_preferred:
        rows: list[dict] = []
        for path in existing_preferred:
            rows.extend(read_csv(path))
        return rows

    report_paths = sorted(results_dir.glob("report_*/benchmark_results.csv"))
    if report_paths:
        rows: list[dict] = []
        for path in report_paths:
            rows.extend(read_csv(path))
        return rows

    nested_paths = sorted(results_dir.glob("*/benchmark_results.csv"))
    if nested_paths:
        rows: list[dict] = []
        for path in nested_paths:
            rows.extend(read_csv(path))
        return rows

    direct = results_dir / "benchmark_results.csv"
    if direct.exists():
        return read_csv(direct)
    return []


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def to_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def write_summary_table(rows: list[dict], output_dir: Path) -> None:
    table_path = output_dir / "summary_table.csv"
    fields = [
        "scenario_name",
        "policy",
        "prediction_algorithm",
        "cost_estimate",
        "cost_instance_hours",
        "sla_violation_rate",
        "tail_latency_p99_ms",
        "resource_efficiency",
        "steps",
        "qps_scale",
        "sla_threshold_ms",
        "boot_delay_steps",
        "max_instances",
    ]
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    md_path = output_dir / "summary_table.md"
    headers = ["scenario", "policy", "algorithm", "cost($)", "inst_hr", "viol_rate", "p99(ms)", "eff"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in sorted(rows, key=lambda x: (x.get("scenario_name", ""), float(x.get("cost_estimate", 0.0)))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scenario_name", "")),
                    str(row.get("policy", "")),
                    str(row.get("prediction_algorithm", "")),
                    f"{to_float(row, 'cost_estimate'):.3f}",
                    f"{to_float(row, 'cost_instance_hours'):.3f}",
                    f"{to_float(row, 'sla_violation_rate'):.5f}",
                    f"{to_float(row, 'tail_latency_p99_ms'):.1f}",
                    f"{to_float(row, 'resource_efficiency'):.3f}",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n")


def write_focused_comparison(rows: list[dict], output_dir: Path) -> None:
    selected = [
        row
        for row in rows
        if row.get("scenario_name", "") != "ablation" and row.get("policy", "") in FOCUSED_POLICIES
    ]
    if not selected:
        return

    lines = [
        "# Focused Comparison",
        "",
        "| scenario | policy | cost($) | viol_rate | p99(ms) | eff |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(selected, key=lambda x: (x.get("scenario_name", ""), x.get("policy", ""))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scenario_name", "")),
                    str(row.get("policy", "")),
                    f"{to_float(row, 'cost_estimate'):.3f}",
                    f"{to_float(row, 'sla_violation_rate'):.5f}",
                    f"{to_float(row, 'tail_latency_p99_ms'):.1f}",
                    f"{to_float(row, 'resource_efficiency'):.3f}",
                ]
            )
            + " |"
        )
    (output_dir / "focused_comparison.md").write_text("\n".join(lines) + "\n")


def scenario_rows(rows: list[dict], scenario_name: str) -> list[dict]:
    return [row for row in rows if row.get("scenario_name", "") == scenario_name]


def filter_rows_by_scenarios(rows: list[dict], scenario_names: set[str]) -> list[dict]:
    return [row for row in rows if row.get("scenario_name", "") in scenario_names]


def top_policy_line(row: dict) -> str:
    return (
        f"- `{row.get('policy', '')}`: cost={to_float(row, 'cost_estimate'):.3f}, "
        f"viol_rate={to_float(row, 'sla_violation_rate'):.5f}, "
        f"p99={to_float(row, 'tail_latency_p99_ms'):.1f}, "
        f"eff={to_float(row, 'resource_efficiency'):.3f}"
    )


def write_full_analysis_report(
    rows: list[dict],
    output_dir: Path,
    windows: dict[str, tuple[int, int]] | None,
    series_by_policy: dict[str, list[dict]],
) -> None:
    lines = [
        "# Full Result Analysis",
        "",
        "This report consolidates the benchmark summary, simplified policy comparison, ablation interpretation, and overscaling diagnostics into one file.",
        "",
    ]

    for scenario in ("main_controlled", "harder_qps", "harder_boot", "harder_capacity", "ablation"):
        current = scenario_rows(rows, scenario)
        if not current:
            continue
        lines.append(f"## {scenario}")
        lines.append("")
        for row in sorted(current, key=lambda item: to_float(item, "cost_estimate")):
            lines.append(top_policy_line(row))
        lines.append("")

        if scenario != "ablation":
            static = next((row for row in current if row.get("policy") == "static"), None)
            reactive = next((row for row in current if row.get("policy") == "reactive"), None)
            util = next((row for row in current if row.get("policy") == "util_base"), None)
            forecast = next((row for row in current if row.get("policy") == "forecast_only"), None)
            sla = next((row for row in current if row.get("policy") == "sla_aware"), None)
            if static and reactive and util and forecast and sla:
                lines.append("Interpretation:")
                lines.append(
                    f"- `static` is the fixed-capacity reference: cost {to_float(static, 'cost_estimate'):.3f}, "
                    f"violation rate {to_float(static, 'sla_violation_rate'):.5f}, p99 {to_float(static, 'tail_latency_p99_ms'):.1f}."
                )
                lines.append(
                    f"- `reactive` remains the cheapest dynamic baseline at {to_float(reactive, 'cost_estimate'):.3f}, but it lags after SLA degradation."
                )
                lines.append(
                    f"- `util_base` is the strongest low-cost baseline with cost {to_float(util, 'cost_estimate'):.3f} and violation rate {to_float(util, 'sla_violation_rate'):.5f}."
                )
                lines.append(
                    f"- `forecast_only` isolates proactive scaling from predicted load pressure; compare its cost {to_float(forecast, 'cost_estimate'):.3f} against `sla_aware` at {to_float(sla, 'cost_estimate'):.3f}."
                )
                lines.append(
                    f"- `sla_aware` is the final proposed policy: it scales from latency pressure, queue pressure, and demand rush, reaching violation rate {to_float(sla, 'sla_violation_rate'):.5f} and p99 {to_float(sla, 'tail_latency_p99_ms'):.1f}."
                )
                lines.append("")
        else:
            baseline = next((row for row in current if row.get("policy") == "sla_aware"), None)
            cheapest = min(current, key=lambda item: to_float(item, "cost_estimate"))
            best_p99 = min(current, key=lambda item: to_float(item, "tail_latency_p99_ms"))
            lines.append("Interpretation:")
            if baseline:
                lines.append(
                    f"- Baseline `sla_aware` cost is {to_float(baseline, 'cost_estimate'):.3f} with p99 {to_float(baseline, 'tail_latency_p99_ms'):.1f}."
                )
            lines.append(f"- Lowest cost ablation: `{cheapest.get('policy', '')}`.")
            lines.append(f"- Lowest p99 ablation: `{best_p99.get('policy', '')}`.")
            lines.append("- The intended reading is signal importance: prediction, queue, latency, and aggressiveness.")
            lines.append("")

    lines.append("## Overscaling")
    lines.append("")
    overscaling_policies = []
    for policy, policy_rows in sorted(series_by_policy.items()):
        prev_instances = None
        episodes = 0
        for row in policy_rows:
            increasing = prev_instances is not None and to_int(row, "instances") > prev_instances
            benign = (
                to_float(row, "utilization") <= 0.35
                and to_float(row, "queue_len") <= 0.0
                and to_int(row, "sla_violation") == 0
            )
            if increasing and benign:
                episodes += 1
            prev_instances = to_int(row, "instances")
        if episodes > 0:
            overscaling_policies.append(f"- `{policy}` shows {episodes} benign scale-up steps.")
    if overscaling_policies:
        lines.extend(overscaling_policies)
    else:
        lines.append("- No overscaling episodes detected under the current diagnostic threshold.")
    lines.append("")

    if windows:
        lines.append("## Case Study Windows")
        lines.append("")
        for name, (start, end) in windows.items():
            lines.append(f"- `{name}` window: steps {start} to {end - 1}")
        lines.append("")

    lines.append("## Bottom Line")
    lines.append("")
    lines.append("- `util_base` is the strongest low-cost baseline.")
    lines.append("- `forecast_only` is the proactive prediction-only baseline.")
    lines.append("- `sla_aware` is the single final proposed policy and should be judged against `util_base` and `forecast_only`.")
    lines.append("- Use the ablation section to explain whether latency pressure, queue pressure, demand rush, or aggressiveness matter most.")
    lines.append("")

    (output_dir / "full_result_analysis.md").write_text("\n".join(lines) + "\n")


def scatter_plot(rows: list[dict], x_key: str, y_key: str, output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 6))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("scenario_name", "custom")].append(row)

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    for idx, (scenario, items) in enumerate(sorted(grouped.items())):
        xs = [to_float(row, x_key) for row in items]
        ys = [to_float(row, y_key) for row in items]
        plt.scatter(xs, ys, label=scenario, s=80, marker=markers[idx % len(markers)], alpha=0.85)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def canonical_main_results_dir(results_dir: Path) -> Path:
    for name in ("report_main_controlled", "report_main"):
        candidate = results_dir / name
        if (candidate / "merged_trace.csv").exists():
            return candidate
    return results_dir


def load_series(results_dir: Path) -> dict[str, list[dict]]:
    series_by_policy: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*_series.csv")):
        rows = read_csv(path)
        if rows:
            series_by_policy[path.stem.replace("_series", "")] = rows
    return series_by_policy


def demand_windows(merged_trace: list[dict], window_size: int) -> dict[str, tuple[int, int]]:
    demand = [to_float(row, "demand_qps") for row in merged_trace]
    if not demand:
        return {"full": (0, 0)}

    best_burst = 0
    best_burst_gain = float("-inf")
    best_stable = 0
    best_stable_std = float("inf")
    best_recovery = 0
    best_recovery_drop = float("-inf")
    limit = max(1, len(demand) - window_size)

    for start in range(limit):
        window = demand[start : start + window_size]
        gain = window[-1] - window[0]
        drop = window[0] - window[-1]
        std = pstdev(window) if len(window) > 1 else 0.0
        if gain > best_burst_gain:
            best_burst_gain = gain
            best_burst = start
        if std < best_stable_std:
            best_stable_std = std
            best_stable = start
        if drop > best_recovery_drop:
            best_recovery_drop = drop
            best_recovery = start

    return {
        "burst": (best_burst, min(len(demand), best_burst + window_size)),
        "stable": (best_stable, min(len(demand), best_stable + window_size)),
        "recovery": (best_recovery, min(len(demand), best_recovery + window_size)),
    }


def plot_window(series_by_policy: dict[str, list[dict]], name: str, start: int, end: int, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    policies = [policy for policy in MAIN_POLICIES if policy in series_by_policy]
    cmap = plt.get_cmap("tab10")
    for idx, policy in enumerate(policies):
        rows = series_by_policy[policy]
        window = rows[start:end]
        if not window:
            continue
        color = cmap(idx % 10)
        x = [to_int(row, "step") for row in window]
        axes[0].plot(x, [to_int(row, "instances") for row in window], color=color, label=policy)
        axes[0].plot(
            x,
            [to_int(row, "target_instances") for row in window],
            color=color,
            linestyle="--",
            alpha=0.55,
            label="_nolegend_",
        )
        axes[1].plot(x, [to_float(row, "latency_p99_ms") for row in window], color=color, label="_nolegend_")
        axes[2].plot(x, [to_int(row, "sla_violation") for row in window], color=color, label="_nolegend_")

    axes[0].set_title(f"{name.title()} Window: Instances vs Target")
    axes[0].set_ylabel("instances")
    axes[1].set_title(f"{name.title()} Window: P99 Latency")
    axes[1].set_ylabel("latency (ms)")
    axes[2].set_title(f"{name.title()} Window: SLA Violation")
    axes[2].set_ylabel("violation")
    axes[2].set_xlabel("step")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_window.png", dpi=180)
    plt.close()


def plot_sensitivity(rows: list[dict], variable: str, output_path: Path, title: str) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pol = row.get("policy", "")
        if pol:
            grouped[pol].append(row)

    plt.figure(figsize=(8, 6))
    for policy, items in sorted(grouped.items()):
        buckets: dict[float, list[float]] = defaultdict(list)
        for row in items:
            xv = to_float(row, variable)
            buckets[xv].append(to_float(row, "sla_violation_rate"))
        points = sorted((x, mean(ys)) for x, ys in buckets.items() if ys)
        if not points:
            continue
        plt.plot([x for x, _ in points], [y for _, y in points], marker="o", label=policy)

    plt.title(title)
    plt.xlabel(variable)
    plt.ylabel("sla_violation_rate")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_case_notes(series_by_policy: dict[str, list[dict]], windows: dict[str, tuple[int, int]], output_dir: Path) -> None:
    lines = [
        "# Case Study Windows",
        "",
        "These windows are selected automatically from `merged_trace.csv`.",
        "",
    ]
    for name, (start, end) in windows.items():
        lines.append(f"- `{name}`: steps {start} to {end - 1}")
    lines.append("")
    lines.append("Suggested report usage:")
    lines.append("- `burst`: discuss whether proactive policies scale before or after latency worsens.")
    lines.append("- `stable`: discuss cost control and downscaling behavior.")
    lines.append("- `recovery`: discuss whether policies safely scale down after demand declines.")
    lines.append("")

    if series_by_policy:
        sample_policy = "sla_aware" if "sla_aware" in series_by_policy else sorted(series_by_policy)[0]
        rows = series_by_policy[sample_policy]
        for name, (start, end) in windows.items():
            window = rows[start:end]
            if not window:
                continue
            avg_latency = mean(to_float(row, "latency_p99_ms") for row in window)
            avg_instances = mean(to_int(row, "instances") for row in window)
            violations = sum(to_int(row, "sla_violation") for row in window)
            lines.append(
                f"- `{name}` sample stats from `{sample_policy}`: avg_instances={avg_instances:.2f}, "
                f"avg_p99={avg_latency:.1f}, violation_steps={violations}"
            )

    (output_dir / "case_windows.md").write_text("\n".join(lines) + "\n")


def detect_overscaling(series_by_policy: dict[str, list[dict]], output_dir: Path) -> None:
    min_episode_len = 3
    max_utilization = 0.35
    max_queue = 0.0
    lines = [
        "# Overscaling Diagnostics",
        "",
        "An overscaling episode is counted when a policy increases instances for consecutive steps while all of the following hold:",
        "",
        f"- utilization <= {max_utilization}",
        f"- queue_len <= {max_queue}",
        "- sla_violation = 0",
        "",
    ]

    found_any = False
    for policy, rows in sorted(series_by_policy.items()):
        current_len = 0
        start_step = None
        prev_instances = None
        episodes: list[tuple[int, int]] = []
        for row in rows:
            instances = to_int(row, "instances")
            benign = (
                to_float(row, "utilization") <= max_utilization
                and to_float(row, "queue_len") <= max_queue
                and to_int(row, "sla_violation") == 0
            )
            increasing = prev_instances is not None and instances > prev_instances
            if benign and increasing:
                if current_len == 0:
                    start_step = to_int(row, "step") - 1
                current_len += 1
            else:
                if current_len >= min_episode_len and start_step is not None:
                    episodes.append((start_step, to_int(row, "step") - 1))
                current_len = 0
                start_step = None
            prev_instances = instances

        if current_len >= min_episode_len and start_step is not None:
            episodes.append((start_step, to_int(rows[-1], "step")))

        if episodes:
            found_any = True
            lines.append(f"## {policy}")
            for start, end in episodes:
                lines.append(f"- overscaling episode from step {start} to {end}")
            lines.append("")

    if not found_any:
        lines.append("No overscaling episodes detected under the current diagnostic thresholds.")
        lines.append("")

    (output_dir / "overscaling_diagnostics.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = discover_result_rows(results_dir)
    if not rows:
        raise RuntimeError(f"missing benchmark results under: {results_dir}")

    write_summary_table(rows, output_dir)
    write_focused_comparison(rows, output_dir)

    non_ablation_rows = filter_rows_by_scenarios(rows, {"main_controlled", "harder_qps", "harder_boot", "harder_capacity"})
    scatter_plot(non_ablation_rows, "cost_estimate", "sla_violation_rate", output_dir / "cost_vs_violation.png", "Cost vs SLA Violation", "cost($)", "sla_violation_rate")
    scatter_plot(non_ablation_rows, "cost_estimate", "tail_latency_p99_ms", output_dir / "cost_vs_p99.png", "Cost vs P99", "cost($)", "tail_latency_p99_ms")
    plot_sensitivity(non_ablation_rows, "qps_scale", output_dir / "sensitivity_qps_scale.png", "Sensitivity: qps_scale vs SLA violation")
    plot_sensitivity(non_ablation_rows, "boot_delay_steps", output_dir / "sensitivity_boot_delay.png", "Sensitivity: boot delay vs SLA violation")
    plot_sensitivity(non_ablation_rows, "max_instances", output_dir / "sensitivity_max_instances.png", "Sensitivity: max instances vs SLA violation")

    main_dir = canonical_main_results_dir(results_dir)
    merged_trace_path = main_dir / "merged_trace.csv"
    series_by_policy = load_series(main_dir)
    windows = None
    if merged_trace_path.exists():
        merged_trace = read_csv(merged_trace_path)
        windows = demand_windows(merged_trace, args.window_size)
        for name, (start, end) in windows.items():
            plot_window(series_by_policy, name, start, end, output_dir)
        write_case_notes(series_by_policy, windows, output_dir)

    detect_overscaling(series_by_policy, output_dir)
    write_full_analysis_report(rows, output_dir, windows, series_by_policy)


if __name__ == "__main__":
    main()
