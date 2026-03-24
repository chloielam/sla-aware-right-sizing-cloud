#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


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
        if row.get("policy", "") in {
            "util_base",
            "sla_required_capacity",
            "forecast_only",
            "sla_aware_tuned",
            "sla_aware_ema_tuned",
        }
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
        "This report consolidates the benchmark summary, focused SLA-aware comparison, scenario-level interpretation, and overscaling diagnostics into one file.",
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
            util = next((row for row in current if row.get("policy") == "util_base"), None)
            req = next((row for row in current if row.get("policy") == "sla_required_capacity"), None)
            forecast = next((row for row in current if row.get("policy") == "forecast_only"), None)
            ema = next((row for row in current if row.get("policy") == "sla_aware_ema_tuned"), None)
            heuristic = next((row for row in current if row.get("policy") == "sla_aware_tuned"), None)
            if util and req and forecast and ema and heuristic:
                lines.append("Interpretation:")
                lines.append(
                    f"- `sla_required_capacity` is the lightweight proactive baseline: cost "
                    f"{to_float(req, 'cost_estimate'):.3f}, violation rate {to_float(req, 'sla_violation_rate'):.5f}, "
                    f"p99 {to_float(req, 'tail_latency_p99_ms'):.1f}."
                )
                lines.append(
                    f"- `forecast_only` isolates prediction-driven proactive scaling with cost "
                    f"{to_float(forecast, 'cost_estimate'):.3f} and p99 {to_float(forecast, 'tail_latency_p99_ms'):.1f}."
                )
                lines.append(
                    f"- `sla_aware_ema_tuned` improves violation rate from {to_float(util, 'sla_violation_rate'):.5f} "
                    f"to {to_float(ema, 'sla_violation_rate'):.5f} while increasing cost from "
                    f"{to_float(util, 'cost_estimate'):.3f} to {to_float(ema, 'cost_estimate'):.3f}."
                )
                lines.append(
                    f"- `sla_aware_tuned` reaches similar SLA outcomes with cost {to_float(heuristic, 'cost_estimate'):.3f}."
                )
                lines.append("")
        else:
            best_p99 = min(current, key=lambda item: to_float(item, "tail_latency_p99_ms"))
            cheapest = min(current, key=lambda item: to_float(item, "cost_estimate"))
            lines.append("Interpretation:")
            lines.append(f"- Lowest p99 variant: `{best_p99.get('policy', '')}`.")
            lines.append(f"- Lowest cost variant: `{cheapest.get('policy', '')}`.")
            lines.append("- The ablation is meaningful only if these variants differ in cost or p99.")
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
    lines.append("- Use `util_base` as the strongest low-cost reactive baseline.")
    lines.append("- Use `sla_required_capacity` as the simple proactive baseline contributed by direct capacity sizing.")
    lines.append("- Use `sla_aware_ema_tuned` as the strongest current SLA-aware candidate.")
    lines.append("- Use the ablation section to discuss aggressiveness, prediction, and weak signal contributions.")
    lines.append("")

    (output_dir / "full_result_analysis.md").write_text("\n".join(lines) + "\n")


def scatter_plot(rows: list[dict], x_key: str, y_key: str, output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 6))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("scenario_name", "custom")].append(row)

    for scenario, items in grouped.items():
        xs = [to_float(row, x_key) for row in items]
        ys = [to_float(row, y_key) for row in items]
        labels = [row.get("policy", "") for row in items]
        plt.scatter(xs, ys, label=scenario, s=70)
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


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
    for policy, rows in sorted(series_by_policy.items()):
        window = rows[start:end]
        if not window:
            continue
        x = [to_int(row, "step") for row in window]
        axes[0].plot(x, [to_int(row, "instances") for row in window], label=policy)
        axes[0].plot(x, [to_int(row, "target_instances") for row in window], linestyle="--", alpha=0.55)
        axes[1].plot(x, [to_float(row, "latency_p99_ms") for row in window], label=policy)
        axes[2].plot(x, [to_int(row, "sla_violation") for row in window], label=policy)

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
        grouped[row.get("policy", "")].append(row)

    plt.figure(figsize=(8, 6))
    for policy, items in sorted(grouped.items()):
        xs = [to_float(row, variable) for row in items]
        ys = [to_float(row, "sla_violation_rate") for row in items]
        points = sorted(zip(xs, ys), key=lambda x: x[0])
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
        sample_policy = "sla_aware_ema_tuned" if "sla_aware_ema_tuned" in series_by_policy else sorted(series_by_policy)[0]
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
    scatter_plot(
        rows,
        "cost_estimate",
        "sla_violation_rate",
        output_dir / "cost_vs_violation.png",
        "Cost vs SLA Violation",
        "cost_estimate ($)",
        "sla_violation_rate",
    )
    scatter_plot(
        rows,
        "cost_estimate",
        "tail_latency_p99_ms",
        output_dir / "cost_vs_p99.png",
        "Cost vs P99 Latency",
        "cost_estimate ($)",
        "tail_latency_p99_ms",
    )

    plot_sensitivity(rows, "qps_scale", output_dir / "sensitivity_qps_scale.png", "Sensitivity: QPS Scale")
    plot_sensitivity(rows, "boot_delay_steps", output_dir / "sensitivity_boot_delay.png", "Sensitivity: Boot Delay")
    plot_sensitivity(rows, "max_instances", output_dir / "sensitivity_max_instances.png", "Sensitivity: Max Instances")

    main_results_dir = results_dir
    if not (main_results_dir / "merged_trace.csv").exists():
        for name in ("report_main_controlled", "report_main"):
            candidate = results_dir / name
            if candidate.exists():
                main_results_dir = candidate
                break

    merged_trace_path = main_results_dir / "merged_trace.csv"
    series_by_policy = load_series(main_results_dir)
    windows: dict[str, tuple[int, int]] | None = None
    if merged_trace_path.exists() and series_by_policy:
        merged_trace = read_csv(merged_trace_path)
        windows = demand_windows(merged_trace, args.window_size)
        for name, (start, end) in windows.items():
            plot_window(series_by_policy, name, start, end, output_dir)
        write_case_notes(series_by_policy, windows, output_dir)
        detect_overscaling(series_by_policy, output_dir)
    write_full_analysis_report(rows, output_dir, windows, series_by_policy)


if __name__ == "__main__":
    main()
