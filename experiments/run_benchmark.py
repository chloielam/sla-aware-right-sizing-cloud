#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.simulator import CloudSimulator, SimulationConfig, TracePoint
from src.policies.sla_aware import SLAAwarePolicy
from src.policies.sla_aware_ema import SLAAwareEMA
from src.policies.static import StaticPolicy
from src.policies.util_threshold import ReactivePolicy, UtilThresholdPolicy


def aggregate_metric(
    path: Path, bucket_seconds: int, filters: dict | None = None, agg: str = "mean"
) -> dict[int, float]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filters:
                mismatch = False
                for k, v in filters.items():
                    if row.get(k) != v:
                        mismatch = True
                        break
                if mismatch:
                    continue
            try:
                ts_raw = float(row["timestamp_anon"])
                val = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            bucket = int(ts_raw) // bucket_seconds * bucket_seconds
            sums[bucket] += val
            counts[bucket] += 1
    if agg == "sum":
        return dict(sums)
    return {k: sums[k] / counts[k] for k in sums}


def with_forward_fill(timestamps: list[int], series: dict[int, float], default: float) -> list[float]:
    out = []
    last = default
    for ts in timestamps:
        if ts in series:
            last = series[ts]
        out.append(last)
    return out


def prepare_trace(
    data_dir: Path,
    processed_dir: Path,
    bucket_seconds: int,
    qps_request_type: str | None,
    qps_agg: str,
    qps_mode: str,
    qps_scale: float,
) -> tuple[list[TracePoint], int]:
    qps_filter = {"request_type": qps_request_type} if qps_request_type else None
    qps = aggregate_metric(data_dir / "qps.csv", bucket_seconds, filters=qps_filter, agg=qps_agg)
    queue = aggregate_metric(data_dir / "queue_rt_raw_anon.csv", bucket_seconds)
    latency = aggregate_metric(data_dir / "controlnet_latency_data_anon.csv", bucket_seconds)
    gpu = aggregate_metric(data_dir / "pod_gpu_duty_cycle_anon.csv", bucket_seconds)
    mem = aggregate_metric(data_dir / "pod_memory_util_anon.csv", bucket_seconds)

    if not qps:
        raise RuntimeError("qps.csv has no usable data")

    if qps_mode == "count":
        qps = {ts: value / max(1.0, float(bucket_seconds)) for ts, value in qps.items()}
    elif qps_mode != "qps":
        raise ValueError(f"Unsupported qps_mode: {qps_mode}")

    if qps_scale != 1.0:
        qps = {ts: value * qps_scale for ts, value in qps.items()}

    timestamps = sorted(qps.keys())
    queue_default = median(queue.values()) if queue else 0.0
    latency_default = median(latency.values()) if latency else 1000.0
    gpu_default = median(gpu.values()) if gpu else 0.0
    mem_default = median(mem.values()) if mem else 0.5

    queue_ff = with_forward_fill(timestamps, queue, queue_default)
    latency_ff = with_forward_fill(timestamps, latency, latency_default)
    gpu_ff = with_forward_fill(timestamps, gpu, gpu_default)
    mem_ff = with_forward_fill(timestamps, mem, mem_default)

    trace = [
        TracePoint(
            timestamp=ts,
            demand_qps=max(0.0, float(qps[ts])),
            external_queue=max(0.0, float(queue_ff[i])),
            external_latency_ms=max(1.0, float(latency_ff[i])),
            gpu_util_pct=max(0.0, float(gpu_ff[i])),
            memory_util=max(0.0, float(mem_ff[i])),
        )
        for i, ts in enumerate(timestamps)
    ]

    processed_dir.mkdir(parents=True, exist_ok=True)
    merged_path = processed_dir / "merged_trace.csv"
    with merged_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "demand_qps",
                "external_queue",
                "external_latency_ms",
                "gpu_util_pct",
                "memory_util",
            ]
        )
        for point in trace:
            writer.writerow(
                [
                    point.timestamp,
                    round(point.demand_qps, 6),
                    round(point.external_queue, 6),
                    round(point.external_latency_ms, 6),
                    round(point.gpu_util_pct, 6),
                    round(point.memory_util, 6),
                ]
            )
    return trace, bucket_seconds


def load_config(path: Path) -> dict:
    content = path.read_text().strip()
    if not content:
        raise RuntimeError(f"empty config file: {path}")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Config {path} is not valid JSON. Use JSON syntax inside .yaml files for this prototype."
        ) from exc


def build_policy(policy_name: str, run_name: str, params: dict):
    key = policy_name.lower().strip()
    if key == "static":
        return StaticPolicy(run_name, params)
    if key in {"util", "util_threshold", "utilization"}:
        return UtilThresholdPolicy(run_name, params)
    if key in {"reactive", "sla_reactive"}:
        return ReactivePolicy(run_name, params)
    if key in {"sla_aware", "sla-aware", "slaaware"}:
        return SLAAwarePolicy(run_name, params)
    if key in {"sla_aware_ema", "sla-aware-ema", "slaawareema", "sla_aware_ewma"}:
        return SLAAwareEMA(run_name, params)
    raise ValueError(f"unknown policy type: {policy_name}")


def build_sim_config(sim_cfg: dict, step_seconds: int, max_steps_override: int | None) -> SimulationConfig:
    max_steps = int(sim_cfg.get("max_steps", 240))
    if max_steps_override is not None:
        max_steps = max_steps_override
    return SimulationConfig(
        min_instances=int(sim_cfg.get("min_instances", 1)),
        max_instances=int(sim_cfg.get("max_instances", 30)),
        initial_instances=int(sim_cfg.get("initial_instances", 8)),
        service_rate_qps_per_instance=float(sim_cfg.get("service_rate_qps_per_instance", 0.08)),
        step_seconds=int(sim_cfg.get("step_seconds", step_seconds)),
        boot_delay_steps=int(sim_cfg.get("boot_delay_steps", 2)),
        cooldown_steps=int(sim_cfg.get("cooldown_steps", 2)),
        base_latency_ms=float(sim_cfg.get("base_latency_ms", 550.0)),
        queue_latency_factor_ms=float(sim_cfg.get("queue_latency_factor_ms", 2.0)),
        utilization_penalty_ms=float(sim_cfg.get("utilization_penalty_ms", 1800.0)),
        external_latency_weight=float(sim_cfg.get("external_latency_weight", 0.05)),
        sla_threshold_ms=float(sim_cfg.get("sla_threshold_ms", 3000.0)),
        cost_per_instance_hour=float(sim_cfg.get("cost_per_instance_hour", 1.25)),
        max_steps=max_steps,
        lookahead_steps=int(sim_cfg.get("lookahead_steps", 4)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SLA right-sizing simulation benchmark.")
    parser.add_argument("--data-dir", default="data", help="directory containing raw CSV traces")
    parser.add_argument("--processed-dir", default="data/processed", help="directory for derived outputs")
    parser.add_argument("--configs", nargs="*", default=[], help="config files to run")
    parser.add_argument("--bucket-seconds", type=int, default=60, help="time-bucket width for preprocessing")
    parser.add_argument("--max-steps", type=int, default=240, help="hard cap on simulation steps")
    parser.add_argument(
        "--write-series",
        action="store_true",
        help="write per-policy simulation series to data/processed/<policy>_series.csv",
    )
    parser.add_argument(
        "--qps-request-type",
        default="all",
        choices=("all", "API Requests", "Generative Requests"),
        help="restrict demand signal to a specific request_type",
    )
    parser.add_argument(
        "--qps-agg",
        default="sum",
        choices=("mean", "sum"),
        help="how to aggregate qps values per bucket",
    )
    parser.add_argument(
        "--qps-mode",
        default="qps",
        choices=("qps", "count"),
        help="qps mode: 'qps' assumes values are per-second, 'count' assumes values are counts within each bucket",
    )
    parser.add_argument(
        "--qps-scale",
        type=float,
        default=1.0,
        help="manual multiplier applied after qps interpretation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = (ROOT / args.data_dir).resolve()
    processed_dir = (ROOT / args.processed_dir).resolve()
    config_paths = [Path(p).resolve() for p in args.configs] if args.configs else sorted((ROOT / "configs").glob("*.yaml"))
    if not config_paths:
        raise RuntimeError("No configs found. Add configs/*.yaml or pass --configs")

    request_type = None if args.qps_request_type == "all" else args.qps_request_type
    trace, inferred_step_seconds = prepare_trace(
        data_dir,
        processed_dir,
        args.bucket_seconds,
        qps_request_type=request_type,
        qps_agg=args.qps_agg,
        qps_mode=args.qps_mode,
        qps_scale=args.qps_scale,
    )
    summaries: list[dict] = []
    for config_path in config_paths:
        conf = load_config(config_path)
        run_name = conf.get("name", config_path.stem)
        policy = build_policy(conf.get("policy", run_name), run_name, conf.get("policy_params", {}))
        sim_config = build_sim_config(conf.get("sim", {}), inferred_step_seconds, args.max_steps)
        simulator = CloudSimulator(sim_config, policy, trace)
        summary, series = simulator.run()
        summary["config"] = config_path.name
        summaries.append(summary)

        if args.write_series:
            out = processed_dir / f"{run_name}_series.csv"
            with out.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(series[0].keys()))
                writer.writeheader()
                writer.writerows(series)

    out_results = processed_dir / "benchmark_results.csv"
    with out_results.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nBenchmark summary")
    print("-" * 88)
    print(
        f"{'policy':16} {'cost($)':>10} {'inst_hr':>10} {'viol_rate':>12} {'p99(ms)':>10} {'eff':>8} {'sim_hr':>8}"
    )
    print("-" * 88)
    for s in sorted(summaries, key=lambda x: x["cost_estimate"]):
        print(
            f"{s['policy'][:16]:16} {s['cost_estimate']:10.3f} {s['cost_instance_hours']:10.3f} "
            f"{s['sla_violation_rate']:12.5f} {s['tail_latency_p99_ms']:10.1f} {s['resource_efficiency']:8.3f} "
            f"{s['simulated_hours']:8.2f}"
        )
    print("-" * 88)
    print(f"Saved merged trace to: {processed_dir / 'merged_trace.csv'}")
    print(f"Saved benchmark results to: {out_results}")


if __name__ == "__main__":
    main()
