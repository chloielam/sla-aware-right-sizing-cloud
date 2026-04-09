# SLA-Aware Right-Sizing for Cloud Cost Optimization

Course project for COMP 6910 (Winter 2026), Memorial University of Newfoundland.

## Overview

This project studies autoscaling tradeoffs for a trace-driven GenAI serving simulator built from the Alibaba dataset. The final project keeps one proposed SLA-aware policy and four comparison baselines:

1. `static`: fixed-capacity baseline
2. `reactive`: scales after SLA degradation
3. `util_base`: utilization-threshold baseline
4. `forecast_only`: proactive prediction-only baseline
5. `sla_aware`: proposed policy that scales from SLA-risk

The final report is intentionally centered on one explainable controller rather than many exploratory variants.

## Problem Statement

Autoscaling faces a direct cost versus latency tradeoff:

- scaling too late increases queueing and SLA violations
- scaling too early wastes instance-hours
- static capacity protects latency but overpays most of the time

The project asks whether a simple SLA-aware controller can outperform standard baselines in a trace-driven simulator.

## Final Policy Story

### `forecast_only`

`forecast_only` is the proactive control baseline. It uses:

- predicted demand
- predicted utilization
- predicted latency

It combines them into a single forecast score and scales early when future pressure looks high.

### `sla_aware`

`sla_aware` is the proposed policy. It scales from three directly interpretable signals:

- `latency_pressure`: how close p99 latency is to the SLA threshold
- `queue_pressure`: how much backlog is accumulating relative to a queue budget
- `demand_rush`: how much predicted demand exceeds current demand

Risk score:

`risk = latency_weight * latency_pressure + queue_weight * queue_pressure + demand_weight * demand_rush`

Scale-up happens only when:

- smoothed risk exceeds a threshold
- and at least one hard SLA-risk signal is present

This keeps the policy explainable in a report or oral defense.

## Repository Structure

```text
sla-aware/
├── analysis/
│   └── generate_report_assets.py
├── configs/
│   ├── forecast_only.yaml
│   ├── reactive.yaml
│   ├── sla_aware.yaml
│   ├── sla_aware_*.yaml
│   ├── static.yaml
│   └── util_base.yaml
├── data/
│   ├── qps.csv
│   ├── queue_rt_raw_anon.csv
│   ├── controlnet_latency_data_anon.csv
│   ├── pod_gpu_duty_cycle_anon.csv
│   └── pod_memory_util_anon.csv
├── experiments/
│   └── run_benchmark.py
├── scripts/
│   └── run_report_experiments.sh
└── src/
    ├── engine/
    │   └── simulator.py
    └── policies/
        ├── base_policy.py
        ├── forecast_only.py
        ├── reactive.py
        ├── sla_aware.py
        ├── static.py
        └── util_threshold.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Prototype

Short debug run:

```bash
python3 experiments/run_benchmark.py --max-steps 240 --write-series --scenario-name debug --result-tag report_debug
```

Outputs go to `data/processed/`:

- `merged_trace.csv`
- `benchmark_results.csv`
- `<policy>_series.csv`

## Main Report Benchmark

Canonical main run:

```bash
python3 experiments/run_benchmark.py \
  --configs configs/static.yaml configs/reactive.yaml configs/util_base.yaml configs/forecast_only.yaml configs/sla_aware.yaml \
  --max-steps 1090 \
  --write-series \
  --scenario-name main_controlled \
  --result-tag report_main_controlled \
  --qps-mode qps \
  --qps-agg sum
```

Stress runs:

```bash
python3 experiments/run_benchmark.py --configs configs/static.yaml configs/reactive.yaml configs/util_base.yaml configs/forecast_only.yaml configs/sla_aware.yaml --max-steps 1090 --write-series --scenario-name harder_qps --result-tag report_harder_qps --qps-mode qps --qps-agg sum --qps-scale 1.35
python3 experiments/run_benchmark.py --configs configs/static.yaml configs/reactive.yaml configs/util_base.yaml configs/forecast_only.yaml configs/sla_aware.yaml --max-steps 1090 --write-series --scenario-name harder_boot --result-tag report_harder_boot --qps-mode qps --qps-agg sum --boot-delay-override 4
python3 experiments/run_benchmark.py --configs configs/static.yaml configs/reactive.yaml configs/util_base.yaml configs/forecast_only.yaml configs/sla_aware.yaml --max-steps 1090 --write-series --scenario-name harder_capacity --result-tag report_harder_capacity --qps-mode qps --qps-agg sum --max-instances-override 18
```

Ablation run:

```bash
python3 experiments/run_benchmark.py \
  --configs configs/sla_aware.yaml configs/sla_aware_no_prediction.yaml configs/sla_aware_no_queue.yaml configs/sla_aware_no_latency.yaml configs/sla_aware_low_aggr.yaml configs/sla_aware_high_aggr.yaml \
  --max-steps 1090 \
  --write-series \
  --scenario-name ablation \
  --result-tag report_ablation \
  --qps-mode qps \
  --qps-agg sum \
  --qps-scale 1.35 \
  --boot-delay-override 4 \
  --max-instances-override 18
```

Or run the full bundle:

```bash
bash scripts/run_report_experiments.sh
```

## Analysis Pipeline

Generate the full report artifact set:

```bash
python3 analysis/generate_report_assets.py --results-dir data/processed --output-dir analysis/output/report_full
```

This produces:

- summary tables in CSV and Markdown
- focused comparison tables for `util_base`, `forecast_only`, and `sla_aware`
- cost vs violation and cost vs p99 scatter plots
- sensitivity plots for `qps_scale`, `boot_delay_steps`, and `max_instances`
- overscaling diagnostics
- representative `burst`, `stable`, and `recovery` time-series figures
- one consolidated `full_result_analysis.md`

## Tracked documentation

`README.md` is the only tracked Markdown document in the repository.

Use it as the canonical reference for:

- project overview
- active policy set
- benchmark commands
- report-generation workflow

Other report notes, drafts, and generated Markdown outputs are kept locally but are intentionally excluded from git.
