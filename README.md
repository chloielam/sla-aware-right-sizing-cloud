# SLA-Aware Right-Sizing for Cloud Cost Optimization

Course project for COMP 6910 (Winter 2026), Memorial University of Newfoundland.

## Overview

Cloud systems are often over-provisioned to avoid Service Level Agreement (SLA) violations, which improves reliability but increases cost. This project studies how an SLA-aware right-sizing strategy can reduce instance cost while maintaining latency targets under dynamic workloads.

The core idea is to compare policy families in a trace-driven simulator:

1. Static provisioning (peak-based capacity)
2. Utilization-threshold scaling
3. Reactive scaling after SLA violations
4. SLA-aware proactive scaling (queue/latency-risk driven)
5. Forecast-only proactive scaling
6. SLA-aware EMA forecasting

## Problem Statement

Common autoscaling approaches are primarily reactive, so they can underperform on bursty traffic:

- scaling too late increases tail latency and SLA breach risk
- scaling too aggressively can cause instability and wasted churn
- static safety margins waste capacity most of the time

This project evaluates the trade-off between cost savings and SLA risk in a reproducible simulation environment.

## Proposed Approach

We will build a Python-based discrete-time cloud simulator with:

- workload replay from trace data
- instance boot delay and cooldown constraints
- policy plug-in interface for multiple right-sizing strategies
- metrics tracking for cost and performance outcomes

The SLA-aware policy predicts near-term latency risk using queue, demand, and latency signals and scales proactively before violations occur. The repository also includes:

- a heuristic risk-based SLA-aware controller
- an EMA-based SLA-aware controller
- a forecast-only control policy for cleaner decomposition of prediction vs risk blending
- tuned SLA-aware report configs focused on diagnosing overscaling versus `util_base`

## Evaluation Metrics

- `Cost`: total instance-hours consumed
- `SLA violation rate`: fraction of requests above SLA latency threshold
- `Tail latency (P99)`: 99th percentile response time
- `Resource efficiency`: average CPU/GPU utilization

## Research Questions

- `RQ1`: Does proactive SLA-aware scaling reduce SLA violations and tail latency compared with static, reactive, and utilization-based baselines under the Alibaba GenAI trace?
- `RQ2`: What is the cost-performance tradeoff of proactive scaling?
- `RQ3`: Which parts of the SLA-aware design matter most: prediction signal, latency risk, queue signal, or aggressiveness?

## Repository Structure

```text
sla-aware/
├── analysis/
│   └── generate_report_assets.py
├── configs/
│   ├── forecast_only.yaml
│   ├── static.yaml
│   ├── util_base.yaml
│   ├── sla_aware.yaml
│   ├── sla_aware_ema.yaml
│   └── ablation configs
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
        ├── static.py
        ├── util_threshold.py
        ├── sla_aware.py
        └── sla_aware_ema.py
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies (add to `requirements.txt` as implementation progresses):

```bash
pip install -r requirements.txt
```

## Run The Prototype

Run all policies with a short, fast simulation horizon (default: 240 steps):

```bash
python3 experiments/run_benchmark.py --max-steps 240 --write-series --scenario-name debug --result-tag report_debug
```

Outputs are written to `data/processed/`:

- `merged_trace.csv`: aligned input trace used by the simulator
- `benchmark_results.csv`: summary metrics for each policy
- `<policy>_series.csv`: per-step simulation series (when `--write-series` is enabled)

Use fewer steps for even faster test runs (for example, ~1 second runtime):

```bash
python3 experiments/run_benchmark.py --max-steps 120
```

Try both demand interpretations:

```bash
python3 experiments/run_benchmark.py --qps-mode qps --qps-agg sum --write-series
python3 experiments/run_benchmark.py --qps-mode count --qps-agg sum --qps-scale 1.0 --write-series
```

If `count` is too aggressive, reduce intensity:

```bash
python3 experiments/run_benchmark.py --qps-mode count --qps-scale 0.25 --qps-agg sum --max-steps 400
```

## Main Report Benchmark

Canonical main run:

```bash
python3 experiments/run_benchmark.py \
  --configs configs/util_base.yaml configs/sla_aware_tuned.yaml configs/sla_aware_ema_tuned.yaml \
  --max-steps 1090 \
  --write-series \
  --scenario-name main_controlled \
  --result-tag report_main_controlled \
  --qps-mode qps \
  --qps-agg sum
```

Recommended robustness checks:

```bash
python3 experiments/run_benchmark.py --configs configs/util_base.yaml configs/sla_aware_tuned.yaml configs/sla_aware_ema_tuned.yaml --max-steps 1090 --write-series --scenario-name harder_qps --result-tag report_harder_qps --qps-mode qps --qps-agg sum --qps-scale 1.35
python3 experiments/run_benchmark.py --configs configs/util_base.yaml configs/sla_aware_tuned.yaml configs/sla_aware_ema_tuned.yaml --max-steps 1090 --write-series --scenario-name harder_boot --result-tag report_harder_boot --qps-mode qps --qps-agg sum --boot-delay-override 4
python3 experiments/run_benchmark.py --configs configs/util_base.yaml configs/sla_aware_tuned.yaml configs/sla_aware_ema_tuned.yaml --max-steps 1090 --write-series --scenario-name harder_capacity --result-tag report_harder_capacity --qps-mode qps --qps-agg sum --max-instances-override 18
```

Ablation run:

```bash
python3 experiments/run_benchmark.py \
  --configs configs/sla_aware_ema_tuned.yaml configs/sla_aware_ema_no_prediction.yaml configs/sla_aware_ema_no_queue_external.yaml configs/sla_aware_ema_no_violation.yaml configs/sla_aware_ema_low_aggr.yaml configs/sla_aware_ema_high_aggr.yaml \
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

Or run the full experiment bundle:

```bash
bash scripts/run_report_experiments.sh
```

## Analysis Pipeline

Generate report-ready tables and figures from any benchmark result directory:

```bash
python3 analysis/generate_report_assets.py --results-dir data/processed/report_main_controlled --output-dir analysis/output/main_controlled
```

Generate bundle-level comparison figures across the main, stress, and ablation runs:

```bash
python3 analysis/generate_report_assets.py --results-dir data/processed --output-dir analysis/output/report_bundle
```

This produces:

- summary tables in CSV and Markdown
- focused comparison tables for `util_base` vs tuned SLA-aware policies
- cost vs violation and cost vs p99 scatter plots
- sensitivity plots for `qps_scale`, `boot_delay_steps`, and `max_instances`
- overscaling diagnostics
- representative `burst`, `stable`, and `recovery` time-series figures

See [REPORTING.md](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/REPORTING.md) for methodology and report-writing guidance.

## Planned Workflow

1. Preprocess trace data into simulator-friendly time series.
2. Implement simulator core (state update, queue dynamics, scaling delay).
3. Implement baseline policies (static, utilization, reactive).
4. Implement SLA-aware and forecast-based predictive policies.
5. Run disciplined main, stress, and ablation experiments.
6. Analyze cost/SLA/latency trade-offs and generate report figures.

## Expected Deliverables

- reproducible simulation code
- benchmark results across policies and scenarios
- figures/tables showing cost vs SLA risk trade-offs
- final report and presentation

## Timeline (from proposal)

- Weeks 1-2: data preprocessing + simulation engine
- Weeks 3-4: baseline policies
- Weeks 5-6: SLA-aware policy
- Weeks 7-8: analysis, visualization, report

## Methodology Notes

The simulator is intentionally simplified. The project should be presented as a trace-driven evaluation framework with explicit modeling assumptions, not as a production-faithful deployment emulator. The latency model, prediction signals, experiment scenarios, and limitations are documented in [REPORTING.md](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/REPORTING.md).

## References

1. Gandhi et al., *Adaptive, model-driven autoscaling for cloud applications*, ICAC 2014.
2. Lim et al., *Automated control for elastic storage*, ICAC 2010.
3. Lin et al., *Understanding diffusion model serving in production*, SoCC 2025/2026.
4. Wu et al., *SLA-based resource allocation for SaaS in cloud computing environments*, CCGrid 2011.
