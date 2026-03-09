# SLA-Aware Right-Sizing for Cloud Cost Optimization

Course project for COMP 6910 (Winter 2026), Memorial University of Newfoundland.

## Overview

Cloud systems are often over-provisioned to avoid Service Level Agreement (SLA) violations, which improves reliability but increases cost. This project studies how an SLA-aware right-sizing strategy can reduce instance cost while maintaining latency targets under dynamic workloads.

The core idea is to compare policy families in a trace-driven simulator:

1. Static provisioning (peak-based capacity)
2. Utilization-threshold scaling
3. Reactive scaling after SLA violations
4. SLA-aware proactive scaling (queue/latency-risk driven)

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

The SLA-aware policy will predict near-term latency risk using queue/workload signals and scale proactively before violations occur.

## Evaluation Metrics

- `Cost`: total instance-hours consumed
- `SLA violation rate`: fraction of requests above SLA latency threshold
- `Tail latency (P99)`: 99th percentile response time
- `Resource efficiency`: average CPU/GPU utilization

## Repository Structure

```text
sla-aware/
├── configs/
│   ├── static.yaml
│   ├── util_base.yaml
│   └── sla_aware.yaml
├── data/
│   ├── qps.csv
│   ├── queue_rt_raw_anon.csv
│   ├── controlnet_latency_data_anon.csv
│   ├── pod_gpu_duty_cycle_anon.csv
│   └── pod_memory_util_anon.csv
├── experiments/
│   └── run_benchmark.py
└── src/
    ├── engine/
    │   └── simulator.py
    └── policies/
        ├── base_policy.py
        ├── static.py
        ├── util_threshold.py
        └── sla_aware.py
```

Note: the repository is currently scaffolded; implementation files are placeholders and will be filled during development.

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
python3 experiments/run_benchmark.py --max-steps 240 --write-series
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

## Planned Workflow

1. Preprocess trace data into simulator-friendly time series.
2. Implement simulator core (state update, queue dynamics, scaling delay).
3. Implement baseline policies (static, utilization, reactive).
4. Implement SLA-aware predictive policy.
5. Run comparable experiments for all policies.
6. Analyze cost/SLA/latency trade-offs and visualize results.

## Expected Deliverables

- reproducible simulation code
- benchmark results across policies
- figures/tables showing cost vs SLA risk trade-offs
- final report and presentation

## Timeline (from proposal)

- Weeks 1-2: data preprocessing + simulation engine
- Weeks 3-4: baseline policies
- Weeks 5-6: SLA-aware policy
- Weeks 7-8: analysis, visualization, report

## References

1. Gandhi et al., *Adaptive, model-driven autoscaling for cloud applications*, ICAC 2014.
2. Lim et al., *Automated control for elastic storage*, ICAC 2010.
3. Lin et al., *Understanding diffusion model serving in production*, SoCC 2025/2026.
4. Wu et al., *SLA-based resource allocation for SaaS in cloud computing environments*, CCGrid 2011.
