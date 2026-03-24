# Report Development Guide

## Research Questions

- `RQ1`: Does proactive SLA-aware scaling reduce SLA violations and tail latency compared with static, reactive, and utilization-based baselines under the Alibaba GenAI trace?
- `RQ2`: What is the cost-performance tradeoff of proactive scaling?
- `RQ3`: Which parts of the SLA-aware design matter most: prediction signal, latency risk, queue signal, or aggressiveness?

## Simulator Methodology

### Trace inputs

The simulator uses bucketed Alibaba trace signals:

- `qps.csv` for workload demand
- `queue_rt_raw_anon.csv` for external queue pressure
- `controlnet_latency_data_anon.csv` for external latency pressure
- `pod_gpu_duty_cycle_anon.csv` for GPU utilization context
- `pod_memory_util_anon.csv` for memory utilization context

The merged trace is written to `data/processed/<run>/merged_trace.csv` for reproducibility.

### Internal latency model

At each simulation step:

- `arrivals = demand_qps * step_seconds`
- `capacity = active_instances * service_rate_qps_per_instance * step_seconds`
- `queue_len = max(0, previous_queue + arrivals - capacity)`
- `utilization = arrivals / capacity`

The internal p99 model is:

`model_p99 = base_latency_ms + queue_latency_factor_ms * queue_len + utilization_penalty_ms * max(0, utilization - 1.0)`

The final p99 used for SLA checks is:

`latency_p99 = model_p99 + external_latency_weight * max(0, external_latency_ms - base_latency_ms)`

### Prediction signals exposed to policies

Policies receive:

- `predicted_demand_qps`: the mean demand over the next `lookahead_steps`
- `predicted_latency_ms`: current model latency plus:
- external latency adjustment,
- a demand-growth penalty based on `(predicted_demand_qps - current_demand_qps)`,
- and a queue-based penalty from the external queue signal

This is a heuristic predictor, not a learned forecasting model.

## Experimental Design

### Main comparison

Policies:

- `util_base`
- `sla_required_capacity`
- `forecast_only`
- `sla_aware_tuned`
- `sla_aware_ema_tuned`

Main run recommendation:

```bash
python3 experiments/run_benchmark.py \
  --configs configs/util_base.yaml configs/sla_required_capacity.yaml configs/forecast_only.yaml configs/sla_aware_tuned.yaml configs/sla_aware_ema_tuned.yaml \
  --max-steps 1090 \
  --write-series \
  --scenario-name main_controlled \
  --result-tag report_main_controlled \
  --qps-mode qps \
  --qps-agg sum
```

### Sensitivity studies

Recommended sensitivities:

- `qps_scale`
- `boot_delay_steps`
- `max_instances`

### Ablation study

Recommended ablations for the best SLA-aware candidate:

- remove prediction contribution
- remove queue/external contribution
- remove recent-violation contribution
- compare lower vs higher aggressiveness

## Analysis Outputs

Use:

```bash
python3 analysis/generate_report_assets.py --results-dir data/processed/report_main_controlled --output-dir analysis/output/main_controlled
```

The analysis script produces:

- `summary_table.csv`
- `summary_table.md`
- `focused_comparison.md`
- `cost_vs_violation.png`
- `cost_vs_p99.png`
- `sensitivity_qps_scale.png`
- `sensitivity_boot_delay.png`
- `sensitivity_max_instances.png`
- `overscaling_diagnostics.md`
- representative `burst`, `stable`, and `recovery` window figures
- `case_windows.md`

Interpret the policy families as:

- `util_base`: strongest low-complexity reactive baseline
- `sla_required_capacity`: direct required-capacity proactive baseline
- `forecast_only`: forecast-driven proactive control without richer SLA-risk blending
- `sla_aware_tuned` and `sla_aware_ema_tuned`: main tuned SLA-aware candidates

## Report Structure

Use this order:

1. `Introduction`
2. `Background and Problem Setup`
3. `Methodology`
4. `Experimental Design`
5. `Results`
6. `Discussion`
7. `Limitations and Future Work`
8. `Conclusion`

## Acceptance Checklist

- all policies evaluated under one canonical main setup
- one table and at least three figures included in the report
- at least one sensitivity analysis included
- at least one ablation study included
- discussion includes one success case and one failure case for SLA-aware scaling
- every reported figure can be regenerated from scripts
