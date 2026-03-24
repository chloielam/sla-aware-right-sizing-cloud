# Technical Setup and Methodology

## Purpose

This file documents the full technical setup of the project:

- simulator structure
- trace preprocessing
- latency and queue formulas
- prediction formulas
- full policy logic
- tuned and ablation coefficients
- experiment scenarios

It is intended to be the exact technical reference for the report.

## 1. System Overview

The project is a discrete-time, trace-driven autoscaling simulator for GenAI serving. At each step, the simulator:

1. reads one aligned trace point
2. computes arrivals and service capacity
3. updates queue length and utilization
4. computes p99 latency
5. creates an observation for the policy
6. asks the policy for a target instance count
7. applies scale-up with boot delay or scale-down with cooldown
8. records cost, SLA violation, and time-series output

The simulator implementation is in [simulator.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/engine/simulator.py).

## 2. Trace Inputs and Preprocessing

### 2.1 Raw trace files

The simulator uses these raw data sources:

- `qps.csv`: workload demand
- `queue_rt_raw_anon.csv`: external queue pressure
- `controlnet_latency_data_anon.csv`: external latency signal
- `pod_gpu_duty_cycle_anon.csv`: GPU utilization context
- `pod_memory_util_anon.csv`: memory utilization context

### 2.2 Bucketing and aggregation

Trace values are bucketed by `bucket_seconds`.

For each metric:

- `bucket = int(timestamp_anon) // bucket_seconds * bucket_seconds`

Aggregation behavior:

- default `qps_agg = sum`
- supported values: `sum`, `mean`

### 2.3 QPS interpretation

The benchmark runner supports:

- `qps_mode = qps`
- `qps_mode = count`

Meaning:

- `qps`: values are treated as per-second QPS already
- `count`: values are treated as counts within the bucket and divided by `bucket_seconds`

Optional scaling:

- after interpretation, `qps_scale` is applied multiplicatively

### 2.4 Forward fill

For queue, latency, GPU, and memory series:

- missing timestamps are forward-filled
- if no previous value exists, the median of the series is used as default

### 2.5 Trace point structure

Each aligned trace point contains:

- `timestamp`
- `demand_qps`
- `external_queue`
- `external_latency_ms`
- `gpu_util_pct`
- `memory_util`

## 3. Simulator State and Core Equations

### 3.1 State variables

The simulator maintains:

- `active`: current active instances
- `pending_scaleups`: delayed scale-up events
- `queue_len`
- `cooldown_until`
- `recent_viol`: recent violation history
- aggregate histories for p99, utilization, and instances

### 3.2 Arrivals and capacity

At each step:

- `arrivals = max(0, demand_qps) * step_seconds`
- `capacity = active_instances * service_rate_qps_per_instance * step_seconds`

### 3.3 Queue update

- `queue_len = max(0, previous_queue_len + arrivals - capacity)`

### 3.4 Utilization

- `utilization = arrivals / capacity`

The simulator clips stored utilization for efficiency tracking to at most `1.5`, but the raw utilization value is still used in policy decisions before clipping in the history statistic.

### 3.5 Internal latency model

The internal p99 latency model is:

`model_p99 = base_latency_ms + queue_latency_factor_ms * queue_len + utilization_penalty_ms * max(0, utilization - 1.0)`

Meaning:

- `base_latency_ms`: latency floor
- `queue_latency_factor_ms * queue_len`: queue penalty
- `utilization_penalty_ms * max(0, utilization - 1.0)`: overload penalty only when utilization exceeds `1.0`

### 3.6 Final p99 latency

The final p99 latency used by the simulator is:

`external_tail_increase = max(0, external_latency_ms - base_latency_ms)`

`latency_p99 = model_p99 + external_latency_weight * external_tail_increase`

Then:

- `latency_p99 = max(1.0, latency_p99)`

### 3.7 SLA violation rule

A step is marked as violating if:

- `latency_p99 > sla_threshold_ms`

Request-weighted SLA violation rate is computed as:

`sla_violation_rate = violating_requests / total_requests`

where violating requests for a step equal all arrivals in that step if the step violated.

### 3.8 Cost model

At each step:

- `billed_instance_hours += active_instances_used_in_step * (step_seconds / 3600)`

Estimated dollar cost:

- `cost_estimate = billed_instance_hours * cost_per_instance_hour`

### 3.9 Resource efficiency

Resource efficiency is reported as:

- `resource_efficiency = min(1.0, average(utilization_history))`

The utilization history is computed from the same active instance count that actually served each step.

## 4. Prediction Signals Exposed to Policies

The policy does not forecast directly from raw files. It receives prediction-related fields through the simulator observation.

### 4.1 Predicted demand

The simulator computes:

- `lookahead = next lookahead_steps of the trace`
- `predicted_demand_qps = mean(lookahead demand_qps)`

If no lookahead exists:

- `predicted_demand_qps = current demand_qps`

### 4.2 Predicted latency

The simulator computes:

`predicted_latency_ms = model_p99 + external_latency_weight * max(0, external_latency_ms - base_latency_ms) + max(0, predicted_demand_qps - current_demand_qps) * 120.0 + external_queue * 0.2`

So predicted latency is a heuristic composed of:

- current modeled latency
- external latency pressure
- a penalty for forecasted demand growth
- an external queue penalty

### 4.3 Recent violation rate

The simulator stores the last `20` step-level SLA outcomes:

- `recent_violation_rate = average(recent_viol)`

## 5. Scaling Mechanics

### 5.1 Scale-up

If `target_instances` exceeds the capacity that is already active or already pending, only the unmet deficit is queued.

Operationally:

- the simulator sums all still-pending scale-up requests
- it computes `scaleup_deficit = target_instances - (active_instances + pending_instances)`
- if `scaleup_deficit > 0`, it queues:
- `(current_step + boot_delay_steps, scaleup_deficit)`

This prevents repeated double-counting of the same desired scale-up while new instances are still booting.

Instances only become active when the ready step arrives.

### 5.2 Scale-down

If `target_instances < active_instances` and the simulator is not in cooldown:

- scale-down is applied immediately
- `cooldown_until = current_step + cooldown_steps`

This asymmetry makes the policies more sensitive to boot delay than to scale-down delay.

The scale-down decision is made during the current step, but the metrics recorded for that step still correspond to the instance count that actually served the current step before the scale-down took effect.

So for each step, the recorded:

- queue length
- utilization
- latency
- `instances`
- and billed instance-hours

all reflect the active capacity used during that step.

The reduced instance count affects subsequent steps.

## 6. Policy Definitions

## 6.1 Static Policy

Implementation: [static.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/static.py)

Rule:

- always return `target_instances` from config

Current config:

- `target_instances = 12`

Main simulation coefficients:

- `min_instances = 2`
- `max_instances = 36`
- `initial_instances = 12`
- `service_rate_qps_per_instance = 0.7`
- `boot_delay_steps = 2`
- `cooldown_steps = 2`
- `base_latency_ms = 500.0`
- `queue_latency_factor_ms = 1.8`
- `utilization_penalty_ms = 2200.0`
- `external_latency_weight = 0.01`
- `sla_threshold_ms = 2300.0`
- `cost_per_instance_hour = 1.25`
- `lookahead_steps = 4`

## 6.2 Reactive Policy

Implementation: [util_threshold.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/util_threshold.py)

Internal state:

- `_viol_hist` with length `history_window`

Definitions:

- `violation = 1.0 if latency_p99_ms > sla_threshold_ms else 0.0`
- `recent = average(_viol_hist)`

Scale-up rule:

- if `latency_p99_ms > sla_threshold_ms`
- or `recent > 0.25`
- then `target += scale_up_step`

Scale-down rule:

- if `latency_p99_ms < sla_threshold_ms * recovery_latency_ratio`
- and `utilization < 0.55`
- and `recent <= quiet_violation_rate`
- then `target -= scale_down_step`

Current coefficients:

- `scale_up_step = 3`
- `scale_down_step = 1`
- `history_window = 8`
- `recovery_latency_ratio = 0.63`
- `quiet_violation_rate = 0.05`

Simulation coefficients:

- `min_instances = 2`
- `max_instances = 36`
- `initial_instances = 9`
- `service_rate_qps_per_instance = 1.0`
- `boot_delay_steps = 2`
- `cooldown_steps = 3`
- `base_latency_ms = 500.0`
- `queue_latency_factor_ms = 1.8`
- `utilization_penalty_ms = 2200.0`
- `external_latency_weight = 0.01`
- `sla_threshold_ms = 2300.0`
- `cost_per_instance_hour = 1.25`
- `lookahead_steps = 4`

## 6.3 Utilization-Threshold Policy (`util_base`)

Implementation: [util_threshold.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/util_threshold.py)

Internal state:

- `_lat_hist`, currently stored but not used in the decision rule

Scale-up rule:

- if `utilization > scale_up_util`
- or `queue_len > queue_up_threshold`
- then `target += scale_up_step`

Scale-down rule:

- if `utilization < scale_down_util`
- and `queue_len < queue_down_threshold`
- and `latency_p99_ms < 0.8 * sla_threshold_ms`
- then `target -= scale_down_step`

Current coefficients:

- `scale_up_util = 0.76`
- `scale_down_util = 0.45`
- `scale_up_step = 2`
- `scale_down_step = 1`
- `queue_up_threshold = 16.0`
- `queue_down_threshold = 5.0`
- `history_window = 10`

Simulation coefficients:

- `min_instances = 2`
- `max_instances = 36`
- `initial_instances = 10`
- `service_rate_qps_per_instance = 1.0`
- `boot_delay_steps = 2`
- `cooldown_steps = 3`
- `base_latency_ms = 500.0`
- `queue_latency_factor_ms = 1.8`
- `utilization_penalty_ms = 2200.0`
- `external_latency_weight = 0.01`
- `sla_threshold_ms = 2300.0`
- `cost_per_instance_hour = 1.25`
- `lookahead_steps = 4`

## 6.4 Forecast-Only Policy

Implementation: [forecast_only.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/forecast_only.py)

Internal state:

- `_forecast_hist` with length `history_window`

Definitions:

- `demand_gap_ratio = max(0, predicted_demand_qps - demand_qps) / max(1.0, demand_qps)`
- `predicted_utilization = utilization * (predicted_demand_qps / demand_qps)` when `demand_qps > 0`
- `predicted_latency_ratio = predicted_latency_ms / sla_threshold_ms`

Forecast score:

`forecast_score = 0.45 * predicted_latency_ratio + 0.35 * demand_gap_ratio + 0.20 * predicted_utilization`

Smoothed score:

- `smooth_forecast = average(_forecast_hist)`

Scale-up rule:

- if `smooth_forecast >= forecast_up_threshold`
- or `predicted_latency_ratio >= 0.95`
- or `predicted_utilization >= predicted_utilization_up`
- or `demand_gap_ratio >= 0.30`
- then `target += scale_up_step`

Scale-down rule:

- if `smooth_forecast <= forecast_down_threshold`
- and `predicted_utilization <= predicted_utilization_down`
- and `latency_p99_ms < 0.75 * sla_threshold_ms`
- then `target -= scale_down_step`

Current coefficients:

- `history_window = 8`
- `forecast_up_threshold = 0.95`
- `forecast_down_threshold = 0.60`
- `predicted_utilization_up = 0.82`
- `predicted_utilization_down = 0.50`
- `scale_up_step = 2`
- `scale_down_step = 1`

Simulation coefficients:

- `min_instances = 2`
- `max_instances = 36`
- `initial_instances = 9`
- `service_rate_qps_per_instance = 1.0`
- `boot_delay_steps = 2`
- `cooldown_steps = 3`
- `base_latency_ms = 500.0`
- `queue_latency_factor_ms = 1.8`
- `utilization_penalty_ms = 2200.0`
- `external_latency_weight = 0.01`
- `sla_threshold_ms = 2300.0`
- `cost_per_instance_hour = 1.25`
- `lookahead_steps = 5`

## 6.5 SLA-Required-Capacity Policy

Implementation: [sla_required_capacity.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/sla_required_capacity.py)

Purpose:

- provide a simple proactive baseline between `util_base` and the richer tuned SLA-aware controllers
- convert current demand and lookahead demand into a direct instance requirement
- keep scale-down conservative so it behaves like a safety-margin capacity planner rather than a high-frequency risk controller

Definitions:

- `planning_demand = max(demand_qps, demand_mix * predicted_demand_qps + (1 - demand_mix) * demand_qps)`
- `safe_capacity_per_instance = service_rate_qps_per_instance * target_utilization`
- `raw_target = ceil(planning_demand / safe_capacity_per_instance)`

Latency headroom rule:

- if `latency_p99_ms >= latency_headroom_ratio * sla_threshold_ms`
- then `raw_target += 1`

Scale-up rule:

- clamp `raw_target` to `[min_instances, max_instances]`
- if `raw_target > active_instances`
- return at most `active_instances + scale_up_cap`

Scale-down rule:

- only allow scale-down when:
- `queue_len <= 0.15 * queue_budget`
- `latency_p99_ms <= 0.70 * sla_threshold_ms`
- `recent_violation_rate <= 0.03`
- `utilization <= 0.80 * target_utilization`
- when allowed, decrease by at most `scale_down_cap` while not going below the direct capacity estimate

Current coefficients:

- `target_utilization = 0.85`
- `demand_mix = 0.70`
- `latency_headroom_ratio = 0.78`
- `scale_up_cap = 1`
- `scale_down_cap = 2`
- `queue_budget = 16.0`

Simulation coefficients:

- `min_instances = 2`
- `max_instances = 36`
- `initial_instances = 10`
- `service_rate_qps_per_instance = 1.0`
- `boot_delay_steps = 2`
- `cooldown_steps = 3`
- `base_latency_ms = 500.0`
- `queue_latency_factor_ms = 1.8`
- `utilization_penalty_ms = 2200.0`
- `external_latency_weight = 0.01`
- `sla_threshold_ms = 2300.0`
- `cost_per_instance_hour = 1.25`
- `lookahead_steps = 4`

Interpretation:

- unlike the tuned SLA-aware policies, this policy does not compute a blended risk score
- unlike `forecast_only`, it converts demand into an explicit required instance count
- it is best treated as a lightweight proactive baseline, not the main headline controller

## 6.6 Tuned Heuristic SLA-Aware Policy

Implementation: [sla_aware.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/sla_aware.py)

Internal state:

- `_risk_hist`
- `_demand_hist`
- `_quiet_hist`

Definitions:

- `trend = (latest_demand - oldest_demand) / oldest_demand`
- `queue_risk = queue_len / queue_budget`
- `trend_risk = max(0, trend)`
- `demand_agg_risk = (predicted_demand_qps - demand_qps) / demand_qps`
- `external_risk = external_queue_signal / (queue_budget * 1.5)`
- `latency_pressure = latency_p99_ms / sla_threshold_ms`
- `softened_latency_pressure = max(0, latency_pressure - 0.70)`
- `predicted_utilization = utilization * (1 + max(0, demand_agg_risk))`
- `capacity_slack_ratio = max(0, 1 / utilization - 1)` when utilization > 0, else very large

Blended risk:

`blended_risk = prediction_weight * softened_latency_pressure + queue_weight * queue_risk + trend_weight * trend_risk + demand_weight * clipped(demand_agg_risk) + violation_weight * recent_violation_rate + external_queue_weight * external_risk + external_latency_component_weight * (external_latency_signal_ms / sla_threshold_ms)`

Scale-up support variables:

- `recent_violation_spike = recent_violation_rate >= 0.10`
- `near_threshold = latency_p99_ms >= 0.95 * sla_threshold_ms`
- `sustained_risk = smooth_risk >= risk_up_threshold or trend_risk >= 0.70`
- `demand_rush = demand_agg_risk >= min_demand_rush_ratio`
- `queue_pressure = queue_len >= 0.75 * queue_budget`
- `high_util = utilization >= 0.82`
- `external_pressure = external_latency_signal_ms >= 0.95 * sla_threshold_ms`
- `hard_signal = near_threshold or queue_pressure or demand_rush`

Quiet-state definition:

- `utilization <= min_scale_up_utilization`
- `queue_len <= 0.15 * queue_budget`
- `recent_violation_rate <= quiet_violation_rate`
- `latency_p99_ms <= 0.55 * sla_threshold_ms`

Overscaling suppression:

- suppress if slack is high, no hard signal, no violation spike, and predicted utilization is low
- suppress repeated scale-up if all values in `_quiet_hist` indicate quiet state

Risk-based scale-up gate:

- `risk_scale_allowed = sustained_risk and (not require_hard_signal_for_risk_scale or hard_signal or predicted_utilization >= 0.85)`

Scale-up rule:

- if not suppressed and any of:
- `near_threshold`
- `recent_violation_spike`
- `demand_rush`
- `queue_pressure`
- `high_util`
- `external_pressure`
- `risk_scale_allowed`
- then `target += scale_up_step`

Scale-down rule:

- if `smooth_risk <= risk_down_threshold`
- and `latency_p99_ms < downscale_latency_ratio * sla_threshold_ms`
- and `utilization < downscale_utilization`
- and `recent_violation_rate <= quiet_violation_rate`
- and `queue_len < downscale_queue_ratio * queue_budget`
- and `predicted_utilization < downscale_utilization`
- then `target -= scale_down_step`

Tuned coefficients:

- `queue_budget = 12.0`
- `risk_up_threshold = 1.18`
- `risk_down_threshold = 0.52`
- `prediction_weight = 0.42`
- `queue_weight = 0.16`
- `trend_weight = 0.05`
- `demand_weight = 0.06`
- `external_queue_weight = 0.03`
- `external_latency_component_weight = 0.04`
- `scale_up_step = 1`
- `scale_down_step = 1`
- `history_window = 12`
- `trend_window = 6`
- `max_slack_ratio_for_scale_up = 2.0`
- `min_scale_up_utilization = 0.45`
- `require_hard_signal_for_risk_scale = true`
- `quiet_steps_before_hold = 5`
- `min_demand_rush_ratio = 0.42`
- `downscale_latency_ratio = 0.82`
- `downscale_utilization = 0.68`
- `downscale_queue_ratio = 0.40`
- `quiet_violation_rate = 0.02`

Simulation coefficients:

- `min_instances = 2`
- `max_instances = 30`
- `initial_instances = 8`
- `service_rate_qps_per_instance = 1.0`
- `boot_delay_steps = 2`
- `cooldown_steps = 3`
- `base_latency_ms = 500.0`
- `queue_latency_factor_ms = 1.8`
- `utilization_penalty_ms = 2200.0`
- `external_latency_weight = 0.01`
- `sla_threshold_ms = 2300.0`
- `cost_per_instance_hour = 1.25`
- `lookahead_steps = 5`

## 6.7 Tuned EMA SLA-Aware Policy

Implementation: [sla_aware_ema.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/sla_aware_ema.py)

Internal state:

- `_risk_hist`
- `_demand_hist`
- `_quiet_hist`
- `_ema_fast`
- `_ema_slow`
- `_prev_pred_err`

EMA update:

- `ema_fast = alpha * demand_now + (1 - alpha) * ema_fast`
- `ema_slow = alpha * demand_now * 0.65 + (1 - alpha) * ema_slow`

Trend estimate:

- `trend = (ema_fast - ema_slow) * trend_gain`

Forecast demand:

- `forecast_demand = predicted_demand_qps + trend`

Definitions:

- `demand_gap = max(0, forecast_demand - demand_qps)`
- `demand_gap_ratio = demand_gap / demand_qps`
- `forecast_gain = forecast_weight * demand_gap_ratio`
- `predicted_utilization = utilization * (forecast_demand / demand_qps)`
- `latency_pressure = latency_p99_ms / sla_threshold_ms`
- `softened_latency_pressure = max(0, latency_pressure - 0.72)`
- `queue_pressure = queue_len / queue_budget`
- `external_risk = external_queue_signal / (queue_budget * 1.3)`
- `ext_latency_ratio = external_latency_signal_ms / sla_threshold_ms`
- `violation_pressure = recent_violation_rate`
- `pred_err = max(0, predicted_demand_qps - demand_qps) / demand_qps`
- `pred_err_delta = max(0, pred_err - prev_pred_err)`
- `vol_risk = min(1.0, std_ratio(demand_history) * volatility_gain)`
- `capacity_slack_ratio = max(0, 1 / utilization - 1)` when utilization > 0

Blended risk:

`blended_risk = latency_weight * softened_latency_pressure + queue_weight * queue_pressure + 0.20 * forecast_gain + predicted_utilization_weight * min(1.0, predicted_utilization) + external_latency_component_weight * ext_latency_ratio + external_queue_weight * external_risk + violation_weight * violation_pressure + volatility_weight * vol_risk + prediction_error_delta_weight * pred_err_delta`

Scale-up support variables:

- `near_threshold = latency_p99_ms >= 0.95 * sla_threshold_ms`
- `queue_alarm = queue_pressure >= 0.85`
- `demand_alarm = demand_gap_ratio >= min_demand_rush_ratio`
- `util_alarm = predicted_utilization >= 0.90 or utilization >= 0.82`
- `hard_signal = near_threshold or queue_alarm or demand_alarm`

Quiet-state definition:

- `utilization <= min_scale_up_utilization`
- `queue_len <= 0.15 * queue_budget`
- `recent_violation_rate <= quiet_violation_rate`
- `latency_p99_ms <= 0.55 * sla_threshold_ms`

Overscaling suppression:

- suppress if slack is high, no hard signal, predicted utilization is low, and recent violation rate is quiet
- suppress repeated scale-up if all values in `_quiet_hist` are quiet

Risk-based scale-up gate:

- `risk_scale_allowed = smooth_risk >= risk_up_threshold and (not require_hard_signal_for_risk_scale or hard_signal or util_alarm)`

Scale-up rule:

- if not suppressed and any of:
- `near_threshold`
- `demand_alarm`
- `predicted_utilization >= 0.90`
- `queue_alarm`
- `utilization >= 0.82`
- `external_risk >= 0.65`
- `risk_scale_allowed`
- then `target += scale_up_step`

Scale-down rule:

- if `smooth_risk <= risk_down_threshold`
- and `latency_p99_ms <= downscale_latency_ratio * sla_threshold_ms`
- and `queue_len <= downscale_queue_ratio * queue_budget`
- and `recent_violation_rate <= quiet_violation_rate`
- and `utilization <= downscale_utilization`
- and `predicted_utilization <= downscale_utilization`
- then `target -= scale_down_step`

Tuned coefficients:

- `queue_budget = 12.0`
- `risk_up_threshold = 1.16`
- `risk_down_threshold = 0.50`
- `forecast_weight = 0.35`
- `ema_alpha = 0.25`
- `trend_gain = 0.90`
- `volatility_gain = 0.20`
- `latency_weight = 0.24`
- `queue_weight = 0.14`
- `predicted_utilization_weight = 0.12`
- `external_latency_component_weight = 0.03`
- `external_queue_weight = 0.02`
- `violation_weight = 0.05`
- `scale_up_step = 1`
- `scale_down_step = 1`
- `history_window = 12`
- `trend_window = 10`
- `max_slack_ratio_for_scale_up = 1.8`
- `min_scale_up_utilization = 0.45`
- `require_hard_signal_for_risk_scale = true`
- `quiet_steps_before_hold = 5`
- `min_demand_rush_ratio = 0.34`
- `downscale_latency_ratio = 0.82`
- `downscale_utilization = 0.66`
- `downscale_queue_ratio = 0.35`
- `quiet_violation_rate = 0.02`

Simulation coefficients:

- `min_instances = 2`
- `max_instances = 30`
- `initial_instances = 8`
- `service_rate_qps_per_instance = 1.0`
- `boot_delay_steps = 2`
- `cooldown_steps = 3`
- `base_latency_ms = 500.0`
- `queue_latency_factor_ms = 1.8`
- `utilization_penalty_ms = 2200.0`
- `external_latency_weight = 0.01`
- `sla_threshold_ms = 2300.0`
- `cost_per_instance_hour = 1.25`
- `lookahead_steps = 5`

## 6.8 Untuned EMA SLA-Aware Baseline

Implementation: [sla_aware_ema.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/sla_aware_ema.py)

This older baseline uses the same EMA-based controller structure as the tuned EMA policy, but with a much more aggressive parameterization.

Important parameter note:

- the active forecast coefficient name is `forecast_weight`
- older notes may refer to `prediction_weight`, but that is not the parameter currently read by the implementation

Current untuned EMA coefficients:

- `queue_budget = 12.0`
- `risk_up_threshold = 0.90`
- `risk_down_threshold = 0.30`
- `forecast_weight = 0.70`
- `ema_alpha = 0.45`
- `trend_gain = 1.80`
- `volatility_gain = 0.30`
- `scale_up_step = 2`
- `scale_down_step = 1`
- `history_window = 12`
- `trend_window = 10`

Simulation coefficients:

- `min_instances = 2`
- `max_instances = 36`
- `initial_instances = 9`
- `service_rate_qps_per_instance = 1.0`
- `boot_delay_steps = 2`
- `cooldown_steps = 3`
- `base_latency_ms = 500.0`
- `queue_latency_factor_ms = 1.8`
- `utilization_penalty_ms = 2200.0`
- `external_latency_weight = 0.01`
- `sla_threshold_ms = 2300.0`
- `cost_per_instance_hour = 1.25`
- `lookahead_steps = 5`

Interpretation:

- this untuned EMA baseline is retained mainly for comparison against the tuned EMA controller
- it is not the preferred headline result in the current report

## 7. EMA Ablation Variants

All ablation variants inherit the tuned EMA structure and change only selected coefficients.

## 7.1 `sla_aware_ema_no_prediction`

Differences from tuned EMA:

- `forecast_weight = 0.0`
- `trend_gain = 0.0`

Everything else remains aligned with the tuned EMA baseline.

## 7.2 `sla_aware_ema_no_queue_external`

Differences:

- `queue_weight = 0.0`
- `external_latency_component_weight = 0.0`
- `external_queue_weight = 0.0`

## 7.3 `sla_aware_ema_no_violation`

Difference:

- `violation_weight = 0.0`

## 7.4 `sla_aware_ema_low_aggr`

Lower-aggressiveness coefficients:

- `risk_up_threshold = 1.24`
- `risk_down_threshold = 0.56`
- `forecast_weight = 0.24`
- `ema_alpha = 0.20`
- `trend_gain = 0.65`
- `volatility_gain = 0.16`
- `latency_weight = 0.20`
- `queue_weight = 0.10`
- `predicted_utilization_weight = 0.10`
- `external_latency_component_weight = 0.02`
- `external_queue_weight = 0.01`
- `violation_weight = 0.04`
- `max_slack_ratio_for_scale_up = 1.5`
- `min_scale_up_utilization = 0.50`
- `min_demand_rush_ratio = 0.38`
- `downscale_latency_ratio = 0.84`
- `downscale_utilization = 0.68`
- `downscale_queue_ratio = 0.40`

## 7.5 `sla_aware_ema_high_aggr`

Higher-aggressiveness coefficients:

- `risk_up_threshold = 1.00`
- `risk_down_threshold = 0.42`
- `forecast_weight = 0.50`
- `ema_alpha = 0.35`
- `trend_gain = 1.30`
- `volatility_gain = 0.24`
- `latency_weight = 0.28`
- `queue_weight = 0.18`
- `predicted_utilization_weight = 0.14`
- `external_latency_component_weight = 0.04`
- `external_queue_weight = 0.03`
- `violation_weight = 0.06`
- `scale_up_step = 2`
- `max_slack_ratio_for_scale_up = 2.2`
- `min_scale_up_utilization = 0.40`
- `min_demand_rush_ratio = 0.30`
- `downscale_latency_ratio = 0.80`
- `downscale_utilization = 0.62`
- `downscale_queue_ratio = 0.32`

## 8. Current Experiment Scenarios

The report experiment script currently runs:

### 8.1 `main_controlled`

- full main policy set
- default QPS scale
- default boot delay
- includes `static`, `reactive`, `util_base`, `sla_required_capacity`, `forecast_only`, `sla_aware_tuned`, and `sla_aware_ema_tuned`

### 8.2 `harder_qps`

- same policy set
- `qps_scale = 1.35`

### 8.3 `harder_boot`

- same policy set
- `boot_delay_steps = 4`

### 8.4 `harder_capacity`

- same policy set
- `max_instances = 18`

### 8.5 `ablation`

- EMA ablation policy set only
- `qps_scale = 1.35`
- `boot_delay_steps = 4`
- `max_instances = 18`

## 9. Output Metrics

Each benchmark summary includes:

- `policy`
- `prediction_algorithm`
- `steps`
- `step_seconds`
- `simulated_hours`
- `avg_instances`
- `cost_instance_hours`
- `cost_estimate`
- `sla_violation_rate`
- `tail_latency_p99_ms`
- `resource_efficiency`

The current report bundle therefore supports two baseline layers:

- classical baselines: `static`, `reactive`, `util_base`
- proactive baselines: `sla_required_capacity` and `forecast_only`

This separation is useful in the report because it lets the tuned SLA-aware policies be compared against both:

- a cheap capacity-planning heuristic
- and a more aggressive forecast-driven proactive controller

Under the corrected simulator accounting, the current headline interpretation is:

- `util_base` is the strongest reactive baseline
- `sla_required_capacity` is a simple proactive baseline
- `forecast_only` is a strong but expensive proactive upper-bound baseline
- `sla_aware_ema_tuned` is the strongest overall SLA-aware candidate in the current setup

Each time-series file includes:

- `step`
- `timestamp`
- `policy`
- `prediction_algorithm`
- `instances`
- `target_instances`
- `arrivals`
- `queue_len`
- `utilization`
- `latency_p99_ms`
- `sla_violation`

## 10. Methodological Limitations

Important limitations of the current methodology:

- the latency model is heuristic, not learned from the trace
- `predicted_demand_qps` and `predicted_latency_ms` are heuristic lookahead signals
- queue and SLA violations are modeled at step level, not per-request with richer service-time distributions
- scale-up is delayed, but scale-down is immediate except for cooldown gating
- GPU and memory signals are present in observations but are not central to the current main policies
- the simulator is intended as a reproducible evaluation framework, not a production-faithful serving emulator

## 11. Recommended Citation Usage in the Report

If this file is used in the report, it can support:

- the simulator methodology section
- the policy formulation section
- the experimental setup section
- the ablation-study section

It is especially useful for listing the exact coefficients used in the tuned SLA-aware controllers and the EMA ablation variants.
It also documents the simpler `sla_required_capacity` baseline so that collaborative contributions can be evaluated in the same experimental framework.
