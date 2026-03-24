# Result Analysis Report

## Purpose

This document interprets the current canonical results after fixing the simulator accounting and pending-scaleup scheduling issues.

The analysis below is based on:

- `report_main_controlled`
- `report_harder_qps`
- `report_harder_boot`
- `report_harder_capacity`
- `report_ablation`

These are the runs now produced by the current `scripts/run_report_experiments.sh` workflow.

## 1. Main Controlled Comparison

Source: `data/processed/report_main_controlled/benchmark_results.csv`

| Policy | Cost($) | Inst-Hr | Viol Rate | P99 (ms) | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sla_required_capacity` | 111.604 | 89.283 | 0.010500 | 1848.2 | 0.555 |
| `reactive` | 114.896 | 91.917 | 0.054860 | 2726.9 | 0.556 |
| `util_base` | 144.917 | 115.933 | 0.009987 | 1676.2 | 0.447 |
| `sla_aware_ema_tuned` | 222.229 | 177.783 | 0.001004 | 636.1 | 0.295 |
| `sla_aware_tuned` | 244.417 | 195.533 | 0.001004 | 641.6 | 0.260 |
| `static` | 272.500 | 218.000 | 0.031825 | 1810.4 | 0.384 |
| `forecast_only` | 523.583 | 418.867 | 0.000000 | 631.5 | 0.186 |

## 2. Main Interpretation

The corrected simulator makes the trade-off cleaner than before.

- `reactive` is still cheap, but clearly too late to protect SLA well.
- `util_base` remains the strongest reactive baseline.
- `sla_required_capacity` is now clearly a cheap proactive baseline, but not a strong latency optimizer.
- `forecast_only` remains extremely protective, but still expensive.
- `sla_aware_ema_tuned` is now the best overall tuned SLA-aware candidate.
- `sla_aware_tuned` is still competitive, but slightly weaker on both cost and p99.

The strongest headline result is:

> the tuned SLA-aware policies preserve most of the latency benefit of aggressive proactive control while using much less capacity than `forecast_only`.

## 3. What the Friend Policy Adds

The `sla_required_capacity` policy is still worth keeping.

It now serves a very clear role in the study:

- cheaper than `util_base`
- dramatically better than `reactive`
- much cheaper than the tuned SLA-aware controllers
- but materially worse on p99 than the tuned policies

In the main run:

- `sla_required_capacity` costs `111.604`
- `util_base` costs `144.917`
- `sla_aware_ema_tuned` costs `222.229`
- `forecast_only` costs `523.583`

So it is a real intermediate baseline:

- better than purely reactive control
- cheaper than tuned proactive control
- but not strong enough to be the headline policy

That is exactly the role it should play in the report.

## 4. Tuned SLA-Aware Policies

### `sla_aware_ema_tuned`

This is the strongest overall policy in the current results.

Compared with `util_base`, it:

- reduces violation rate from `0.009987` to `0.001004`
- reduces p99 from `1676.2 ms` to `636.1 ms`
- increases cost from `144.917` to `222.229`

That is a substantial improvement in SLA behavior for a moderate increase in cost.

### `sla_aware_tuned`

This remains a valid tuned candidate, but it is now dominated by `sla_aware_ema_tuned`:

- higher cost: `244.417` vs `222.229`
- slightly worse p99: `641.6 ms` vs `636.1 ms`
- same violation rate: `0.001004`

So if the report needs one main SLA-aware policy, it should now be `sla_aware_ema_tuned`.

## 5. Forecast-Only Baseline

`forecast_only` still proves an important point:

- pure proactive prediction can nearly eliminate violations
- but the cost is still very high

At `523.583`, it is:

- far more expensive than `sla_aware_ema_tuned`
- far more expensive than `sla_required_capacity`

This makes `forecast_only` useful as a strong proactive control baseline, but not as the preferred final policy.

## 6. Robustness Checks

## 6.1 Higher QPS

Source: `data/processed/report_harder_qps/benchmark_results.csv`

Key results:

- `sla_required_capacity`: cost `146.792`, viol `0.02279`, p99 `2230.9`
- `util_base`: cost `196.396`, viol `0.01333`, p99 `1871.8`
- `sla_aware_ema_tuned`: cost `296.333`, viol `0.00000`, p99 `641.6`
- `sla_aware_tuned`: cost `318.979`, viol `0.00000`, p99 `636.9`
- `forecast_only`: cost `574.438`, viol `0.00000`, p99 `636.1`
- `static`: viol `0.657333`, p99 `21301.1`

Interpretation:

- `static` fails badly under heavier load
- the simple proactive baseline degrades substantially
- `util_base` also degrades
- the tuned SLA-aware policies remain robust

This is a strong argument for proactive control under higher offered load.

## 6.2 Larger Boot Delay

Source: `data/processed/report_harder_boot/benchmark_results.csv`

Key results:

- `sla_required_capacity`: cost `110.667`, viol `0.03211`, p99 `2149.3`
- `util_base`: cost `144.083`, viol `0.01165`, p99 `1848.2`
- `forecast_only`: cost `190.583`, viol `0.00000`, p99 `641.2`
- `sla_aware_ema_tuned`: cost `198.667`, viol `0.00361`, p99 `688.0`
- `sla_aware_tuned`: cost `214.479`, viol `0.00100`, p99 `647.4`

Interpretation:

- longer boot delay is now a more meaningful stressor than before
- even tuned proactive policies degrade somewhat
- `forecast_only` becomes much cheaper here because the corrected scheduler no longer over-queues pending scale-ups
- `sla_aware_tuned` is slightly safer than `sla_aware_ema_tuned` in this specific stress case

This is useful for the report because it shows that the tuned controllers are strong, but not completely insensitive to actuation delay.

## 6.3 Capacity Cap

Source: `data/processed/report_harder_capacity/benchmark_results.csv`

Key results:

- `sla_required_capacity`: cost `111.604`, viol `0.01050`, p99 `1848.2`
- `util_base`: cost `144.917`, viol `0.00999`, p99 `1676.2`
- `sla_aware_ema_tuned`: cost `214.438`, viol `0.00100`, p99 `636.1`
- `sla_aware_tuned`: cost `234.146`, viol `0.00100`, p99 `641.6`
- `forecast_only`: cost `300.250`, viol `0.00000`, p99 `631.5`

Interpretation:

- the tuned SLA-aware policies remain strong even under the lower max-instance cap
- `forecast_only` is less overprovisioned than in the unconstrained case, but is still not the best trade-off
- `sla_aware_ema_tuned` remains the strongest tuned option

## 7. EMA Ablation

Source: `data/processed/report_ablation/benchmark_results.csv`

| Policy | Cost($) | Viol Rate | P99 (ms) |
| --- | ---: | ---: | ---: |
| `sla_aware_ema_low_aggr` | 213.208 | 0.00537 | 1317.3 |
| `sla_aware_ema_no_prediction` | 221.479 | 0.00611 | 1147.7 |
| `sla_aware_ema_no_queue_external` | 224.646 | 0.00654 | 1050.9 |
| `sla_aware_ema_tuned` | 225.854 | 0.00654 | 1135.4 |
| `sla_aware_ema_no_violation` | 225.854 | 0.00654 | 1135.4 |
| `sla_aware_ema_high_aggr` | 398.125 | 0.00000 | 636.9 |

## 7.1 What the ablation now says

After the simulator fix, the ablation became more informative.

Clear observations:

- aggressiveness is still the strongest lever
- prediction removal hurts
- queue/external removal changes behavior, but does not collapse performance
- recent-violation weighting still has almost no effect in this setup

Interpretation:

- `high_aggr` buys near-perfect protection at very high cost
- `low_aggr` is cheapest, but sacrifices too much latency and violation performance
- `no_prediction` is worse than tuned EMA, which supports the value of forecasted demand
- `no_violation` being identical to tuned EMA suggests that recent violation rate is not an important differentiator in the current workload regime

This is now a much better ablation story than before, because the variants no longer look almost identical.

## 8. Overall Report Framing

The corrected final framing should be:

1. `reactive` is too late for strong SLA protection.
2. `util_base` is the best low-complexity reactive baseline.
3. `sla_required_capacity` is a useful simple proactive baseline.
4. `forecast_only` shows the power and cost of pure proactive prediction.
5. `sla_aware_ema_tuned` is the best overall SLA-aware policy in the current setup.

That yields a stronger and more defensible final conclusion:

> proactive scaling helps, but not all proactive methods are equally efficient. A simple required-capacity heuristic reduces violations cheaply, while a tuned EMA-based SLA-aware controller gives the best overall cost-latency-SLA trade-off among the current policies.

## 9. Remaining Limitations

The main limitations are still:

- heuristic latency and prediction models
- simulation-only evaluation
- some ablation terms remain weakly expressed
- tuned proactive control still costs more than the cheapest baselines

Those limitations should stay in the report, but they no longer obscure the main result.
