# Tuned Result Analysis Report

## Purpose

This document explains:

- what was changed in the SLA-aware redesign,
- why those changes were needed,
- what the new benchmark results show,
- and how those results should be used in the final report.

This analysis replaces the earlier interpretation where the SLA-aware policies were so aggressive that they overscaled immediately and made the ablation study uninformative.

## 1. Why the redesign was needed

In the earlier version of the project, both `sla_aware` and `sla_aware_ema` had a serious failure mode:

- they scaled up almost immediately from the start of the trace,
- they kept scaling even when utilization was very low,
- they reached very high instance counts too easily,
- and most ablation variants produced nearly identical results.

That created two problems:

1. The SLA-aware policies looked effective only because they were overprovisioning.
2. The ablation study did not actually explain which parts of the SLA-aware design mattered.

Because the project is centered on SLA-aware scaling, this was a major issue. The redesign was therefore intended to make the SLA-aware controllers operate in a real tradeoff regime instead of a saturated one.

## 2. What was changed

### 2.1 Retuning the SLA-aware policies

Both SLA-aware controllers were redesigned moderately rather than replaced.

For `sla_aware_tuned`:

- the scale-up threshold was raised,
- latency pressure was softened so it only mattered more strongly when closer to the SLA threshold,
- queue and external weights were reduced,
- demand-rush conditions were made stricter,
- and scale-up based purely on smoothed risk was no longer allowed unless stronger conditions were present.

For `sla_aware_ema_tuned`:

- `risk_up_threshold` was increased,
- `forecast_weight`, `trend_gain`, and `ema_alpha` were reduced,
- `scale_up_step` was reduced,
- and smoothed risk was no longer enough on its own to justify scale-up.

### 2.2 Adding anti-overscaling guards

Both tuned policies were given explicit overscaling controls:

- slack-aware suppression when capacity headroom is already large,
- quiet-state detection across multiple recent steps,
- prevention of repeated scale-up during low-utilization, low-risk periods,
- and more reachable downscale conditions.

This was the most important design change. The purpose was to stop the policy from continuing to climb simply because some blended risk score remained slightly elevated.

### 2.3 Creating a controlled report setup

The report experiment structure was also narrowed so that the main comparison focuses on:

- `util_base`
- `sla_aware_tuned`
- `sla_aware_ema_tuned`

instead of mixing too many baseline families into the main report argument.

The new scenarios are:

- `main_controlled`
- `harder_qps`
- `harder_boot`
- `harder_capacity`

### 2.4 Retuning the ablation study

The ablation study was moved into a harder regime with:

- higher `qps_scale`,
- larger boot delay,
- and reduced `max_instances`.

This was done specifically to ensure that policy component differences would have a visible effect on behavior.

## 3. Main controlled result

From `data/processed/report_main_controlled/benchmark_results.csv`:

| Policy | Cost($) | Inst-Hr | Viol Rate | P99 (ms) | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `util_base` | 149.021 | 119.217 | 0.008146 | 1462.6 | 0.4248 |
| `sla_aware_tuned` | 264.292 | 211.433 | 0.001004 | 636.1 | 0.2385 |
| `sla_aware_ema_tuned` | 250.688 | 200.550 | 0.001004 | 636.9 | 0.2638 |

### Interpretation

This is a strong improvement over the earlier project state.

The tuned SLA-aware policies now:

- still outperform `util_base` on SLA violation rate,
- still greatly reduce p99 latency,
- but no longer do so through the extreme runaway scaling seen previously.

At the same time:

- the improvement is not free,
- the cost is meaningfully higher than `util_base`,
- so the result is now a real cost-SLA tradeoff rather than a trivial “scale everything up” result.

Among the tuned variants:

- `sla_aware_ema_tuned` is currently the stronger candidate,
- because it achieves nearly the same SLA and p99 outcome as `sla_aware_tuned`,
- while using slightly fewer instance-hours and showing better efficiency.

## 4. Early-trace behavior after tuning

The first 20-35 rows of the series files show the most important behavioral improvement.

### `util_base`

`util_base` still behaves like the low-cost baseline:

- it scales down aggressively early,
- it stays lean under low load,
- and it only begins scaling up when load becomes more substantial.

### `sla_aware_tuned`

`sla_aware_tuned` no longer ramps upward immediately from the beginning:

- it stays around `7-8` instances for a long early period,
- it does not climb monotonically toward the cap,
- and its early scaling behavior is much more restrained.

### `sla_aware_ema_tuned`

`sla_aware_ema_tuned` is even more conservative early:

- it scales down from `7` toward `2`,
- stays restrained during low-load periods,
- and only begins to scale when stronger demand signals appear.

This confirms that the redesign fixed the earlier pathological behavior where SLA-aware policies began scaling almost from step 0 regardless of actual pressure.

## 5. Harder scenario results

### 5.1 Higher-QPS scenario

From `data/processed/report_harder_qps/benchmark_results.csv`:

| Policy | Cost($) | Viol Rate | P99 (ms) |
| --- | ---: | ---: | ---: |
| `util_base` | 201.708 | 0.005704 | 1799.9 |
| `sla_aware_tuned` | 349.000 | 0.000000 | 631.5 |
| `sla_aware_ema_tuned` | 329.813 | 0.000000 | 636.1 |

### Interpretation

This is one of the strongest results in the entire project.

Under a harder offered-load setting:

- `util_base` remains a good baseline but still has nonzero violations,
- both tuned SLA-aware policies eliminate violations entirely,
- and they maintain dramatically lower tail latency.

This supports the strongest positive claim available in the report:

> under more demanding conditions, tuned SLA-aware scaling can provide clear SLA protection beyond what utilization-threshold scaling provides.

Again, the cost is higher, but now the benefit is very clear.

### 5.2 Larger boot delay

From `data/processed/report_harder_boot/benchmark_results.csv`:

| Policy | Cost($) | Viol Rate | P99 (ms) |
| --- | ---: | ---: | ---: |
| `util_base` | 159.604 | 0.008146 | 1534.7 |
| `sla_aware_tuned` | 262.792 | 0.000000 | 636.1 |
| `sla_aware_ema_tuned` | 250.500 | 0.000000 | 636.9 |

### Interpretation

This supports the proactive-scaling argument well:

- when boot delay increases, proactive policies are supposed to have an advantage,
- and that is exactly what the result shows.

The tuned SLA-aware policies preserve SLA performance even when delayed scaling becomes more expensive.

### 5.3 Capacity-limited scenario

From `data/processed/report_harder_capacity/benchmark_results.csv`:

| Policy | Cost($) | Viol Rate | P99 (ms) |
| --- | ---: | ---: | ---: |
| `util_base` | 149.021 | 0.008146 | 1462.6 |
| `sla_aware_tuned` | 235.500 | 0.000000 | 636.1 |
| `sla_aware_ema_tuned` | 219.625 | 0.000000 | 636.1 |

### Interpretation

This is also important because it shows that the SLA-aware result is not purely an artifact of giving the policy unlimited headroom.

Even when maximum capacity is reduced:

- both tuned SLA-aware variants still keep violation at zero,
- and `sla_aware_ema_tuned` remains the cheaper of the two tuned SLA-aware options.

That makes the result more credible.

## 6. Overscaling diagnostics

From `analysis/output/main_controlled/overscaling_diagnostics.md`:

- `sla_aware_tuned` still has several overscaling episodes
- `sla_aware_ema_tuned` also still has several overscaling episodes

### Interpretation

This means the redesign improved the problem but did not eliminate it completely.

That is actually useful for the report:

- before tuning, overscaling was immediate and overwhelming,
- after tuning, overscaling still exists, but it is later and more localized,
- so the policies are now interpretable rather than obviously broken.

This gives the report a balanced conclusion:

- tuned SLA-aware control is now effective and meaningfully improved,
- but cost control is still incomplete.

## 7. Ablation study after redesign

From `data/processed/report_ablation/benchmark_results.csv`:

| Variant | Cost($) | Viol Rate | P99 (ms) |
| --- | ---: | ---: | ---: |
| `sla_aware_ema_tuned` | 257.479 | 0.000000 | 727.7 |
| `no_prediction` | 247.000 | 0.000000 | 750.8 |
| `no_queue_external` | 251.833 | 0.000000 | 690.5 |
| `no_violation` | 257.479 | 0.000000 | 727.7 |
| `low_aggr` | 240.396 | 0.000000 | 813.4 |
| `high_aggr` | 399.375 | 0.000000 | 636.9 |

### Interpretation

This is the second major success of the redesign.

Before the redesign, the ablation results were nearly flat and could not support any real interpretation.

Now the ablations clearly show:

### Aggressiveness matters a lot

- `low_aggr` is cheaper but much worse on p99
- `high_aggr` is far more expensive but gets the best p99

This means the cost-SLA tradeoff is now visible and measurable.

### Prediction matters somewhat

- removing prediction reduces cost slightly,
- but increases p99 from `727.7` to `750.8`

This suggests the predictive part of the controller is contributing, though not overwhelmingly.

### Queue/external signals may be misweighted

- removing queue/external signals slightly improves p99 here

That is an interesting negative result. It suggests these signals may currently be noisy, redundant, or weighted incorrectly under this trace.

### Recent-violation history currently adds almost nothing

- `no_violation` is effectively identical to the base tuned EMA policy

This suggests the recent-violation term is not important in the current tuned regime.

This is exactly the kind of ablation study that the project needed. It now supports real claims about which components matter and which do not.

## 8. Best-supported conclusions now

The strongest conclusions supported by the new results are:

### 8.1 `util_base` is still the strongest practical low-cost baseline

It remains much cheaper than the tuned SLA-aware policies and behaves efficiently.

### 8.2 Tuned SLA-aware policies now provide a real, defensible SLA advantage

They:

- reduce violation rate substantially,
- reduce p99 dramatically,
- and perform especially well under harder conditions.

### 8.3 `sla_aware_ema_tuned` is the best SLA-aware candidate

It is currently preferable to `sla_aware_tuned` because:

- its SLA result is nearly identical,
- it is slightly cheaper,
- and its efficiency is somewhat better.

### 8.4 The redesign made the ablation study meaningful

This is one of the most important achievements of the redesign.

The project can now say something specific:

- controller aggressiveness is a major factor,
- prediction contributes modestly,
- recent-violation input contributes little,
- and queue/external terms may need further reconsideration.

## 9. What the final report should now say

The report can now move from a weak conclusion to a strong one.

### Earlier weak conclusion

Earlier, the honest conclusion was:

> SLA-aware policies improve SLA only because they overprovision, and their components cannot be meaningfully distinguished.

### New stronger conclusion

Now the better conclusion is:

> After retuning, SLA-aware policies no longer overscale pathologically from the beginning of the trace. They achieve significantly lower SLA violation rates and substantially lower p99 latency than the utilization-threshold baseline, especially under harder load and scaling-delay conditions. Among the tuned variants, the EMA-based controller provides the best overall SLA-aware tradeoff. The redesigned ablation study also shows that controller aggressiveness is the dominant factor, prediction contributes modestly, and some auxiliary risk signals remain weak or poorly aligned under the current trace-model setting.

That is a much stronger report result.

## 10. Remaining limitations

The redesigned project is stronger, but there are still limitations:

- overscaling has been reduced, not eliminated
- the cost gap versus `util_base` is still real
- all ablation variants still achieve zero violation in the harder ablation scenario, so p99 and cost remain the main differentiators
- queue/external signals may need further tuning or justification
- the result is still tied to a simplified trace-driven simulator, not a real deployment

These limitations should remain explicit in the report.

## 11. Bottom line

The redesign was successful.

It did three important things:

1. It fixed the worst early overscaling behavior.
2. It made the SLA-aware policies competitive in a more realistic way.
3. It made the ablation study interpretable.

The project now supports a credible main claim:

> tuned SLA-aware scaling can provide meaningful SLA protection beyond a utilization-threshold baseline, but the main remaining challenge is cost control rather than basic effectiveness.

This is now strong enough to serve as the core result of the SLA-aware section in the final report.
