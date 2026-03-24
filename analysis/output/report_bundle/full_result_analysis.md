# Full Result Analysis

This report consolidates the benchmark summary, focused SLA-aware comparison, scenario-level interpretation, and overscaling diagnostics into one file.

## main_controlled

- `sla_required_capacity`: cost=111.417, viol_rate=0.00318, p99=1525.2, eff=0.536
- `reactive`: cost=116.292, viol_rate=0.05237, p99=2661.7, eff=0.546
- `util_base`: cost=149.021, viol_rate=0.00815, p99=1462.6, eff=0.425
- `sla_aware_ema_tuned`: cost=250.688, viol_rate=0.00100, p99=636.9, eff=0.264
- `sla_aware_tuned`: cost=264.292, viol_rate=0.00100, p99=636.1, eff=0.238
- `static`: cost=272.500, viol_rate=0.03182, p99=1810.4, eff=0.384
- `forecast_only`: cost=622.167, viol_rate=0.00000, p99=627.4, eff=0.138

Interpretation:
- `sla_required_capacity` is the lightweight proactive baseline: cost 111.417, violation rate 0.00318, p99 1525.2.
- `forecast_only` isolates prediction-driven proactive scaling with cost 622.167 and p99 627.4.
- `sla_aware_ema_tuned` improves violation rate from 0.00815 to 0.00100 while increasing cost from 149.021 to 250.688.
- `sla_aware_tuned` reaches similar SLA outcomes with cost 264.292.

## harder_qps

- `sla_required_capacity`: cost=146.125, viol_rate=0.00679, p99=1941.0, eff=0.562
- `reactive`: cost=153.458, viol_rate=0.05128, p99=2896.6, eff=0.575
- `util_base`: cost=201.708, viol_rate=0.00570, p99=1799.9, eff=0.429
- `static`: cost=272.500, viol_rate=0.65733, p99=21301.1, eff=0.511
- `sla_aware_ema_tuned`: cost=329.812, viol_rate=0.00000, p99=636.1, eff=0.270
- `sla_aware_tuned`: cost=349.000, viol_rate=0.00000, p99=631.5, eff=0.241
- `forecast_only`: cost=645.625, viol_rate=0.00000, p99=627.4, eff=0.170

Interpretation:
- `sla_required_capacity` is the lightweight proactive baseline: cost 146.125, violation rate 0.00679, p99 1941.0.
- `forecast_only` isolates prediction-driven proactive scaling with cost 645.625 and p99 627.4.
- `sla_aware_ema_tuned` improves violation rate from 0.00570 to 0.00000 while increasing cost from 201.708 to 329.812.
- `sla_aware_tuned` reaches similar SLA outcomes with cost 349.000.

## harder_boot

- `sla_required_capacity`: cost=119.667, viol_rate=0.01738, p99=2067.9, eff=0.517
- `reactive`: cost=142.542, viol_rate=0.04213, p99=2579.7, eff=0.510
- `util_base`: cost=159.604, viol_rate=0.00815, p99=1534.7, eff=0.404
- `sla_aware_ema_tuned`: cost=250.500, viol_rate=0.00000, p99=636.9, eff=0.261
- `sla_aware_tuned`: cost=262.792, viol_rate=0.00000, p99=636.1, eff=0.237
- `static`: cost=272.500, viol_rate=0.03182, p99=1810.4, eff=0.384
- `forecast_only`: cost=618.062, viol_rate=0.00000, p99=627.4, eff=0.141

Interpretation:
- `sla_required_capacity` is the lightweight proactive baseline: cost 119.667, violation rate 0.01738, p99 2067.9.
- `forecast_only` isolates prediction-driven proactive scaling with cost 618.062 and p99 627.4.
- `sla_aware_ema_tuned` improves violation rate from 0.00815 to 0.00000 while increasing cost from 159.604 to 250.500.
- `sla_aware_tuned` reaches similar SLA outcomes with cost 262.792.

## harder_capacity

- `sla_required_capacity`: cost=111.417, viol_rate=0.00318, p99=1525.2, eff=0.536
- `reactive`: cost=116.292, viol_rate=0.05237, p99=2661.7, eff=0.546
- `util_base`: cost=149.021, viol_rate=0.00815, p99=1462.6, eff=0.425
- `sla_aware_ema_tuned`: cost=219.625, viol_rate=0.00000, p99=636.1, eff=0.284
- `sla_aware_tuned`: cost=235.500, viol_rate=0.00000, p99=636.1, eff=0.258
- `static`: cost=272.500, viol_rate=0.03182, p99=1810.4, eff=0.384
- `forecast_only`: cost=327.292, viol_rate=0.00000, p99=627.4, eff=0.220

Interpretation:
- `sla_required_capacity` is the lightweight proactive baseline: cost 111.417, violation rate 0.00318, p99 1525.2.
- `forecast_only` isolates prediction-driven proactive scaling with cost 327.292 and p99 627.4.
- `sla_aware_ema_tuned` improves violation rate from 0.00815 to 0.00000 while increasing cost from 149.021 to 219.625.
- `sla_aware_tuned` reaches similar SLA outcomes with cost 235.500.

## ablation

- `sla_aware_ema_low_aggr`: cost=240.396, viol_rate=0.00000, p99=813.4, eff=0.352
- `sla_aware_ema_no_prediction`: cost=247.000, viol_rate=0.00000, p99=750.8, eff=0.341
- `sla_aware_ema_no_queue_external`: cost=251.833, viol_rate=0.00000, p99=690.5, eff=0.335
- `sla_aware_ema_tuned`: cost=257.479, viol_rate=0.00000, p99=727.7, eff=0.329
- `sla_aware_ema_no_violation`: cost=257.479, viol_rate=0.00000, p99=727.7, eff=0.329
- `sla_aware_ema_high_aggr`: cost=399.375, viol_rate=0.00000, p99=636.9, eff=0.246

Interpretation:
- Lowest p99 variant: `sla_aware_ema_high_aggr`.
- Lowest cost variant: `sla_aware_ema_low_aggr`.
- The ablation is meaningful only if these variants differ in cost or p99.

## Overscaling

- `forecast_only` shows 155 benign scale-up steps.
- `reactive` shows 2 benign scale-up steps.
- `sla_aware_ema_tuned` shows 113 benign scale-up steps.
- `sla_aware_tuned` shows 66 benign scale-up steps.
- `sla_required_capacity` shows 8 benign scale-up steps.
- `util_base` shows 23 benign scale-up steps.

## Case Study Windows

- `burst` window: steps 261 to 290
- `stable` window: steps 1056 to 1085
- `recovery` window: steps 309 to 338

## Bottom Line

- Use `util_base` as the strongest low-cost reactive baseline.
- Use `sla_required_capacity` as the simple proactive baseline contributed by direct capacity sizing.
- Use `sla_aware_ema_tuned` as the strongest current SLA-aware candidate.
- Use the ablation section to discuss aggressiveness, prediction, and weak signal contributions.

