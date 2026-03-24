# Full Result Analysis

This report consolidates the benchmark summary, focused SLA-aware comparison, scenario-level interpretation, and overscaling diagnostics into one file.

## main_controlled

- `sla_required_capacity`: cost=111.604, viol_rate=0.01050, p99=1848.2, eff=0.555
- `reactive`: cost=114.896, viol_rate=0.05486, p99=2726.9, eff=0.556
- `util_base`: cost=144.917, viol_rate=0.00999, p99=1676.2, eff=0.447
- `sla_aware_ema_tuned`: cost=222.229, viol_rate=0.00100, p99=636.1, eff=0.295
- `sla_aware_tuned`: cost=244.417, viol_rate=0.00100, p99=641.6, eff=0.260
- `static`: cost=272.500, viol_rate=0.03182, p99=1810.4, eff=0.384
- `forecast_only`: cost=523.583, viol_rate=0.00000, p99=631.5, eff=0.186

Interpretation:
- `sla_required_capacity` is the lightweight proactive baseline: cost 111.604, violation rate 0.01050, p99 1848.2.
- `forecast_only` isolates prediction-driven proactive scaling with cost 523.583 and p99 631.5.
- `sla_aware_ema_tuned` improves violation rate from 0.00999 to 0.00100 while increasing cost from 144.917 to 222.229.
- `sla_aware_tuned` reaches similar SLA outcomes with cost 244.417.

## harder_qps

- `sla_required_capacity`: cost=146.792, viol_rate=0.02279, p99=2230.9, eff=0.578
- `reactive`: cost=150.062, viol_rate=0.05991, p99=2976.7, eff=0.592
- `util_base`: cost=196.396, viol_rate=0.01333, p99=1871.8, eff=0.455
- `static`: cost=272.500, viol_rate=0.65733, p99=21301.1, eff=0.511
- `sla_aware_ema_tuned`: cost=296.333, viol_rate=0.00000, p99=641.6, eff=0.305
- `sla_aware_tuned`: cost=318.979, viol_rate=0.00000, p99=636.9, eff=0.265
- `forecast_only`: cost=574.438, viol_rate=0.00000, p99=636.1, eff=0.213

Interpretation:
- `sla_required_capacity` is the lightweight proactive baseline: cost 146.792, violation rate 0.02279, p99 2230.9.
- `forecast_only` isolates prediction-driven proactive scaling with cost 574.438 and p99 636.1.
- `sla_aware_ema_tuned` improves violation rate from 0.01333 to 0.00000 while increasing cost from 196.396 to 296.333.
- `sla_aware_tuned` reaches similar SLA outcomes with cost 318.979.

## harder_boot

- `sla_required_capacity`: cost=110.667, viol_rate=0.03211, p99=2149.3, eff=0.568
- `reactive`: cost=115.479, viol_rate=0.05725, p99=2781.2, eff=0.557
- `util_base`: cost=144.083, viol_rate=0.01165, p99=1848.2, eff=0.460
- `forecast_only`: cost=190.583, viol_rate=0.00000, p99=641.2, eff=0.321
- `sla_aware_ema_tuned`: cost=198.667, viol_rate=0.00361, p99=688.0, eff=0.335
- `sla_aware_tuned`: cost=214.479, viol_rate=0.00100, p99=647.4, eff=0.299
- `static`: cost=272.500, viol_rate=0.03182, p99=1810.4, eff=0.384

Interpretation:
- `sla_required_capacity` is the lightweight proactive baseline: cost 110.667, violation rate 0.03211, p99 2149.3.
- `forecast_only` isolates prediction-driven proactive scaling with cost 190.583 and p99 641.2.
- `sla_aware_ema_tuned` improves violation rate from 0.01165 to 0.00361 while increasing cost from 144.083 to 198.667.
- `sla_aware_tuned` reaches similar SLA outcomes with cost 214.479.

## harder_capacity

- `sla_required_capacity`: cost=111.604, viol_rate=0.01050, p99=1848.2, eff=0.555
- `reactive`: cost=114.896, viol_rate=0.05486, p99=2726.9, eff=0.556
- `util_base`: cost=144.917, viol_rate=0.00999, p99=1676.2, eff=0.447
- `sla_aware_ema_tuned`: cost=214.438, viol_rate=0.00100, p99=636.1, eff=0.302
- `sla_aware_tuned`: cost=234.146, viol_rate=0.00100, p99=641.6, eff=0.269
- `static`: cost=272.500, viol_rate=0.03182, p99=1810.4, eff=0.384
- `forecast_only`: cost=300.250, viol_rate=0.00000, p99=631.5, eff=0.245

Interpretation:
- `sla_required_capacity` is the lightweight proactive baseline: cost 111.604, violation rate 0.01050, p99 1848.2.
- `forecast_only` isolates prediction-driven proactive scaling with cost 300.250 and p99 631.5.
- `sla_aware_ema_tuned` improves violation rate from 0.00999 to 0.00100 while increasing cost from 144.917 to 214.438.
- `sla_aware_tuned` reaches similar SLA outcomes with cost 234.146.

## ablation

- `sla_aware_ema_low_aggr`: cost=213.208, viol_rate=0.00537, p99=1317.3, eff=0.419
- `sla_aware_ema_no_prediction`: cost=221.479, viol_rate=0.00611, p99=1147.7, eff=0.401
- `sla_aware_ema_no_queue_external`: cost=224.646, viol_rate=0.00654, p99=1050.9, eff=0.395
- `sla_aware_ema_tuned`: cost=225.854, viol_rate=0.00654, p99=1135.4, eff=0.393
- `sla_aware_ema_no_violation`: cost=225.854, viol_rate=0.00654, p99=1135.4, eff=0.393
- `sla_aware_ema_high_aggr`: cost=398.125, viol_rate=0.00000, p99=636.9, eff=0.247

Interpretation:
- Lowest p99 variant: `sla_aware_ema_high_aggr`.
- Lowest cost variant: `sla_aware_ema_low_aggr`.
- The ablation is meaningful only if these variants differ in cost or p99.

## Overscaling

- `forecast_only` shows 130 benign scale-up steps.
- `reactive` shows 2 benign scale-up steps.
- `sla_aware_ema_tuned` shows 89 benign scale-up steps.
- `sla_aware_tuned` shows 59 benign scale-up steps.
- `sla_required_capacity` shows 12 benign scale-up steps.
- `util_base` shows 17 benign scale-up steps.

## Case Study Windows

- `burst` window: steps 261 to 290
- `stable` window: steps 1056 to 1085
- `recovery` window: steps 309 to 338

## Bottom Line

- Use `util_base` as the strongest low-cost reactive baseline.
- Use `sla_required_capacity` as the simple proactive baseline contributed by direct capacity sizing.
- Use `sla_aware_ema_tuned` as the strongest current SLA-aware candidate.
- Use the ablation section to discuss aggressiveness, prediction, and weak signal contributions.

