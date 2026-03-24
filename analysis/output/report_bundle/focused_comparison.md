# Focused Comparison

| scenario | policy | cost($) | viol_rate | p99(ms) | eff |
| --- | --- | --- | --- | --- | --- |
| ablation | sla_aware_ema_tuned | 257.479 | 0.00000 | 727.7 | 0.329 |
| harder_boot | forecast_only | 618.062 | 0.00000 | 627.4 | 0.141 |
| harder_boot | sla_aware_ema_tuned | 250.500 | 0.00000 | 636.9 | 0.261 |
| harder_boot | sla_aware_tuned | 262.792 | 0.00000 | 636.1 | 0.237 |
| harder_boot | sla_required_capacity | 119.667 | 0.01738 | 2067.9 | 0.517 |
| harder_boot | util_base | 159.604 | 0.00815 | 1534.7 | 0.404 |
| harder_capacity | forecast_only | 327.292 | 0.00000 | 627.4 | 0.220 |
| harder_capacity | sla_aware_ema_tuned | 219.625 | 0.00000 | 636.1 | 0.284 |
| harder_capacity | sla_aware_tuned | 235.500 | 0.00000 | 636.1 | 0.258 |
| harder_capacity | sla_required_capacity | 111.417 | 0.00318 | 1525.2 | 0.536 |
| harder_capacity | util_base | 149.021 | 0.00815 | 1462.6 | 0.425 |
| harder_qps | forecast_only | 645.625 | 0.00000 | 627.4 | 0.170 |
| harder_qps | sla_aware_ema_tuned | 329.812 | 0.00000 | 636.1 | 0.270 |
| harder_qps | sla_aware_tuned | 349.000 | 0.00000 | 631.5 | 0.241 |
| harder_qps | sla_required_capacity | 146.125 | 0.00679 | 1941.0 | 0.562 |
| harder_qps | util_base | 201.708 | 0.00570 | 1799.9 | 0.429 |
| main_controlled | forecast_only | 622.167 | 0.00000 | 627.4 | 0.138 |
| main_controlled | sla_aware_ema_tuned | 250.688 | 0.00100 | 636.9 | 0.264 |
| main_controlled | sla_aware_tuned | 264.292 | 0.00100 | 636.1 | 0.238 |
| main_controlled | sla_required_capacity | 111.417 | 0.00318 | 1525.2 | 0.536 |
| main_controlled | util_base | 149.021 | 0.00815 | 1462.6 | 0.425 |
