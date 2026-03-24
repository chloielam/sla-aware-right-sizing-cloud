| scenario | policy | algorithm | cost($) | inst_hr | viol_rate | p99(ms) | eff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main_controlled | util_base | util_base | 149.021 | 119.217 | 0.00815 | 1462.6 | 0.425 |
| main_controlled | sla_aware_ema_tuned | ema_forecast | 250.688 | 200.550 | 0.00100 | 636.9 | 0.264 |
| main_controlled | sla_aware_tuned | heuristic_risk | 264.292 | 211.433 | 0.00100 | 636.1 | 0.238 |
