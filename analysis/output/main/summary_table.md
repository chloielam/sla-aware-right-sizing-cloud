| scenario | policy | algorithm | cost($) | inst_hr | viol_rate | p99(ms) | eff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main | reactive | reactive | 116.292 | 93.033 | 0.05237 | 2661.7 | 0.546 |
| main | util_base | util_base | 149.021 | 119.217 | 0.00815 | 1462.6 | 0.425 |
| main | static | static | 272.500 | 218.000 | 0.03182 | 1810.4 | 0.384 |
| main | forecast_only | forecast_only | 622.167 | 497.733 | 0.00000 | 627.4 | 0.138 |
| main | sla_aware | heuristic_risk | 809.062 | 647.250 | 0.00000 | 627.4 | 0.090 |
| main | sla_aware_ema | ema_forecast | 812.854 | 650.283 | 0.00000 | 627.4 | 0.090 |
