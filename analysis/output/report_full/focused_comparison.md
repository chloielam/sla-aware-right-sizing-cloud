# Focused Comparison

| scenario | policy | cost($) | viol_rate | p99(ms) | eff |
| --- | --- | --- | --- | --- | --- |
| ablation | sla_aware_ema_tuned | 225.854 | 0.00654 | 1135.4 | 0.393 |
| harder_boot | forecast_only | 190.583 | 0.00000 | 641.2 | 0.321 |
| harder_boot | sla_aware_ema_tuned | 198.667 | 0.00361 | 688.0 | 0.335 |
| harder_boot | sla_aware_tuned | 214.479 | 0.00100 | 647.4 | 0.299 |
| harder_boot | sla_required_capacity | 110.667 | 0.03211 | 2149.3 | 0.568 |
| harder_boot | util_base | 144.083 | 0.01165 | 1848.2 | 0.460 |
| harder_capacity | forecast_only | 300.250 | 0.00000 | 631.5 | 0.245 |
| harder_capacity | sla_aware_ema_tuned | 214.438 | 0.00100 | 636.1 | 0.302 |
| harder_capacity | sla_aware_tuned | 234.146 | 0.00100 | 641.6 | 0.269 |
| harder_capacity | sla_required_capacity | 111.604 | 0.01050 | 1848.2 | 0.555 |
| harder_capacity | util_base | 144.917 | 0.00999 | 1676.2 | 0.447 |
| harder_qps | forecast_only | 574.438 | 0.00000 | 636.1 | 0.213 |
| harder_qps | sla_aware_ema_tuned | 296.333 | 0.00000 | 641.6 | 0.305 |
| harder_qps | sla_aware_tuned | 318.979 | 0.00000 | 636.9 | 0.265 |
| harder_qps | sla_required_capacity | 146.792 | 0.02279 | 2230.9 | 0.578 |
| harder_qps | util_base | 196.396 | 0.01333 | 1871.8 | 0.455 |
| main_controlled | forecast_only | 523.583 | 0.00000 | 631.5 | 0.186 |
| main_controlled | sla_aware_ema_tuned | 222.229 | 0.00100 | 636.1 | 0.295 |
| main_controlled | sla_aware_tuned | 244.417 | 0.00100 | 641.6 | 0.260 |
| main_controlled | sla_required_capacity | 111.604 | 0.01050 | 1848.2 | 0.555 |
| main_controlled | util_base | 144.917 | 0.00999 | 1676.2 | 0.447 |
