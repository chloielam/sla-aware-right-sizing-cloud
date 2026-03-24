# Case Study Windows

These windows are selected automatically from `merged_trace.csv`.

- `burst`: steps 261 to 290
- `stable`: steps 1056 to 1085
- `recovery`: steps 309 to 338

Suggested report usage:
- `burst`: discuss whether proactive policies scale before or after latency worsens.
- `stable`: discuss cost control and downscaling behavior.
- `recovery`: discuss whether policies safely scale down after demand declines.

- `burst` sample stats from `sla_aware_ema_tuned`: avg_instances=6.47, avg_p99=557.2, violation_steps=0
- `stable` sample stats from `sla_aware_ema_tuned`: avg_instances=2.00, avg_p99=553.6, violation_steps=0
- `recovery` sample stats from `sla_aware_ema_tuned`: avg_instances=17.87, avg_p99=543.8, violation_steps=0
