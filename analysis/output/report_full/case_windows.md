# Case Study Windows

These windows are selected automatically from `merged_trace.csv`.

- `burst`: steps 261 to 290
- `stable`: steps 1056 to 1085
- `recovery`: steps 309 to 338

Suggested report usage:
- `burst`: discuss whether proactive policies scale before or after latency worsens.
- `stable`: discuss cost control and downscaling behavior.
- `recovery`: discuss whether policies safely scale down after demand declines.

- `burst` sample stats from `reactive`: avg_instances=3.80, avg_p99=889.5, violation_steps=2
- `stable` sample stats from `reactive`: avg_instances=2.00, avg_p99=553.6, violation_steps=0
- `recovery` sample stats from `reactive`: avg_instances=10.50, avg_p99=614.0, violation_steps=1
