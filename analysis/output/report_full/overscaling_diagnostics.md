# Overscaling Diagnostics

An overscaling episode is counted when a policy increases instances for consecutive steps while all of the following hold:

- utilization <= 0.35
- queue_len <= 0.0
- sla_violation = 0

No overscaling episodes detected under the current diagnostic thresholds.

