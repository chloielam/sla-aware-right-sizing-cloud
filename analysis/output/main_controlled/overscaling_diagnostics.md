# Overscaling Diagnostics

An overscaling episode is counted when a policy increases instances for consecutive steps while all of the following hold:

- utilization <= 0.35
- queue_len <= 0.0
- sla_violation = 0

## sla_aware_ema_tuned
- overscaling episode from step 272 to 278
- overscaling episode from step 312 to 316
- overscaling episode from step 347 to 352
- overscaling episode from step 663 to 675
- overscaling episode from step 761 to 766
- overscaling episode from step 785 to 788

## sla_aware_tuned
- overscaling episode from step 325 to 328
- overscaling episode from step 424 to 427
- overscaling episode from step 672 to 676
- overscaling episode from step 785 to 788

