# Overscaling Diagnostics

An overscaling episode is counted when a policy increases instances for consecutive steps while all of the following hold:

- utilization <= 0.35
- queue_len <= 0.0
- sla_violation = 0

## forecast_only
- overscaling episode from step 26 to 30
- overscaling episode from step 77 to 80
- overscaling episode from step 250 to 253
- overscaling episode from step 267 to 272
- overscaling episode from step 339 to 342
- overscaling episode from step 495 to 498
- overscaling episode from step 514 to 517
- overscaling episode from step 552 to 555
- overscaling episode from step 590 to 593
- overscaling episode from step 638 to 641
- overscaling episode from step 894 to 898
- overscaling episode from step 982 to 987

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

