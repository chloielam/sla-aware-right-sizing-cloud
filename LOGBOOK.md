# Software Project Logbook

**Project:** SLA-aware autoscaling simulation for Alibaba Generative AI trace dataset  
**Repository:** `sla-aware`  
**Author:** Truclam  
**Period Covered:** Initial debugging to current simulator/policy refinement stage

## Log Entries

### 2026-02-20

**Task / Goal**  
Investigate why different autoscaling policies produce nearly the same `viol_rate`.

**What was done**  
- Reviewed benchmark summaries across `static`, `reactive`, `util_base`, and `sla_aware`.
- Compared expected policy behavior against observed outputs.
- Identified that violations were either unrealistically similar, too low, or too high across policies.

**Files changed**  
- None

**Result**  
- Confirmed the simulation behavior was suspicious and required full setup verification.

**Issue / Observation**  
- Different policies were not separating in the way they should.
- Suspected problems in trace interpretation, simulator-policy coupling, or policy configuration.

**Next step**  
Verify assumptions about dataset inputs and benchmark configuration.

### 2026-02-22

**Task / Goal**  
Define expected policy behavior to use as a debugging baseline.

**What was done**  
- Established expected qualitative behavior for each policy:
- `static` should keep instance count flat.
- `reactive` should scale only after latency or SLA degrades.
- `util_base` should adapt more smoothly based on utilization or thresholds.
- `sla_aware` should scale earlier to protect SLA.

**Files changed**  
- None

**Result**  
- Created a clear verification target for future simulator checks.

**Issue / Observation**  
- Without a concrete expected pattern, the benchmark summary alone was too hard to interpret.

**Next step**  
Check whether the workload trace is being interpreted correctly.

### 2026-02-25

**Task / Goal**  
Re-check dataset semantics after new README information was added under `data/`.

**What was done**  
- Used the dataset README to understand the meaning of the QPS file.
- Re-evaluated how QPS input mode and aggregation were being used in the simulation.
- Reconsidered whether the step values and offered load were being interpreted correctly.

**Files changed**  
- None

**Result**  
- Improved understanding of how the Alibaba trace should feed into the simulator.

**Issue / Observation**  
- Earlier benchmark results may have been distorted by incorrect assumptions about the QPS trace.

**Next step**  
Re-run the benchmark with corrected trace interpretation assumptions.

### 2026-02-27

**Task / Goal**  
Verify whether benchmark behavior becomes more realistic after rechecking dataset usage.

**What was done**  
- Re-ran the benchmark using the revised interpretation of QPS input behavior.
- Compared policy summaries again after setup adjustments.

**Files changed**  
- Benchmark configuration files were updated during this phase.

**Result**  
- Policy behaviors started to separate more plausibly.
- `reactive` began to look worse on latency and SLA than more adaptive policies.

**Issue / Observation**  
- Summary metrics alone were still not enough to determine whether scaling logic was truly correct.

**Next step**  
Inspect per-step series outputs instead of relying only on summary tables.

### 2026-03-01

**Task / Goal**  
Define a direct series-based sanity check for all policies.

**What was done**  
- Chose a set of columns to compare in each `*_series.csv` file:
- `step` or `timestamp`
- `instances`
- `violation` or `sla_violation`
- `p99_ms` or latency column
- `target_instances` if present
- Defined the expected pattern for `static`, `reactive`, `util_base`, and `sla_aware`.

**Files changed**  
- None

**Result**  
- Established a concrete step-by-step validation method for simulator behavior.

**Issue / Observation**  
- The existing series output did not yet expose enough policy intent to distinguish target scaling from actual scaling.

**Next step**  
Add `target_instances` to series output.

### 2026-03-02

**Task / Goal**  
Improve simulator observability by including target scaling decisions in time-series output.

**What was done**  
- Updated the simulator time-series writing logic to include `target_instances`.

**Files changed**  
- [simulator.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/engine/simulator.py)

**Result**  
- The series files could now show both desired scale and actual active instances.

**Issue / Observation**  
- This exposed whether policy logic or simulator application was responsible for poor behavior.

**Next step**  
Use the new series output to compare policy traces directly.

### 2026-03-03

**Task / Goal**  
Check whether policy traces match expected behavior patterns.

**What was done**  
- Compared the first sections of the `*_series.csv` files.
- Looked for:
- flat instance counts in `static`,
- delayed jumps in `reactive`,
- smoother adaptation in `util_base`,
- earlier or more aggressive action in `sla_aware`.

**Files changed**  
- None

**Result**  
- The project moved from vague suspicion to concrete policy-by-policy diagnosis.

**Issue / Observation**  
- Some results looked better, but a new pattern appeared where violation rates became too low across most policies.

**Next step**  
Investigate whether the simulation had become too easy or overly forgiving.

### 2026-03-04

**Task / Goal**  
Investigate why `viol_rate` became near zero for most policies.

**What was done**  
- Reviewed benchmark outputs where all policies except `reactive` had near-zero violations.
- Considered whether the SLA threshold, capacity model, or latency sensitivity had become too forgiving.

**Files changed**  
- Configuration files were adjusted during this phase.

**Result**  
- Confirmed that the project had shifted from "all policies look the same" to "too many policies look unrealistically perfect."

**Issue / Observation**  
- A low `viol_rate` is not automatically wrong, but it is suspicious if almost every policy achieves it easily.

**Next step**  
Focus on the relationship between scaling and SLA improvement.

### 2026-03-05

**Task / Goal**  
Investigate why `util_base` sometimes outperformed `sla_aware` despite using fewer resources.

**What was done**  
- Compared benchmark outputs showing `util_base` with lower violation than `sla_aware`.
- Treated this as either a setup issue, a policy-quality issue, or both.
- Defined a direct check using `instances`, `target_instances`, `latency_p99_ms`, and `sla_violation`.

**Files changed**  
- None

**Result**  
- Identified the key diagnostic question: does scaling happen early enough, and does it actually reduce p99 and violations soon after?

**Issue / Observation**  
- If `target_instances` increases but p99 does not improve, scaling may not be relieving the bottleneck.

**Next step**  
Tune the `sla_aware` decision rules.

### 2026-03-06

**Task / Goal**  
Improve the original SLA-aware policy so it reacts earlier to risk.

**What was done**  
- Revised the `sla_aware` policy logic.
- Strengthened triggers involving:
- near-threshold latency,
- sustained risk/trend,
- demand surge,
- queue pressure,
- utilization pressure,
- and stronger downscale protection.

**Files changed**  
- [sla_aware.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/sla_aware.py)

**Result**  
- The policy became more proactive and less purely reactive.

**Issue / Observation**  
- Improvement was not consistent enough across runs. `sla_aware` still did not clearly dominate `util_base`.

**Next step**  
Try a different prediction/control approach instead of only tuning thresholds.

### 2026-03-07

**Task / Goal**  
Create a new SLA-aware variant with a different prediction algorithm.

**What was done**  
- Implemented a new policy variant: `sla_aware_ema`.
- Used EMA-style forecasting with fast and slow demand smoothing.
- Incorporated trend, volatility, predicted utilization, and blended risk signals.

**Files changed**  
- [sla_aware_ema.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/policies/sla_aware_ema.py)

**Result**  
- Added an alternative SLA-aware controller that could be benchmarked against the original.

**Issue / Observation**  
- A new policy file alone was not enough; the benchmark and config system also needed to recognize it.

**Next step**  
Integrate the new variant into benchmark execution and configuration.

### 2026-03-08

**Task / Goal**  
Integrate the new EMA-based policy into the benchmark pipeline.

**What was done**  
- Updated benchmark policy registration to recognize `sla_aware_ema`.
- Added configuration support for the new variant.

**Files changed**  
- [run_benchmark.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/experiments/run_benchmark.py)
- [sla_aware_ema.yaml](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/configs/sla_aware_ema.yaml)

**Result**  
- The new policy could be executed in the same workflow as the other policies.

**Issue / Observation**  
- Output files now needed a way to indicate which prediction algorithm generated which results.

**Next step**  
Add algorithm metadata to series and summary outputs.

### 2026-03-09

**Task / Goal**  
Improve output traceability across multiple SLA-aware variants.

**What was done**  
- Updated simulator outputs to include `prediction_algorithm` in:
- time-series rows,
- benchmark summary rows.

**Files changed**  
- [simulator.py](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/src/engine/simulator.py)

**Result**  
- CSV outputs could now distinguish between the original SLA-aware policy and the EMA-based version.

**Issue / Observation**  
- Policy comparison became easier, but interpretation still depended on whether increased scaling actually improved SLA metrics.

**Next step**  
Run the updated benchmark and compare resource usage against p99 and violation.

### 2026-03-10

**Task / Goal**  
Verify that the new SLA-aware variant changes behavior in a measurable way.

**What was done**  
- Re-ran the benchmark with series writing enabled.
- Compared `sla_aware` and `sla_aware_ema` outputs.

**Files changed**  
- Generated benchmark output files were updated under `data/processed/`.

**Result**  
- The EMA-based policy produced different instance-hours and costs from the original SLA-aware policy.
- This confirmed the new variant was operational.

**Issue / Observation**  
- Some runs still showed very low violation for multiple policies, so interpretation had to rely on both cost and time-series behavior.

**Next step**  
Continue comparing whether earlier scaling leads to lower p99 and fewer violations.

### 2026-03-12

**Task / Goal**  
Clarify what simulation horizon should be used in practice.

**What was done**  
- Considered the meaning of step count and discussed whether shorter runs such as `240` steps or longer runs such as `1090` steps should be used by default.

**Files changed**  
- None

**Result**  
- Recognized that short runs are useful for debugging and long runs are better for realistic benchmarking.

**Issue / Observation**  
- Different horizons can change how stable the policy comparison looks.

**Next step**  
Use shorter runs for quick debugging and longer runs for final evaluation.

### 2026-03-14

**Task / Goal**  
Clean up repository hygiene around generated benchmark outputs.

**What was done**  
- Identified that files under `data/processed/` were being staged for git.
- Reviewed how `.gitignore` should be updated so generated files are not pushed to origin.
- Considered the required cleanup using cached removal from git tracking.

**Files changed**  
- No file edit recorded in this conversation, but the intended target was `.gitignore`.

**Result**  
- Established that processed benchmark outputs should not be kept under version control.

**Issue / Observation**  
- Generated CSV files created noise in commits and produced line-ending warnings during staging.

**Next step**  
Update `.gitignore` and remove tracked processed outputs from the index.

### 2026-03-16

**Task / Goal**  
Clean up recent commit history after an incorrect commit sequence.

**What was done**  
- Reviewed git steps for:
- removing the most recent commit while keeping changes staged,
- amending the next commit,
- and rewriting an older commit message if needed.

**Files changed**  
- None

**Result**  
- Established a clean workflow for fixing commit history without losing work.

**Issue / Observation**  
- History rewrite requires force-pushing if commits have already been pushed.

**Next step**  
Apply git cleanup only after confirming the final tracked file set is correct.

### 2026-03-19

**Task / Goal**  
Document project progress in a standard software logbook format.

**What was done**  
- Reconstructed the development history from debugging, simulator changes, policy tuning, benchmark updates, and repository cleanup work.
- Converted the narrative history into a structured engineering log format.

**Files changed**  
- [LOGBOOK.md](/Users/truclam/Documents/MUN/Winter%202026/COMP6910/Project/sla-aware/LOGBOOK.md)

**Result**  
- The project now has a conventional software logbook with dated entries, tasks, actions, results, observations, and next steps.

**Issue / Observation**  
- Some exact dates are approximate because the conversation history does not include precise timestamps for every intermediate change.

**Next step**  
Continue updating this logbook after each benchmark run, policy revision, or git cleanup task.

## Current Status Summary

**What is working**
- Policy outputs are more differentiated than at the start.
- `target_instances` is included in the series output.
- `prediction_algorithm` is included in series and summary output.
- A second SLA-aware variant (`sla_aware_ema`) is implemented and benchmarkable.

**Open concerns**
- `sla_aware` does not always outperform `util_base`.
- Very low `viol_rate` across multiple policies still needs careful interpretation.
- Final git cleanup around generated files should be completed before pushing.

**Recommended ongoing logbook practice**
- Add one entry per work session.
- Keep each entry short and factual.
- Record exact commands or benchmark settings when they materially affect results.
- Record changed files only when edits were actually made.
