#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAIN_CONFIGS=(
  configs/util_base.yaml
  configs/sla_aware_tuned.yaml
  configs/sla_aware_ema_tuned.yaml
)

ABLATION_CONFIGS=(
  configs/sla_aware_ema_tuned.yaml
  configs/sla_aware_ema_no_prediction.yaml
  configs/sla_aware_ema_no_queue_external.yaml
  configs/sla_aware_ema_no_violation.yaml
  configs/sla_aware_ema_low_aggr.yaml
  configs/sla_aware_ema_high_aggr.yaml
)

python3 experiments/run_benchmark.py \
  --configs "${MAIN_CONFIGS[@]}" \
  --max-steps 1090 \
  --write-series \
  --scenario-name main_controlled \
  --result-tag report_main_controlled \
  --qps-mode qps \
  --qps-agg sum

python3 experiments/run_benchmark.py \
  --configs "${MAIN_CONFIGS[@]}" \
  --max-steps 1090 \
  --write-series \
  --scenario-name harder_qps \
  --result-tag report_harder_qps \
  --qps-mode qps \
  --qps-agg sum \
  --qps-scale 1.35

python3 experiments/run_benchmark.py \
  --configs "${MAIN_CONFIGS[@]}" \
  --max-steps 1090 \
  --write-series \
  --scenario-name harder_boot \
  --result-tag report_harder_boot \
  --qps-mode qps \
  --qps-agg sum \
  --boot-delay-override 4

python3 experiments/run_benchmark.py \
  --configs "${MAIN_CONFIGS[@]}" \
  --max-steps 1090 \
  --write-series \
  --scenario-name harder_capacity \
  --result-tag report_harder_capacity \
  --qps-mode qps \
  --qps-agg sum \
  --max-instances-override 18

python3 experiments/run_benchmark.py \
  --configs "${ABLATION_CONFIGS[@]}" \
  --max-steps 1090 \
  --write-series \
  --scenario-name ablation \
  --result-tag report_ablation \
  --qps-mode qps \
  --qps-agg sum \
  --qps-scale 1.35 \
  --boot-delay-override 4 \
  --max-instances-override 18

python3 analysis/generate_report_assets.py --results-dir data/processed/report_main_controlled --output-dir analysis/output/main_controlled
python3 analysis/generate_report_assets.py --results-dir data/processed --output-dir analysis/output/report_bundle
