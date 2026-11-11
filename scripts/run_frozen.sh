#!/usr/bin/env bash
# Frozen-backbone pipeline (CPU is sufficient): features, fusion heads, protocol study, risk control, tables, figures.
set -euo pipefail
export RMBFF_WEIGHTS=${RMBFF_WEIGHTS:-weights}
PY=${PY:-python}
$PY -m rmbff.data --fer2013 "${FER2013_ROOT:-data/fer2013}" --cmd "${CMD_ROOT:-data/cmd}"
PY=$PY bash scripts/extract_all.sh
PY=$PY bash scripts/extract_aug.sh
$PY -m rmbff.efficiency --threads 2 --out outputs/efficiency.csv
$PY -m rmbff.train_head --exp fer_loss          # objective selection (validation only)
$PY -m rmbff.select_loss
$PY -m rmbff.train_head --exp fer_tune fer_tune_bd # width/dropout/weight decay and branch dropout (validation only)
$PY -m rmbff.select_loss                          # writes outputs/selection.json
$PY -m rmbff.train_head --exp fer_main fer_subsets fer_fusion fer_noproj fer_dedup fer_sensitivity \
    cross_fer2cmd cmd_group cmd_random fer_robust fer_robust_mv fer_robust_single
$PY -m rmbff.analyze
$PY -m rmbff.protocol_study --models v mv --seeds 0 1 2 3 4
$PY -m rmbff.riskexp --reps 50
$PY -m rmbff.fig_framework_aej
$PY -m rmbff.figures_aej
