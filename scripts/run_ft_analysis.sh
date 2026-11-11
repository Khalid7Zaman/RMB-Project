#!/usr/bin/env bash
# After slurm_finetune.sh, slurm_ft_failure.sh and slurm_ft_eval.sh: risk control, tables and figures (CPU).
set -euo pipefail
export RMBFF_RUNS=outputs/runs_ft
PY=${PY:-python}
$PY -m rmbff.riskexp --reps 50 --tag _ft
$PY -m rmbff.ft_tables
RMBFF_TAG=_ft $PY -m rmbff.figures_aej
