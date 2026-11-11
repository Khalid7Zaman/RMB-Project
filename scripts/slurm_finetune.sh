#!/bin/bash
# End-to-end fine-tuning of all single-backbone and fused models. 90 array tasks:
#   0-29  FER2013 official split   (6 models x 5 seeds)
#   30-59 FER2013 -> DFERCD        (6 models x 5 seeds)
#   60-89 DFERCD group-disjoint CV (6 models x 5 folds)
# Each task writes outputs/runs_ft/<exp>/<run>/ with config, metrics, curves, val/test posteriors.
#SBATCH --job-name=rmbff-ft
#SBATCH --gres=gpu:1
# Add '#SBATCH --partition=<gpu partition>' and module loads as required by your cluster.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-89
#SBATCH --output=logs/%x_%A_%a.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rmbff   # adapt to your environment
set -euo pipefail
mkdir -p logs
export RMBFF_WEIGHTS=${RMBFF_WEIGHTS:-weights}
MODELS=("m" "v" "r" "s" "m v" "m v r s"); TAGS=("m" "v" "r" "s" "mv" "mrsv")
ID=$SLURM_ARRAY_TASK_ID; BLOCK=$((ID / 30)); J=$((ID % 30)); MI=$((J / 5)); K=$((J % 5))
HP=$(python - <<PY
import json; h=json.load(open('outputs/selection.json'))['hp']; t='${TAGS[$MI]}'; x=h.get(t, h['mrsv'])
print(x['loss'], x['branch_drop'] if len(t)>1 else 0.0)
PY
)
LOSS=$(echo $HP | cut -d' ' -f1); BD=$(echo $HP | cut -d' ' -f2)
COMMON="--branches ${MODELS[$MI]} --loss $LOSS --branch_drop $BD --epochs 30 --bs 64 --workers 8 --resume"
case $BLOCK in
  0) python -m rmbff.finetune --protocol fer_official --seed $K $COMMON --out outputs/runs_ft/fer_main/branches-${TAGS[$MI]}_seed-$K ;;
  1) python -m rmbff.finetune --protocol fer_to_cmd --seed $K $COMMON --out outputs/runs_ft/cross_fer2cmd/branches-${TAGS[$MI]}_seed-$K ;;
  2) python -m rmbff.finetune --protocol cmd_group --fold $K --seed 0 $COMMON --out outputs/runs_ft/cmd_group/branches-${TAGS[$MI]}_fold-$K ;;
esac
