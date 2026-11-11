#!/bin/bash
# Branch-failure experiment with fine-tuned edge models (R2). 10 array tasks:
#   0-4 : RMB-FF-MV trained WITHOUT branch dropout (q = 0.00), seeds 0-4
#   5-9 : RMB-FF-MV trained WITH    branch dropout (q = 0.25), seeds 0-4
# Same recipe as slurm_finetune.sh (loss from outputs/selection.json, 30 epochs, batch 64).
# After training: fp32 scoring of the healthy model, then fp32 scoring with the ViT branch removed.
#SBATCH --job-name=rmbff-fail
#SBATCH --gres=gpu:1
# Add '#SBATCH --partition=<gpu partition>' and module loads as required by your cluster.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rmbff
set -euo pipefail
mkdir -p logs
export RMBFF_WEIGHTS=${RMBFF_WEIGHTS:-weights}
ID=$SLURM_ARRAY_TASK_ID; QS=("0.0" "0.25"); Q=${QS[$((ID / 5))]}; K=$((ID % 5))
LOSS=$(python -c "import json;h=json.load(open('outputs/selection.json'))['hp'];print(h.get('mv',h['mrsv'])['loss'])")
OUT=outputs/runs_ft/fer_robust_mv/branches-mv_bd-${Q}_seed-$K
python -m rmbff.finetune --protocol fer_official --seed $K --branches m v --loss $LOSS --branch_drop $Q \
    --epochs 30 --bs 64 --workers 8 --resume --out $OUT
python -m rmbff.ft_eval --run $OUT
python -m rmbff.ft_eval --run $OUT --drop v
