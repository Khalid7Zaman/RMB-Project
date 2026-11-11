#!/bin/bash
# Re-score all 90 fine-tuned runs in full fp32 from best.pt (no retraining). One run per array task.
#SBATCH --job-name=rmbff-eval32
#SBATCH --gres=gpu:1
# Add '#SBATCH --partition=<gpu partition>' and module loads as required by your cluster.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-89%16
#SBATCH --output=logs/%x_%A_%a.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rmbff
set -euo pipefail
export RMBFF_WEIGHTS=${RMBFF_WEIGHTS:-weights}
python -m rmbff.ft_eval --index $SLURM_ARRAY_TASK_ID
