# Reproducing the results

All commands are run from the repository root.

## 1. Environment, data and weights

```bash
bash scripts/setup_env.sh                       # conda env "rmbff"
python scripts/download_weights.py --out weights
export RMBFF_WEIGHTS=$PWD/weights
```

Place the datasets as described in `data/README.md`. The split manifests in `outputs/manifests/` are the ones used
in the paper; `python -m rmbff.data --fer2013 data/fer2013 --cmd data/cmd` rebuilds identical manifests.

## 2. Frozen backbones (CPU)

```bash
bash scripts/run_frozen.sh
```

This extracts frozen features, selects objectives and hyper-parameters on validation data only
(`outputs/selection.json`), trains all fusion heads (5 seeds or folds), runs the split-protocol study, the
risk-control experiments, and writes `tables/*.csv` and `figures/frozen/`.

## 3. End-to-end fine-tuning (SLURM GPU cluster)

```bash
mkdir -p logs
sbatch scripts/slurm_finetune.sh     # 90 runs: 6 models x (FER2013 5 seeds, FER2013->DFERCD 5 seeds, DFERCD 5 folds)
sbatch scripts/slurm_ft_failure.sh   # 10 runs: edge model with branch dropout q = 0 / 0.25, ViT branch removed at test
sbatch scripts/slurm_ft_eval.sh      # re-scores all runs in fp32 from the selected checkpoint
```

Add the partition and module lines required by your cluster to the three scripts. Each run writes
`outputs/runs_ft/<experiment>/<run>/` with `config.json`, `metrics.json`, `curves.csv` and the validation/test
posteriors.

## 4. Risk control, tables and figures for the fine-tuned models (CPU)

```bash
bash scripts/run_ft_analysis.sh
```

This writes `tables/R1_cascade_ft.csv`, `R2_failure_ft.csv`, `R3_adapt_ft.csv`, `T_main_fer2013_ft.csv`,
`FT_vs_frozen.csv`, `FT_failure_acc.csv` and Figs. 4–7 in `figures/`.

## Result files in this repository

| File | Content |
|---|---|
| `tables/T_main_fer2013_ft.csv` | FER2013 recognition, fine-tuned (Table 3) |
| `tables/E9_protocol.csv`, `tables/E8_cmd_cv.csv` | split-protocol study (Table 4, Fig. 3) |
| `tables/R1_cascade_ft.csv` | certified wrong-action rate, decisiveness, cost (Table 5, Figs. 4–5) |
| `tables/R2_failure_ft.csv`, `tables/FT_failure_acc.csv` | branch and tier failures (Fig. 6) |
| `tables/R3_adapt_ft.csv`, `tables/R3_adapt_raw_ft.csv` | person-adaptive calibration (Table 6, Fig. 7) |
| `tables/FT_vs_frozen.csv` | frozen versus fine-tuned backbones (Table 7) |
| `tables/FT_runs_all.csv` | metrics of all 90 fine-tuned runs |
| `tables/*` without `_ft` | frozen-backbone results |
