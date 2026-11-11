# RMB-FF: Risk-controlled multi-backbone fusion for facial emotion recognition in IoT-enabled human–robot interaction

This repository contains the code, configurations, split manifests, result tables and figures for the paper
*"Risk-controlled multi-backbone fusion for trustworthy facial emotion recognition in IoT-enabled human–robot
interaction"*.

RMB-FF organises heterogeneous ImageNet-pretrained backbones (MobileNetV3-L, ViT-S/16, RegNetY-1.6GF,
SE-ResNet-50) as a robot → edge → cloud cascade, maps facial emotions to robot actions, and certifies the
escalation thresholds of all tiers jointly with Learn-then-Test, so that the wrong-action rate stays below a
chosen level ε with probability at least 1 − δ. An online person-adaptive update keeps the error rate near ε
under identity and dataset shift, and branch dropout with per-configuration recalibration handles failing
backbones.

## Repository structure

```
rmbff/                 Python package
  data.py              manifests, official FER2013 split, DFERCD group-disjoint folds, duplicate audit
  backbones.py         timm backbones loaded from local weights
  model.py, losses.py  fusion head with branch dropout; CE, class-balanced focal and logit-adjusted losses
  extract.py, train_head.py, select_loss.py   frozen-backbone features, fusion heads, validation-only selection
  finetune.py, ft_eval.py                     end-to-end fine-tuning and fp32 scoring (GPU)
  riskcontrol.py       action map, Hoeffding–Bentkus p-values, Learn-then-Test with fixed-sequence testing,
                       cascade evaluation, person-adaptive update
  riskexp.py           risk-control experiments (cascade, failures, person adaptation)
  protocol_study.py    effect of the split protocol on FER2013
  analyze.py, ft_tables.py, figures_aej.py, fig_framework_aej.py   tables and figures
configs/experiments.yaml   experiment definitions
scripts/               environment setup, pipelines and SLURM job scripts
outputs/               split manifests, selected hyper-parameters, efficiency measurements, checksums
tables/                all result tables (CSV); suffix _ft = fine-tuned backbones
figures/               figures of the paper (PDF, SVG, PNG); figures/frozen = frozen-backbone versions
refs/                  bibliography (RIS, BibTeX, EndNote XML)
docs/REPRODUCE.md      step-by-step reproduction
```

## Quick start

```bash
bash scripts/setup_env.sh
python scripts/download_weights.py --out weights && export RMBFF_WEIGHTS=$PWD/weights
# place the datasets as described in data/README.md, then see docs/REPRODUCE.md
```

## Data

FER2013 is publicly available. DFERCD contains identifiable faces and is not distributed with this repository;
it is available from the corresponding author on reasonable request for non-commercial research. No face images,
model weights or trained checkpoints are stored here.

## Citation

If you use this code, please cite the paper (citation details will be added after publication).

## License

MIT License (see `LICENSE`).
