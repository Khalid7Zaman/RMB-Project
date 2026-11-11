#!/usr/bin/env bash
# Regime A feature extraction (CPU or GPU). Resumable.
set -e
export RMBFF_WEIGHTS=${RMBFF_WEIGHTS:-weights}
T=${THREADS:-2}
${PY:-python} -m rmbff.extract --manifest outputs/manifests/fer2013.csv --dataset fer2013 --threads $T
${PY:-python} -m rmbff.extract --manifest outputs/manifests/cmd_groupfolds.csv --dataset cmd --threads $T
for c in gauss_noise blur low_light jpeg low_res occlusion; do
  python -m rmbff.extract --manifest outputs/manifests/fer2013.csv --dataset fer2013 --subset test \
         --variant $c --severity 3 --threads $T
done
