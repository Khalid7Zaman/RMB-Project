#!/usr/bin/env bash
# Augmented copies of FER2013 (protocol-sensitivity experiment E9). Severity is ignored for aug_* variants.
set -e
export RMBFF_WEIGHTS=${RMBFF_WEIGHTS:-weights}
for v in aug_flip aug_rot; do
  ${PY:-python} -m rmbff.extract --manifest outputs/manifests/fer2013.csv --dataset fer2013 --variant $v --severity 0 --branches m v --threads ${THREADS:-2}
done
