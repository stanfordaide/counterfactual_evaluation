#!/bin/bash
#SBATCH --job-name=viola_merge
#SBATCH --output=logs/viola_merge_%A_%a.log
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
# One array task computes viola_iou for a strided 1/16 of the master-CSV rows (~165 rows, few min).
# Run from 04_evaluation_and_analysis/. The master CSV is the internal output of
# analysis/recover_cf_decoded_mapping.py (output/ich/cf_2640_samples_recovered.csv).
W=${VIOLA_DIR:-/path/to/viola31}
PY=${PYTHON:-python}
GLOB="$W/*/shard_*/out/predictions_violai_3_1_*/*_decoded.nii.gz"
P=$W/merge_partials; mkdir -p $P
$PY analysis/viola_detect_array.py shard "$GLOB" \
  output/ich/cf_2640_samples_recovered.csv \
  $P/part_${SLURM_ARRAY_TASK_ID}.csv ${SLURM_ARRAY_TASK_ID} 16
