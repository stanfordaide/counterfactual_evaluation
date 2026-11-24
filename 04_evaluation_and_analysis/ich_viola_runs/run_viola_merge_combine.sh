#!/bin/bash
#SBATCH --job-name=viola_combine
#SBATCH --output=logs/viola_combine_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
# Merge the 16 partials into the recovered master CSV + report synthetic Viola detection by volume.
# Run from 04_evaluation_and_analysis/.
W=${VIOLA_DIR:-/path/to/viola31}
PY=${PYTHON:-python}
$PY analysis/viola_detect_array.py combine \
  "$W/merge_partials/part_*.csv" \
  output/ich/cf_2640_samples_recovered.csv \
  output/ich/cf_2640_samples_recovered.csv
echo "=== VIOLA MERGE COMPLETE $(date) ==="
