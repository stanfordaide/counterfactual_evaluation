#!/bin/bash
#SBATCH --job-name=fid_boot
#SBATCH --output=fidelity/logs/fid_boot_%A_%a.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#
# Per-config FID bootstrap (paired, float64 on CPU). Run AFTER:
#   1) sbatch --array=0-3 fidelity/run_fid_overlap.sh   (extract overlapping-set features)
#   2) python fidelity/fid_bootstrap_prep.py            (shared counts + real reference)
# then:
#   sbatch --array=0-3 fidelity/run_fid_bootstrap.sh
#   python fidelity/fid_bootstrap_combine.py            (table + test)
#
# array id -> config: 0 text | 1 text_demo | 2 text_mask | 3 text_demo_mask
set -e
CFGS=(text text_demo text_mask text_demo_mask)
CFG=${CFGS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "Date: $(date)  Host: $(hostname)  Job: $SLURM_JOB_ID  Config: $CFG"
# Activate the Python environment (requirements.txt) before you submit.
# Submit from the 01_pe_diffusion folder.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
python fidelity/fid_bootstrap_worker.py --config "$CFG"
