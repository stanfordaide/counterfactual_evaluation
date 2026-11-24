#!/bin/bash
#SBATCH --job-name=phase5
#SBATCH --array=0-149%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/counterfactual_ct/03_counterfactual_masks/pe/phase5_cohort_assembly/output/logs/slurm_%A_%a.out
#SBATCH --error=/path/to/counterfactual_ct/03_counterfactual_masks/pe/phase5_cohort_assembly/output/logs/slurm_%A_%a.err

# Phase 5: Counterfactual Cohort Assembly
# 150 patients, 16 configs each (severity-stratified grid + one extra severe).
# Edit /path/to/counterfactual_ct (your checkout) and add the partition or
# account options of your cluster before you submit.

cd /path/to/counterfactual_ct/03_counterfactual_masks/pe

# Create logs directory if it doesn't exist
mkdir -p /path/to/counterfactual_ct/03_counterfactual_masks/pe/phase5_cohort_assembly/output/logs

# Read the selected 150-patient list (written by select_patients.py)
PATIENT_LIST="/path/to/counterfactual_ct/03_counterfactual_masks/pe/phase5_cohort_assembly/output/selected_patient_ids.txt"
PATIENT_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PATIENT_LIST")

if [ -z "$PATIENT_ID" ]; then
    echo "ERROR: No patient found at index $SLURM_ARRAY_TASK_ID in $PATIENT_LIST"
    exit 1
fi

echo "========================================="
echo "Phase 5: Processing patient $PATIENT_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID / 149"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================="

conda run --no-capture-output -n maisi python phase5_cohort_assembly/run_single_patient.py \
    "$PATIENT_ID" \
    --output-dir /path/to/counterfactual_ct/03_counterfactual_masks/pe/phase5_cohort_assembly/output

EXIT_CODE=$?

echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"
echo "========================================="

exit $EXIT_CODE
