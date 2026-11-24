#!/bin/bash
#SBATCH --job-name=phase6_apes
#SBATCH --array=0-149%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/counterfactual_ct/03_counterfactual_masks/pe/phase6_evaluation/output_apes/logs/slurm_%A_%a.out
#SBATCH --error=/path/to/counterfactual_ct/03_counterfactual_masks/pe/phase6_evaluation/output_apes/logs/slurm_%A_%a.err

# Phase 6: Evaluate PE-Detector-2 (APES) output vs intended PE masks.
# Runs evaluation for one patient per array task (150 patients).
# After all array jobs complete, run aggregation:
#   conda run -n maisi python phase6_evaluation/run_phase6_apes.py \
#       --aggregate-only \
#       --output-dir /path/to/counterfactual_ct/03_counterfactual_masks/pe/phase6_evaluation/output_apes

cd /path/to/counterfactual_ct/03_counterfactual_masks/pe

# Ensure logs directory exists
mkdir -p /path/to/counterfactual_ct/03_counterfactual_masks/pe/phase6_evaluation/output_apes/logs

PATIENT_LIST="/path/to/counterfactual_ct/03_counterfactual_masks/pe/phase5_cohort_assembly/output/selected_patient_ids.txt"
PATIENT_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PATIENT_LIST")

PREDICTIONS_CSV="/path/to/controlnet_predictions_counterfactuals.csv"   # written by 01_pe_diffusion/scripts/controlnet_prediction_csv.py
GRID_BASE="/path/to/counterfactual_ct/03_counterfactual_masks/pe/phase5_cohort_assembly/output/patients"
APES_SEG_DIR="/path/to/pe_segmentation_APES_cf"   # one APES mask per generated CT
OUTPUT_DIR="/path/to/counterfactual_ct/03_counterfactual_masks/pe/phase6_evaluation/output_apes"

if [ -z "$PATIENT_ID" ]; then
    echo "ERROR: No patient found at index $SLURM_ARRAY_TASK_ID in $PATIENT_LIST"
    exit 1
fi

echo "========================================="
echo "Phase 6 APES: Evaluating patient $PATIENT_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID / 149"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================="

conda run --no-capture-output -n maisi python phase6_evaluation/run_phase6_apes.py \
    --patient-id "$PATIENT_ID" \
    --output-dir "$OUTPUT_DIR" \
    --predictions-csv "$PREDICTIONS_CSV" \
    --grid-base-dir "$GRID_BASE" \
    --apes-seg-dir "$APES_SEG_DIR"

EXIT_CODE=$?

echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"
echo "========================================="

exit $EXIT_CODE
