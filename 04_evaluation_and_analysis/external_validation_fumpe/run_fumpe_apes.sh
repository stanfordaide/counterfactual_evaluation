#!/bin/bash
#SBATCH --job-name=fumpe_apes
#SBATCH --output=logs/fumpe_apes_%j.log
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

# APES Acute-PE Segmenter (Dataset010_lungthrombus, fold 1) on FUMPE, same config as
# the counterfactual evaluation.
# Run in an environment with nnUNetv2 installed.
DATA=${CF_RAW:-/path/to/private_data}/FUMPE

export nnUNet_results="/path/to/APES_ThrombusSegmenter/nnUNet/nnUNet_results"

nnUNetv2_predict \
    -i $DATA/nnunet_input \
    -o $DATA/pred_apes \
    -d 10 \
    -c 3d_fullres \
    -f 1 \
    --verbose
echo "FUMPE APES prediction DONE"
