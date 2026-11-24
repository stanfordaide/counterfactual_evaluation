#!/bin/bash
#SBATCH --job-name=cadpe_nnunet
#SBATCH --output=logs/cadpe_nnunet_%j.log
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

# nnUNet3D-ResEncXL PE model (PulmonaryEmbolismSegmentation) on CAD-PE, same config as
# the FUMPE / counterfactual evaluation. CAD-PE (Gonzalez 2020) has radiologist
# per-embolus reference masks, so the strict +/-5-slice overlap is directly comparable
# to the synthetic data (2nd real external validation after FUMPE).
# Run in an environment with nnUNetv2 installed.
DATA=${CF_RAW:-/path/to/private_data}/CAD_PE

export nnUNet_results="/path/to/PulmonaryEmbolismSegmentation/dataset/nnUNet_results"
export nnUNet_raw="$DATA/nnunet_input"
export nnUNet_preprocessed="$DATA/nnunet_input"

nnUNetv2_predict \
    -i $DATA/nnunet_input \
    -o $DATA/pred_nnunet \
    -d Dataset256_PE \
    -c 3d_fullres \
    -tr nnUNetTrainer_500epochs \
    -p nnUNetResEncUNetPlans_80G \
    -f all \
    --verbose
echo "CAD-PE nnUNet prediction DONE"
