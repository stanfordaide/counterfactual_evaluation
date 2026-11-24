#!/bin/bash
#SBATCH --job-name=fumpe_nnunet
#SBATCH --output=logs/fumpe_nnunet_%j.log
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

# nnUNet3D-ResEncXL PE model (PulmonaryEmbolismSegmentation) on FUMPE, same config
# as the counterfactual evaluation. External validation: FUMPE has manual GT
# so the strict +/-5-slice overlap is directly comparable to the synthetic data.
# Run in an environment with nnUNetv2 installed.
DATA=${CF_RAW:-/path/to/private_data}/FUMPE

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
echo "FUMPE nnUNet prediction DONE"
