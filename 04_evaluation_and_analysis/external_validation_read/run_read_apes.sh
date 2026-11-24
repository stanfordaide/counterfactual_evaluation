#!/bin/bash
#SBATCH --job-name=read_apes
#SBATCH --output=logs/read_apes_%j.log
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

# APES Acute-PE Segmenter (Dataset010_lungthrombus, fold 1) on READ, same config as
# FUMPE / CAD-PE / counterfactual.
# Run in an environment with nnUNetv2 installed.
DATA=${CF_RAW:-/path/to/private_data}/READ

export nnUNet_results="/path/to/APES_ThrombusSegmenter/nnUNet/nnUNet_results"

nnUNetv2_predict \
    -i $DATA/nnunet_input \
    -o $DATA/pred_apes \
    -d 10 \
    -c 3d_fullres \
    -f 1 \
    --verbose
echo "READ APES prediction DONE"
