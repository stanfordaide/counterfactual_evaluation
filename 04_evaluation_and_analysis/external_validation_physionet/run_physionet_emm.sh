#!/bin/bash
#SBATCH --job-name=phys_emm
#SBATCH --output=logs/phys_emm_%j.log
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=1:00:00

# 3D ResNet-101 classifier (ICH-Detector-2) on the 36 PhysioNet ICH-positive real head CTs,
# through the classifier's own, unmodified inference code (ich_emm/inference.py; import main(),
# pass our config). Produces physionet_emm.csv (file_id, emm_prob).
# Run in the classifier's environment. EMM_DIR = the ich_emm code folder.
# Run from 04_evaluation_and_analysis/ (the config path below is relative to it).
CFG=$(pwd)/external_validation_physionet/inf_config_physionet.yml
cd ${EMM_DIR:-/path/to/ich_emm}      # for its relative imports
python -c "from inference import main; main('$CFG')"
echo "=== PhysioNet EMM inference DONE $(date) ==="
