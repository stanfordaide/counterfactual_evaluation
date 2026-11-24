#!/bin/bash
#SBATCH --job-name=viola_phys
#SBATCH --output=logs/viola_phys_%j.log
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=1:30:00

# Viola-UNet v3.1 (ICH-Detector-1) on the 36 PhysioNet ICH-positive real head CTs. Same
# self-contained install as the counterfactual run (VIOLA_DIR, see ich_viola_runs/).
W=${VIOLA_DIR:-/path/to/viola31}
PHYS=${CF_RAW:-/path/to/private_data}/physionet_ich
export LD_LIBRARY_PATH=$W/rootfs/opt/conda/lib:$LD_LIBRARY_PATH
cd $W/viola_app/workspace
$W/rootfs/opt/conda/bin/python main.py \
  --input_dir=$PHYS/viola_in \
  --predict_dir=$PHYS/viola_out \
  --batch_size=4
echo "=== RESULT CSV ==="; cat "$(find $PHYS/viola_out -name predictions_viola3-1.csv)"
echo "=== PhysioNet Viola DONE $(date) ==="
