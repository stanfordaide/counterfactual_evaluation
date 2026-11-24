#!/bin/bash
#SBATCH --job-name=viola_cf
#SBATCH --output=logs/viola_cf_%A_%a.log
#SBATCH --array=0-7
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=16:00:00
# Viola-UNet v3.1 (ICH-Detector-1) on the ICH counterfactual CTs, 8 shards.
# VIOLA_DIR = self-contained Viola-UNet v3.1 install (rootfs/ + viola_app/) with the decoded
# counterfactual CTs split into shards/shard_<k>/in/.
W=${VIOLA_DIR:-/path/to/viola31}
export LD_LIBRARY_PATH=$W/rootfs/opt/conda/lib:$LD_LIBRARY_PATH
cd $W/viola_app/workspace
S=$W/shards/shard_${SLURM_ARRAY_TASK_ID}
echo "task $SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES nvol=$(ls $S/in | wc -l)"
time $W/rootfs/opt/conda/bin/python main.py --input_dir=$S/in --predict_dir=$S/out --batch_size=8
echo "task $SLURM_ARRAY_TASK_ID DONE masks=$(find $S/out -name '*.nii*' | wc -l)"
