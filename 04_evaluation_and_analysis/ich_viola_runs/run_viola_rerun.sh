#!/bin/bash
#SBATCH --job-name=viola_rerun
#SBATCH --output=logs/viola_rerun_%A_%a.log
#SBATCH --array=0-7
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=6:00:00

# Re-run Viola-UNet v3.1 on the 291 CFs the main array did not reach before the 16 h
# time limit. Re-sharded 8-way (~37 CFs each) so each task finishes in ~2 h. Same env/entrypoint
# as run_viola_array.sh; outputs go to $VIOLA_DIR/rerun/shard_$k/out/.
W=${VIOLA_DIR:-/path/to/viola31}
export LD_LIBRARY_PATH=$W/rootfs/opt/conda/lib:$LD_LIBRARY_PATH
cd $W/viola_app/workspace
S=$W/rerun/shard_${SLURM_ARRAY_TASK_ID}
echo "task $SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES nvol=$(ls $S/in | wc -l)"
time $W/rootfs/opt/conda/bin/python main.py --input_dir=$S/in --predict_dir=$S/out --batch_size=8
echo "task $SLURM_ARRAY_TASK_ID DONE masks=$(find $S/out -name '*.nii*' | wc -l)"
