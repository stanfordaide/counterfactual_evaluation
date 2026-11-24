#!/bin/bash
#SBATCH --job-name=fid_overlap
#SBATCH --output=fidelity/logs/fid_overlap_%A_%a.log
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#
# Extract RadImageNet 2.5D features for each ablation config on the overlapping set (3447 images)
# (same images as the CLIPScore test). The point-estimate FID + CIs come from the bootstrap worker,
# NOT from this script's built-in FID (which here runs on a small real stub and is meaningless).
#
# Real features are NOT extracted here: they already exist (fid_real_v2_clean) and are consumed once
# by fid_bootstrap_prep.py (see run_fid_score.sh). Pointing --real_filelist at a small stub keeps each array task from
# re-loading all 3447 real volumes.
#
#   sbatch --array=0-3 fidelity/run_fid_overlap.sh
#
# array id -> config: 0 text | 1 text_demo | 2 text_mask | 3 text_demo_mask
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

R=${CF_RAW:-/path/to/private_data}/pe_training_data_all
FL=$R/fid_overlap_filelists
CFGS=(text text_demo text_mask text_demo_mask)
ROOTS=(
  "$R/predictions/unet_3d_text_all_l2_loss"
  "$R/predictions/unet_3d_text_all_l2_loss_best_demo"
  "$R/predictions_controlnet/controlnet_text_segmentation_best"
  "$R/predictions_controlnet_best_demo_pe_all_v2_clean"
)
i=${SLURM_ARRAY_TASK_ID:-0}
CFG=${CFGS[$i]}
SROOT=${ROOTS[$i]}
PORT=$((25301 + i))

echo "Date: $(date)  Host: $(hostname)  Job: $SLURM_JOB_ID  Config: $CFG  GPUs: $CUDA_VISIBLE_DEVICES"
# Activate the Python environment (requirements.txt) before you submit.
# Submit from the 01_pe_diffusion folder.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p fidelity/logs

torchrun --nproc_per_node=4 --master_port=$PORT scripts/compute_fid_2-5d_ct_old.py \
  --model_name "radimagenet_resnet50" \
  --real_dataset_root "${CF_RAW:-/path/to/private_data}/CTPA/" \
  --real_filelist "$FL/real_stub_filelist.txt" \
  --real_features_dir "fid_real_v2_clean/" \
  --synth_dataset_root "$SROOT/" \
  --synth_filelist "$FL/filelist_$CFG.txt" \
  --synth_features_dir "fid_overlap_$CFG/" \
  --enable_center_slices_ratio 0.4 \
  --enable_padding True \
  --enable_center_cropping True \
  --enable_resampling_spacing "1.0x1.0x1.0" \
  --ignore_existing False \
  --num_images 4000 \
  --output_root "$R/" \
  --target_shape "512x512x512"
