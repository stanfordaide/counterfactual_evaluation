#!/bin/bash
#SBATCH --job-name=FID_calculation
#SBATCH --output=fidelity/logs/fid_calculation_%j.log
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00

# FID score computation for 3448 QC-passed v2_clean controlnet predictions
# Same subset as controlnet_predictions_qc.csv
# This run writes the real reference features (fid_real_v2_clean/) that fid_bootstrap_prep.py reads.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print job details
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Activate the Python environment (requirements.txt) before you submit.
# Submit from the 01_pe_diffusion folder.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=${CF_RAW:-/path/to/private_data}/pe_training_data_all

torchrun --nproc_per_node=4 --master_port=25133 scripts/compute_fid_2-5d_ct_old.py \
  --model_name "radimagenet_resnet50" \
  --real_dataset_root "${CF_RAW:-/path/to/private_data}/CTPA/" \
  --real_filelist "$R/real_file_list_fid_v2_clean.txt" \
  --real_features_dir "fid_real_v2_clean/" \
  --synth_dataset_root "$R/predictions_controlnet_best_demo_pe_all_v2_clean/" \
  --synth_filelist "$R/synth_file_list_fid_v2_clean.txt" \
  --synth_features_dir "fid_synth_v2_clean/" \
  --enable_center_slices_ratio 0.4 \
  --enable_padding True \
  --enable_center_cropping True \
  --enable_resampling_spacing "1.0x1.0x1.0" \
  --ignore_existing False \
  --num_images 3448 \
  --output_root "$R/" \
  --target_shape "512x512x512"
