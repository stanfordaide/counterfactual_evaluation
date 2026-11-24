# 01: PE diffusion model (MAISI ControlNet)

This folder trains and runs the PE generator. The generator is built on NVIDIA MAISI
(`maisi3d-rflow`, a 3D latent diffusion model with a rectified-flow scheduler). It makes a CT
pulmonary angiogram (CTPA) from a segmentation label map and a text prompt. We fine-tuned the
text-conditioned diffusion U-Net and the ControlNet. The MAISI autoencoder is used as is.

`fidelity/` holds the scripts for the fidelity metrics of Extended Data Table 1 (2.5D FID and
BiomedCLIP CLIPScore). The counterfactual label maps come from
[`../03_counterfactual_masks/`](../03_counterfactual_masks/).

## What you need

```bash
cd 01_pe_diffusion                       # run every command from this folder
pip install -r requirements.txt          # the GPU steps of the paper ran with Python 3.11
export PYTHONPATH=$(pwd):$(pwd)/scripts:$PYTHONPATH
```

The generator needs a GPU.

| Item | Where to get it | How the code finds it |
|---|---|---|
| Fine-tuned U-Net and ControlNet | HuggingFace (`<HF_URL>`, link to be added): `models/pe/diffusion_unet.pt`, `models/pe/controlnet.pt` | paths in the environment JSON |
| MAISI autoencoder | [`nvidia/NV-Generate-CT`](https://huggingface.co/nvidia/NV-Generate-CT), file `models/autoencoder_v1.pt` | path in the environment JSON |
| ClinicalT5-base | `luqh/ClinicalT5-base` on HuggingFace; make a PyTorch copy with `python scripts/convert_t5.py` | `CF_CLINICALT5` |
| nnU-Net PE detector (PE-Detector-1) | from its authors (see the paper) | `CF_NNUNET_RESULTS` (the `nnUNet_results` folder), `CF_NNUNET_INPUT` (used as `nnUNet_raw` and `nnUNet_preprocessed`) |
| Conditioning masks | HuggingFace: `dataset/pe/conditioning_masks/` | datalist JSON |
| Training and fidelity data (INSPECT CTPA, impressions, generated test scans) | not released | `CF_RAW` |

The training scripts log to Weights & Biases (`WANDB_API_KEY`). Radiology impressions are not
released, so you write the `text_prompt` of each datalist entry yourself. The released models
use this format (placeholders only):

```
"<age> year old <race> <sex>. IMPRESSION: 1. <finding>."
```

## Generate a CT from a released conditioning mask

1. Write a datalist JSON, for example `my_datalist.json`:

```json
{"test": [
  {"image_id": "<cf_id>",
   "segmentation": "pe/conditioning_masks/<cf_id>.nii.gz",
   "embedding": "pe/conditioning_masks/<cf_id>.nii.gz",
   "dim": [<nx>, <ny>, <nz>],
   "spacing": [<sx>, <sy>, <sz>],
   "pe_status": 1,
   "text_prompt": "<age> year old <race> <sex>. IMPRESSION: 1. <finding>."}
]}
```

   `dim` is the mask shape and `spacing` is its voxel size in mm, from the NIfTI header. The
   paths are relative to `data_base_dir`.

2. Edit `configs/environment_maisi_controlnet_inference_demo.json`: the three weight paths,
   `data_base_dir` (your local `dataset/` folder), `json_data_list` and `output_dir`.
3. Run:

```bash
# one GPU
python scripts/infer_controlnet.py -g 1 \
  -e ./configs/environment_maisi_controlnet_inference_demo.json \
  -t ./configs/config_maisi_controlnet_train.json
# several GPUs
torchrun --nproc_per_node=4 scripts/infer_controlnet.py -g 4 --skip-existing \
  -e ./configs/environment_maisi_controlnet_inference_demo.json \
  -t ./configs/config_maisi_controlnet_train.json
```

Outputs in `output_dir`:

- `<image_id>_pred_<time>_image.nii.gz`: the generated CT.
- `<image_id>_pred_<time>_label.nii.gz`: the label map resized to `dim` (the generator input).
- `<image_id>_pred_<time>_pe_seg.nii.gz`: the nnU-Net PE detector output (only with `pe_status` = 1).

To match the paper's counterfactual run, set `autoencoder_sliding_window_infer_overlap` to
0.67 in the `-t` file (it is 0.4 in the shipped file).

## Train the models

The training data are not released. These steps show how the released models were made.

```bash
# 1. ClinicalT5 PyTorch copy
CF_CLINICALT5=/path/to/clinicalt5-pytorch python scripts/convert_t5.py

# 2. Optional: find interleaved CTs (two series merged into one volume); writes interleaved_cts.csv
python scripts/detect_interleaved_cts.py --ct-dir $CF_RAW/CTPA --output-csv interleaved_cts.csv --workers 8

# 3. VAE latents: writes <embedding_base_dir>/<name>_emb.nii.gz
python create_pe_training_data.py --num_gpus 8 --env_config ./configs/environment_maisi_diff_model_text_demo.json

# 4. Vessel and airway masks for the ControlNet label maps, then point the datalist to the merged masks
python scripts/run_total_segmentator.py --dataset_json <dataset.json> --ct_dir $CF_RAW/CTPA --output_dir <out> --num_gpus 4
python scripts/update_controlnet_datalist.py

# 5. Diffusion U-Net: writes the checkpoints to model_dir
torchrun --nproc_per_node=8 scripts/train_diff_model_text_demo.py \
  --env_config ./configs/environment_maisi_diff_model_text_demo.json \
  --model_config ./configs/config_maisi_diff_model.json \
  --model_def ./configs/config_maisi3d-rflow.json --num_gpus 8

# 6. ControlNet, with the U-Net of step 5 frozen: writes the checkpoints to model_dir
torchrun --nproc_per_node=8 scripts/train_controlnet_seg_text.py -g 8 \
  -e ./configs/environment_maisi_controlnet_train_demo.json \
  -c ./configs/config_maisi3d-rflow.json \
  -t ./configs/config_maisi_controlnet_train.json

# 7. Quality control of generated scans: writes qc_results.csv
python scripts/qc_generated_scans.py --pred-dir <output_dir> --output-csv qc_results.csv --patch-size 80 --overlap 0.67
```

Edit the paths in the environment JSON files before each step.

## Fidelity metrics (Extended Data Table 1)

The inputs (real test CTs, generated test scans, prompts) are not released. Submit the `sbatch`
scripts from this folder after `mkdir -p fidelity/logs`.

```bash
# 1. Manifest of a ControlNet run
python scripts/controlnet_prediction_csv.py <pred_dir> <datalist.json> -o <pred_dir>/controlnet_predictions.csv
# 2. CLIPScore per configuration: writes <dir>/clipscore_merged.csv
torchrun --nproc_per_node=4 scripts/compute_biomedclip_scores.py --csv_path <manifest> --output_dir <dir>
# 3. CLIPScore table and tests: fidelity_clip_table.csv, fidelity_clip_test.csv
python fidelity/fidelity_stats.py
# 4. File lists, real-scan features, features per configuration
python fidelity/build_fid_filelists.py
sbatch fidelity/run_fid_score.sh
sbatch --array=0-3 fidelity/run_fid_overlap.sh
# 5. Bootstrap (B = 1,000): fidelity_fid_table.csv (FID, 95% CI), fidelity_fid_test.csv
python fidelity/fid_bootstrap_prep.py
sbatch --array=0-3 fidelity/run_fid_bootstrap.sh
python fidelity/fid_bootstrap_combine.py
```

<details><summary>Details</summary>

**Released checkpoints.** U-Net: lowest-training-loss checkpoint, `epoch` = 193 (`n_epochs` =
200). ControlNet: lowest-training-loss checkpoint, `epoch` = 14 (`n_epochs` = 40). The values
are read from the checkpoint files (1-based). `infer_controlnet.py` reads the keys
`unet_state_dict` and `scale_factor` from the U-Net file and `controlnet_state_dict` from the
ControlNet file. `autoencoder_v1.pt` is identical to MAISI `autoencoder_epoch273.pt`.

**Other weights.** Fine-tuning starts from the MAISI checkpoints `diff_unet_3d_rflow.pt`,
`controlnet_3d_rflow.pt` and `autoencoder_epoch273.pt`. `train_diff_model_text*.py` loads
ClinicalT5 directly with `from_flax=True`. `infer_controlnet.py` loads the detector from
`$CF_NNUNET_RESULTS/Dataset256_PE/nnUNetTrainer_500epochs__nnUNetResEncUNetPlans_80G__3d_fullres`
(`checkpoint_final.pth`, folds `all`) before it generates any scan. FID needs the RadImageNet
ResNet-50 weights at `$CF_RAW/pe_training_data_all/models/ResNet50.pt`. BiomedCLIP is loaded
from the HuggingFace hub.

**`CF_RAW` layout.** `$CF_RAW/CTPA/<image_id>.nii.gz` (INSPECT CTPA) and
`$CF_RAW/pe_training_data_all/` (label CSVs, embeddings, generated test scans, FID features).

**Labels** (`configs/label_dict_inspect.json`, 0 is background): 1 to 118 are the
TotalSegmentator `total` classes and the `body` outline (118); 119 = pulmonary vessels and
120 = airways (TotalSegmentator `lung_vessels` task); 121 = pulmonary embolism. The ControlNet
receives the label map as an 8-channel bit-plane encoding. Its loss weights label 121 by 100.

**Prompts.** In file and option names, `demo` means the demographic prefix.
`train_diff_model_text_demo.py` and `run_batch_inference.py --demo` build
`"<age> year old <race> <sex>. "` from the datalist fields `age`, `race` and `gender`.
`train_controlnet_seg_text.py` and `infer_controlnet.py` use `text_prompt` as it is.
`diff_model_infer_text.py` needs `--prompts`.

**Inference.** `check_input` in `scripts/sample.py` accepts `dim[0] = dim[1]` in {256, 384, 512},
`dim[2]` in {128, 256, 384, 512, 640, 768}, `spacing[0] = spacing[1]` from 0.5 to 3.0 mm and
`spacing[2]` from 0.5 to 5.0 mm. The `embedding` file is read but not used, so it can point to
the mask. Settings come from the `controlnet_infer` block of the `-t` file: 30 steps and an
80 x 80 x 80 decoding window. The CT and the saved label have the affine of the generated grid:
the affine of the input label map with each axis scaled by (label size / `dim`). The released
scans were made before this fix; their headers were corrected in the released dataset. On NFS,
the nnU-Net clean-up can print `OSError: Device or resource busy`; the outputs are still
written.

**Training.** Step 3 reads `Final_Impressions_labels.csv` and `Final_Impressions.csv` in
`$CF_RAW/pe_training_data_all/`. Step 4 runs the TotalSegmentator `lung_vessels` task; the
`total` and `body` tasks, the nnU-Net PE masks and the merge into one label map are not in this
folder. U-Net: lr 1e-4, 200 epochs. `train_diff_model_text.py` is the same script without the
demographic prefix (text-only ablation). ControlNet: lr 1e-5, 40 epochs;
`environment_maisi_controlnet_train.json` is the text + mask ablation. The interleaved CTs of
step 2 were left out of the ControlNet training data of the released model. QC defaults:
banding ratio 1.15, through-plane change 35 HU, body standard deviation 180 to 400 HU, lung
fraction 0.005, heart fraction 0.002; `--overlap` must match the decoding overlap.

**Fidelity.** Four configurations (FID key / CLIPScore key): `text` / `text`, `text_demo` /
`text+demo`, `text_mask` / `text+mask`, `text_demo_mask` / `text+demo+mask` (the released
models). U-Net-only scans come from `run_batch_inference.py`, which calls
`diff_model_infer_text.py` once per test case; edit its paths first. The QC manifest
`controlnet_predictions_qc.csv` was made by hand from the manifest of the released
configuration (3,528 rows): the 80 scans that failed QC or had a corrupt file were removed,
which leaves 3,448 rows. `fidelity_stats.py` uses the common QC-passed set (3,447 scans in the
paper). The two lists read by `run_fid_score.sh` (`real_file_list_fid_v2_clean.txt`,
`synth_file_list_fid_v2_clean.txt`) and `fid_overlap_filelists/real_stub_filelist.txt` (50
lines of the real list) were also written by hand. The bootstrap worker runs on the CPU in
float64 (peak memory 46 GB, about 20 minutes per configuration with 16 threads). The FID that
`compute_fid_2-5d_ct_old.py` prints in `run_fid_overlap.sh` uses the small real stub; ignore
it. The ICH FID and CLIPScore come from head-CT versions of these scripts, which are not in
this repository.

</details>
