# 02 ICH diffusion model

This folder trains and runs the head CT generator for intracranial hemorrhage (ICH). The
generator is a fine-tuned MAISI model (3D latent diffusion for CT, rectified-flow version).
It makes a non-contrast head CT from a segmentation label map (head anatomy plus the
hemorrhage, label 66) and a text prompt (demographic prefix plus a report impression). The
prompt is encoded by a frozen ClinicalT5-base encoder. The MAISI VAE is not retrained.

The counterfactual label maps and prompts come from
[`../03_counterfactual_masks/ich/`](../03_counterfactual_masks/ich/).

## What you need

```bash
cd 02_ich_diffusion                      # run every command from this folder
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$(pwd)/scripts:$PYTHONPATH
export CF_CLINICALT5=/path/to/ClinicalT5-base/model              # T5 encoder weights (same checkpoint as PE)
export CF_CLINICALT5_TOKENIZER=/path/to/ClinicalT5-base/tokenizer  # tokenizer files (spiece.model)
export WANDB_API_KEY=...                                         # training only (Weights & Biases)
```

A GPU is needed. To generate scans without training, download the weights and use only step 8:

| File | Content |
|---|---|
| `models/ich/diffusion_unet.pt` (HuggingFace, `<HF_URL>`, link to be added) | ICH diffusion U-Net |
| `models/ich/controlnet.pt` (HuggingFace) | ICH ControlNet |
| `models/autoencoder_v1.pt` from [`nvidia/NV-Generate-CT`](https://huggingface.co/nvidia/NV-Generate-CT) | MAISI VAE, same file for PE and ICH |

The training data are internal head CT studies and their report impressions. The scans, the
reports, the text prompts and the datalists are not released. The prompt format is
(placeholders only):

```
<AGE> years old <race> <sex>. IMPRESSION:<impression text>
```

Every path in `configs/` is a `/path/to/...` placeholder. Edit the paths before you run. The
data-preparation scripts (steps 1 to 5) have their paths at the top of the file or as
command-line defaults.

## Run

Steps 1 to 7 build the training data and train the two models. Step 8 generates head CTs.

```bash
# 1. List the CT volumes: writes ich_training_data_all.json next to the input table
python create_ich_training_data.py --task create_dataset_json

# 2. Segment anatomy and hemorrhage with TotalSegmentator, one run per task:
#    writes <output_base_dir>/total_segmentator_<task>/.../<case>_segmentation_<task>.nii.gz
for t in brain_structures craniofacial_structures cerebral_bleed; do
  python scripts/run_total_segmentator_headct.py --task $t --num_gpus 1 \
    --ct_dir /path/to/head_ct_niis \
    --dataset_json_path /path/to/ich_datalists/ich_training_data_all.json \
    --output_base_dir /path/to/total_segmentator \
    --model_dir /path/to/totalsegmentator_weights
done

# 3. Merge the labels: set the `task` variable in the script and run the four tasks in this order.
#    Writes the label maps *_segmentation_brain_craniofacial_ICH_merged.nii.gz
python scripts/merge_total_segmentator_labels_headct_batch.py                          # task = "create_image_list_merge_brain_craniofacial"
python scripts/merge_total_segmentator_labels_headct_batch.py <id_list.txt> <start> <end>  # task = "merge_brain_craniofacial"
python scripts/merge_total_segmentator_labels_headct_batch.py                          # task = "create_image_list_merge_brain_craniofacial_ICH"
python scripts/merge_total_segmentator_labels_headct_batch.py <id_list.txt> <start> <end>  # task = "merge_brain_craniofacial_ICH"

# 4. Encode the CTs with the frozen VAE: writes <embedding_base_dir>/<relative path>_emb.nii.gz
python create_ich_training_data.py --task extract_vae_features --num_gpus <N>

# 5. Split the data and build the datalists: set the `task` variable and run once per task
#    (create_train_val_test_split, fill_in_dim_and_spacing, create_diffusion_model_training_data,
#    create_test_data_json). Writes diffusion_train_headct.json and the test-set datalist
python data/create_headct_training_datalists.py

# 6. Fine-tune the diffusion U-Net: writes <model_filename>_epoch_<k>.pt and <model_filename>_best.pt
torchrun --standalone --nproc_per_node=<N> scripts/train_diff_model_train_text_headct.py --num_gpus <N>

# 7. Fine-tune the ControlNet with the U-Net frozen:
#    writes <model_dir>/weighted_loss_10.0/<exp_name>_epoch_<k>.pt and <exp_name>_best.pt
torchrun --nproc_per_node=<N> scripts/train_controlnet_seg_text_headct.py --gpus <N>

# 8. Generate head CTs (one GPU)
python scripts/infer_controlnet_headct_infer_embed_then_decode.py
```

For step 8, set `json_data_list` in the inference environment file to the test-set datalist
or to the counterfactual datalist from `../03_counterfactual_masks/ich/`. Each entry needs
`image` (used for the output name), `embedding` (any readable NIfTI; loaded but not used),
`segmentation`, `dim`, `spacing` and `text_prompt`.

Outputs of step 8:

- `<output_dir>/embeddings/`: latents and the label maps resized to `dim` (the generator input).
- `<output_dir>/decoded_images/*_image_embed_decoded.nii.gz`: the CTs (int16, HU).

The decoding pass reads every `*_image_embed.nii.gz` file in `<output_dir>/embeddings/`, so use
a new `output_dir` for each run.

The code that de-faces the released head CTs will be added to this repository.

<details><summary>Details</summary>

**Environment.** `requirements.txt` installs the environment of `../01_pe_diffusion/` (which
includes TotalSegmentator) and adds `sentencepiece` for `T5Tokenizer`. The TotalSegmentator
`brain_structures` task needs a license key. Set `WANDB_ENTITY` to choose the W&B entity.

**Released checkpoints.** U-Net: lowest-training-loss checkpoint (epoch 144); the run used a
200-epoch schedule and stopped after epoch 154. ControlNet: lowest-training-loss checkpoint
(epoch 30) of a completed 50-epoch run. Training starts from the MAISI checkpoints
`diff_unet_3d_rflow.pt` and `controlnet_3d_rflow.pt`.

**Configuration files.**

| File | Step | Keys the script reads |
|---|---|---|
| `environment_maisi_diff_model_headct.json` | 4 | `data_base_dir`, `embedding_base_dir`, `json_data_list`, `trained_autoencoder_path` |
| `environment_maisi_diff_model_with_text_ICH_piecewise_linear_mapping_CenterWeightedMSE.json` | 6 | `json_data_list`, `model_dir`, `model_filename`, `existing_ckpt_filepath` |
| `environment_maisi_controlnet_train_with_text_ICH_seg_piecewise_linear_mapping_CenterWeightedMSE.json` | 7 | `data_base_dir`, `json_data_list`, `model_dir`, `tfevent_path`, `exp_name`, `trained_diffusion_path`, `trained_controlnet_path` |
| `environment_maisi_controlnet_infer_with_text_ICH_seg_piecewise_linear_mapping_CenterWeightedMSE.json` | 8 | `data_base_dir`, `json_data_list`, `output_dir`, `trained_autoencoder_path`, `trained_diffusion_path`, `trained_controlnet_path` |
| `config_maisi3d-rflow.json` | 4, 6 to 8 | network definitions and the rectified-flow scheduler |
| `config_maisi_diff_model_headct.json` | 4, 6 | U-Net training: Adam, lr 1e-4, 200 epochs, batch size 1 |
| `config_maisi_controlnet_train_headct_weighted_label.json` | 7, 8 | ControlNet training: AdamW, lr 1e-4, 50 epochs, label 66 weight 10. Inference: 30 steps, VAE decoding window 80 x 80 x 80 with overlap 0.4 |

Both training scripts use a polynomial learning-rate decay (power 2) over all epochs.

**Data preparation.** Step 1 needs a table with one row per head CT and the column
`ct_nifti_rel_path` (set `all_info_csv` in the script). The merge tasks of step 3 take an id
list and a start and end index, so large sets can run as parallel jobs. Step 5 splits by
patient into 70 / 10 / 20 % with strata of 10-year age bin, sex and race (`data/data_utils.py`,
seed 42); its input table has the columns `Patient Id`, `Age`, `Gender`, `Race`,
`ct_nifti_rel_path`, `Impression` and `LLM Label`. `dim` and `spacing` of each datalist entry
are set to one of two presets: 512 x 512 x 64 with 0.43 x 0.43 x 2.9 mm, or 512 x 512 x 512
with 0.43 x 0.43 x 0.29 mm (the preset whose slice count is nearer to the scan). The
training-set builder stops at the first scan that has no latent or no label map (`break`). The
test-set builder skips such scans. The prompt builder writes age as an integer and race and
sex in lower case.

**What we changed in MAISI.**

1. Intensity transform for the brain window. Before VAE encoding, HU values go through a
   piecewise-linear map with breakpoints [-1000, -100, 0, 80, 300, 2000] HU mapped to
   [-1000, -600, -200, 400, 700, 1000], then the standard MAISI scaling of [-1000, 1000] to
   [0, 1] (`ct_piecewise_forward_torch`). After decoding, the inverse map gives HU in
   [-1000, 2000] (`revise_transform`).
2. Volume size. Each axis is resized to the nearest multiple of 128. If the result is
   512 x 512 x 512, or any axis is above 512, the volume is resized to 512 x 512 x 384.
3. Text conditioning. The U-Net and the ControlNet get the ClinicalT5 token embeddings (768
   channels) through cross-attention (`scripts/text_controlnet.py`).
4. U-Net loss. L2 loss in which the central half of each spatial axis of the latent is weighted
   by 2 (`CenterWeightedMSE` in `scripts/loss_functions.py`).
5. ControlNet loss. L2 loss in which latent voxels with label 66 have weight 10.
6. Two-pass inference. The script samples and saves all latents, frees the U-Net, ControlNet
   and text encoder, and then decodes every latent with the VAE.

**Label map** (`scripts/merge_total_segmentator_labels_headct_batch.py`): 1 to 16 =
TotalSegmentator `brain_structures`; 21, 22, 23, 25, 26, 27 = `craniofacial_structures` label
+ 20 (mandible, lower teeth, skull, maxillary sinus, frontal sinus, upper teeth); 24 = all
other voxels in the head mask; 66 = hemorrhage (`cerebral_bleed`). The head mask is the
TotalSegmentator `head` or `skull` region after one binary closing and four dilations. Label
66 replaces any other label.

**Headers.** Latents, label maps and CTs have the affine of the generated grid: the affine of
the input label map with each axis scaled by (label size / `dim`). The released scans were
made before this fix; their headers were corrected in the released dataset.

</details>
