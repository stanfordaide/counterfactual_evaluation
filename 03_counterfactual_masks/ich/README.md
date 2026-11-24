# ICH counterfactual masks

This folder builds the intracranial hemorrhage (ICH) conditioning masks. A real hemorrhage mask from a donor scan is moved into the anatomy segmentation of an ICH-negative source scan by skull registration. The result is the conditioning mask for the ICH diffusion model in [`../../02_ich_diffusion/`](../../02_ich_diffusion/). No CT intensities are used or changed in this step.

| Script | Purpose |
|--------|---------|
| `generate_ich_counterfactual_masks.py` | Registers each donor to its recipient and writes the counterfactual mask (ICH = label 66) |
| `prepare_controlnet_input.py` | Computes the ICH volume of each counterfactual mask, builds the text prompt and writes `controlnet_input.json` for ControlNet inference |

## What you need

The conditioning masks and source anatomy masks of the paper are on HuggingFace (`<HF_URL>`, link to be added), with the list of pairs (`ich/counterfactual_pairs.json`) and the donor table (`ich/donor_info.csv`). You do not need this code to use them.

To run the code you need:

- anatomy segmentations of the ICH-negative source scans and of the donor scans, and a binary hemorrhage mask per donor. The donor masks of the paper are not in the current release.
- a pairing JSON in the input format below.
- for the prompts only: a donor table with the columns `ich_idx` and `subtype` (the HuggingFace table `ich/donor_info.csv` names the index `donor_idx`), and a private CSV with the report impression and demographics of each source scan (`--patient_info_csv`, default `$CF_RAW/ich/patient_info.csv`, columns `patient_id`, `impression`, `age`, `gender`, `race`). `patient_id` must match the file names in `src_healthy_seg_path`. The impressions and demographics are not released.

## Run

```bash
cd 03_counterfactual_masks/ich
pip install -r requirements.txt

# 1. Counterfactual masks: one file per pair, at the `segmentation` path of the entry
python generate_ich_counterfactual_masks.py \
  --json_path /path/to/ich_pairs_2640.json \
  --temp_dir /path/to/temp_dir

# 2. Volumes and prompts: writes controlnet_input.json next to the pairing JSON
python prepare_controlnet_input.py \
  --pairs_json /path/to/ich_pairs_2640.json \
  --ich_info_scv /path/to/ich_info.csv \
  --patient_info_csv /path/to/patient_info.csv
```

Outputs: step 1 writes one counterfactual mask (uint16, ICH = label 66) per pair. Step 2 writes `controlnet_input.json`, the datalist for step 8 of `../../02_ich_diffusion/`. The option name `--ich_info_scv` is spelled this way in the code.

## Input format

```json
{
  "training": [
    {
      "src_healthy_seg_path": "source_anatomy_masks/ICH001.nii.gz",
      "src_doner_seg_path": "donor_masks/brain/12.nii.gz",
      "src_ich_seg_path": "donor_masks/ich/12.nii.gz",
      "segmentation": "conditioning_masks/ICH001_12.nii.gz"
    }
  ]
}
```

| Key | Content |
|-----|---------|
| `src_healthy_seg_path` | Anatomy segmentation of the ICH-negative source scan (recipient) |
| `src_doner_seg_path` | Anatomy segmentation of the donor scan. The key name is misspelled in the code; keep it as written. |
| `src_ich_seg_path` | Binary ICH mask of the donor |
| `segmentation` | Output path of the counterfactual mask |

Paths are opened as written, so give absolute paths or run from the folder they are relative to. `prepare_controlnet_input.py` reads the source id from the file name of `src_healthy_seg_path` and the donor index from the file name of `src_ich_seg_path`.

## Data used in the paper

- Source anatomy: 150 ICH-negative head CTs. Each segmentation is the merge of the TotalSegmentator `brain_structures` and `craniofacial_structures` outputs (`02_ich_diffusion/scripts/merge_total_segmentator_labels_headct_batch.py`).
- Donor library: 165 hemorrhage masks. The `dataset` value `SU300` marks the 91 masks that, in the words of the paper, "were derived from expert annotations". The paper continues: "The remaining 74 masks were obtained via automated segmentation [42], including 41 cases from the internal Stanford head CT dataset and 33 from the RSNA intracranial hemorrhage dataset" (`dataset` values `Stanford` and `RSNA`; both TotalSegmentator `cerebral_bleed`). Each donor also has its merged anatomy segmentation, used only for the skull registration.
- Pairs: each donor was pasted into 16 source scans (165 x 16 = 2,640 pairs). Each source scan received 17 or 18 donors. 2,601 of the 2,640 counterfactual masks have an ICH volume above 0 mL. These 2,601 form the evaluation cohort. The step that drew the pairs is not part of this folder.

<details><summary>Details</summary>

**Notes.** Use one `--temp_dir` per process if you split the list over several jobs. `generate_ich_counterfactual_masks.py` processes the entries in order and has no per-entry error handling.

**Labels.**

| Label | Structure |
|-------|-----------|
| 0 | Background |
| 1-16 | Brain structures (TotalSegmentator `brain_structures`, in order: brainstem, subarachnoid space, venous sinuses, septum pellucidum, cerebellum, caudate nucleus, lentiform nucleus, insular cortex, internal capsule, ventricle, central sulcus, frontal lobe, parietal lobe, occipital lobe, temporal lobe, thalamus) |
| 21-27 | Craniofacial structures (TotalSegmentator `craniofacial_structures` label + 20): 21 mandible, 22 lower teeth, 23 skull, 25 maxillary sinus, 26 frontal sinus, 27 upper teeth |
| 24 | Head: the remaining voxels inside the head mask |
| 66 | ICH (written by `generate_ich_counterfactual_masks.py`) |

The donor ICH mask is binary (1 = ICH). Labels 1-16 are the recipient region that may receive ICH (`--brain_start`, `--brain_end`).

**Method (`generate_ich_counterfactual_masks.py`).**

For each entry of the pairing JSON:

1. Reorient the three input masks to the closest canonical orientation and make the affine orthonormal (copies in `--temp_dir`).
2. Make the skull mask (label 23) of the recipient and of the donor. With `--use_head_with_skull`, label 24 is added.
3. Crop each skull mask to its bounding box plus `--margin_mm` (25 mm), compute signed distance maps and resample them to `--reg_spacing_mm` (6 mm) isotropic voxels.
4. Register the donor (moving) to the recipient (fixed): rigid 3D transform, mean-squares metric, regular-step gradient descent, `--iters` (60) iterations, 2 resolution levels. With `--affine`, an affine step follows.
5. Resample the donor ICH mask into the recipient grid (nearest neighbor), keep only voxels inside the recipient brain labels 1-16, and write them as label 66.
6. Save the result (uint16) at the `segmentation` path of the entry.

The paper used the defaults: rigid registration driven by the skull only (no `--affine`, no `--use_head_with_skull`).

**Arguments.**

| Argument | Default | Meaning |
|----------|---------|---------|
| `--json_path` | `/path/to/ich_pairs.json` | Pairing JSON |
| `--temp_dir` | `/path/to/temp_dir` | Folder for the reoriented copies |
| `--skull_label` | 23 | Skull label |
| `--head_label` | 24 | Head label (used only with `--use_head_with_skull`) |
| `--ich_in_label` | 1 | ICH value in the donor mask |
| `--ich_out_label` | 66 | ICH label in the output |
| `--brain_start`, `--brain_end` | 1, 16 | Recipient labels that may receive ICH |
| `--use_head_with_skull` | off | Register on skull + head |
| `--margin_mm` | 25 | Crop margin around the skull |
| `--reg_spacing_mm` | 6 | Registration voxel size |
| `--iters` | 60 | Registration iterations |
| `--affine` | off | Add an affine step after the rigid step |

**Text prompt (`prepare_controlnet_input.py`).**

For each pair, the script:

- computes the ICH volume from label 66 of the counterfactual mask (resampled to 512 x 512 x 512 by nearest neighbor if needed, voxel size 0.43 x 0.43 x 0.29 mm, rounded to 2 decimals),
- reads the subtype(s) of the donor from the donor table,
- takes the report impression of the source scan and removes its first numbered finding (`remove_original_ich_info`). The finding is found with the regular expression `1\.\s+(.+?)(?=\s*\d+\.|$)`, but the text removed is the literal `"1.  "` (with two spaces) followed by the match. If the spacing after `1.` is different, nothing is removed. If the regular expression finds no numbered finding, the standard negative ICH sentence written in the script is removed instead,
- replaces the literal `IMPRESSION:` with the demographic prefix (`<age> years old <race> <gender>.`), `IMPRESSION:` and a template sentence that names the subtype(s) and the total volume. The `impression` column must therefore contain `IMPRESSION:`. If it does not, the prompt gets neither the prefix nor the hemorrhage sentence,
- writes `controlnet_input.json` next to the pairing JSON. Each entry keeps the pair keys and adds `image` and `embedding` (both set to the mask path; the loader needs these keys), `dim`, `spacing`, `text_prompt`, `ich_subtype`, `ich_volume_ml`, `patient_id`, `age`, `gender` and `race`.

These two steps reproduce the original pipeline: the same regular expression and literal were used to clean the impressions of the source scans, and the same `IMPRESSION:` replacement was used to insert the hemorrhage sentence.

</details>
