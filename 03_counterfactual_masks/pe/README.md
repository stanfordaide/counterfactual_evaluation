# PE counterfactual masks

This folder builds the pulmonary embolism (PE) conditioning masks. It starts from a real PE-negative CT pulmonary angiogram (CTPA) and its segmentation. It places synthetic emboli inside the pulmonary arterial tree on a fixed grid of 16 configurations per patient, and checks each mask automatically. The diffusion model in [`../../01_pe_diffusion/`](../../01_pe_diffusion/) then generates one CT per mask. Phase 6 compares the detector output on the generated CTs with the intended emboli and writes the cohort tables used by [`../../04_evaluation_and_analysis/`](../../04_evaluation_and_analysis/).

```
PE-negative CTPA -> Phase 1: vessel tree -> Phase 2: 16-config grid -> Phase 3: PE masks -> Phase 4: checks
                    Phase 5: cohort assembly (runs phases 1-4 per patient, impressions, master table)
                    Phase 6: evaluation of detector output vs intended PE (after 01_pe_diffusion inference)
```

## What you need

The masks and generated CTs of the paper are on HuggingFace (`<HF_URL>`, link to be added). You do not need this code to use them.

To run the code you need data that is not part of this release: the INSPECT CTPA scans, their TotalSegmentator masks and their report impressions. Everything runs on a CPU.

| Variable | Used for | Default |
|----------|----------|---------|
| `CF_RAW` | INSPECT data root (layout in Details) | `/path/to/private_data` |
| `CF_OUT` | Output root of the stand-alone Phase 1 and Phase 2 scripts | `/path/to/cf_mask_pipeline_output` |

`select_patients.py` also needs a CSV of PE-classifier predictions on the INSPECT test set (`CLASSIFIER_CSV` in the script; columns `image_id`, `target`, `predicted_label`, `predicted_score`).

The `.sbatch` files contain `/path/to/counterfactual_ct` placeholders and no partition or account lines. Edit them for your cluster. `conda run -n maisi` is a placeholder for your conda environment.

## Run

```bash
cd 03_counterfactual_masks/pe
pip install -r requirements.txt

# 1. Select the PE-negative patients
python phase5_cohort_assembly/select_patients.py --run-lobe-qc --finalize

# 2. Phases 1-4 and impressions, one patient per array task
mkdir -p phase5_cohort_assembly/output/logs
sbatch phase5_cohort_assembly/run_phase5.sbatch
#    or one patient:
python phase5_cohort_assembly/run_single_patient.py <STUDY_ID> --output-dir phase5_cohort_assembly/output

# 3. Master table and diffusion datalist
python phase5_cohort_assembly/aggregate_results.py --output-dir phase5_cohort_assembly/output

# 4. After CT generation in 01_pe_diffusion: evaluation tables for PE-Detector-1 and PE-Detector-2
mkdir -p phase6_evaluation/output/logs phase6_evaluation/output/vessel_overlay/logs \
         phase6_evaluation/output_apes/logs phase6_evaluation/output_apes/vessel_overlay/logs
sbatch phase6_evaluation/run_phase6.sbatch
sbatch phase6_evaluation/run_phase6_apes.sbatch
sbatch phase6_evaluation/run_vessel_overlay.sbatch
sbatch phase6_evaluation/run_vessel_overlay_apes.sbatch
#    after all array tasks finish:
python phase6_evaluation/run_phase6.py --aggregate-only --output-dir phase6_evaluation/output
python phase6_evaluation/run_phase6_apes.py --aggregate-only --output-dir phase6_evaluation/output_apes
python phase6_evaluation/run_vessel_overlay.py --aggregate-only --output-dir phase6_evaluation/output/vessel_overlay
python phase6_evaluation/run_vessel_overlay_apes.py --aggregate-only --output-dir phase6_evaluation/output_apes/vessel_overlay
```

`<STUDY_ID>` is an INSPECT study id. The `mkdir` lines are needed because SLURM opens the log files before the script starts.

## Outputs

| Step | Output |
|------|--------|
| 1 | `phase5_cohort_assembly/output/selected_patient_ids.txt` (and `selected_patients.csv`, `selected_patients_full.json`) |
| 2 | `phase5_cohort_assembly/output/patients/{id}/`: vessel graph (`phase1/`), configuration grid (`phase2/`), PE masks `{config_id}_pe_mask.nii.gz` and merged label maps `{config_id}_combined_seg.nii.gz` (`phase3/`), validation report and QC images (`phase4/`), `impressions.json` |
| 3 | `phase5_cohort_assembly/output/master_metadata.csv` and `counterfactual_datalist.json` (the datalist for `01_pe_diffusion`) |
| 4 | Four tables under `phase6_evaluation/`: `output/evaluation_all_configs.csv` and `output/vessel_overlay/vessel_overlay_all_configs.csv` (PE-Detector-1), and the same two under `output_apes/` (PE-Detector-2) |

The files of steps 1 to 3 that hold report text or demographics are not released (`counterfactual_datalist.json`, `selected_patients*`, `impressions.json`, and the `text_prompt`, age, sex and race columns of `master_metadata.csv`).

The analysis in `04_evaluation_and_analysis` reads a reduced copy of the four tables of step 4 from `resources/cohort/`: `evaluation_all_configs{,_apes}.csv` and `vessel_overlay_all_configs{,_apes}.csv` (see the resources section of `04_evaluation_and_analysis/README.md`). The `error` column (`no_intended_pe`, `missing_label`, `missing_pe_seg`) marks configurations that cannot be scored.

## Cohort used in the paper

- 150 PE-negative INSPECT test patients were selected (75 per sex, age-matched).
- 2,388 configurations were generated. 2,359 passed the four automated checks (`master_metadata.csv`, column `all_checks_pass`).
- 12 configurations were excluded: the label file was missing (1) or the PE-Detector-1 output was missing (11).
- Final cohort: 2,347 counterfactuals from 150 patients (`final_cohort.csv`, built by `04_evaluation_and_analysis/analysis_data/make_final_cohort.py`).
- Location (most proximal embolus): lobar 37%, subsegmental 31%, segmental 25%, central 6%. Severity: mild 76%, moderate 15%, severe 9%. Acute morphology 56%, chronic 44%. Median intended embolus volume 0.50 mL.

<details><summary>Details</summary>

**`CF_RAW` layout.**

```
$CF_RAW/pe_training_data_all/
  total_segmentator_all/{id}_segmentation_merged.nii.gz          anatomy (lung lobes 10-14)
  total_segmentator_all_lung_pe/{id}_segmentation_merged.nii.gz  merged mask that receives labels 119 and 121
  total_segmentator_lung_vessels_v2/{id}_segmentation_lung_vessels_v2.nii.gz   v2 vessel labels
  total_segmentator_lung_vessels/{id}_segmentation_lung_vessels.nii.gz         vessel QC in select_patients.py
  pe_datalist_spacing_all_test_with_demo_4_controlnet.json       test datalist with report impressions
```

**Seed.** Each phase can also run alone, for example `python phase1_vessel_tree/run_phase1.py <STUDY_ID>`. If `--seed` is not given, the grid seed is `hash(patient_id)`. Python randomizes string hashes per process unless `PYTHONHASHSEED` is set, so a new run gives a different random choice of vessel edges than the released masks.

**Labels in the conditioning mask.**

| Label | Structure |
|-------|-----------|
| 10-14 | Lung lobes (TotalSegmentator `total`: LUL, LLL, RUL, RML, RLL) |
| 119 | Pulmonary vessels (the Phase 1 `vessels_cleaned` mask) |
| 120 | Airways |
| 121 | Synthetic PE |

All other labels come from the merged TotalSegmentator mask of the source scan. The PE (121) is placed inside the vessel mask (119). For placement, Phase 1 labels each vessel edge as artery, vein or unknown from the TotalSegmentator v2 lung-vessel labels (3 = arteries, 4 = veins). Segmental and subsegmental emboli go into artery or unknown edges, not vein edges, unless no other edge is available (see `phase2_counterfactuals/README.md`). In the conditioning mask, arteries and veins share label 119.

Central emboli are placed on generation-0 edges (radius 8 mm or more) of this vessel tree. Artery or unknown edges are preferred; other edges are used only if none is available.

The released masks were made with an earlier version of this code that defined the central vessel seed differently. The seed definition has since been updated to improve central counterfactuals generated in the future.

**Phase 1: vessel tree (`phase1_vessel_tree/`).** `run_phase1.py` runs six steps: vessel mask from the v2 lung-vessel labels (3, 4) with small-component removal and closing; 3D skeleton and radius map; NetworkX graph (nodes are branch points and endpoints, edges are vessel segments); artery, vein or unknown per edge (majority of v2 labels along the edge); generation by radius: central (>= 8 mm), lobar (4-8 mm), segmental (2-4 mm), subsegmental (< 2 mm); lobe per edge from the anatomy mask. Output: `{patient_id}_vessel_graph_final.pkl`.

**Phase 2: counterfactual grid (`phase2_counterfactuals/`).** `counterfactual_grid.py` builds 16 configurations per patient over location, occlusion (partial 30-50%, near-complete 70-90%, complete), burden and morphology (acute: concentric; chronic: eccentric). Each configuration gets a Qanadli-style score (20 segments, maximum 40; partial or near-complete = 1, complete = 2). See [`phase2_counterfactuals/README.md`](phase2_counterfactuals/README.md) for the full grid.

**Phase 3: PE masks (`phase3_mask_construction/`).** For each target vessel segment: take the lumen voxels, group them into cross-sections along the vessel axis, and fill them from the center (acute) or from the wall (chronic) to the target occlusion. Ends are tapered, the mask is smoothed with a Gaussian kernel and thresholded, and the result stays inside the vessel. The vessel mask is written as label 119 and the PE as label 121 into the anatomy mask.

**Phase 4: checks (`phase4_validation/`)**

| Check | Criterion |
|-------|-----------|
| Containment | At least 99% of the PE voxels inside the vessel mask (`containment_fraction >= 0.99`) |
| Volume | Within the bounds for the location (`VOLUME_BOUNDS` in `automated_checks.py`): subsegmental 0.0001-60 mL, segmental 0.0001-10 mL, lobar 0.001-30 mL, central 0.01-50 mL |
| Shape | From the inertia tensor of the PE voxels: sphericity below 0.7; with 3 or more connected components, elongation above 0.2 and sphericity below 0.5. A mask with 1 to 3 voxels fails |
| Anatomy | Label 121 voxel count equals the PE voxel count; no PE voxel on the trachea (label 16 of the source mask, the only airway label checked); labels outside the PE and vessel masks unchanged |

`visual_qc.py` writes QC slice images. `run_phase4.py` runs the checks for one patient.

**Phase 5: cohort assembly (`phase5_cohort_assembly/`).** `select_patients.py` selects the PE-negative patients: true negatives of a PE classifier, vessel QC (at least 5,000 vessel voxels), optional lobe QC (`--run-lobe-qc`: at least 4 lobes with at least 2 mL of vessels), 75 per sex, age-matched. `run_single_patient.py` runs phases 1-4 for one patient, with retries, and writes the impression text: `impression_generator.py` replaces the "no PE" statement of the source impression with a template sentence for the inserted PE.

Retries: if fewer than 16 configurations pass, `generate_replacement_configs` (Phase 2) makes replacement configurations, for up to 3 rounds. A replacement keeps the location, burden and occlusion of the failed configuration and uses edges in a different lobe, drawn from the larger half of the candidate edges. Round 2 raises the occlusion one level. Round 3 also changes an isolated burden to multi-unilateral (3 edges). Replacements are always chronic (eccentric fill): the code does not pass the `acute` flag, so the default `acute=False` applies. In the paper cohort, the 23 counterfactuals numbered above `cf016` are all chronic.

**Phase 6: evaluation (`phase6_evaluation/`)**

These scripts run after the CTs are generated with `01_pe_diffusion/scripts/infer_controlnet.py`. They compare the intended PE (label 121 of the generated `_label` file) with the detector output. The four `.sbatch` files run one patient per array task; `--aggregate-only` then writes the `*_all_configs.csv` table.

The detector output is on the grid of the generated CT (D slices). Runs made before the generator fix in `01_pe_diffusion` saved the `_label` file on the source grid (nz slices). Every quantity that uses the prediction (overlap, detected volume, edge detection) is computed on the CT grid: the scripts put the label on that grid with the generator's slice map (`generator_slice_map` in `utils/nifti_io.py`: generated slice k = label slice floor(k x nz / D)), and sizes and distances use the label spacing with z scaled by nz / D. The properties of the intended mask alone (`intended_*`, `vessel_voxels`, `n_intended_components`, `n_edges_with_intended_pe`) are computed on the full label grid with the label spacing. `intended_volume_ml` equals `pe_volume_ml` of `master_metadata.csv`. For new runs the two grids are the same.

Inputs:
- `--predictions-csv`: the CSV written by `01_pe_diffusion/scripts/controlnet_prediction_csv.py`. Only the columns `image_id` and `prediction_filename` are read. The `_label` and `_pe_seg` files must sit next to each `_image` file.
- `--grid-base-dir`: `phase5_cohort_assembly/output/patients` (vessel graph in `phase1/`, grid JSON in `phase2/`).
- `--apes-seg-dir` (APES scripts only): one binary APES mask per generated CT, named like the generated image without the `_image` suffix. We ran the public APES model to make these masks. That step is not part of this folder.

The acute/chronic shape category uses: acute if at most 35% of the PE voxels are within 1 mm of the vessel wall and the mean normalized depth is at least 0.15; chronic if at least 65% are within 1 mm or the mean depth is at most 0.06; otherwise indeterminate (`_compute_morphology` in `run_vessel_overlay.py`).

</details>
