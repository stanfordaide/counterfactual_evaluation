# 04 Evaluation and analysis

This folder computes every number of the paper. It holds two kinds of code:

- **Statistics.** Detection rates, confidence intervals, models and tests. They run on a CPU
  from the tables in `../resources/`. No download is needed.
- **Detector scoring.** Code that scores the PE and ICH detectors against the intended lesion
  masks and writes the detection tables. These steps need CT volumes, masks or detector
  predictions. Their outputs are in `../resources/`, so every statistic runs without them.

The detector models and the scripts that draw the paper figures are not included.

## What you need

```bash
cd 04_evaluation_and_analysis        # run every command below from this folder
pip install -r requirements.txt      # the paper used Python 3.12, statsmodels 0.14, SciPy 1.15
```

For the statistics, leave all environment variables unset. `CF_DATA` (default `../resources`)
points to the tables. `CF_RAW` points to data that is not shipped and is used only by the
detector-scoring steps. `ICH_SEG` (default `viola`) selects the ICH segmentation detector.

## Run all CPU statistics

```bash
python analysis_data/make_final_cohort.py
python analysis/cohort_descriptives.py
python patient_bootstrap/text_numbers.py
python patient_bootstrap/figure_numbers.py
python analysis/qanadli_distribution.py
python analysis/ich_subtype_distribution.py
python analysis/detection_operating_limits_ci.py
python analysis/radiologist_ci.py
python analysis/radiologist_scores.py
python analysis/reader_confirmed_and_disease_free.py
python analysis/disease_free_fp_size.py
python analysis/refit_pe_drivers.py
python analysis/run_ich_glmm.py
python analysis/comorbidity_extraction_kappa.py
python analysis/comorbidity_gee.py
python analysis/effusion_interaction.py
```

The 16 scripts take less than a minute in total. Each script prints its results and writes
its tables to `output/` (not tracked).

## What comes out

| Paper item | Script | Output in `output/` |
|---|---|---|
| Analysis cohort: 2,347 PE counterfactuals, 150 patients | `analysis_data/make_final_cohort.py` | `final_cohort.csv` (identical to `../resources/cohort/final_cohort.csv`) |
| Cohort numbers (Results), Supplementary Table 1, cohort funnel from 2,388 generated configurations to 2,347 (Methods) | `analysis/cohort_descriptives.py` | printed |
| Qanadli score median, IQR and bands (Supplementary Table 1) | `analysis/qanadli_distribution.py` | `qanadli_distribution.csv` |
| ICH subtypes, single and combined (Results) | `analysis/ich_subtype_distribution.py` | `ich_subtype_distribution.csv` |
| Radiologist review with 95% CIs (Results, Fig. 2a); overall sensitivity and sensitivity by volume, location, severity, subtype and number of subtypes (Results) | `patient_bootstrap/text_numbers.py` | `patient_bootstrap/manuscript_numbers.csv` |
| Radiologist mean ± SD scores | `analysis/radiologist_scores.py` | printed |
| Within-patient resemblance with exact CIs (Fig. 2b) | `analysis/radiologist_ci.py` | `radiologist_ci.csv` (`resemblance` rows) |
| Cumulative 90% sensitivity volume thresholds | `analysis/detection_operating_limits_ci.py` | `detection_operating_limits_ci.csv` (`cum_threshold_p90_mL` rows: `pct` is the 90% threshold; `ci_low`, `ci_high` are the 80% and 95% thresholds) |
| Fig. 4a/4b bars; ICH detection by volume within subtype with trend tests (Fig. 6b); external datasets, synthetic vs real rates (Fig. 7a, 7b) | `patient_bootstrap/figure_numbers.py` | `patient_bootstrap/ci_comparison.csv` (paper CI: `boot_lo`, `boot_hi`), `ich_volume_subtype_catrend.csv` (Fig. 6b trend test), `ich_volume_subtype_extraVintra.csv`, `external_counts.csv`, `pvalue_comparison.csv` (paper p value: `p_clustered`) |
| Specificity and predicted volume on disease-free scans (Extended Data Table 2) | `analysis/disease_free_fp_size.py`, `analysis/reader_confirmed_and_disease_free.py` | `pe_disease_free_fp_size.csv`, `pe_disease_free_scans.csv` |
| Detection on radiologist-confirmed counterfactuals (Extended Data Table 3); top realism score on the real CTPA scans; LLM concordance grades | `analysis/reader_confirmed_and_disease_free.py` | `reader_confirmed_detection.csv`, `concordance_grades.csv`, printed |
| PE driver odds ratios (Fig. 5a) | `analysis/refit_pe_drivers.py` | `detector_outputs/{nnunet,apes}/glmm_results.csv` |
| ICH driver odds ratios (Fig. 5b) | `analysis/run_ich_glmm.py` | `ich/ich_glmm_results_viola.csv`, `ich/ich_glmm_results_emm.csv` |
| Co-occurring findings: prevalence and agreement of the two extractions (Supplementary Table 2) | `analysis/comorbidity_extraction_kappa.py` | printed |
| Co-occurring findings and PE detection, odds ratios (Fig. 6a) | `analysis/comorbidity_gee.py` | `comorbidity/comorbidity_glmm_{apes,nnunet}.csv` |
| Detector x pleural effusion interaction (Results) | `analysis/effusion_interaction.py` | `effusion_interaction.csv` |

Some outputs also hold columns that the paper does not use: Clopper-Pearson CIs on all
observations (`cp_lo`, `cp_hi` and the rate rows of `radiologist_ci.csv` and
`detection_operating_limits_ci.csv`), unclustered tests (`p_original`), variational-Bayes
columns (`*_vb`), `ich_volume_subtype_trend.csv` and `ich/ich_scope_or_viola.csv`.

Extended Data Table 1 (PE FID and CLIPScore) is computed in `../01_pe_diffusion/fidelity/`.

## Detector scoring (code only)

These steps need data that is not in this repository. Their outputs are in `../resources/`.

| Output table | Scripts | Needs |
|---|---|---|
| PE detection tables `detector_outputs/{nnunet,apes}/analysis_dataframe.csv` | `compute_slice_overlap.py`, `python -m analysis.run_slicematch_evaluation`, `python -m analysis.run_apes_slicematch_evaluation` | HuggingFace masks, detector predictions |
| `detector_outputs/disease_free_{predictions,regions}.csv` | `analysis/disease_free_predictions.py` | detector predictions |
| `ich/cf_2640_samples_recovered.csv` | `analysis/recover_cf_decoded_mapping.py`, `analysis/viola_detect.py`, `analysis/viola_detect_array.py`, `ich_viola_runs/*.sh` | HuggingFace volumes, detector runs |
| `external_validation/<dataset>/<dataset>_lesions.csv` | `external_validation_{fumpe,cadpe,read}/`: `convert_*.py`, `run_*_{nnunet,apes}.sh`, `evaluate_*_lesions.py` (FUMPE also `fumpe_clinical_metadata.py`) | raw datasets |
| `external_validation/physionet/physionet_detection.csv` | `external_validation_physionet/`: `build_physionet.py`, `run_physionet_viola.sh`, `run_physionet_emm.sh`, `evaluate_physionet.py` | raw dataset |
| `radiologist/concordance_judged.csv` | `analysis_data/concordance/prep_pairs.py`, `aggregate.py` | free text |
| `comorbidity/ich_comorbidity_llm.csv` | `analysis_data/ich_comorbidity_prep.py`, `ich_comorbidity_assemble.py` | free text |
| INSPECT split sizes (Methods; printed) | `analysis/inspect_split_sizes.py` | INSPECT datalists (not shipped) |

<details><summary>Details</summary>

**Environment variables.**

| Variable | Default | Use |
|---|---|---|
| `CF_DATA` | `../resources` | Tabular inputs (this repository). |
| `ICH_SEG` | `viola` | `viola` = ICH-Detector-1 (Viola-UNet). `totalseg` selects TotalSegmentator, which is not used in the paper. `text_numbers.py`, `figure_numbers.py`, `reader_confirmed_and_disease_free.py` and `disease_free_fp_size.py` always use `viola`. |
| `CF_RAW` | `/path/to/private_data` | Data that is not shipped (detector runs, raw INSPECT and external datasets, free text). |
| `VIOLA_DIR`, `EMM_DIR`, `PYTHON` | placeholders | Detector installs and interpreter for the SLURM launchers. |

**Detection criteria.**

- **PE (both detectors).** An intended embolus mask is detected when the predicted mask has at
  least one voxel within ±5 axial slices of it (`compute_slice_overlap.py`, margin 5). This is
  ±5 slices, not ±5 mm. The slices are counted on the grid of the generated CT. The generator
  resized each label map from nz to D slices (nearest neighbour; x and y unchanged), so generated
  slice k shows label slice floor(k x nz / D). The scoring puts the intended mask on the
  generated grid with this slice map (`generator_slice_map`) before it compares slices. Each
  counterfactual is scored once: its intended mask holds all emboli of the configuration. On
  FUMPE, CAD-PE and READ, each annotated embolus is scored separately (column `det_m5` of the
  lesion tables).
- **ICH-Detector-1 (Viola-UNet).** Detected when at least one predicted hemorrhage voxel
  overlaps the intended hemorrhage mask: `viola_iou > 0` (PhysioNet: `detected_viola_coloc`).
- **ICH-Detector-2 (3D ResNet-101).** Detected when the hemorrhage probability is at least
  0.5: `emm_prob >= 0.5` (PhysioNet: `detected_emm`).
- **Cohorts.** PE: the frozen list `cohort/final_cohort.csv` (2,347 counterfactuals, 150
  patients). ICH: the rows of `ich/cf_2640_samples_recovered.csv` with intended volume > 0
  (2,601 counterfactuals, 150 patients).

**Statistics.**

- 95% CIs of detection rates and radiologist proportions: percentile bootstrap that resamples
  whole clusters, B = 2,000, seed 0 (`patient_bootstrap/boot.py`). Clusters are source patients
  (counterfactuals, radiologist review) or scans (FUMPE, CAD-PE, READ; PhysioNet has one scan
  per patient). When every observation in a group is detected, or none is, the CI is
  Clopper-Pearson on the number of clusters.
- Proportions with one observation per patient (within-patient resemblance): exact
  Clopper-Pearson (`analysis/radiologist_ci.py`). Some other outputs also hold Clopper-Pearson
  columns on all observations. These are for reference only; the paper does not use them (see
  the table above).
- Trend across ordered volume bins (Fig. 6b): logistic GEE of detection on the bin score,
  independence working correlation, sandwich SE clustered on patient, Wald test, then
  Benjamini-Hochberg across subtypes.
- Driver and co-occurring-finding models (Figs. 5 and 6a): population-average logistic GEE,
  independence working correlation, sandwich SE clustered on patient, Benjamini-Hochberg FDR.
- Run-to-run differences. The bootstrap outputs are deterministic. The model outputs can differ
  between machines in the last digits (relative difference below 1e-5). This applies to the
  reported columns, to the variational-Bayes reference columns (`*_vb`, not used in the
  paper), and more strongly to the `se` of a term that is marked `separated`. For a separated
  term no odds ratio is reported.

**Resources layout (`../resources/`, `CF_DATA`).**

All tables keep the row order of the internal tables, so the bootstrap results do not change.
They contain no report text, no radiologist free text, no LLM rationales, no demographics, no
file paths and no timestamps.

IDs: PE patients keep their INSPECT ids (`PE...`); a PE counterfactual is `config_id`
(`<patient>_cf<k>`). ICH patients are `ICH001` to `ICH150` (order preserving: sorted internal
ids map to sorted release ids). An ICH counterfactual is `cf_id` = `<patient>_<ich_idx>`, where
`ich_idx` (1 to 165) is the donor hemorrhage mask. Its `image` is `<cf_id>__<suffix>`: the
suffix holds the donor dataset, subtype codes and volume (for example
`ICH001_162__SU300_IPH_1.26ml.nii.gz`; `SU300` marks the 91 donor masks derived from expert
annotations, see `../03_counterfactual_masks/ich/README.md`). Subtypes are parsed from it with
the regular expression `(SDH|EDH|SAH|IPH|IVH)`. Radiologist ICH scans are
`<patient>_read_study_<k>`; `k = 0` is the synthetic scan without hemorrhage.

```
cohort/
  final_cohort.csv                  frozen PE cohort (2,347 rows): config_id, patient_id,
                                    location, burden, severity_category, qanadli_score, occlusion
  master_metadata.csv               every generated PE configuration (2,388 rows): lesion
                                    properties, pe_volume_ml, the four validation checks
                                    (c1..c4, all_checks_pass)
  evaluation_all_configs{,_apes}.csv   per-configuration volumes for PE-Detector-1 (no suffix)
                                    and PE-Detector-2 (_apes): intended_volume_ml (intended mask
                                    on the grid of the conditioning mask; equal to pe_volume_ml
                                    of master_metadata.csv), detected_volume_ml (all predicted
                                    voxels x voxel volume of the generated CT), intended lobes,
                                    error flag
  vessel_overlay_all_configs{,_apes}.csv  vessel-overlay morphology of the intended mask per
                                    configuration, and the error flag used by
                                    make_final_cohort.py
detector_outputs/
  nnunet/analysis_dataframe.csv     PE-Detector-1 detection table (one row per counterfactual of
  apes/analysis_dataframe.csv       the cohort): pe_detected (±5 slices on the generated grid),
                                    lesion properties, morphology of the intended mask (shape
                                    category, occlusion, vessel generation)
  disease_free_predictions.csv      one row per disease-free scan of the 150 patients and
                                    detector (600 rows): detector, scans (synthetic or real),
                                    patient_id, n_voxels, voxel_mm3
  disease_free_regions.csv          one row per predicted 3D region on those scans: detector,
                                    scans, patient_id, region_voxels
radiologist/
  pe/ratings_per_scan.csv           patient_id, scan_id, q1 anatomical quality, q3, q4 PE present,
                                    q5_described (free-text diagnosis given), q6a, q6b, q8 realism
  pe/ratings_per_patient.csv        q2 within-patient resemblance, q3, q9 (scan chosen as real)
  pe/blinding_map.csv               display ids of the 60 baseline scans and their scan_type
  pe/manifest.csv                   the 190 reviewed counterfactuals and 60 baselines: scan_type,
                                    config_id, lesion properties, selection_reason
  ich/ratings_per_scan.csv          as for PE (q4 ICH present, q5_described)
  ich/ratings_per_patient.csv       q2, q3
  ich/radiologist_data.csv          patient_id, scan_id, ich_volume_ml, is_baseline
  concordance_judged.csv            PE and ICH confirmed and described counterfactuals:
                                    intended and reader component sets, overlap (tier set),
                                    tierset_grade, concordance (LLM grade 0/1/2), type_match
comorbidity/
  comorbidity_llm.csv               PE co-occurring findings per source patient (150 rows): 14
                                    flags, effusion_size, n_comorbid, n_cf (counterfactuals of
                                    the patient in the cohort)
  comorbidity_pass2.csv             second, independent extraction (same 14 flags) for 147 of
                                    the 150 patients; used only for Supplementary Table 2
  ich_comorbidity_llm.csv           ICH co-occurring findings per counterfactual (14 flags), with
                                    cf_id, image, patient_id, ich_idx, volume_ml, subtype
ich/
  cf_2640_samples_recovered.csv     ICH counterfactual table: cf_id, patient_id, ich_idx, image,
                                    ich_volume(ml), emm_prob (ICH-Detector-2), viola_iou
                                    (ICH-Detector-1), total_segmentator_iou_recovered (not in the paper)
external_validation/
  fumpe/fumpe_lesions.csv           one row per annotated embolus and detector: case, model,
  cadpe/cadpe_lesions.csv           embolus, vol_ml, det_m5 (±5 slices), det_voxel; FUMPE also
  read/read_lesions.csv             the published Qanadli score of the case
  physionet/physionet_detection.csv one row per PhysioNet CT-ICH scan with a hemorrhage mask:
                                    gt_volume_ml, subtypes, Viola-UNet overlap and detection,
                                    emm_prob, detected_emm (TotalSegmentator columns not used)
```

Notes on the PE cohort tables:

- They are the Phase 6 tables of `../03_counterfactual_masks/pe/phase6_evaluation/` without the
  mask-path columns and without the columns that describe the predicted mask in detail (Dice,
  detected lobes, detected-vessel statistics). No statistic needs these columns.
- The detection decision of the paper is `pe_detected` of the two `analysis_dataframe.csv`
  tables. `detected_voxels` and `pe_detected` of `vessel_overlay_all_configs{,_apes}.csv` mean
  "any predicted voxel in the scan". The detection-table pipeline reads them, but no paper
  number uses them.
- The tables have 2,363 rows for 2,359 configurations, because 4 configurations were generated
  twice. Every script keeps the first row.

**Code-only steps.**

- **PE detection tables.** `compute_slice_overlap.py` needs a vessel-overlay CSV that still has
  the mask paths (`label_path` = conditioning mask with PE label 121, `pe_seg_path` = detector
  prediction on the grid of the generated CT). The x and y sizes of the two masks must match;
  the script stops with an error if they do not. The conditioning masks are in the HuggingFace
  dataset; the predictions come from running the detectors. The runners
  (`python -m analysis.run_slicematch_evaluation`, `python -m analysis.run_apes_slicematch_evaluation`)
  read this CSV and the disease-free detection results from `CF_RAW/pe_detector_runs/{nnunet,apes}/`.
  They also write their own diagnostic plots (`analysis/plotting.py`).
- **Disease-free scans.** `analysis/disease_free_predictions.py` reads the predicted masks when
  they are under `CF_RAW`; otherwise it reads the two per-scan tables in `detector_outputs/`.
  It takes `voxel_mm3` from the mask header, so the detectors must run on CTs with a correct
  header (see `../01_pe_diffusion/README.md`).
- **ICH.** Viola-UNet runs on the decoded counterfactual CTs (`ich_viola_runs/`), then
  `viola_detect_array.py` merges `viola_iou` into the internal table built by
  `recover_cf_decoded_mapping.py`. The public table is that table with private columns removed
  and ids re-keyed.
- **Free text.** The concordance pairs, the LLM grading and the ICH co-occurring-finding labels
  use report impressions and radiologist diagnoses, which are not released. The grading step
  itself has no script: a large language model (see Methods) read each batch of pairs and wrote
  one grade per pair.
- **SLURM launchers.** The `*.sh` launchers write their logs to `logs/` under the folder you
  submit from. SLURM opens these files before the script starts, so run `mkdir -p logs` before
  `sbatch`.

</details>
