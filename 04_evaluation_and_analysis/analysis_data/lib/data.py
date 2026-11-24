"""Unified data layer for the counterfactual-evaluation analyses.

Evaluation modalities across two clinical domains:

  PE  segmentation : APES + nnUNet3D-ResEncXL, strict axial-slice-overlap (m5, ±5 slices) detection, with
                     full subgroup metadata + precomputed stratified tables. (PE is
                     evaluated with these two segmentation detectors only.)
  ICH seg + clf    : the ICH counterfactual table: Viola-UNet (ICH-Detector-1, default)
                     or TotalSegmentator cerebral_bleed (task of Rusche et al. 2023) detection
                     + a 3D ResNet-101 classifier (ICH-Detector-2; I3D-inflated torchvision
                     resnet101, RSNA-trained; the ResNet-101 member of Fang et al. 2025's
                     EMM framework run on its own, so `emm_prob` is a single
                     network, not the 5-model ensemble), volume + subtype.

  Radiologist      : completed ratings by two radiologists in their fifth year of residency
                     (PE 250 scans, ICH 270), scan_type recovered from blinding map / id suffix.

Positives-only convention: every counterfactual carries the intended pathology,
so the metric is sensitivity / detection rate (+ Clopper-Pearson CI).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[3] / "resources"))

# ----------------------------------------------------------------- PE seg ---
# Detection = strict axial-SLICE overlap. m0 = exact same slice; m5 (the primary
# criterion used by every figure) dilates each intended-PE slice by +/-5 slices
# (an 11-slice window), so a predicted PE within 5 slices of the intended lesion
# counts. This is +/-5 SLICES, NOT +/-5 mm (at ~3 mm z-spacing it is ~+/-15 mm).
PE_SEG_DIR = {
    "apes": RES / "detector_outputs/apes",
    "nnunet": RES / "detector_outputs/nnunet",
}
PE_SEG_LABEL = {"apes": "PE-Detector-2", "nnunet": "PE-Detector-1"}

PE_LOCATION_ORDER = ["subsegmental", "segmental", "lobar", "central"]
PE_SEVERITY_ORDER = ["mild", "moderate", "severe", "massive"]
PE_BURDEN_ORDER = ["isolated", "multi-unilateral", "multi-bilateral", "extensive"]
PE_OCCLUSION_ORDER = ["partial", "near-complete", "complete"]


PE_EVAL_CONFIGS = RES / "cohort/evaluation_all_configs.csv"


def _det_col(df: pd.DataFrame) -> str:
    return "detected" if "detected" in df.columns else "pe_detected"


# Frozen final cohort (cohort/final_cohort.csv). Built and documented by
# analysis_data/make_final_cohort.py: configs valid for both detectors, with a
# segmentation result from both, and keeping every radiologist-rated config. Every PE analysis is restricted to this frozen set, so the
# two detectors are scored on identical configs.
FROZEN_COHORT_CSV = RES / "cohort/final_cohort.csv"


def frozen_configs() -> set:
    return set(pd.read_csv(FROZEN_COHORT_CSV)["config_id"].astype(str))


def _pe_volume_map() -> dict:
    """config_id -> intended PE volume (mL), from cohort/evaluation_all_configs.csv (the
    Phase 6 evaluation table); the detection table has voxels and Qanadli score but not mL."""
    e = pd.read_csv(PE_EVAL_CONFIGS, usecols=["config_id", "intended_volume_ml"])
    return dict(zip(e["config_id"], e["intended_volume_ml"]))


def load_pe_seg(model: str = "apes") -> pd.DataFrame:
    """Per-counterfactual PE segmentation detection table (strict slice-overlap)."""
    df = pd.read_csv(PE_SEG_DIR[model] / "analysis_dataframe.csv").copy()
    key = "config_id" if "config_id" in df.columns else "image_id"
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    df = df[df[key].astype(str).isin(frozen_configs())].reset_index(drop=True)   # FROZEN cohort
    df["detected"] = df[_det_col(df)].astype(bool)
    df["model"] = model
    df["volume_ml"] = df[key].map(_pe_volume_map())
    return df


# ------------------------------------------------------------------- ICH ----
# Collision-recovered ICH counterfactual table (analysis/recover_cf_decoded_mapping.py):
# corrected emm_prob, total_segmentator_iou_recovered, and viola_iou (merged by
# analysis/viola_detect_array.py after the Viola-UNet run).
ICH_CSV = RES / "ich/cf_2640_samples_recovered.csv"
# The ICH segmentation detector is selected by the ICH_SEG env var:
#   viola (default)    -> Viola-UNet v3.1 (all-subtype ICH segmenter), col viola_iou
#                         (ICH-Detector-1 in the paper)
#   totalseg           -> TotalSegmentator cerebral_bleed (nnU-Net; Rusche et al., J Clin Med
#                         2023;12(7):2631), col total_segmentator_iou_recovered (not in the paper)
# Output tables get the ICH_SEG_SUFFIX so the two detector variants never overwrite.
# Classifier is always the 3D ResNet-101 (emm_prob; the single ResNet-101 member of Fang et al.'s
# EMM framework, npj Digit Med 2025;8:608), unchanged.
ICH_SEG = os.environ.get("ICH_SEG", "viola").lower()
_ICH_SEG_IOU_COL = {"totalseg": "total_segmentator_iou_recovered", "viola": "viola_iou"}[ICH_SEG]
ICH_SEG_SUFFIX = "_viola" if ICH_SEG == "viola" else ""
ICH_SEG_LABEL = {"totalseg": "TotalSegmentator", "viola": "ICH-Detector-1"}[ICH_SEG]
# The ICH classifier is an I3D-inflated 3D ResNet-101 (column prefix "emm"; emm_prob is a single
# ResNet-101, not the 5-model ensemble the name implies).
ICH_CLF_LABEL = "ICH-Detector-2"
# Only TotalSegmentator's cerebral_bleed task is intracerebral-scoped (extra-axial SDH/EDH/SAH are
# outside its training scope). Viola-UNet is an all-subtype segmenter.
ICH_SEG_SCOPED = ICH_SEG == "totalseg"

# The five ICH volume bins of the paper.
ICH_VOLUME_BINS = [-0.001, 5, 15, 30, 50, 1e6]
ICH_VOLUME_LABELS = ["<5 mL", "5-15 mL", "15-30 mL", "30-50 mL", ">50 mL"]
ICH_SUBTYPE_ORDER = ["EDH", "SDH", "SAH", "IPH", "IVH", "Mixed"]
ICH_COMBO_ORDER = ["1 subtype", "2 subtypes", "3 subtypes", "4 subtypes"]
ICH_CLF_THRESH = 0.5   # emm_prob >= this counts as a classifier 'detection'

_ICH_SUBTYPE_FRAGMENT = re.compile(r"(SDH|EDH|SAH|IPH|IVH)")


def _ich_codes(image_path: str) -> list[str]:
    stem = str(image_path).split("/")[-1]
    return list(dict.fromkeys(_ICH_SUBTYPE_FRAGMENT.findall(stem)))


def _ich_subtype(image_path: str) -> str:
    matches = _ich_codes(image_path)
    if not matches:
        return "Other"
    return matches[0] if len(matches) == 1 else "Mixed"


def load_ich() -> pd.DataFrame:
    df = pd.read_csv(ICH_CSV)
    df["detected"] = (df[_ICH_SEG_IOU_COL] > 0)                          # segmentation co-localization; ICH_SEG picks Viola (default) vs TotalSeg (recovered)
    df["score"] = df["emm_prob"]                                          # classifier
    df["clf_detected"] = df["emm_prob"] >= ICH_CLF_THRESH                 # classifier detection
    df["volume_ml"] = df["ich_volume(ml)"]
    codes = df["image"].apply(_ich_codes)
    df["subtype"] = codes.apply(lambda m: "Other" if not m else
                                (m[0] if len(m) == 1 else "Mixed"))
    df["n_subtypes"] = codes.apply(len).clip(1, 4)
    df["combo"] = df["n_subtypes"].map(
        {1: "1 subtype", 2: "2 subtypes", 3: "3 subtypes", 4: "4 subtypes"})
    # Left-closed bins (right=False), so the labels are exact: "<5 mL" is strict (< 5) and a
    # 5.0 mL hemorrhage falls in "5-15 mL".
    df["volume_bin"] = pd.cut(df["volume_ml"], bins=ICH_VOLUME_BINS,
                              labels=ICH_VOLUME_LABELS, right=False)
    if "gender" in df.columns:          # demographics are not part of the public release
        df["sex"] = df["gender"]
    return df


# ----------------------------------------------------------- Radiologist ----
PE_RATINGS_SCAN = RES / "radiologist/pe/ratings_per_scan.csv"
PE_RATINGS_PAT = RES / "radiologist/pe/ratings_per_patient.csv"
PE_BLINDING = RES / "radiologist/pe/blinding_map.csv"
ICH_RATINGS_SCAN = RES / "radiologist/ich/ratings_per_scan.csv"
ICH_RATINGS_PAT = RES / "radiologist/ich/ratings_per_patient.csv"

SCAN_TYPE_ORDER = ["real", "no_pathology", "cf"]


def _pe_scan_type_map() -> dict[str, str]:
    blind = pd.read_csv(PE_BLINDING)
    return dict(zip(blind["display_scan_id"], blind["scan_type"]))


def load_radiologist_scan() -> pd.DataFrame:
    """Per-scan radiologist ratings for PE and ICH with scan_type, harmonized
    column names (q4_pathology_present, q6a_pathology_concordance, ...)."""
    pe = pd.read_csv(PE_RATINGS_SCAN)
    tmap = _pe_scan_type_map()
    pe["scan_type"] = pe["scan_id"].map(tmap).fillna("cf")
    pe.loc[pe["scan_type"] == "nope_synthetic", "scan_type"] = "no_pathology"
    pe["condition"] = "PE"
    pe = pe.rename(columns={
        "q4_pe_present": "q4_pathology_present",
        "q5_pe_description": "q5_pathology_description",
        "q6a_pe_concordance": "q6a_pathology_concordance",
    })

    ich = pd.read_csv(ICH_RATINGS_SCAN)
    ich["scan_type"] = np.where(
        ich["scan_id"].str.endswith("_read_study_0"), "no_pathology", "cf")
    ich["condition"] = "ICH"
    ich = ich.rename(columns={
        "q4_ich_present": "q4_pathology_present",
        "q5_ich_description": "q5_pathology_description",
        "q6a_ich_concordance": "q6a_pathology_concordance",
    })

    keep = [
        "condition", "patient_id", "scan_id", "scan_type",
        "q1_anatomical_quality", "q3_interslice_consistency",
        "q4_pathology_present", "q6a_pathology_concordance",
        "q6b_other_findings_concordance", "q8_overall_realism",
        "q7_artifact_description",
    ]
    keep_pe = [c for c in keep if c in pe.columns]
    keep_ich = [c for c in keep if c in ich.columns]
    return pd.concat([pe[keep_pe], ich[keep_ich]], ignore_index=True)


def load_radiologist_patient() -> pd.DataFrame:
    pe = pd.read_csv(PE_RATINGS_PAT)
    pe["condition"] = "PE"
    ich = pd.read_csv(ICH_RATINGS_PAT)
    ich["condition"] = "ICH"
    keep = ["condition", "patient_id", "q2_within_scan_consistency",
            "q3_plausible_variations"]
    return pd.concat([pe[keep], ich[keep]], ignore_index=True)


if __name__ == "__main__":
    print("== PE seg ==")
    for m in ("apes", "nnunet"):
        d = load_pe_seg(m)
        print(f"  {PE_SEG_LABEL[m]:8s}: {len(d):5d} CFs, {d['detected'].mean()*100:.1f}% detected")
    print("== ICH ==")
    i = load_ich()
    print(f"  {len(i)} CFs, seg det {i['detected'].mean()*100:.1f}%, "
          f"emm_prob {i['score'].min():.3f}-{i['score'].max():.3f}")
    print(i.groupby("subtype", observed=False)["detected"].agg(["mean", "count"]))
    print("== Radiologist ==")
    s = load_radiologist_scan()
    print(s.groupby(["condition", "scan_type"]).size())
    p = load_radiologist_patient()
    print(p.groupby("condition").size())
