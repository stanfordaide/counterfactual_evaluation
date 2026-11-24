"""Build and FREEZE the final PE counterfactual analysis cohort.

The frozen cohort is the set of configs that:
  1. passed both models' validity checks (vessel-overlay error == valid: an
     intended PE was present and both the label and predicted-seg files exist),
  2. have a real segmentation RESULT in BOTH APES and nnU-Net (present in both
     analysis tables).
PLUS any config a radiologist already evaluated is KEPT even if it would fail the
above (by design - never drop a radiologist-evaluated result). In practice all 190
radiologist CFs already pass, so the union is a no-op, but the rule is enforced.

Validity note (stress-test integrity): a `missing_pe_seg`/`missing_label` flag
means the FILE is absent, NOT that the model predicted "no PE". A genuine model
miss writes an empty mask -> pe_detected=False and is KEPT. So this cohort does
not drop any of the models' true misses.

Output: output/final_cohort.csv. It is identical to the frozen list shipped in
CF_DATA/cohort/final_cohort.csv, which every analysis reads (analysis_data/lib/data.py).
To use a regenerated list, copy it to CF_DATA/cohort/final_cohort.csv.

    python analysis_data/make_final_cohort.py
"""
from __future__ import annotations
import os

from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
GEN = REPO / "cohort"
APES_VESSEL = GEN / "vessel_overlay_all_configs_apes.csv"
NN_VESSEL = GEN / "vessel_overlay_all_configs.csv"
APES_ADF = REPO / "detector_outputs/apes/analysis_dataframe.csv"
NN_ADF = REPO / "detector_outputs/nnunet/analysis_dataframe.csv"
RAD = REPO / "radiologist/pe/ratings_per_scan.csv"
OUT = Path(__file__).resolve().parents[1] / "output" / "final_cohort.csv"

def _valid(vessel_csv: str) -> set:
    v = pd.read_csv(vessel_csv)
    v["error"] = v["error"].fillna("valid")
    return set(v.loc[v["error"] == "valid", "config_id"].astype(str))


def _adf_configs(adf: Path) -> set:
    return set(pd.read_csv(adf)["config_id"].astype(str))


def main():
    apes_valid, nn_valid = _valid(APES_VESSEL), _valid(NN_VESSEL)
    apes_cfg, nn_cfg = _adf_configs(APES_ADF), _adf_configs(NN_ADF)
    rad = set(pd.read_csv(RAD)["scan_id"].astype(str)) if RAD.exists() else set()

    common = apes_valid & nn_valid & apes_cfg & nn_cfg
    # radiologist-keep exception (union); all radiologist CFs that have a result
    rad_keep = (rad & apes_cfg & nn_cfg) - common
    cohort = common | rad_keep

    # attach metadata from the APES analysis table
    a = pd.read_csv(APES_ADF); a["config_id"] = a["config_id"].astype(str)
    cols = [c for c in ["config_id", "patient_id", "location", "burden",
                        "severity_category", "qanadli_score", "occlusion"] if c in a.columns]
    out = (a[a["config_id"].isin(cohort)][cols]
           .drop_duplicates("config_id").sort_values("config_id").reset_index(drop=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"FROZEN cohort -> {OUT}")
    print(f"  {len(out)} configs / {out['patient_id'].nunique()} patients")
    print(f"  radiologist-keep exceptions added: {len(rad_keep)} (expected 0)")
    print(f"  location: {out['location'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
