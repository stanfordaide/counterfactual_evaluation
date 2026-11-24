"""Descriptive numbers of the PE and ICH counterfactual cohorts.

Prints the cohort numbers of the paper that no other script prints. Each line names the
part of the paper that it backs:
  - Results, first paragraph: counterfactuals per patient (PE, ICH), PE severity and location
    shares, PE and ICH volume range and median.
  - Supplementary Table 1: PE configuration counts and % by location, clot burden, occlusion,
    severity and laterality; embolus volume and Qanadli score, median and IQR.
  - Methods, PE counterfactual generation: largest embolus volume per location, and the
    cohort funnel from the generated configurations to the final counterfactuals.

Inputs (CF_DATA, default ../resources): cohort/final_cohort.csv, cohort/master_metadata.csv,
cohort/vessel_overlay_all_configs.csv (PE-Detector-1 error flags) and
ich/cf_2640_samples_recovered.csv. The PE embolus volume is `pe_volume_ml` of
master_metadata.csv (the volume of the PE mask at the Phase 4 checks). The ICH volume is
`ich_volume(ml)`; the ICH cohort is the rows with a volume above 0 mL.
Per-patient means are given with the sample standard deviation (ddof = 1).

Nothing is written. Run from 04_evaluation_and_analysis/:
    python analysis/cohort_descriptives.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_data" / "lib"))
import data  # noqa: E402

RES = data.RES
PE_LOCATIONS = ["lobar", "subsegmental", "segmental", "central"]
PE_BURDENS = ["extensive", "isolated", "multi_bilateral", "multi_unilateral"]
PE_OCCLUSIONS = ["complete", "near_complete", "partial"]
PE_SEVERITIES = ["mild", "moderate", "severe"]
# "central" laterality = no lung lobe recorded ("Not assigned" in Supplementary Table 1)
PE_LATERALITIES = ["bilateral", "unilateral_right", "unilateral_left", "central"]


def counts(s: pd.Series, order: list) -> str:
    n = len(s)
    vc = s.value_counts()
    return ", ".join(f"{k} {vc.get(k, 0):,} ({100 * vc.get(k, 0) / n:.1f}%)" for k in order)


def median_iqr(x: pd.Series, fmt: str) -> str:
    q1, med, q3 = np.percentile(x.to_numpy(float), [25, 50, 75])
    return f"median {med:{fmt}} (IQR {q1:{fmt}}-{q3:{fmt}})"


def per_patient(df: pd.DataFrame) -> str:
    n = df.groupby("patient_id").size()
    return f"{n.mean():.1f} ± {n.std():.1f} counterfactuals per patient"


def main() -> None:
    cohort = pd.read_csv(RES / "cohort/final_cohort.csv")
    master = pd.read_csv(RES / "cohort/master_metadata.csv")
    meta = master.set_index("config_id").loc[cohort["config_id"]]
    vol = meta["pe_volume_ml"]
    loc = cohort["location"].to_numpy()

    ich = data.load_ich()
    ich = ich[ich["volume_ml"] > 0]

    print("== Results, first paragraph ==")
    print(f"PE: {len(cohort):,} counterfactuals across {cohort.patient_id.nunique()} patients "
          f"({per_patient(cohort)})")
    print(f"ICH: {len(ich):,} counterfactuals across {ich.patient_id.nunique()} patients "
          f"({per_patient(ich)})")
    sev = cohort["severity_category"].value_counts(normalize=True) * 100
    print("PE severity: " + ", ".join(f"{k} {sev[k]:.0f}%" for k in PE_SEVERITIES))
    lshare = cohort["location"].value_counts(normalize=True) * 100
    print("PE location: " + ", ".join(f"{k} {lshare[k]:.0f}%" for k in PE_LOCATIONS))
    print(f"PE intended clot volume: {vol.min():.3f} to {vol.max():.1f} mL (median {vol.median():.2f} mL)")
    iv = ich["volume_ml"]
    print(f"ICH volume: {iv.min():.2f} to {iv.max():.1f} mL (median {iv.median():.1f} mL)")

    print("\n== Supplementary Table 1 (n = %s) ==" % f"{len(cohort):,}")
    print("Anatomical location: " + counts(cohort["location"], PE_LOCATIONS))
    print("Clot burden: " + counts(cohort["burden"], PE_BURDENS))
    print("Vessel occlusion: " + counts(cohort["occlusion"], PE_OCCLUSIONS))
    print("Severity by Qanadli index: " + counts(cohort["severity_category"], PE_SEVERITIES))
    print("Laterality: " + counts(meta["laterality"], PE_LATERALITIES)
          + "   (central = not assigned)")
    print("Embolus volume, mL: " + median_iqr(vol, ".2f"))
    print("Qanadli score: " + median_iqr(cohort["qanadli_score"], ".0f"))

    print("\n== Methods, PE counterfactual generation ==")
    print(f"Configurations: {len(cohort):,} counterfactuals, "
          f"{per_patient(cohort).replace(' counterfactuals per patient', '')} per subject")
    maxima = {k: vol[loc == k].max() for k in ["subsegmental", "segmental", "lobar", "central"]}
    print("Largest retained configuration volume: "
          + ", ".join(f"{k} {v:.2f} mL" if v < 10 else f"{k} {v:.1f} mL" for k, v in maxima.items()))

    passed = master[master["all_checks_pass"]]
    print(f"Generated: {len(master):,} configurations across {master.patient_id.nunique()} subjects; "
          f"{len(passed):,} passed the automated validation")
    vo = pd.read_csv(RES / "cohort/vessel_overlay_all_configs.csv", low_memory=False)
    vo = vo.drop_duplicates("config_id", keep="first")
    vo = vo[vo.config_id.isin(passed.config_id)]
    err = vo["error"].fillna("valid").value_counts()
    n_err = int(err.drop("valid").sum())
    print(f"Excluded: {n_err} configurations: no embolus in the conditioning label "
          f"{err.get('no_intended_pe', 0)}, label file missing {err.get('missing_label', 0)}, "
          f"PE-Detector-1 output missing {err.get('missing_pe_seg', 0)}")
    left = vo[vo["error"].fillna("valid") == "valid"]
    print(f"Left: {len(left):,} counterfactuals across {left.patient_id.nunique()} subjects "
          f"(final_cohort.csv: {len(cohort):,} / {cohort.patient_id.nunique()}; "
          f"same ids: {set(left.config_id) == set(cohort.config_id)})")


if __name__ == "__main__":
    main()
