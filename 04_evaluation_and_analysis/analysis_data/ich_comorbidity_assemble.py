"""Assemble ICH co-morbidity labels -> per-config table + first-look detection stats (Methods).

Merges the LLM label batches (CF_RAW/ich_comorbidity/ich_labels_b*.json, one JSON list of
{imp_id, <flag>: 0/1} per batch), maps unique impressions back to all 2,640 CFs (via
ich_config_impression.csv from ich_comorbidity_prep.py), joins the detection outcomes
(`detected` = ICH_SEG detector, `clf_detected` = 3D ResNet-101), volume, subtype, source
patient_id and the donor hemorrhage-mask index (ich_idx). Prints flag prevalence + crude
detection at config / patient / donor level. The per-patient flag counts are the ICH
co-occurring findings listed in the Methods.

Code only: the label batches and the internal table hold impression text. The public copy of
the output is CF_DATA/comorbidity/ich_comorbidity_llm.csv: same rows and flags, ids re-keyed,
file names sanitized, and without the detection columns (those were computed before the
collision recovery; use the ICH counterfactual table for detection).

Writes output/ich_comorbidity_llm.csv (one row per CF).
"""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import data

OUT = Path(__file__).resolve().parents[1] / "output"
LABELS = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "ich_comorbidity"
ICH_INTERNAL = OUT / "ich" / "cf_2640_samples_recovered.csv"   # internal table (recover_cf_decoded_mapping.py)

FLAGS = ["atrophy", "encephalomalacia_gliosis", "old_infarct", "small_vessel_disease",
         "calcification", "hydrocephalus_ventenlarge", "mass_lesion", "mass_effect",
         "midline_shift", "edema", "calvarial_fracture", "postsurgical",
         "motion_artifact", "sinus_mastoid_disease"]


def load_labels() -> pd.DataFrame:
    recs = []
    for f in sorted(glob.glob(str(LABELS / "ich_labels_b*.json"))):
        recs.extend(json.load(open(f)))
    lab = pd.DataFrame(recs).drop_duplicates("imp_id").sort_values("imp_id")
    for c in FLAGS:
        lab[c] = pd.to_numeric(lab.get(c), errors="coerce").fillna(0).astype(int)
    return lab[["imp_id"] + FLAGS]


def main():
    lab = load_labels()
    print(f"labels: {len(lab)} unique impressions (expected 58)")

    cmap = pd.read_csv(LABELS / "ich_config_impression.csv")         # image, imp_id
    cfg = cmap.merge(lab, on="imp_id", how="left")
    cfg[FLAGS] = cfg[FLAGS].fillna(0).astype(int)                    # imp_id=-1 -> all 0

    ich = data.load_ich().reset_index(drop=True)          # public table (same row order)
    raw = pd.read_csv(ICH_INTERNAL)
    assert len(raw) == len(ich)
    ich["image"] = raw["image"].astype(str).values        # internal file name = join key of cmap
    ich["donor"] = ich["ich_idx"].values                  # donor hemorrhage-mask index

    m = ich.merge(cfg, on="image", how="left")
    m[FLAGS] = m[FLAGS].fillna(0).astype(int)
    m["n_comorbid"] = m[FLAGS].sum(axis=1)
    keep = ["image", "patient_id", "donor", "detected", "clf_detected", "score",
            "volume_ml", "subtype", "n_subtypes"] + FLAGS + ["n_comorbid"]
    m[keep].to_csv(OUT / "ich_comorbidity_llm.csv", index=False)
    print(f"wrote {OUT/'ich_comorbidity_llm.csv'} ({len(m)} CFs)")

    print(f"\nbaseline detection: TotalSeg {m.detected.mean()*100:.1f}%  "
          f"EMM {m.clf_detected.mean()*100:.1f}%")
    print("\n=== flag prevalence (configs / patients / donors) + crude detection ===")
    print(f"{'flag':26s} {'cfg':>4s} {'pat':>4s} {'don':>4s}  {'TS+':>6s} {'TS-':>6s}"
          f"  {'EMM+':>6s} {'EMM-':>6s}")
    for f in FLAGS:
        pos = m[m[f] == 1]
        neg = m[m[f] == 0]
        npat = m.loc[m[f] == 1, "patient_id"].nunique()
        ndon = m.loc[m[f] == 1, "donor"].nunique()
        ts_p = pos.detected.mean() * 100 if len(pos) else np.nan
        ts_n = neg.detected.mean() * 100
        em_p = pos.clf_detected.mean() * 100 if len(pos) else np.nan
        em_n = neg.clf_detected.mean() * 100
        print(f"{f:26s} {len(pos):4d} {npat:4d} {ndon:4d}  {ts_p:6.1f} {ts_n:6.1f}"
              f"  {em_p:6.1f} {em_n:6.1f}")


if __name__ == "__main__":
    main()
