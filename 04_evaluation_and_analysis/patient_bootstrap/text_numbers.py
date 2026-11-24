"""Manuscript Results numbers with patient-level bootstrap 95% CIs (same method as the figure
CIs, boot.ci_arrays: resample source patients, B = 2,000, seed 0; all/none detected ->
approximate Clopper-Pearson on the number of patients).

Covers the Results paragraphs on the radiologist review, PE detection and ICH detection.
Run:  python patient_bootstrap/text_numbers.py  -> output/patient_bootstrap/manuscript_numbers.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["ICH_SEG"] = "viola"
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("CF_DATA", HERE.parents[1] / "resources"))
sys.path[:0] = [str(HERE), str(HERE.parent / "analysis_data" / "lib")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import boot as BOOT  # noqa: E402
import data  # noqa: E402

rows = []


def add(section, label, y, cl):
    y = np.asarray(y, float)
    lo, hi = BOOT.ci_arrays(y, cl, label=label)
    rows.append(dict(section=section, label=label, k=int(y.sum()), n=len(y),
                     patients=len(pd.unique(np.asarray(cl))), pct=round(100 * y.mean(), 1),
                     lo=round(100 * lo, 1), hi=round(100 * hi, 1)))


# ------------------------------------------------------------------ radiologist review
scan = data.load_radiologist_scan()
cf = scan[scan.scan_type == "cf"]
for cond in ["PE", "ICH"]:
    d = cf[cf.condition == cond]
    for q in ["q1_anatomical_quality", "q8_overall_realism"]:
        v = pd.to_numeric(d[q], errors="coerce"); keep = v.notna() & (v >= 0)
        add("radiologist", f"{cond} {q} >= 1", (v[keep] >= 1), d.loc[keep, "patient_id"])
    top = 2 if cond == "PE" else 3
    v = pd.to_numeric(d["q8_overall_realism"], errors="coerce"); keep = v.notna() & (v >= 0)
    add("radiologist", f"{cond} realism top score ({top})", (v[keep] == top), d.loc[keep, "patient_id"])
    v = pd.to_numeric(d["q4_pathology_present"], errors="coerce"); keep = v.notna()
    add("radiologist", f"{cond} lesion confirmed", (v[keep] >= 1), d.loc[keep, "patient_id"])

conc = pd.read_csv(RES / "radiologist/concordance_judged.csv")
conc["overlap"] = conc["overlap"].astype(str).str.lower().isin(["true", "1"])
pe_c = conc[conc.modality == "PE"].copy()
pe_c["patient_id"] = pe_c.scan_id.str.split("_cf").str[0]
add("radiologist", "PE location concordance (confirmed, free text)", pe_c.overlap, pe_c.patient_id)
rd = pd.read_csv(RES / "radiologist/ich/radiologist_data.csv")[["scan_id", "patient_id"]]
ich_c = conc[conc.modality == "ICH"].merge(rd, on="scan_id", how="left")
assert ich_c.patient_id.notna().all()
add("radiologist", "ICH subtype concordance (confirmed, free text)", ich_c.overlap, ich_c.patient_id)

# ------------------------------------------------------------------ PE detection
for m, lab in [("nnunet", "PE-Detector-1"), ("apes", "PE-Detector-2")]:
    d = data.load_pe_seg(m); d = d[d.volume_ml > 0]
    add("PE", f"{lab} overall", d.detected, d.patient_id)
    vb = pd.cut(d.volume_ml, [0, 0.5, 1, 2.5, 5, 1e6])           # same bins as the Fig. 4a figure code
    for name, mask in [("<0.5 mL", vb.cat.codes == 0), (">5 mL", vb.cat.codes == 4)]:
        add("PE", f"{lab} volume {name}", d.detected[mask], d.patient_id[mask])
    for loc in ["central", "segmental", "subsegmental"]:
        mk = d.location == loc
        add("PE", f"{lab} location {loc}", d.detected[mk], d.patient_id[mk])
    for sev in ["mild", "severe"]:
        mk = d.severity_category == sev
        add("PE", f"{lab} severity {sev}", d.detected[mk], d.patient_id[mk])

# ------------------------------------------------------------------ ICH detection
d = data.load_ich(); d = d[d.volume_ml > 0].copy()
d["d1"] = d.detected.astype(bool); d["d2"] = d.emm_prob >= data.ICH_CLF_THRESH
for col, lab in [("d1", "ICH-Detector-1"), ("d2", "ICH-Detector-2")]:
    add("ICH", f"{lab} overall", d[col], d.patient_id)
    for st in ["IPH", "IVH"]:
        mk = d.subtype == st
        add("ICH", f"{lab} subtype {st}", d[col][mk], d.patient_id[mk])
    for name, mask in [("<5 mL", d.volume_ml < 5), (">50 mL", d.volume_ml >= 50)]:
        add("ICH", f"{lab} volume {name}", d[col][mask], d.patient_id[mask])
    for k in [1, 4]:
        mk = d.n_subtypes == k
        add("ICH", f"{lab} {k} subtype(s)", d[col][mk], d.patient_id[mk])

out = pd.DataFrame(rows)
BOOT.OUT_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(BOOT.OUT_DIR / "manuscript_numbers.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
