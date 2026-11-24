"""Per-embolus FUMPE evaluation - the unit that matches the synthetic per-config data.

FUMPE GT masks are per-scan (all emboli merged). To compare against the synthetic
per-lesion detectability profile, we split each GT mask into 3D connected components
(= individual emboli) and score each embolus with the SAME strict ±5-slice overlap
criterion as the synthetic data (does the model's PE mask share an axial slice, within
±5, with this embolus). In the reliable (large-volume) regime a big embolus is detected
on its own slices, so slice-borrowing from neighbouring emboli is negligible.

Also emits a voxel-intersection detection (pred mask actually overlaps the embolus) as
a stricter robustness variant.

Outputs output/external_validation/fumpe/fumpe_lesions.csv (case × model × embolus: volume + det_m5 + det_voxel).
Run with maisi env (has scipy.ndimage + nibabel). Excludes the 2 empty-GT scans.
"""
from __future__ import annotations
import os

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

DATA = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "FUMPE"
GT = DATA / "gt_nii"
PRED = {"nnunet": DATA / "pred_nnunet", "apes": DATA / "pred_apes"}
RESULTS = Path(__file__).resolve().parents[1] / "output" / "external_validation" / "fumpe"
MANIFEST = RESULTS / "fumpe_manifest.csv"
MARGIN = 5
MIN_LESION_ML = 0.01   # drop annotation specks


def _lesions(case: str, vox_ml: float):
    gt = nib.load(GT / f"{case}.nii.gz").get_fdata() > 0
    lab, n = ndimage.label(gt)
    out = []
    for k in range(1, n + 1):
        comp = lab == k
        vol = int(comp.sum()) * vox_ml
        if vol < MIN_LESION_ML:
            continue
        gsl = set(np.where(comp.any(axis=(0, 1)))[0])
        out.append((k, comp, vol, gsl))
    return out


def main():
    man = pd.read_csv(MANIFEST)
    man = man[man.pe_voxels > 0].copy()
    clin = pd.read_csv(RESULTS / "fumpe_clinical.csv")[["case", "qanadli"]]
    man = man.merge(clin, on="case", how="left")
    rows = []
    for _, m in man.iterrows():
        case = m["case"]
        vox_ml = m["spacing_x"] * m["spacing_y"] * m["spacing_z"] / 1000.0
        lesions = _lesions(case, vox_ml)
        for model, pdir in PRED.items():
            pf = pdir / f"{case}.nii.gz"
            if not pf.exists():
                continue
            pred = nib.load(pf).get_fdata() > 0
            det_sl = set(np.where(pred.any(axis=(0, 1)))[0])
            for k, comp, vol, gsl in lesions:
                zz = min(comp.shape[2], pred.shape[2])
                dil = {s2 for s in gsl for s2 in range(s - MARGIN, s + MARGIN + 1)}
                voxel = bool((comp[:, :, :zz] & pred[:, :, :zz]).any())
                rows.append({"case": case, "model": model, "embolus": k,
                             "vol_ml": vol, "qanadli": m["qanadli"],
                             "det_m5": len(dil & det_sl) > 0, "det_voxel": voxel})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "fumpe_lesions.csv", index=False)
    print(f"wrote {len(df)} lesion rows ({df.case.nunique()} scans) -> fumpe_lesions.csv")
    for model in ["nnunet", "apes"]:
        s = df[df.model == model]
        if not len(s):
            print(f"  {model}: no predictions"); continue
        s = s.copy(); s["vb"] = pd.cut(s.vol_ml, [0, 0.1, 0.5, 2, 5, 1e6],
                                       labels=["<0.1", "0.1-0.5", "0.5-2", "2-5", ">5"])
        g = s.groupby("vb", observed=True).agg(n=("det_m5", "size"), m5=("det_m5", "mean"))
        print(f"  {model}: " + "  ".join(f"{b}:{r.m5*100:.0f}%(n={int(r.n)})"
                                         for b, r in g.iterrows()))
    return df


if __name__ == "__main__":
    main()
