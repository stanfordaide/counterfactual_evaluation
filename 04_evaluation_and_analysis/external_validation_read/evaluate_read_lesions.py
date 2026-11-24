"""Per-embolus READ evaluation - the unit that matches the synthetic per-config data.

READ GT masks are binary per-scan (all emboli merged), so - like FUMPE - we split each
mask into 3D connected components (individual emboli) and score each with the SAME strict
±5-slice overlap criterion as the synthetic + FUMPE + CAD-PE evaluations (does the model's
PE mask share an axial slice, within ±5, with this embolus), plus a voxel-intersection
variant. READ has no severity annotation, so volume is the detectability axis.

Outputs output/external_validation/read/read_lesions.csv (case × model × embolus: vol_ml + det_m5 + det_voxel).
Run with maisi env (scipy.ndimage + nibabel). Excludes empty-GT scans.
"""
from __future__ import annotations
import os

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

DATA = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "READ"
GT = DATA / "gt_nii"
PRED = {"nnunet": DATA / "pred_nnunet", "apes": DATA / "pred_apes"}
RESULTS = Path(__file__).resolve().parents[1] / "output" / "external_validation" / "read"
MANIFEST = RESULTS / "read_manifest.csv"
MARGIN = 5
MIN_LESION_ML = 0.01


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
                             "vol_ml": vol, "det_m5": len(dil & det_sl) > 0,
                             "det_voxel": voxel})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "read_lesions.csv", index=False)
    print(f"wrote {len(df)} lesion rows ({df.case.nunique()} scans) -> read_lesions.csv")
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
