"""Per-embolus CAD-PE evaluation - the unit that matches the synthetic per-config data.

CAD-PE reference masks are INSTANCE-labeled (each embolus = a distinct integer id), so
unlike FUMPE (where we ran connected components) we use the radiologist's own embolus
separation directly. Each embolus is scored with the SAME strict ±5-slice overlap
criterion as the synthetic data (does the model's PE mask share an axial slice, within
±5, with this embolus), plus a stricter voxel-intersection variant.

CAD-PE has no Qanadli / severity annotation, so volume is the only detectability axis
(the dominant one - matches the synthetic model card). PE-negative scans (empty mask)
are excluded.

Outputs output/external_validation/cadpe/cadpe_lesions.csv (case × model × embolus: vol_ml + det_m5 + det_voxel).
Run with maisi env (nibabel + numpy).
"""
from __future__ import annotations
import os

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

DATA = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "CAD_PE"
GT = DATA / "gt_nii"                                  # instance-labeled reference
PRED = {"nnunet": DATA / "pred_nnunet", "apes": DATA / "pred_apes"}
RESULTS = Path(__file__).resolve().parents[1] / "output" / "external_validation" / "cadpe"
MANIFEST = RESULTS / "cadpe_manifest.csv"
MARGIN = 5
MIN_LESION_ML = 0.01   # drop annotation specks


def _lesions(case: str, vox_ml: float):
    """Each reference instance id > 0 is one embolus. nibabel axis 2 = z (slices)."""
    gt = np.rint(nib.load(GT / f"{case}.nii.gz").get_fdata()).astype(int)
    out = []
    for lid in [int(v) for v in np.unique(gt) if v > 0]:
        comp = gt == lid
        vol = int(comp.sum()) * vox_ml
        if vol < MIN_LESION_ML:
            continue
        gsl = set(np.where(comp.any(axis=(0, 1)))[0])
        out.append((lid, comp, vol, gsl))
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
            for lid, comp, vol, gsl in lesions:
                zz = min(comp.shape[2], pred.shape[2])
                dil = {s2 for s in gsl for s2 in range(s - MARGIN, s + MARGIN + 1)}
                voxel = bool((comp[:, :, :zz] & pred[:, :, :zz]).any())
                rows.append({"case": case, "model": model, "embolus": lid,
                             "vol_ml": vol, "det_m5": len(dil & det_sl) > 0,
                             "det_voxel": voxel})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "cadpe_lesions.csv", index=False)
    print(f"wrote {len(df)} lesion rows ({df.case.nunique()} scans) -> cadpe_lesions.csv")
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
