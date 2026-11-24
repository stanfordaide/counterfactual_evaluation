"""Score Viola-UNet + EMM detection on the 36 PhysioNet ICH-positive real head CTs vs the binary
radiologist GT, join to GT volume + subtype, and write the per-scan results table for the
transfer figure.

Detection criteria (matched to the synthetic side, exactly as INSTANCE):
  Viola-UNet (segmentation) - CO-LOCALIZATION: predicted ICH mask (any subtype label > 0) shares
    >= 1 voxel with the GT ICH mask. This matches the synthetic `viola_iou > 0` co-localization
    rule (analysis/viola_detect.py), so both sides are apples-to-apples.
  EMM (classifier) - emm_prob >= 0.5 (the nominal ICH model-card cutoff).

GT is aligned to the CT grid; Viola predicts on the input grid. GT is nearest-resampled to the
prediction grid on the rare mismatch (same guard as analysis/viola_detect.py::seg_iou).

Code only: needs the Viola-UNet masks (run_physionet_viola.sh) and the classifier probabilities
(run_physionet_emm.sh). The output table, without age and adult flag, is shipped as
CF_DATA/external_validation/physionet/physionet_detection.csv and is what
patient_bootstrap/figure_numbers.py (Fig. 7b) reads. TotalSegmentator masks are optional
(not used in the paper); their columns stay empty when the masks are absent.

Run:  python external_validation_physionet/evaluate_physionet.py
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "physionet_ich"
VIOLA_OUT = WORK / "viola_out"
TS_OUT = WORK / "totalseg_out"
RESULTS = HERE.parent / "output" / "physionet"
MANIFEST = RESULTS / "physionet_manifest.csv"
EMM_CSV = RESULTS / "physionet_emm.csv"
OUT = RESULTS / "physionet_detection.csv"
EMM_THRESH = 0.5


def _pid(name: str) -> int | None:
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def _viola_masks() -> dict[int, str]:
    """pid -> Viola prediction mask path (exclude the CSV; masks are NIfTI)."""
    out = {}
    for f in glob.glob(str(VIOLA_OUT / "**" / "*.nii*"), recursive=True):
        p = _pid(f)
        if p is not None:
            out.setdefault(p, f)
    return out


def _ts_masks() -> dict[int, str]:
    """pid -> TotalSegmentator cerebral_bleed mask path."""
    out = {}
    for f in glob.glob(str(TS_OUT / "*_segmentation_cerebral_bleed.nii.gz")):
        p = _pid(f)
        if p is not None:
            out.setdefault(p, f)
    return out


def _emm_map() -> dict[int, float]:
    if not EMM_CSV.exists():
        return {}
    e = pd.read_csv(EMM_CSV)
    fn = [c for c in e.columns if any(k in c.lower() for k in ("file", "name", "image"))][0]
    pc = [c for c in e.columns if "prob" in c.lower() or c.lower() in ("pred", "score")][0]
    return {_pid(str(r[fn])): float(r[pc]) for _, r in e.iterrows() if _pid(str(r[fn])) is not None}


def _coloc(gt_path: str, mask_path: str):
    g = nib.load(gt_path).get_fdata() > 0
    v = nib.load(mask_path).get_fdata()
    if v.shape != g.shape:
        g = zoom(g, [v.shape[i] / g.shape[i] for i in range(3)], order=0) > 0
    vb = v > 0
    inter = int(np.logical_and(vb, g).sum())
    union = int(np.logical_or(vb, g).sum())
    return int(vb.sum()), inter, (inter / union if union else 0.0)


def main() -> None:
    man = pd.read_csv(MANIFEST)
    vmasks = _viola_masks()
    tsmasks = _ts_masks()
    emm = _emm_map()
    if not vmasks:
        print(f"  (no Viola masks yet in {VIOLA_OUT})")
    if not tsmasks:
        print(f"  (no TotalSeg masks yet in {TS_OUT})")
    if not emm:
        print(f"  (no EMM csv yet at {EMM_CSV})")

    def _seg_rec(prefix: str, mk, gt_path):
        if not mk:
            return {f"{prefix}_pred_voxels": np.nan, f"{prefix}_overlap_voxels": np.nan,
                    f"{prefix}_iou": np.nan, f"detected_{prefix}_coloc": np.nan,
                    f"detected_{prefix}_any": np.nan}
        pv, inter, iou = _coloc(gt_path, mk)
        return {f"{prefix}_pred_voxels": pv, f"{prefix}_overlap_voxels": inter,
                f"{prefix}_iou": iou, f"detected_{prefix}_coloc": inter > 0,
                f"detected_{prefix}_any": pv > 0}

    rows = []
    for _, m in man.iterrows():
        pid = int(m["pid"])
        rec = {"pid": pid, "case": m["case"], "gt_volume_ml": float(m["gt_volume_ml"]),
               "age": float(m["age"]) if m["age"] == m["age"] else np.nan,
               "is_adult": bool(m["is_adult"]),
               "subtypes": m["subtypes"], "extra_axial": bool(m["extra_axial"]),
               "intra_axial": bool(m["intra_axial"]), "subarachnoid": bool(m["subarachnoid"])}
        rec.update(_seg_rec("viola", vmasks.get(pid), m["mask_path"]))
        rec.update(_seg_rec("ts", tsmasks.get(pid), m["mask_path"]))
        p = emm.get(pid, np.nan)
        rec.update({"emm_prob": p, "detected_emm": (p >= EMM_THRESH) if p == p else np.nan})
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} scans -> {OUT}")

    bins = [-0.001, 0.5, 5, 30, 1e6]
    labs = ["<0.5", "0.5-5", "5-30", ">=30"]
    df["vb"] = pd.cut(df.gt_volume_ml, bins, labels=labs)

    def _rate(g, col):
        return 100 * g[col].mean() if g[col].notna().any() else float("nan")

    def summarize(sub, label):
        print(f"\n{'=' * 66}\n{label}   (n={len(sub)})\n{'=' * 66}")
        for name, col in [("Viola", "detected_viola_coloc"), ("TotalSeg", "detected_ts_coloc"),
                          ("EMM", "detected_emm")]:
            d = sub[col].notna()
            if d.any():
                extra = (f"  |  any-prediction {_rate(sub[d], col.replace('coloc', 'any')):.1f}%"
                         if "coloc" in col else "")
                print(f"  {name:9s} (n={int(d.sum())}): {_rate(sub[d], col):.1f}%{extra}")
        print("  by GT volume bin (Viola | TotalSeg | EMM):")
        for b in labs:
            g = sub[sub.vb == b]
            if len(g):
                print(f"    {b:7s} (n={len(g):2d}):  Viola {_rate(g,'detected_viola_coloc'):5.1f}%   "
                      f"TotalSeg {_rate(g,'detected_ts_coloc'):5.1f}%   EMM {_rate(g,'detected_emm'):5.1f}%")
        print("  by location -- Viola vs TotalSeg extra-axial contrast (Viola | TotalSeg | EMM):")
        for lab, mask in [("extra-axial (EDH/SDH)", sub.extra_axial),
                          ("intra-axial (IPH/IVH)", sub.intra_axial),
                          ("subarachnoid", sub.subarachnoid)]:
            g = sub[mask]
            if len(g):
                print(f"    {lab:22s} (n={len(g):2d}):  Viola {_rate(g,'detected_viola_coloc'):5.1f}%   "
                      f"TotalSeg {_rate(g,'detected_ts_coloc'):5.1f}%   EMM {_rate(g,'detected_emm'):5.1f}%")

    summarize(df[df.is_adult], "ADULTS >=18  (PRIMARY cohort)")
    summarize(df, "ALL 36 positives  (sensitivity; includes 17 pediatric)")


if __name__ == "__main__":
    main()
