"""Compute Viola-UNet ICH co-localization detection on the synthetic counterfactuals,
replicating the criterion used for the ICH segmentation analysis (seg_IOU of the first ICH
analysis script):

    viola_iou = |(Viola pred > 0) ∩ (intended mask == 66)| / |union|,   detected = viola_iou > 0

The intended (label-66) mask is nearest-neighbour upsampled to the Viola prediction grid,
identical to the reference pipeline, so viola_iou is directly comparable. Viola output masks are
labeled (0=BG, 1-5=EDH/SDH/SAH/IPH/IVH); we threshold > 0 for the co-localization (any-subtype)
criterion.

Collision-proof join
--------------------
Each counterfactual row is matched to its Viola mask via the UNIQUE decoded-image filename -- the
`decoded_image` column produced by analysis/recover_cf_decoded_mapping.py, which resolves the 42
source-duplicate segmentation-basename collisions with the NIfTI affine fingerprint. Viola writes
one mask per decoded image under the full decoded filename (timestamp included), so this join is
exact and never falls back to the ambiguous segmentation basename. This is precisely the
first-match error that affected the original TotalSegmentator/EMM basename merges; here it cannot
happen because (a) the output names are unique and (b) we join on the recovered decoded filename.

Code only (needs the Viola-UNet output masks and the conditioning masks). The SLURM array
version is viola_detect_array.py (launchers in ich_viola_runs/).

Usage:
  python analysis/viola_detect.py "<viola_out_glob>" <recovered_master_csv> <out_csv>
  (recovered_master_csv must have the `decoded_image` column; run recover_cf_decoded_mapping.py first)
"""
import glob
import os
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def seg_iou(src_seg: str, viola_mask: str) -> float:
    v = nib.load(viola_mask).get_fdata()
    if v.sum() == 0:
        return 0.0
    s = (nib.load(src_seg).get_fdata() == 66)
    zf = [v.shape[i] / s.shape[i] for i in range(3)]
    s = zoom(s, zf, order=0) > 0
    vf = v > 0
    inter = np.logical_and(s, vf).sum()
    uni = np.logical_or(s, vf).sum()
    return float(inter / uni) if uni > 0 else 0.0


def _segbase_from_viola(fn: str) -> str:
    """Fallback key: viola mask filename -> the SEG_BASENAME.nii.gz (only for non-colliding rows)."""
    b = os.path.basename(fn)
    return b.split(".nii.gz_pred_")[0] + ".nii.gz"


def run(viola_glob: str, master_csv: str, out_csv: str) -> None:
    import pandas as pd
    df = pd.read_csv(master_csv)
    if "decoded_image" not in df.columns:
        raise SystemExit(
            "master_csv lacks the 'decoded_image' column; run "
            "analysis/recover_cf_decoded_mapping.py first to build the affine-recovered mapping."
        )
    masks = [f for f in glob.glob(viola_glob) if f.endswith((".nii", ".nii.gz"))]
    # Primary index: exact unique decoded filename (Viola preserves the full input name).
    by_name = {os.path.basename(m): m for m in masks}
    # Fallback index: SEG_BASENAME -> masks, used only when decoded_image is absent AND the
    # basename is unambiguous (single candidate). Never first-matches a colliding basename.
    by_base: dict[str, list[str]] = {}
    for m in masks:
        by_base.setdefault(_segbase_from_viola(m), []).append(m)
    print(f"{len(masks)} viola masks -> {len(by_name)} unique decoded names")

    df["viola_iou"] = np.nan
    df["viola_detected"] = False
    df["viola_mask"] = ""
    matched = ambiguous = 0
    for i, row in df.iterrows():
        dec = str(row.get("decoded_image", "") or "")
        c = by_name.get(dec)
        if c is None:
            # fall back to basename ONLY if unambiguous
            cands = by_base.get(os.path.basename(str(row["segmentation"])), [])
            if len(cands) == 1:
                c = cands[0]
            elif len(cands) > 1:
                ambiguous += 1
                continue
        if c is None:
            continue  # viola mask not produced (yet)
        iou = seg_iou(row["segmentation"], c)
        df.at[i, "viola_iou"] = iou
        df.at[i, "viola_detected"] = iou > 0
        df.at[i, "viola_mask"] = os.path.basename(c)
        matched += 1
        if matched % 200 == 0:
            print(f"  {matched} matched...")
    df.to_csv(out_csv, index=False)
    m = df["viola_iou"].notna()
    print(f"matched {matched}/{len(df)} rows"
          + (f"  (skipped {ambiguous} colliding rows with no decoded_image)" if ambiguous else ""))
    if m.any():
        print(f"viola detection rate over matched: {(df.loc[m, 'viola_iou'] > 0).mean():.3f} "
              f"(n={int(m.sum())})")
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
