"""Recover the exact decoded image (hence EMM prediction and Viola/TotalSeg mask) for every
ICH counterfactual row, resolving the 42 segmentation-basename collisions via the NIfTI affine.

Background
----------
The 2,640 ICH counterfactual rows are keyed in the master CSV by the conditioning
segmentation's *full path* (unique). The EMM predictions (cf_2640_samples_npy.csv) and the
decoded images / segmentation masks are instead named after the DECODED image, whose stem is
only the segmentation BASENAME. For 42 source-duplicate pairs -- the same source scan+date+donor
enrolled under two patient IDs -- that basename is non-unique, so the original basename join
(in the first ICH analysis script, for TotalSegmentator and EMM) silently took the
first match for BOTH rows in the pair.

Each decoded image inherits its source scan's full NIfTI affine (voxel spacing + origin), which
is unique per scan (the Z-origin differs by ~1000 mm between two source scans). Matching each row's
segmentation affine to the decoded image's affine therefore recovers the row->decoded-image
link exactly. Verified: 42/42 clean bijection (zooms resolve 25 pairs, the affine origin the
remaining 17).

This module rebuilds, non-destructively (writes a *_recovered.csv):
  - `decoded_image`      : the unique decoded filename for each row
  - `emm_prob_recovered` : re-merged from cf_2640_samples_npy.csv by the recovered decoded name

Code only: needs the internal ICH prediction folder (CF_RAW/ich_predictions/: the master CSV,
the classifier prediction CSV and the decoded images). The public table
CF_DATA/ich/cf_2640_samples_recovered.csv is this script's output (after the Viola-UNet merge,
analysis/viola_detect_array.py) with the private columns removed and the ids re-keyed.

Usage:
  python analysis/recover_cf_decoded_mapping.py
"""
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

BASE = str(Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "ich_predictions")
MASTER = BASE + "/cf_2640_samples_all_info_with_emm_prob.csv"
NPY = BASE + "/cf_2640_samples_npy.csv"
DECODED_GLOB = BASE + "/cf_*/decoded_images/*_image_embed_decoded.nii.gz"
OUT = str(Path(__file__).resolve().parents[1] / "output" / "ich" / "cf_2640_samples_recovered.csv")


def affine_key(path: str) -> tuple:
    """Fingerprint = (voxel zooms, affine origin), rounded. Unique per source scan."""
    im = nib.load(path)  # lazy: header only, no voxel data read
    a = im.affine
    return tuple(np.round(np.concatenate([np.array(im.header.get_zooms()[:3]), a[:3, 3]]), 1))


def _segbase_from_decoded(decoded_path: str) -> str:
    """decoded filename -> the SEG_BASENAME.nii.gz join key."""
    return os.path.basename(decoded_path).split(".nii.gz_pred_")[0] + ".nii.gz"


def _decoded_to_fileid(decoded_basename: str) -> str:
    """decoded: '..._image_embed_decoded.nii.gz'  ->  npy file_id: '..._image_embed_decoded.nii'."""
    return decoded_basename[:-3] if decoded_basename.endswith(".nii.gz") else decoded_basename


def build_decoded_map(df: pd.DataFrame) -> pd.DataFrame:
    decoded = glob.glob(DECODED_GLOB)
    dmap: dict[str, list[str]] = {}
    for d in decoded:
        dmap.setdefault(_segbase_from_decoded(d), []).append(d)
    n_dec = len(decoded)
    n_coll_keys = sum(1 for v in dmap.values() if len(v) > 1)
    print(f"{n_dec} decoded images -> {len(dmap)} basename keys ({n_coll_keys} colliding)")

    names, resolved, unmatched = [], 0, 0
    for _, row in df.iterrows():
        base = os.path.basename(str(row["segmentation"]))
        cands = dmap.get(base, [])
        if len(cands) == 1:
            chosen = cands[0]
        elif not cands:
            chosen = None
            unmatched += 1
        else:
            seg_k = affine_key(row["segmentation"])
            match = [c for c in cands if affine_key(c) == seg_k]
            chosen = match[0] if len(match) == 1 else None
            if chosen is not None:
                resolved += 1
            else:
                unmatched += 1
        names.append(os.path.basename(chosen) if chosen else "")
    df["decoded_image"] = names
    print(f"colliding rows resolved by affine fingerprint: {resolved}")
    print(f"rows with NO unique decoded match: {unmatched}")
    return df


def remerge_emm(df: pd.DataFrame) -> pd.DataFrame:
    npy = pd.read_csv(NPY)
    emm_by_fileid = dict(zip(npy["file_id"], npy["emm_prob"]))
    df["emm_prob_recovered"] = [
        emm_by_fileid.get(_decoded_to_fileid(d), np.nan) if d else np.nan
        for d in df["decoded_image"]
    ]
    return df


if __name__ == "__main__":
    df = pd.read_csv(MASTER)
    df = build_decoded_map(df)
    df = remerge_emm(df)

    old, new = df["emm_prob"].values, df["emm_prob_recovered"].values
    miss = df["emm_prob_recovered"].isna().sum()
    changed = ~np.isclose(old, new, equal_nan=True)
    det_old, det_new = old > 0.5, new > 0.5
    print("\n--- EMM re-merge impact ---")
    print(f"rows with recovered emm_prob missing: {miss}")
    print(f"rows whose emm_prob changed: {int(changed.sum())}")
    print(f"detection (>0.5) flips: {int((det_old != det_new).sum())}")
    print(f"EMM detected count: old={int(det_old.sum())} -> new={int(det_new.sum())} "
          f"(delta {int(det_new.sum() - det_old.sum()):+d})")
    # restrict to the analysis cohort (intended volume > 0)
    vol = df["ich_volume(ml)"] > 0
    print(f"\nwithin volume>0 cohort (n={int(vol.sum())}):")
    print(f"  EMM detection rate: old={det_old[vol].mean():.4f} -> new={det_new[vol].mean():.4f}")

    # Make the affine-recovered emm_prob the primary column; keep the original first-match
    # values under `emm_prob_firstmatch` for transparency. `total_segmentator_*` columns are
    # left untouched (TotalSegmentator is retired in favor of Viola, whose masks join on the
    # new `decoded_image` column via the same fingerprint).
    df["emm_prob_firstmatch"] = df["emm_prob"]
    df["emm_prob"] = df["emm_prob_recovered"]
    df = df.drop(columns=["emm_prob_recovered"])
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")
