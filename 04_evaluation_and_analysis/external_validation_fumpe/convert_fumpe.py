"""Convert FUMPE (Kaggle andrewmvd mirror) to nnU-Net input + aligned GT NIfTI.

FUMPE layout (raw/FUMPE/):
  CT_scans/PATxxx/*.dcm   - DICOM CTPA series (HU)
  GroundTruth/PATxxx.mat  - radiologist PE mask, MATLAB 'Mask' [Y, X, Z] uint8 {0,1}

For each patient we:
  1. Read the DICOM series with SimpleITK (spatial sort) -> CT image in HU.
  2. Save it as nnU-Net input  <NNU_IN>/FUMPE_<nnn>_0000.nii.gz.
  3. Load the .mat mask, transpose to [Z, Y, X], and pick the z-ordering (as-is
     vs reversed) whose PE voxels land on clot/blood HU rather than air - a
     per-patient safety net against DICOM-order vs .mat-order mismatch. Save the
     aligned GT with the CT geometry to <GT>/FUMPE_<nnn>.nii.gz.

Because GT is written with the CT geometry and nnU-Net preserves input geometry,
the GT and both models' predictions share a voxel grid - so the strict ±N-slice
overlap (compute_slice_overlap, intended_label=1) is directly comparable to the
synthetic evaluation. Volumes (from GT) drive the subgroup axis.

Run with the maisi env (has SimpleITK + pydicom + scipy):
  /path/to/miniconda/envs/maisi/bin/python external_validation_fumpe/convert_fumpe.py
"""
from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import SimpleITK as sitk

RAW = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "FUMPE" / "raw" / "FUMPE"
DATA = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "FUMPE"
NNU_IN = DATA / "nnunet_input"
GT = DATA / "gt_nii"
MANIFEST = Path(__file__).resolve().parents[1] / "output" / "external_validation" / "fumpe" / "fumpe_manifest.csv"


def _plausible_score(ct_zyx: np.ndarray, mask_zyx: np.ndarray) -> tuple[float, float]:
    """Fraction of PE voxels at clot/blood HU and the air fraction (lower=better)."""
    hu = ct_zyx[mask_zyx].astype(float)
    if hu.size == 0:
        return (0.0, 1.0)
    plausible = float(((hu >= -100) & (hu <= 400)).mean())
    air = float((hu < -500).mean())
    return plausible, air


def convert_one(pat: str, idx: int) -> dict:
    series_dir = RAW / "CT_scans" / pat
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir))
    reader = sitk.ImageSeriesReader(); reader.SetFileNames(files)
    img = reader.Execute()                      # HU, spatial-sorted
    ct = sitk.GetArrayFromImage(img)            # [z, y, x]

    mask = sio.loadmat(RAW / "GroundTruth" / f"{pat}.mat")["Mask"]  # [y, x, z]
    mask_zyx = np.transpose(mask, (2, 0, 1)).astype(np.uint8)       # [z, y, x]

    # pick z-order by HU plausibility (robust to per-patient order mismatch)
    z = min(ct.shape[0], mask_zyx.shape[0])
    ct_c, m_asis = ct[:z], mask_zyx[:z].astype(bool)
    m_rev = mask_zyx[:z][::-1].astype(bool)
    p_as, a_as = _plausible_score(ct_c, m_asis)
    p_rev, a_rev = _plausible_score(ct_c, m_rev)
    reversed_ = (p_rev - a_rev) > (p_as - a_as)
    chosen = (m_rev if reversed_ else m_asis).astype(np.uint8)
    plaus, air = (p_rev, a_rev) if reversed_ else (p_as, a_as)

    case = f"FUMPE_{idx:03d}"
    # CT -> nnU-Net input
    NNU_IN.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(NNU_IN / f"{case}_0000.nii.gz"))
    # GT with CT geometry (truncate CT geometry to z if needed via same array size)
    gt_img = sitk.GetImageFromArray(chosen)
    ref = img[:, :, :z] if z != ct.shape[0] else img
    gt_img.CopyInformation(ref)
    GT.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(gt_img, str(GT / f"{case}.nii.gz"))

    sx, sy, sz = img.GetSpacing()
    vox_ml = sx * sy * sz / 1000.0
    pe_vox = int(chosen.sum())
    pe_slices = int((chosen.sum(axis=(1, 2)) > 0).sum())
    return {"patient": pat, "case": case, "n_slices": int(ct.shape[0]),
            "spacing_x": sx, "spacing_y": sy, "spacing_z": sz,
            "pe_voxels": pe_vox, "pe_volume_ml": pe_vox * vox_ml,
            "pe_slices": pe_slices, "z_reversed": bool(reversed_),
            "plausible_frac": round(plaus, 3), "air_frac": round(air, 3)}


def main():
    pats = sorted(p.name for p in (RAW / "CT_scans").iterdir() if p.is_dir())
    rows = []
    for i, pat in enumerate(pats, 1):
        r = convert_one(pat, i)
        rows.append(r)
        flag = "  <-- CHECK (low plausible / z-rev)" if (r["plausible_frac"] < 0.5 or r["z_reversed"]) else ""
        print(f"  {r['case']} {pat}: {r['n_slices']} sl, spacing z={r['spacing_z']:.2f}, "
              f"PE {r['pe_volume_ml']:.2f} mL / {r['pe_slices']} sl, "
              f"plausible={r['plausible_frac']}{flag}")
    df = pd.DataFrame(rows)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    print(f"\nwrote {len(df)} cases -> {MANIFEST}")
    print(f"PE volume mL: median {df.pe_volume_ml.median():.2f}, "
          f"range {df.pe_volume_ml.min():.2f}-{df.pe_volume_ml.max():.2f}")
    print(f"z-reversed patients: {df.z_reversed.sum()}; "
          f"low-plausible (<0.5): {(df.plausible_frac < 0.5).sum()}")


if __name__ == "__main__":
    main()
