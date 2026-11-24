"""Convert READ (Nature Sci Data 2023, s41597-023-02374-x) to nnU-Net input + GT NIfTI.

READ = 40 CTPA scans from two scanners (20 Toshiba Aquilion + 20 GE Revolution), each a
DICOM series with a binary pixel-level PE mask (NIfTI). Layout under raw/:
  Files .../CT scans (DICOM files)/{TOSHIBA,GE} (DICOM files)/<NNTS|NNGE>/*.dcm
  Files .../Ground truth (pixel level segmentation - NiFTI files)/{TOSHIBA,GE} (NiFTI files)/<NNTS|NNGE>.nii.gz

Masks are binary {0,1} and share the DICOM grid (verified), so - like FUMPE - per-embolus
detection uses 3D connected components on the mask; the strict ±5-slice overlap is the
same criterion as the synthetic + FUMPE + CAD-PE evaluations.

For each scan we read the DICOM series (HU), write it as nnU-Net input, and write the mask
on the CT grid (CopyInformation) so GT and predictions share a voxel grid.

Run with the maisi env (SimpleITK + pydicom):
  /path/to/miniconda/envs/maisi/bin/python external_validation_read/convert_read.py
"""
from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

DATA = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "READ"
BASE = DATA / "raw" / ("Files Pixel-level annotated dataset of computed tomography "
                       "images of acute pulmonary embolism")
DICOM_ROOT = BASE / "CT scans (DICOM files)"
NIFTI_ROOT = BASE / "Ground truth (pixel level segmentation - NiFTI files)"
NNU_IN = DATA / "nnunet_input"
GT = DATA / "gt_nii"
MANIFEST = Path(__file__).resolve().parents[1] / "output" / "external_validation" / "read" / "read_manifest.csv"
SCANNERS = [("TOSHIBA", "DICOM files", "NiFTI files"), ("GE", "DICOM files", "NiFTI files")]


def _plausible(ct: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    hu = ct[mask].astype(float)
    if hu.size == 0:
        return (1.0, 0.0)
    return (float(((hu >= -100) & (hu <= 400)).mean()), float((hu < -500).mean()))


def convert_one(scanner: str, scan_dir: Path) -> dict:
    scan = scan_dir.name                                             # e.g. 01TS
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(scan_dir))
    reader = sitk.ImageSeriesReader(); reader.SetFileNames(files)
    img = reader.Execute()                                          # HU
    ref = sitk.ReadImage(str(NIFTI_ROOT / f"{scanner} (NiFTI files)" / f"{scan}.nii.gz"))
    if img.GetSize() != ref.GetSize():
        ref = sitk.Resample(ref, img, sitk.Transform(), sitk.sitkNearestNeighbor,
                            0, ref.GetPixelID())
    ref.CopyInformation(img)
    mask = (sitk.GetArrayFromImage(ref) > 0).astype(np.uint8)      # binary [z, y, x]
    ct = sitk.GetArrayFromImage(img)

    case = f"READ_{scan}"
    NNU_IN.mkdir(parents=True, exist_ok=True); GT.mkdir(parents=True, exist_ok=True)
    out_img, out_gt = NNU_IN / f"{case}_0000.nii.gz", GT / f"{case}.nii.gz"
    if not out_img.exists():
        sitk.WriteImage(img, str(out_img))
    if not out_gt.exists():
        gt_img = sitk.GetImageFromArray(mask); gt_img.CopyInformation(img)
        sitk.WriteImage(gt_img, str(out_gt))

    sx, sy, sz = img.GetSpacing(); vox_ml = sx * sy * sz / 1000.0
    pe_vox = int(mask.sum())
    plaus, air = _plausible(ct, mask.astype(bool))
    return {"scan": scan, "scanner": scanner, "case": case, "n_slices": int(ct.shape[0]),
            "spacing_x": sx, "spacing_y": sy, "spacing_z": sz,
            "pe_voxels": pe_vox, "pe_volume_ml": pe_vox * vox_ml,
            "pe_slices": int((mask.sum(axis=(1, 2)) > 0).sum()),
            "plausible_frac": round(plaus, 3), "air_frac": round(air, 3)}


def main():
    rows = []
    for scanner, _dsub, _nsub in SCANNERS:
        ddir = DICOM_ROOT / f"{scanner} (DICOM files)"
        for scan_dir in sorted(p for p in ddir.iterdir() if p.is_dir()):
            r = convert_one(scanner, scan_dir)
            rows.append(r)
            flag = "  <-- CHECK" if r["plausible_frac"] < 0.5 and r["pe_voxels"] else ""
            print(f"  {r['case']}: {r['n_slices']} sl, z={r['spacing_z']:.2f}mm, "
                  f"PE {r['pe_volume_ml']:.2f} mL, plausible={r['plausible_frac']}{flag}")
    df = pd.DataFrame(rows)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    pos = df[df.pe_voxels > 0]
    print(f"\nwrote {len(df)} cases -> {MANIFEST}")
    print(f"PE-positive {len(pos)}, PE-negative {len(df) - len(pos)}")
    print(f"PE volume mL (positives): median {pos.pe_volume_ml.median():.2f}, "
          f"range {pos.pe_volume_ml.min():.2f}-{pos.pe_volume_ml.max():.2f}")
    print(f"z-spacing counts: {df.spacing_z.round(2).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
