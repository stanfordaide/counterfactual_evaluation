"""Convert CAD-PE (Gonzalez et al. 2020, arXiv:2003.13440) to nnU-Net input + GT NIfTI.

CAD-PE layout (raw/):
  images/NNN.nrrd        - CTPA series in HU (NNN = 001..091)
  rs/0NNNRefStd.nrrd     - radiologist reference standard, INSTANCE-labeled: each
                           embolus has a distinct integer id (1,2,3,...); 0 = background.
                           Some scans are PE-negative (all-zero mask).

Unlike FUMPE (DICOM + .mat, needed a z-order safety net), CAD-PE images and masks are
both NRRD carrying full geometry, so alignment is geometric. For each case we:
  1. Read image (HU) and reference mask with SimpleITK.
  2. Write image as nnU-Net input  <NNU_IN>/CADPE_<nnn>_0000.nii.gz.
  3. Write the INSTANCE-labeled reference on the image grid to <GT>/CADPE_<nnn>.nii.gz
     (resampled nearest-neighbour only if the mask grid differs from the image).
     Keeping instance ids means the per-embolus evaluation uses the radiologist's own
     embolus separation rather than connected components.

A HU-plausibility check (PE voxels should sit on clot/blood HU, not air) is reported as
a diagnostic; CAD-PE masks are geometry-aligned so it should pass everywhere.

Run with the maisi env (SimpleITK):
  /path/to/miniconda/envs/maisi/bin/python external_validation_cadpe/convert_cadpe.py
"""
from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

DATA = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "CAD_PE"
RAW = DATA / "raw"
IMAGES = RAW / "images"
RS = RAW / "rs"
NNU_IN = DATA / "nnunet_input"
GT = DATA / "gt_nii"
MANIFEST = Path(__file__).resolve().parents[1] / "output" / "external_validation" / "cadpe" / "cadpe_manifest.csv"


def _plausible_score(ct_zyx: np.ndarray, mask_zyx: np.ndarray) -> tuple[float, float]:
    hu = ct_zyx[mask_zyx].astype(float)
    if hu.size == 0:
        return (1.0, 0.0)
    return (float(((hu >= -100) & (hu <= 400)).mean()), float((hu < -500).mean()))


def _mask_path(stem: str) -> Path:
    """Reference mask for an image. Two naming schemes coexist in CAD-PE: the 40
    numeric challenge cases ('001' -> '0001RefStd.nrrd') and the 51 post-challenge
    cases ('e0032' -> 'e0032RefStd.nrrd')."""
    cands = [f"{stem}RefStd.nrrd"]
    if stem.isdigit():
        cands.append(f"{int(stem):04d}RefStd.nrrd")
    for c in cands:
        if (RS / c).exists():
            return RS / c
    raise FileNotFoundError(f"no RefStd mask for image {stem}")


def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-3)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-2)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-3))


def convert_one(num: str, idx: int) -> dict:
    img = sitk.ReadImage(str(IMAGES / f"{num}.nrrd"))                 # HU
    ref = sitk.ReadImage(str(_mask_path(num)))                       # instance labels

    if not _same_grid(img, ref):
        # resample instance mask onto the image grid (nearest -> preserve ids)
        ref = sitk.Resample(ref, img, sitk.Transform(), sitk.sitkNearestNeighbor,
                            0, ref.GetPixelID())
    ref.CopyInformation(img)                                         # guarantee grid match
    lab = sitk.GetArrayFromImage(ref).astype(np.uint16)             # [z, y, x]
    ct = sitk.GetArrayFromImage(img)                                # [z, y, x]

    case = f"CADPE_{idx:03d}"
    NNU_IN.mkdir(parents=True, exist_ok=True); GT.mkdir(parents=True, exist_ok=True)
    out_img, out_gt = NNU_IN / f"{case}_0000.nii.gz", GT / f"{case}.nii.gz"
    if not out_img.exists():
        sitk.WriteImage(img, str(out_img))                          # resume: skip if done
    if not out_gt.exists():
        gt_img = sitk.GetImageFromArray(lab); gt_img.CopyInformation(img)
        sitk.WriteImage(gt_img, str(out_gt))

    sx, sy, sz = img.GetSpacing(); vox_ml = sx * sy * sz / 1000.0
    ids = [int(v) for v in np.unique(lab) if v > 0]
    pe_vox = int((lab > 0).sum())
    plaus, air = _plausible_score(ct, lab > 0)
    return {"num": num, "case": case, "n_slices": int(ct.shape[0]),
            "spacing_x": sx, "spacing_y": sy, "spacing_z": sz,
            "n_emboli": len(ids), "pe_voxels": pe_vox,
            "pe_volume_ml": pe_vox * vox_ml,
            "pe_slices": int(((lab > 0).sum(axis=(1, 2)) > 0).sum()),
            "plausible_frac": round(plaus, 3), "air_frac": round(air, 3)}


def main():
    nums = sorted(p.stem for p in IMAGES.glob("*.nrrd"))
    rows = []
    for i, num in enumerate(nums, 1):
        r = convert_one(num, i)
        rows.append(r)
        flag = "  <-- CHECK (low plausible)" if r["plausible_frac"] < 0.5 and r["pe_voxels"] else ""
        print(f"  {r['case']} img{num}: {r['n_slices']} sl, z={r['spacing_z']:.2f}mm, "
              f"{r['n_emboli']} emboli, PE {r['pe_volume_ml']:.2f} mL, "
              f"plausible={r['plausible_frac']}{flag}")
    df = pd.DataFrame(rows)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    pos = df[df.pe_voxels > 0]
    print(f"\nwrote {len(df)} cases -> {MANIFEST}")
    print(f"PE-positive {len(pos)}, PE-negative {len(df) - len(pos)}; "
          f"total emboli {int(df.n_emboli.sum())}")
    print(f"PE volume mL (positives): median {pos.pe_volume_ml.median():.2f}, "
          f"range {pos.pe_volume_ml.min():.2f}-{pos.pe_volume_ml.max():.2f}")
    print(f"low-plausible (<0.5) positives: {((df.plausible_frac < 0.5) & (df.pe_voxels > 0)).sum()}")


if __name__ == "__main__":
    main()
