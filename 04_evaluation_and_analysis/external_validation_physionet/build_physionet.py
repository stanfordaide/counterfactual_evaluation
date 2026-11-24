"""Prepare PhysioNet CT-ICH (Hssayeni et al. v1.3.1) for the Viola-UNet + EMM external ICH
detectability-transfer validation. This is the leakage-free replacement for INSTANCE (which is
Viola's training set); Hssayeni is neither INSTANCE (clean for Viola) nor RSNA (clean for EMM).

Positives-only: the 36 scans with a radiologist GT hemorrhage mask (matches the
synthetic model-card VOLUME transfer). The dataset also holds 39 hemorrhage-negative scans
(verified: mask-empty <=> all-slice No_Hemorrhage, 75/75 agreement) which are left aside here but
remain available for a specificity add-on.

Unlike INSTANCE (DICOM + separate -sim npy), PhysioNet ships aligned NIfTI CT + binary masks
(label 254), so setup is minimal:
  - Viola input: symlink each positive CT NIfTI into viola_in/  (Viola reads a NIfTI folder).
  - EMM input:   save each positive CT as (1, X, Y, Z) float32 raw-HU .npy in emm_npy/
                 (exactly the format ich_emm/dataset.load_image_keep_zdim consumes: squeeze ->
                 resize 480 -> window -> 30-slice windows).
  - Manifest:    per-scan GT volume + subtypes (from hemorrhage_diagnosis_raw_ct.csv) + demog.

Code only: needs the public PhysioNet CT-ICH v1.3.1 download under CF_RAW/physionet_ich/.
The demographics file is read only to flag adults for an optional sensitivity analysis; the
public resource table (CF_DATA/external_validation/physionet/physionet_detection.csv) does not
carry age.

Run:  python external_validation_physionet/build_physionet.py
"""
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

WORK = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "physionet_ich"
D = WORK / "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1"
VIOLA_IN = WORK / "viola_in"
EMM_NPY = WORK / "emm_npy"
HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "output" / "physionet" / "physionet_manifest.csv"

SUB = ["Intraventricular", "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural"]
EXTRA_AXIAL = {"Epidural", "Subdural"}      # bleeds TotalSegmentator is structurally blind to
INTRA_AXIAL = {"Intraparenchymal", "Intraventricular"}


def _demographics() -> pd.DataFrame:
    demo = pd.read_csv(D / "Patient_demographics.csv")
    demo = demo.rename(columns={demo.columns[0]: "pid", demo.columns[1]: "age",
                                demo.columns[2]: "gender"})
    demo = demo[pd.to_numeric(demo["pid"], errors="coerce").notna()].copy()
    demo["pid"] = demo["pid"].astype(int)
    demo["age"] = pd.to_numeric(demo["age"], errors="coerce")   # infants recorded as fractional yr
    return demo.set_index("pid")[["age", "gender"]]


def main() -> None:
    VIOLA_IN.mkdir(parents=True, exist_ok=True)
    EMM_NPY.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    diag = pd.read_csv(D / "hemorrhage_diagnosis_raw_ct.csv")
    demo = _demographics()
    rows = []
    for mk in sorted(glob.glob(str(D / "masks" / "*.nii*"))):
        pid = int(os.path.basename(mk).split(".")[0])
        ct = D / "ct_scans" / os.path.basename(mk)
        m = nib.load(mk)
        md = m.get_fdata() > 0
        vox_ml = float(np.prod(m.header.get_zooms())) / 1000.0
        gt_ml = float(md.sum()) * vox_ml
        if gt_ml <= 0:
            continue                                     # positives only

        g = diag[diag["PatientNumber"] == pid]
        subs = [s for s in SUB if g[s].sum() > 0]

        link = VIOLA_IN / f"{pid:03d}.nii"
        if not link.exists():
            os.symlink(str(ct), str(link))
        npy_path = EMM_NPY / f"{pid:03d}.npy"
        if not npy_path.exists():
            arr = nib.load(str(ct)).get_fdata().astype(np.float32)  # (X, Y, Z) raw HU
            np.save(npy_path, arr[None])                            # (1, X, Y, Z)

        age = float(demo.loc[pid, "age"]) if pid in demo.index and demo.loc[pid, "age"] == demo.loc[pid, "age"] else np.nan
        rows.append({
            "pid": pid, "case": f"{pid:03d}", "ct_path": str(ct), "mask_path": mk,
            "gt_volume_ml": round(gt_ml, 4), "gt_voxels": int(md.sum()),
            "subtypes": "+".join(subs), "n_subtypes": len(subs),
            "extra_axial": any(s in EXTRA_AXIAL for s in subs),
            "intra_axial": any(s in INTRA_AXIAL for s in subs),
            "subarachnoid": "Subarachnoid" in subs,
            "age": age, "is_adult": bool(age >= 18) if age == age else False,
            "gender": demo.loc[pid, "gender"] if pid in demo.index else "",
            "spacing_z": round(float(m.header.get_zooms()[2]), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST, index=False)
    print(f"{len(df)} positive scans -> {MANIFEST}")
    print(f"viola_in symlinks: {len(list(VIOLA_IN.glob('*.nii')))}   emm_npy: {len(list(EMM_NPY.glob('*.npy')))}")
    b = [-0.001, 0.5, 5, 30, 1e6]
    df["vb"] = pd.cut(df.gt_volume_ml, b, labels=["<0.5", "0.5-5", "5-30", ">=30"])
    print("\nGT volume bins:")
    print(df["vb"].value_counts().reindex(["<0.5", "0.5-5", "5-30", ">=30"]).to_string())
    print(f"\nextra-axial (EDH/SDH) scans: {int(df.extra_axial.sum())}  "
          f"intra-axial (IPH/IVH): {int(df.intra_axial.sum())}  SAH: {int(df.subarachnoid.sum())}")
    ad = df[df.is_adult]
    print(f"\nadults (>=18): {len(ad)}  (PRIMARY cohort)   pediatric (<18): {len(df)-len(ad)}"
          f"   age missing: {int(df.age.isna().sum())}")
    print("adult-only volume bins:")
    print(ad["vb"].value_counts().reindex(["<0.5", "0.5-5", "5-30", ">=30"]).to_string())


if __name__ == "__main__":
    main()
