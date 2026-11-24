"""PE-detector output on the disease-free scans, one row per scan (Extended Data Table 2).

The two Extended Data Table 2 scripts (reader_confirmed_and_disease_free.py and
disease_free_fp_size.py) need, for each detector and scan type, the predicted
embolus mask of every disease-free scan of the cohort patients:
  - PE-Detector-1 / PE-Detector-2 on the disease-free synthetic scan of each patient,
  - PE-Detector-1 / PE-Detector-2 on the real disease-free CTPA of each patient.

The predicted masks are detector outputs and are not shipped. This module reads them when
they are available under CF_RAW (the mask-reading code is the original code of both scripts)
and writes two small tables; otherwise it reads the same tables from CF_DATA:
  detector_outputs/disease_free_predictions.csv  detector, scans, patient_id,
      n_voxels (predicted voxels, mask > 0.5), voxel_mm3 (|det| of the affine 3x3)
  detector_outputs/disease_free_regions.csv      detector, scans, patient_id,
      region_voxels (one row per 3D connected component of a non-empty mask)
Rows keep the original scan order (sorted patient ids), so the bootstrap is unchanged.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
RAW = Path(os.environ.get("CF_RAW", "/path/to/private_data"))
OUT = Path(__file__).resolve().parents[1] / "output"

BASE = f"{RAW}/pe_training_data_all"
NOPE, NOPE_AP, REAL_NN = (f"{BASE}/pe_segmentation_synthetic_nope_round4",
                          f"{BASE}/pe_segmentation_APES_synthetic_selected", f"{BASE}/pe_segmentation")
APES_REAL_CSV = f"{BASE}/pe_seg_eval_results_apes_real.csv"
# 9 real scans that part 5 of the APES real-CTPA run never reached were run separately
# (same model and settings).
APES_MISSING9 = f"{RAW}/pe_segmentation_APES_real_missing9"
SCANS = "detector_outputs/disease_free_predictions.csv"
REGIONS = "detector_outputs/disease_free_regions.csv"


def neg_paths(pids):
    """(detector, scan type) -> {patient_id: predicted-mask path}, as in the original scripts."""
    apes_real = pd.read_csv(APES_REAL_CSV).set_index("impression_id")["seg_path"].to_dict()
    apes_nope = {os.path.basename(f).split("_pred_")[0]: f for f in glob.glob(NOPE_AP + "/*.nii.gz")}
    for _p in pids:
        if not (isinstance(apes_real.get(_p), str) and os.path.exists(apes_real[_p])):
            apes_real[_p] = f"{APES_MISSING9}/{_p}.nii.gz"
    return {
        ("PE-Detector-1", "Synthetic, disease-free"): {p: f"{NOPE}/{p}.nii.gz" for p in pids},
        ("PE-Detector-2", "Synthetic, disease-free"): {p: apes_nope.get(p) for p in pids},
        ("PE-Detector-1", "Real, disease-free"): {p: f"{REAL_NN}/{p}.nii.gz" for p in pids},
        ("PE-Detector-2", "Real, disease-free"): {p: apes_real.get(p) for p in pids},
    }


def _from_masks(pids):
    import nibabel as nib
    from scipy import ndimage
    scans, regions = [], []
    for (model, cond), paths in neg_paths(pids).items():
        for p, f in paths.items():
            if not (isinstance(f, str) and os.path.exists(f)):
                continue
            img = nib.load(f)
            m = np.asarray(img.dataobj) > 0.5
            scans.append({"detector": model, "scans": cond, "patient_id": p,
                          "n_voxels": int(m.sum()),
                          "voxel_mm3": float(abs(np.linalg.det(img.affine[:3, :3])))})
            if m.any():
                lab, _ = ndimage.label(m)
                for nv in np.bincount(lab.ravel())[1:]:
                    regions.append({"detector": model, "scans": cond, "patient_id": p,
                                    "region_voxels": int(nv)})
        print(f"{model} {cond}: read {sum(r['detector'] == model and r['scans'] == cond for r in scans)} masks",
              flush=True)
    s, r = pd.DataFrame(scans), pd.DataFrame(regions)
    (OUT / "detector_outputs").mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / SCANS, index=False)
    r.to_csv(OUT / REGIONS, index=False)
    print(f"wrote {OUT / SCANS} and {OUT / REGIONS}")
    return s, r


def load(pids):
    """Per-scan table and per-region table for the given (sorted) patient ids."""
    if Path(BASE).exists():
        return _from_masks(pids)
    s, r = pd.read_csv(RES / SCANS), pd.read_csv(RES / REGIONS)
    assert set(s.patient_id) <= set(pids), "disease-free table has patients outside the cohort"
    return s, r
