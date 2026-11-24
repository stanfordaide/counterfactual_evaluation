"""NIfTI I/O helpers and path utilities for the PE dataset."""

import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

# ── Root paths ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(os.environ.get("CF_RAW", "/path/to/private_data"))
ANATOMY_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_all"
VESSEL_V2_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_lung_vessels_v2"


def anatomy_path(patient_id: str) -> Path:
    return ANATOMY_DIR / f"{patient_id}_segmentation_merged.nii.gz"


def vessel_v2_path(patient_id: str) -> Path:
    return VESSEL_V2_DIR / f"{patient_id}_segmentation_lung_vessels_v2.nii.gz"


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a NIfTI file, returning (data, affine, header)."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.affine, img.header


def generator_slice_map(nz: int, d: int) -> np.ndarray:
    """Label slice index of each generated slice k = 0..d-1.

    The PE generator (01_pe_diffusion/scripts/infer_controlnet.py) resized the label map from
    nz to d slices with torch F.interpolate(mode="nearest"); x and y were not resized. This is
    the same index rule: floor(k * (nz / d)) in float32, at most nz - 1. Use it to put a label
    map on the grid of the generated CT and of the detector prediction.
    """
    if nz < 1 or d < 1:
        raise ValueError(f"label and prediction grids cannot be related: nz={nz}, D={d}")
    scale = np.float32(nz) / np.float32(d)
    idx = np.floor(np.arange(d, dtype=np.float32) * scale).astype(np.int64)
    return np.minimum(idx, nz - 1)


def save_nifti(data: np.ndarray, affine: np.ndarray, header: Optional[nib.Nifti1Header], path: Path):
    """Save a NIfTI file preserving the original affine and header."""
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(path))


# Convenience aliases for path functions
def get_anatomy_path(patient_id: str) -> Path:
    """Get anatomy mask path for a patient."""
    return anatomy_path(patient_id)


def get_vessel_v2_path(patient_id: str) -> Path:
    """Get v2 vessel segmentation path (arteries/veins/airways split)."""
    return vessel_v2_path(patient_id)
