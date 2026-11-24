"""3D skeletonization of vessel masks.

Extracts the centerline/skeleton from cleaned vessel masks.
"""

import os
import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage
from typing import Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pe/ folder (for utils)
from utils.nifti_io import load_nifti, save_nifti


def compute_skeleton(vessel_mask: np.ndarray) -> np.ndarray:
    """
    Compute 3D skeleton of vessel mask.

    Args:
        vessel_mask: Binary vessel mask

    Returns:
        skeleton: Binary skeleton mask
    """
    print("Computing 3D skeleton (this may take a while)...")
    skeleton = skeletonize(vessel_mask.astype(bool))
    n_skeleton_voxels = np.sum(skeleton)
    print(f"  Skeleton voxels: {n_skeleton_voxels}")
    return skeleton.astype(np.uint8)


def compute_distance_transform(vessel_mask: np.ndarray, spacing: Tuple[float, ...]) -> np.ndarray:
    """
    Compute distance transform (radius at each point) for the vessel mask.

    Args:
        vessel_mask: Binary vessel mask
        spacing: Voxel spacing in mm (x, y, z)

    Returns:
        distance_transform: Distance to nearest boundary in mm
    """
    print("Computing distance transform...")
    dt = ndimage.distance_transform_edt(vessel_mask, sampling=spacing)
    return dt.astype(np.float32)


def skeletonize_vessels(vessel_mask: np.ndarray, spacing: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skeletonize vessel mask and compute radius at skeleton points.

    Args:
        vessel_mask: Binary vessel mask
        spacing: Voxel spacing in mm

    Returns:
        skeleton: Binary skeleton
        radius_at_skeleton: Radius (mm) at each skeleton voxel
    """
    # Compute skeleton
    skeleton = compute_skeleton(vessel_mask)

    # Compute distance transform for radius estimation
    dt = compute_distance_transform(vessel_mask, spacing)

    # Extract radius at skeleton points
    radius_at_skeleton = np.zeros_like(dt)
    radius_at_skeleton[skeleton > 0] = dt[skeleton > 0]

    print(f"  Radius range at skeleton: {radius_at_skeleton[skeleton > 0].min():.2f} - {radius_at_skeleton[skeleton > 0].max():.2f} mm")

    return skeleton, radius_at_skeleton


DEFAULT_OUTPUT_DIR = os.environ.get("CF_OUT", "/path/to/cf_mask_pipeline_output") + "/phase1_vessel_tree/output"


def skeletonize_for_patient(patient_id: str, save_output: bool = True,
                            output_dir: str = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load cleaned vessel mask and compute skeleton for a patient.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Load cleaned vessel mask
    cleaned_path = f"{output_dir}/{patient_id}_vessels_cleaned.nii.gz"
    print(f"Loading cleaned vessel mask: {cleaned_path}")

    vessel_mask, affine, header = load_nifti(cleaned_path)
    spacing = header.get_zooms()[:3]
    print(f"  Spacing: {spacing} mm")

    # Skeletonize
    skeleton, radius = skeletonize_vessels(vessel_mask.astype(bool), spacing)

    stats = {
        'vessel_voxels': int(np.sum(vessel_mask)),
        'skeleton_voxels': int(np.sum(skeleton)),
        'radius_min': float(radius[skeleton > 0].min()),
        'radius_max': float(radius[skeleton > 0].max()),
        'radius_mean': float(radius[skeleton > 0].mean()),
    }

    if save_output:
        # Save skeleton
        skel_path = f"{output_dir}/{patient_id}_skeleton.nii.gz"
        save_nifti(skeleton, affine, header, skel_path)
        print(f"Saved skeleton: {skel_path}")

        # Save radius map
        radius_path = f"{output_dir}/{patient_id}_radius.nii.gz"
        save_nifti(radius, affine, header, radius_path)
        print(f"Saved radius map: {radius_path}")

        stats['skeleton_path'] = skel_path
        stats['radius_path'] = radius_path

    return skeleton, radius, affine, header, stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {os.path.basename(__file__)} <STUDY_ID>")
    patient_id = sys.argv[1]
    skeleton, radius, affine, header, stats = skeletonize_for_patient(patient_id)
    print(f"\nSkeletonization complete. Stats: {stats}")
