"""Vessel-overlay analysis for APES PE segmentations on the counterfactuals.

Same analysis as run_vessel_overlay.py but uses PE segmentations from the
APES model (binary masks, 1=PE, 0=background) instead of the original
pe_seg outputs.

APES segmentations live in a separate directory and use a different naming
convention: {patient}_{config}_pred_{timestamp}.nii.gz (no _pe_seg suffix).

Usage:
    python phase6_evaluation/run_vessel_overlay_apes.py --patient-id <STUDY_ID>
    python phase6_evaluation/run_vessel_overlay_apes.py --all
    python phase6_evaluation/run_vessel_overlay_apes.py --aggregate-only
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import (binary_dilation, binary_erosion,
                            distance_transform_edt, label as ndimage_label)
from scipy.spatial import cKDTree

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "phase1_vessel_tree"))

from utils.nifti_io import load_nifti, DATA_ROOT, generator_slice_map

# ── Paths (APES segmentations) ──────────────────────────────────────────────
PREDICTIONS_DIR = DATA_ROOT / "pe_training_data_all" / "predictions_controlnet_counterfactuals"
PREDICTIONS_CSV = PREDICTIONS_DIR / "controlnet_predictions_counterfactuals.csv"
GRID_BASE_DIR = Path(PROJECT_ROOT) / "phase5_cohort_assembly" / "output" / "patients"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output_apes" / "vessel_overlay"

# APES segmentation directory (binary masks: 1=PE, 0=background)
APES_SEG_DIR = DATA_ROOT / "pe_training_data_all" / "pe_segmentation_APES_cf"

PE_LABEL = 121
VESSEL_LABEL = 119

# Max distance (mm) from a PE voxel to the nearest skeleton voxel to count
# it as "on-vessel".  Generous enough for thick vessels.
MAX_VESSEL_DIST_MM = 15.0

GENERATION_NAMES = ["central", "lobar", "segmental", "subsegmental"]
VESSEL_TYPES = ["likely_artery", "likely_vein", "unknown"]
LOBE_NAMES = ["LUL", "LLL", "RUL", "RML", "RLL"]


# ── File index (reuse pattern from run_phase6) ──────────────────────────────

def build_file_index(predictions_csv: Path,
                     patient_id: Optional[str] = None,
                     apes_seg_dir: Path = APES_SEG_DIR) -> List[Dict]:
    """Parse predictions CSV to derive label + APES pe_seg paths per config.

    Label paths are derived from the prediction filename (same directory).
    PE segmentation paths point to the APES directory, where files are named
    {stem}.nii.gz (without the _image or _pe_seg suffix).
    """
    entries = []
    with open(predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            pred_path = row["prediction_filename"]
            pid = img_id.rsplit("_cf", 1)[0]
            if patient_id and pid != patient_id:
                continue
            label_path = pred_path.replace("_image.nii.gz", "_label.nii.gz")

            # APES naming: same stem as prediction but without _image suffix
            pred_stem = Path(pred_path).name.replace("_image.nii.gz", ".nii.gz")
            pe_seg_path = str(apes_seg_dir / pred_stem)

            entries.append({
                "image_id": img_id,
                "patient_id": pid,
                "config_id": img_id,
                "prediction_path": pred_path,
                "label_path": label_path,
                "pe_seg_path": pe_seg_path,
            })
    return entries


def load_config_metadata(patient_id: str, config_id: str) -> Dict:
    """Load config metadata from counterfactual grid JSON."""
    grid_path = GRID_BASE_DIR / patient_id / "phase2" / f"{patient_id}_counterfactual_grid.json"
    if not grid_path.exists():
        return {}
    with open(grid_path) as f:
        grid = json.load(f)
    for cfg in grid.get("configs", []):
        if cfg.get("config_id") == config_id:
            return cfg
    return {}


# ── Vessel graph helpers ─────────────────────────────────────────────────────

def load_vessel_graph(patient_id: str) -> Tuple[Optional[object], Optional[np.ndarray]]:
    """Load the final vessel graph and spacing for a patient.

    Returns (graph, spacing) or (None, None) if not found.
    """
    pkl_path = (GRID_BASE_DIR / patient_id / "phase1" /
                f"{patient_id}_vessel_graph_final.pkl")
    if not pkl_path.exists():
        return None, None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data["spacing"]


def build_edge_kdtree(G, spacing: np.ndarray):
    """Build a KD-tree over all skeleton voxel coords (in mm).

    Returns:
        tree: cKDTree over (N, 3) mm-space skeleton coords
        edge_ids: array of edge indices (1-based) parallel to tree data
        edge_list: list of (u, v, data) tuples indexed by edge_id - 1
    """
    coords_mm = []
    edge_ids = []
    edge_list = list(G.edges(data=True))

    for idx, (u, v, d) in enumerate(edge_list):
        for vox in d.get("segment_coords", []):
            coords_mm.append(np.array(vox, dtype=np.float64) * spacing)
            edge_ids.append(idx)

    if not coords_mm:
        return None, np.array([]), edge_list

    coords_mm = np.array(coords_mm)
    edge_ids = np.array(edge_ids, dtype=np.int32)
    tree = cKDTree(coords_mm)
    return tree, edge_ids, edge_list


def load_radius_map(patient_id: str):
    """Load per-voxel vessel radius map (optional - used for radius stats)."""
    radius_path = (GRID_BASE_DIR / patient_id / "phase1" /
                   f"{patient_id}_radius.nii.gz")
    if not radius_path.exists():
        return None
    data, _, _ = load_nifti(radius_path)
    return data


def load_reference_affine(patient_id: str) -> Optional[np.ndarray]:
    """Load affine from original-space vessel volume (same space as graph)."""
    vessels_path = (GRID_BASE_DIR / patient_id / "phase1" /
                    f"{patient_id}_vessels_cleaned.nii.gz")
    if not vessels_path.exists():
        return None
    _, affine, _ = load_nifti(vessels_path)
    return affine


def _align_to_reference(data: np.ndarray, data_affine: np.ndarray,
                        ref_affine: np.ndarray) -> np.ndarray:
    """Flip axes of data so it matches the reference coordinate space.

    Compares the sign of diagonal elements in both affines. If an axis
    direction is flipped, reverses that axis in the data array.
    """
    for axis in range(3):
        if np.sign(data_affine[axis, axis]) != np.sign(ref_affine[axis, axis]):
            data = np.flip(data, axis=axis)
    return data


# ── Occlusion & morphology helpers ────────────────────────────────────────────

OCCLUSION_THRESH_COMPLETE = 0.75       # >75% vessel filled → complete
OCCLUSION_THRESH_NEAR_COMPLETE = 0.25  # 25-75% → near_complete; <25% → partial
WALL_ADHERENCE_DIST_MM = 1.0           # PE voxels within 1 mm of wall → wall-adherent


def _compute_morphology(result: Dict, pe_mask: np.ndarray,
                        lumen_mask: np.ndarray, spacing: np.ndarray,
                        mapped_edges: np.ndarray, edge_list: list,
                        prefix: str = "detected"):
    """Compute occlusion fraction, acute/chronic shape features for PE.

    Writes keys with the given *prefix* into *result*.
    Operates on a cropped ROI for efficiency.

    *lumen_mask* is the true vessel lumen (vessel label 119 | intended PE
    label 121).  Both intended and detected PE are analyzed relative to
    this same lumen.

    Metrics:
        {prefix}_occlusion_frac   - PE / lumen in PE vicinity
        {prefix}_occlusion_cat    - complete / near_complete / partial
        {prefix}_wall_dist_mean_mm  - mean distance of PE voxels to lumen wall
        {prefix}_wall_adherence_frac - fraction of PE voxels within 1 mm of wall
        {prefix}_centrality       - mean normalized depth (0 = wall, 1 = center)
        {prefix}_compactness      - isoperimetric ratio  (higher = rounder)
        {prefix}_eccentricity     - PCA elongation (0 = sphere, 1 = line)
        {prefix}_centroid_offset_norm - PE centroid offset / lumen radius
        {prefix}_xs_coverage_mean - mean cross-sectional PE / lumen
        {prefix}_shape_category   - acute / chronic / indeterminate
    """
    nan = float("nan")
    pe_voxels = int(np.sum(pe_mask > 0))

    if pe_voxels == 0:
        for key in ["occlusion_frac", "occlusion_cat",
                     "wall_dist_mean_mm", "wall_adherence_frac",
                     "centrality", "compactness", "eccentricity",
                     "shape_category"]:
            result[f"{prefix}_{key}"] = nan if key != "occlusion_cat" and key != "shape_category" else ""
        return

    # ── Crop to ROI around PE for efficiency ──────────────────────────────
    # Generous padding so the ROI captures the surrounding vessel lumen
    # even when PE is slightly off-vessel.
    coords = np.argwhere(pe_mask > 0)
    pad = 30  # voxels (~18 mm at 0.6 mm spacing)
    slices = tuple(
        slice(max(0, coords[:, i].min() - pad),
              min(pe_mask.shape[i], coords[:, i].max() + pad + 1))
        for i in range(3)
    )
    pe_roi = pe_mask[slices]
    lumen_roi = lumen_mask[slices]

    # ── Occlusion fraction ────────────────────────────────────────────────
    # Expand the PE region slightly (3-voxel dilation) to capture the
    # surrounding vessel lumen, then compute PE ∩ lumen / lumen.
    pe_dilated = binary_dilation(pe_roi > 0, iterations=3)
    lumen_in_region = int(np.sum(lumen_roi[pe_dilated]))
    pe_in_lumen = int(np.sum((pe_roi > 0) & (lumen_roi > 0) & pe_dilated))

    if lumen_in_region > 0:
        occ_frac = pe_in_lumen / lumen_in_region
    else:
        occ_frac = nan

    result[f"{prefix}_occlusion_frac"] = occ_frac

    if np.isnan(occ_frac):
        result[f"{prefix}_occlusion_cat"] = ""
    elif occ_frac >= OCCLUSION_THRESH_COMPLETE:
        result[f"{prefix}_occlusion_cat"] = "complete"
    elif occ_frac >= OCCLUSION_THRESH_NEAR_COMPLETE:
        result[f"{prefix}_occlusion_cat"] = "near_complete"
    else:
        result[f"{prefix}_occlusion_cat"] = "partial"

    # ── Wall distance & centrality ────────────────────────────────────────
    # distance_transform_edt(input): for each NONZERO element, distance
    # to the nearest ZERO element.
    # With input = lumen_roi:
    #   inside lumen (nonzero) → distance to nearest background = wall dist
    #   outside lumen (zero) → 0
    dist_to_wall = distance_transform_edt(
        lumen_roi, sampling=spacing
    )

    pe_in_lumen_mask = (pe_roi > 0) & (lumen_roi > 0)
    pe_inside_count = int(np.sum(pe_in_lumen_mask))

    if pe_inside_count > 0:
        wall_dists = dist_to_wall[pe_in_lumen_mask]
        result[f"{prefix}_wall_dist_mean_mm"] = float(np.mean(wall_dists))
        result[f"{prefix}_wall_adherence_frac"] = float(
            np.mean(wall_dists <= WALL_ADHERENCE_DIST_MM)
        )

        # Centrality: normalize by max depth in the local vessel cross-section.
        # Use the max dist_to_wall in the ROI as proxy for vessel center depth.
        max_depth = float(np.max(dist_to_wall))
        if max_depth > 0:
            result[f"{prefix}_centrality"] = float(
                np.mean(wall_dists) / max_depth
            )
        else:
            result[f"{prefix}_centrality"] = nan
    else:
        result[f"{prefix}_wall_dist_mean_mm"] = nan
        result[f"{prefix}_wall_adherence_frac"] = nan
        result[f"{prefix}_centrality"] = nan

    # ── Compactness (isoperimetric ratio) ─────────────────────────────────
    # Sphericity ∝ V^(2/3) / A.  Higher = rounder (more acute-like).
    eroded = binary_erosion(pe_roi > 0)
    surface = (pe_roi > 0) & ~eroded
    surface_count = int(np.sum(surface))
    voxel_vol = float(np.prod(spacing))
    volume_mm3 = pe_voxels * voxel_vol
    # Approximate surface area: each surface voxel contributes ~(min_spacing)^2
    surface_area_mm2 = surface_count * float(np.min(spacing)) ** 2
    if surface_area_mm2 > 0:
        # Normalized so that a perfect sphere = 1
        compactness = ((np.pi ** (1.0 / 3.0)) *
                       (6.0 * volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2
    else:
        compactness = nan
    result[f"{prefix}_compactness"] = compactness

    # ── Eccentricity (PCA) ────────────────────────────────────────────────
    # 0 = sphere, 1 = maximally elongated (line)
    pe_coords_mm = np.argwhere(pe_roi > 0).astype(np.float64) * spacing
    if len(pe_coords_mm) >= 4:
        centred = pe_coords_mm - pe_coords_mm.mean(axis=0)
        cov = np.cov(centred.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eigvals[0] > 0:
            eccentricity = 1.0 - eigvals[-1] / eigvals[0]
        else:
            eccentricity = 0.0
    else:
        eccentricity = nan
    result[f"{prefix}_eccentricity"] = eccentricity

    # ── Centroid offset (concentric vs eccentric filling) ─────────────────
    # Distance between the PE centroid and the lumen centroid, normalized
    # by the mean lumen radius.  Low = concentric (acute), high = eccentric
    # (chronic / mural).
    lumen_coords_mm = np.argwhere(lumen_roi > 0).astype(np.float64) * spacing
    if len(pe_coords_mm) >= 1 and len(lumen_coords_mm) >= 1:
        pe_centroid = pe_coords_mm.mean(axis=0)
        lumen_centroid = lumen_coords_mm.mean(axis=0)
        offset_mm = float(np.linalg.norm(pe_centroid - lumen_centroid))
        # Normalize by max lumen depth (= vessel center depth ≈ radius)
        max_depth = float(np.max(dist_to_wall[lumen_roi > 0])) if np.any(lumen_roi > 0) else 0.0
        if max_depth > 0:
            centroid_offset_norm = offset_mm / max_depth
        else:
            centroid_offset_norm = nan
    else:
        offset_mm = nan
        centroid_offset_norm = nan
    result[f"{prefix}_centroid_offset_mm"] = offset_mm
    result[f"{prefix}_centroid_offset_norm"] = centroid_offset_norm

    # ── Cross-sectional coverage ──────────────────────────────────────────
    # For each unique z-slice with PE, compute PE_area / lumen_area.
    # Reports mean and max cross-sectional occlusion. This distinguishes
    # concentric filling (high coverage) from eccentric crescent (low).
    pe_z_slices = np.unique(np.argwhere(pe_roi > 0)[:, 0])
    xs_coverages = []
    for z in pe_z_slices:
        lumen_area = int(np.sum(lumen_roi[z] > 0))
        pe_area = int(np.sum(pe_roi[z] > 0))
        if lumen_area > 0:
            xs_coverages.append(pe_area / lumen_area)
    if xs_coverages:
        result[f"{prefix}_xs_coverage_mean"] = float(np.mean(xs_coverages))
        result[f"{prefix}_xs_coverage_max"] = float(np.max(xs_coverages))
    else:
        result[f"{prefix}_xs_coverage_mean"] = nan
        result[f"{prefix}_xs_coverage_max"] = nan

    # ── Composite acute / chronic classification ──────────────────────────
    # Acute PE: concentric/central filling, compact, moderate-to-high
    #           cross-sectional coverage, low centroid offset.
    # Chronic PE: eccentric, elongated, hugs the wall, low XS coverage.
    # Thresholds calibrated for synthetic PE in subsegmental–lobar vessels.
    xs_mean = result.get(f"{prefix}_xs_coverage_mean", nan)
    wall_adh = result.get(f"{prefix}_wall_adherence_frac", nan)
    centrality_val = result.get(f"{prefix}_centrality", nan)

    if any(np.isnan(x) for x in [eccentricity, wall_adh, xs_mean]):
        result[f"{prefix}_shape_category"] = "indeterminate"
    elif wall_adh <= 0.35 and centrality_val >= 0.15:
        result[f"{prefix}_shape_category"] = "acute"
    elif wall_adh >= 0.65 or centrality_val <= 0.06:
        result[f"{prefix}_shape_category"] = "chronic"
    else:
        result[f"{prefix}_shape_category"] = "indeterminate"


# ── Per-config evaluation ────────────────────────────────────────────────────

def evaluate_config_vessel_overlay(
    label_path: str,
    pe_seg_path: str,
    config_meta: Dict,
    tree: cKDTree,
    edge_ids: np.ndarray,
    edge_list: list,
    spacing: np.ndarray,
    ref_affine: np.ndarray,
    radius_map: Optional[np.ndarray] = None,
) -> Dict:
    """Compute vessel-overlay metrics for one counterfactual config.

    Args:
        label_path: Path to conditioning label NIfTI (PE=121, vessel=119).
        pe_seg_path: Path to detected PE segmentation NIfTI.
        config_meta: Config metadata dict from grid JSON.
        tree: KD-tree over skeleton voxel coords (mm).
        edge_ids: Edge index for each skeleton point in the tree.
        edge_list: List of (u, v, data) edge tuples.
        spacing: Voxel spacing in mm (original space).
        ref_affine: Affine from the original-space volumes (vessel graph space).
        radius_map: Optional per-voxel radius volume.

    Returns:
        Dict with all vessel-overlay metrics.
    """
    result = {}

    # Copy metadata
    for key in ["config_id", "patient_id", "location", "occlusion", "burden",
                "severity_category", "qanadli_score", "n_segments_affected",
                "target_lobes", "laterality"]:
        result[key] = config_meta.get(key, "")
    tl = result.get("target_lobes", "")
    if isinstance(tl, list):
        result["target_lobes"] = ";".join(tl)

    # Load volumes and align to original coordinate space (vessel graph space)
    label_data, label_affine, label_header = load_nifti(label_path)
    pe_seg_data, pe_seg_affine, _ = load_nifti(pe_seg_path)

    # Shape alignment (same logic as run_phase6): the prediction is on the generated CT
    # grid. Put the label on that grid with the generator's slice map, in the stored
    # orientation (before any flip). Positions on that grid use the graph spacing with
    # z scaled by z_label / z_seg.
    # Two kinds of output:
    #   - properties of the intended mask alone (intended_*, vessel_voxels,
    #     n_edges_with_intended_pe): full label grid, graph spacing;
    #   - everything that uses the prediction (detected_*, edge detection): generated grid.
    if label_data.shape[:2] != pe_seg_data.shape[:2]:
        result["error"] = "XY shape mismatch"
        return result
    z_label, z_seg = label_data.shape[2], pe_seg_data.shape[2]
    label_full = label_data                                              # label grid
    label_data = label_data[:, :, generator_slice_map(z_label, z_seg)]   # generated grid
    seg_spacing = np.array([spacing[0], spacing[1], spacing[2] * z_label / z_seg], dtype=np.float64)

    if ref_affine is not None:
        if z_label != z_seg and np.sign(label_affine[2, 2]) != np.sign(ref_affine[2, 2]):
            raise ValueError("label and prediction grids cannot be related: slice axis flipped "
                             "relative to the vessel graph and resized (not supported)")
        label_full = _align_to_reference(label_full, label_affine, ref_affine)
        label_data = _align_to_reference(label_data, label_affine, ref_affine)
        pe_seg_data = _align_to_reference(pe_seg_data, pe_seg_affine, ref_affine)

    vol_spacing = tuple(float(s) for s in label_header.get_zooms()[:3])

    # Extract masks: intended PE and vessels on the label grid; the same two masks and
    # the detected PE on the generated grid (suffix _gen)
    intended_pe = (label_full == PE_LABEL).astype(np.uint8)
    vessel_mask = (label_full == VESSEL_LABEL).astype(np.uint8)
    intended_pe_gen = (label_data == PE_LABEL).astype(np.uint8)
    vessel_mask_gen = (label_data == VESSEL_LABEL).astype(np.uint8)
    detected_pe = (pe_seg_data > 0).astype(np.uint8)

    intended_voxels = int(np.sum(intended_pe))
    detected_voxels = int(np.sum(detected_pe))
    vessel_voxels = int(np.sum(vessel_mask))
    result["intended_voxels"] = intended_voxels
    result["detected_voxels"] = detected_voxels
    result["vessel_voxels"] = vessel_voxels
    result["pe_detected"] = detected_voxels > 0

    if int(np.sum(intended_pe_gen)) == 0:
        result["error"] = "no_intended_pe"
        return result

    # ── Vessel containment (label-based) ──────────────────────────────────
    # Fraction of PE voxels that sit on vessel label (119) voxels
    intended_on_vessel = int(np.sum((intended_pe > 0) & (vessel_mask > 0)))
    detected_on_vessel = int(np.sum((detected_pe > 0) & (vessel_mask_gen > 0)))

    result["intended_on_vessel_voxels"] = intended_on_vessel
    result["intended_on_vessel_frac"] = intended_on_vessel / intended_voxels
    result["detected_on_vessel_voxels"] = detected_on_vessel
    result["detected_on_vessel_frac"] = (
        detected_on_vessel / detected_voxels if detected_voxels > 0 else float("nan")
    )

    # ── Map PE voxels to vessel edges via KD-tree ─────────────────────────
    if tree is None or len(edge_ids) == 0:
        result["error"] = "no_vessel_graph"
        return result

    def _map_pe_to_edges(pe_mask, grid_spacing):
        """Map PE voxels to nearest vessel edges. Returns (edge_indices, distances)."""
        coords = np.argwhere(pe_mask > 0)  # (N, 3)
        if len(coords) == 0:
            return np.array([], dtype=np.int32), np.array([])
        coords_mm = coords.astype(np.float64) * grid_spacing
        dists, tree_idx = tree.query(coords_mm)
        mapped_edge = edge_ids[tree_idx]
        return mapped_edge, dists

    # --- Intended PE → edges ---
    intended_edges, intended_dists = _map_pe_to_edges(intended_pe, np.asarray(spacing, dtype=np.float64))
    intended_edges_gen, _ = _map_pe_to_edges(intended_pe_gen, seg_spacing)   # for edge detection

    # --- Detected PE → edges ---
    detected_edges, detected_dists = _map_pe_to_edges(detected_pe, seg_spacing)

    # ── Distance statistics ───────────────────────────────────────────────
    if len(intended_dists) > 0:
        result["intended_dist_mean_mm"] = float(np.mean(intended_dists))
        result["intended_dist_median_mm"] = float(np.median(intended_dists))
    else:
        result["intended_dist_mean_mm"] = float("nan")
        result["intended_dist_median_mm"] = float("nan")

    if len(detected_dists) > 0:
        result["detected_dist_mean_mm"] = float(np.mean(detected_dists))
        result["detected_dist_median_mm"] = float(np.median(detected_dists))
        # Fraction within max distance
        result["detected_near_vessel_frac"] = float(
            np.mean(detected_dists <= MAX_VESSEL_DIST_MM)
        )
    else:
        result["detected_dist_mean_mm"] = float("nan")
        result["detected_dist_median_mm"] = float("nan")
        result["detected_near_vessel_frac"] = float("nan")

    # ── Per-generation breakdown ──────────────────────────────────────────
    def _generation_breakdown(mapped_edges, prefix):
        """Count PE voxels mapped to each vessel generation."""
        gen_counts = {g: 0 for g in GENERATION_NAMES}
        for eidx in mapped_edges:
            _, _, d = edge_list[eidx]
            gn = d.get("generation_name", "unknown")
            if gn in gen_counts:
                gen_counts[gn] += 1
        total = len(mapped_edges) if len(mapped_edges) > 0 else 1
        for g in GENERATION_NAMES:
            result[f"{prefix}_gen_{g}_voxels"] = gen_counts[g]
            result[f"{prefix}_gen_{g}_frac"] = gen_counts[g] / total

    _generation_breakdown(intended_edges, "intended")
    _generation_breakdown(detected_edges, "detected")

    # ── Per-vessel-type breakdown ─────────────────────────────────────────
    def _vessel_type_breakdown(mapped_edges, prefix):
        """Count PE voxels mapped to each vessel type."""
        vtype_counts = {vt: 0 for vt in VESSEL_TYPES}
        for eidx in mapped_edges:
            _, _, d = edge_list[eidx]
            vt = d.get("vessel_type_str", "unknown")
            if vt in vtype_counts:
                vtype_counts[vt] += 1
            else:
                vtype_counts["unknown"] += 1
        total = len(mapped_edges) if len(mapped_edges) > 0 else 1
        for vt in VESSEL_TYPES:
            result[f"{prefix}_vtype_{vt}_voxels"] = vtype_counts[vt]
            result[f"{prefix}_vtype_{vt}_frac"] = vtype_counts[vt] / total

    _vessel_type_breakdown(intended_edges, "intended")
    _vessel_type_breakdown(detected_edges, "detected")

    # ── Per-lobe breakdown (vessel-based) ─────────────────────────────────
    def _lobe_breakdown(mapped_edges, prefix):
        """Count PE voxels mapped to each lobe via vessel assignment."""
        lobe_counts = {l: 0 for l in LOBE_NAMES}
        for eidx in mapped_edges:
            _, _, d = edge_list[eidx]
            ln = d.get("lobe_name", "")
            if ln in lobe_counts:
                lobe_counts[ln] += 1
        total = len(mapped_edges) if len(mapped_edges) > 0 else 1
        for l in LOBE_NAMES:
            result[f"{prefix}_lobe_{l}_voxels"] = lobe_counts[l]
            result[f"{prefix}_lobe_{l}_frac"] = lobe_counts[l] / total

    _lobe_breakdown(intended_edges, "intended")
    _lobe_breakdown(detected_edges, "detected")

    # ── Vessel radius at PE sites ─────────────────────────────────────────
    def _radius_stats(mapped_edges, prefix):
        """Compute mean radius of vessel edges carrying PE."""
        radii = []
        for eidx in mapped_edges:
            _, _, d = edge_list[eidx]
            r = d.get("radius_mean", float("nan"))
            if not np.isnan(r):
                radii.append(r)
        if radii:
            result[f"{prefix}_radius_mean_mm"] = float(np.mean(radii))
            result[f"{prefix}_radius_median_mm"] = float(np.median(radii))
            result[f"{prefix}_radius_std_mm"] = float(np.std(radii))
        else:
            result[f"{prefix}_radius_mean_mm"] = float("nan")
            result[f"{prefix}_radius_median_mm"] = float("nan")
            result[f"{prefix}_radius_std_mm"] = float("nan")

    _radius_stats(intended_edges, "intended")
    _radius_stats(detected_edges, "detected")

    # ── Edge-level detection ──────────────────────────────────────────────
    # Which unique edges carry intended PE? For each, was any detected PE
    # also mapped to the same edge?
    # (n_edges_with_intended_pe: label grid; the comparison with the detected edges: generated grid)
    intended_edge_set = set(intended_edges_gen.tolist()) if len(intended_edges_gen) > 0 else set()
    detected_edge_set = set(detected_edges.tolist()) if len(detected_edges) > 0 else set()

    result["n_edges_with_intended_pe"] = len(set(intended_edges.tolist()))
    result["n_edges_with_detected_pe"] = len(detected_edge_set)
    edges_both = intended_edge_set & detected_edge_set
    result["n_edges_intended_and_detected"] = len(edges_both)
    result["edge_detection_rate"] = (
        len(edges_both) / len(intended_edge_set)
        if intended_edge_set else float("nan")
    )

    # ── Target-edge analysis ──────────────────────────────────────────────
    # From config metadata, find which edges were explicitly targeted
    target_segments = config_meta.get("segments", [])
    target_edge_keys = set()
    for seg in target_segments:
        ek = seg.get("edge_key")
        if ek:
            target_edge_keys.add(tuple(ek))

    # Map edge_key → edge_list index
    target_edge_indices = set()
    for idx, (u, v, d) in enumerate(edge_list):
        if (u, v) in target_edge_keys or (v, u) in target_edge_keys:
            target_edge_indices.add(idx)

    result["n_target_edges"] = len(target_edge_indices)
    target_detected = target_edge_indices & detected_edge_set
    result["n_target_edges_detected"] = len(target_detected)
    result["target_edge_detection_rate"] = (
        len(target_detected) / len(target_edge_indices)
        if target_edge_indices else float("nan")
    )

    # ── Occlusion and shape/morphology analysis ─────────────────────────
    # Intended properties from config metadata
    target_segments = config_meta.get("segments", [])
    intended_acute = any(seg.get("acute", False) for seg in target_segments)
    result["intended_occlusion"] = config_meta.get("occlusion", "")
    result["intended_acute"] = intended_acute

    # True vessel lumen = vessel (119) + intended PE (121), since PE
    # replaces vessel voxels in the conditioning label.
    true_lumen = ((vessel_mask > 0) | (intended_pe > 0)).astype(np.uint8)              # label grid
    true_lumen_gen = ((vessel_mask_gen > 0) | (intended_pe_gen > 0)).astype(np.uint8)  # generated grid

    _compute_morphology(result, detected_pe, true_lumen_gen, seg_spacing,
                        detected_edges, edge_list, prefix="detected")
    _compute_morphology(result, intended_pe, true_lumen, np.asarray(spacing, dtype=np.float64),
                        intended_edges, edge_list, prefix="intended")

    result["label_path"] = label_path
    result["pe_seg_path"] = pe_seg_path

    return result


# ── CSV columns ──────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "patient_id", "config_id", "location", "occlusion", "burden",
    "severity_category", "qanadli_score", "n_segments_affected",
    "target_lobes", "laterality",
    "intended_voxels", "detected_voxels", "vessel_voxels", "pe_detected",
    # Vessel containment (label-based)
    "intended_on_vessel_voxels", "intended_on_vessel_frac",
    "detected_on_vessel_voxels", "detected_on_vessel_frac",
    # Distance to nearest vessel skeleton
    "intended_dist_mean_mm", "intended_dist_median_mm",
    "detected_dist_mean_mm", "detected_dist_median_mm",
    "detected_near_vessel_frac",
    # Per-generation (intended)
    "intended_gen_central_voxels", "intended_gen_central_frac",
    "intended_gen_lobar_voxels", "intended_gen_lobar_frac",
    "intended_gen_segmental_voxels", "intended_gen_segmental_frac",
    "intended_gen_subsegmental_voxels", "intended_gen_subsegmental_frac",
    # Per-generation (detected)
    "detected_gen_central_voxels", "detected_gen_central_frac",
    "detected_gen_lobar_voxels", "detected_gen_lobar_frac",
    "detected_gen_segmental_voxels", "detected_gen_segmental_frac",
    "detected_gen_subsegmental_voxels", "detected_gen_subsegmental_frac",
    # Per-vessel-type (intended)
    "intended_vtype_likely_artery_voxels", "intended_vtype_likely_artery_frac",
    "intended_vtype_likely_vein_voxels", "intended_vtype_likely_vein_frac",
    "intended_vtype_unknown_voxels", "intended_vtype_unknown_frac",
    # Per-vessel-type (detected)
    "detected_vtype_likely_artery_voxels", "detected_vtype_likely_artery_frac",
    "detected_vtype_likely_vein_voxels", "detected_vtype_likely_vein_frac",
    "detected_vtype_unknown_voxels", "detected_vtype_unknown_frac",
    # Per-lobe (intended)
    "intended_lobe_LUL_voxels", "intended_lobe_LUL_frac",
    "intended_lobe_LLL_voxels", "intended_lobe_LLL_frac",
    "intended_lobe_RUL_voxels", "intended_lobe_RUL_frac",
    "intended_lobe_RML_voxels", "intended_lobe_RML_frac",
    "intended_lobe_RLL_voxels", "intended_lobe_RLL_frac",
    # Per-lobe (detected)
    "detected_lobe_LUL_voxels", "detected_lobe_LUL_frac",
    "detected_lobe_LLL_voxels", "detected_lobe_LLL_frac",
    "detected_lobe_RUL_voxels", "detected_lobe_RUL_frac",
    "detected_lobe_RML_voxels", "detected_lobe_RML_frac",
    "detected_lobe_RLL_voxels", "detected_lobe_RLL_frac",
    # Radius at PE
    "intended_radius_mean_mm", "intended_radius_median_mm", "intended_radius_std_mm",
    "detected_radius_mean_mm", "detected_radius_median_mm", "detected_radius_std_mm",
    # Edge-level detection
    "n_edges_with_intended_pe", "n_edges_with_detected_pe",
    "n_edges_intended_and_detected", "edge_detection_rate",
    "n_target_edges", "n_target_edges_detected", "target_edge_detection_rate",
    # Occlusion & shape - intended (from metadata + measured)
    "intended_occlusion", "intended_acute",
    "intended_occlusion_frac", "intended_occlusion_cat",
    "intended_wall_dist_mean_mm", "intended_wall_adherence_frac",
    "intended_centrality", "intended_compactness", "intended_eccentricity",
    "intended_centroid_offset_mm", "intended_centroid_offset_norm",
    "intended_xs_coverage_mean", "intended_xs_coverage_max",
    "intended_shape_category",
    # Occlusion & shape - detected (measured)
    "detected_occlusion_frac", "detected_occlusion_cat",
    "detected_wall_dist_mean_mm", "detected_wall_adherence_frac",
    "detected_centrality", "detected_compactness", "detected_eccentricity",
    "detected_centroid_offset_mm", "detected_centroid_offset_norm",
    "detected_xs_coverage_mean", "detected_xs_coverage_max",
    "detected_shape_category",
    # Paths
    "label_path", "pe_seg_path", "error",
]


# ── Patient-level driver ─────────────────────────────────────────────────────

def evaluate_patient(patient_id: str, file_index: List[Dict],
                     output_dir: Path) -> List[Dict]:
    """Run vessel overlay analysis for all configs of one patient."""
    patient_entries = [e for e in file_index if e["patient_id"] == patient_id]
    if not patient_entries:
        print(f"  No entries for {patient_id}")
        return []

    print(f"\nVessel overlay: {patient_id} ({len(patient_entries)} configs)",
          flush=True)

    # Load vessel graph once per patient
    G, graph_spacing = load_vessel_graph(patient_id)
    if G is None:
        print(f"  SKIP {patient_id}: no vessel graph", flush=True)
        return []

    tree, edge_ids, edge_list = build_edge_kdtree(G, graph_spacing)
    radius_map = load_radius_map(patient_id)
    ref_affine = load_reference_affine(patient_id)

    results = []
    for i, entry in enumerate(patient_entries):
        config_id = entry["config_id"]
        label_path = entry["label_path"]
        pe_seg_path = entry["pe_seg_path"]

        if not Path(label_path).exists():
            print(f"  SKIP {config_id}: missing label", flush=True)
            results.append({"config_id": config_id, "patient_id": patient_id,
                            "error": "missing_label"})
            continue
        if not Path(pe_seg_path).exists():
            print(f"  SKIP {config_id}: missing pe_seg", flush=True)
            results.append({"config_id": config_id, "patient_id": patient_id,
                            "error": "missing_pe_seg"})
            continue

        config_meta = load_config_metadata(patient_id, config_id)
        if not config_meta:
            config_meta = {"config_id": config_id, "patient_id": patient_id}

        try:
            result = evaluate_config_vessel_overlay(
                label_path, pe_seg_path, config_meta,
                tree, edge_ids, edge_list, graph_spacing,
                ref_affine, radius_map,
            )
            det_str = "YES" if result.get("pe_detected") else "no"
            on_v = result.get("detected_on_vessel_frac", float("nan"))
            on_v_str = f"{on_v:.2f}" if not np.isnan(on_v) else "N/A"
            print(f"  [{i+1}/{len(patient_entries)}] {config_id}: "
                  f"det={det_str}, on_vessel={on_v_str}, "
                  f"edge_det_rate={result.get('edge_detection_rate', 'N/A')}",
                  flush=True)
        except Exception as e:
            print(f"  ERROR {config_id}: {e}", flush=True)
            result = {"config_id": config_id, "patient_id": patient_id,
                      "error": str(e)}

        results.append(result)

    # Write per-patient CSV
    per_config_dir = output_dir / "per_config"
    per_config_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_config_dir / f"{patient_id}_vessel_overlay.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"  Wrote {csv_path} ({len(results)} rows)")
    return results


# ── Aggregation ──────────────────────────────────────────────────────────────

def compute_aggregates(output_dir: Path):
    """Concatenate per-patient CSVs and compute summary statistics."""
    per_config_dir = output_dir / "per_config"
    csv_files = sorted(per_config_dir.glob("*_vessel_overlay.csv"))
    if not csv_files:
        print("No per-patient vessel overlay CSVs found.")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    all_df = pd.concat(dfs, ignore_index=True)
    all_path = output_dir / "vessel_overlay_all_configs.csv"
    all_df.to_csv(all_path, index=False)
    print(f"\nConcatenated {len(all_df)} configs → {all_path}")

    # Filter to valid configs
    valid = all_df[all_df["intended_voxels"] > 0].copy()
    det = valid[valid["pe_detected"]].copy()
    print(f"  Valid: {len(valid)}, Detected: {len(det)}")

    # Summary
    lines = []
    lines.append("=" * 70)
    lines.append("VESSEL OVERLAY ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total configs: {len(valid)}")
    lines.append(f"Configs with detected PE: {len(det)}")
    lines.append("")

    # Vessel containment
    lines.append("Vessel containment (detected PE on vessel label):")
    on_v = det["detected_on_vessel_frac"].dropna()
    lines.append(f"  Mean:   {on_v.mean():.3f}")
    lines.append(f"  Median: {on_v.median():.3f}")
    lines.append(f"  Std:    {on_v.std():.3f}")
    lines.append("")

    # Distance to nearest vessel
    lines.append("Distance from detected PE to nearest vessel skeleton (mm):")
    dd = det["detected_dist_mean_mm"].dropna()
    lines.append(f"  Mean of means:   {dd.mean():.2f}")
    lines.append(f"  Median of means: {dd.median():.2f}")
    lines.append("")

    # Near-vessel fraction
    nv = det["detected_near_vessel_frac"].dropna()
    lines.append(f"Fraction of detected PE within {MAX_VESSEL_DIST_MM}mm of vessel:")
    lines.append(f"  Mean:   {nv.mean():.3f}")
    lines.append(f"  Median: {nv.median():.3f}")
    lines.append("")

    # Per-generation detected PE distribution (mean fractions)
    lines.append("Detected PE distribution by vessel generation (mean frac):")
    for g in GENERATION_NAMES:
        col = det[f"detected_gen_{g}_frac"].dropna()
        lines.append(f"  {g:15s}  {col.mean():.3f}")
    lines.append("")

    # Per-vessel-type
    lines.append("Detected PE distribution by vessel type (mean frac):")
    for vt in VESSEL_TYPES:
        col = det[f"detected_vtype_{vt}_frac"].dropna()
        lines.append(f"  {vt:15s}  {col.mean():.3f}")
    lines.append("")

    # Radius at PE
    lines.append("Vessel radius at detected PE sites (mm):")
    rd = det["detected_radius_mean_mm"].dropna()
    lines.append(f"  Mean of means:   {rd.mean():.2f}")
    lines.append(f"  Median of means: {rd.median():.2f}")
    lines.append("")

    # Edge-level detection
    lines.append("Edge-level detection:")
    edr = det["edge_detection_rate"].dropna()
    lines.append(f"  Edge detection rate (intended→detected):  mean={edr.mean():.3f}  median={edr.median():.3f}")
    tedr = det["target_edge_detection_rate"].dropna()
    lines.append(f"  Target edge detection rate:               mean={tedr.mean():.3f}  median={tedr.median():.3f}")
    lines.append("")

    # Occlusion
    lines.append("Detected PE occlusion:")
    occ = det["detected_occlusion_frac"].dropna()
    lines.append(f"  Mean occlusion fraction: {occ.mean():.3f}")
    lines.append(f"  Median:                  {occ.median():.3f}")
    occ_cat = det["detected_occlusion_cat"].value_counts()
    for cat in ["complete", "near_complete", "partial"]:
        n = occ_cat.get(cat, 0)
        lines.append(f"  {cat:15s}  n={n}  ({n/len(det)*100:.1f}%)")
    lines.append("")

    # Occlusion concordance: intended vs detected category
    lines.append("Occlusion concordance (intended → detected category):")
    occ_match = det.apply(
        lambda r: r.get("intended_occlusion", "") == r.get("detected_occlusion_cat", ""),
        axis=1)
    lines.append(f"  Exact match rate: {occ_match.mean():.3f}  ({occ_match.sum()}/{len(det)})")
    lines.append("")

    # Shape / acute vs chronic
    lines.append("Detected PE shape (acute / chronic):")
    shape_cat = det["detected_shape_category"].value_counts()
    for cat in ["acute", "chronic", "indeterminate"]:
        n = shape_cat.get(cat, 0)
        lines.append(f"  {cat:15s}  n={n}  ({n/len(det)*100:.1f}%)")
    wall_adh = det["detected_wall_adherence_frac"].dropna()
    lines.append(f"  Wall adherence (mean):  {wall_adh.mean():.3f}")
    centrality = det["detected_centrality"].dropna()
    lines.append(f"  Centrality (mean):      {centrality.mean():.3f}")
    ecc = det["detected_eccentricity"].dropna()
    lines.append(f"  Eccentricity (mean):    {ecc.mean():.3f}")
    comp = det["detected_compactness"].dropna()
    lines.append(f"  Compactness (mean):     {comp.mean():.3f}")
    lines.append("")

    # Shape concordance: intended acute vs detected shape
    lines.append("Shape concordance (intended acute flag → detected shape):")
    for acute_val in [True, False]:
        sub = det[det["intended_acute"] == acute_val]
        if len(sub) > 0:
            label = "acute" if acute_val else "chronic"
            sc = sub["detected_shape_category"].value_counts()
            parts = [f"{cat}={sc.get(cat, 0)}" for cat in ["acute", "chronic", "indeterminate"]]
            lines.append(f"  Intended {label:8s} (n={len(sub)}): {', '.join(parts)}")
    lines.append("")

    # By location
    lines.append("Vessel containment by PE location:")
    for loc in ["central", "lobar", "segmental", "subsegmental"]:
        sub = det[det["location"] == loc]
        if len(sub) > 0:
            ov = sub["detected_on_vessel_frac"].dropna()
            lines.append(f"  {loc:15s}  mean={ov.mean():.3f}  n={len(sub)}")
    lines.append("")

    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = output_dir / "vessel_overlay_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved to {summary_path}")

    # Aggregate CSV grouped by location/severity/etc
    group_vars = ["location", "occlusion", "burden", "severity_category"]
    metric_cols = [
        "detected_on_vessel_frac", "detected_near_vessel_frac",
        "detected_dist_mean_mm", "detected_radius_mean_mm",
        "edge_detection_rate", "target_edge_detection_rate",
        "detected_occlusion_frac", "detected_wall_adherence_frac",
        "detected_centrality", "detected_compactness", "detected_eccentricity",
    ]
    agg_rows = []
    for gvar in group_vars:
        for gval, gdf in det.groupby(gvar, observed=True):
            row = {"group_variable": gvar, "group_value": str(gval),
                   "n_configs": len(gdf)}
            for mc in metric_cols:
                col = pd.to_numeric(gdf[mc], errors="coerce")
                row[f"{mc}_mean"] = col.mean()
                row[f"{mc}_median"] = col.median()
            agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_path = output_dir / "vessel_overlay_aggregate.csv"
    agg_df.to_csv(agg_path, index=False, float_format="%.4f")
    print(f"  Aggregate → {agg_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vessel overlay analysis for counterfactual PE segmentations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient-id", type=str, help="Evaluate one patient")
    group.add_argument("--all", action="store_true", help="Evaluate all patients")
    group.add_argument("--aggregate-only", action="store_true",
                       help="Aggregate existing per-patient CSVs")

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--predictions-csv", type=str, default=None)
    parser.add_argument("--grid-base-dir", type=str, default=None)
    parser.add_argument("--apes-seg-dir", type=str, default=None,
                        help="Path to APES segmentation directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else PREDICTIONS_CSV
    apes_seg_dir = Path(args.apes_seg_dir) if args.apes_seg_dir else APES_SEG_DIR

    global GRID_BASE_DIR
    if args.grid_base_dir:
        GRID_BASE_DIR = Path(args.grid_base_dir)

    if args.aggregate_only:
        compute_aggregates(output_dir)
        return

    file_index = build_file_index(predictions_csv,
                                  patient_id=args.patient_id if args.patient_id else None,
                                  apes_seg_dir=apes_seg_dir)
    print(f"File index: {len(file_index)} configs")

    if args.patient_id:
        evaluate_patient(args.patient_id, file_index, output_dir)
    else:
        patient_ids = sorted(set(e["patient_id"] for e in file_index))
        print(f"Processing {len(patient_ids)} patients")
        for pid in patient_ids:
            evaluate_patient(pid, file_index, output_dir)
        compute_aggregates(output_dir)


if __name__ == "__main__":
    main()
