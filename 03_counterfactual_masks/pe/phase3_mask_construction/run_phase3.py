"""Phase 3: PE Mask Construction - Orchestrator.

Reads a patient's vessel graph (Phase 1) and counterfactual grid (Phase 2),
then constructs a binary PE segmentation mask for each configuration.

Usage:
    python phase3_mask_construction/run_phase3.py <STUDY_ID>
    python phase3_mask_construction/run_phase3.py <STUDY_ID> --config-ids cf001 cf005 cf021
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)
# Phase 1 modules must be importable for pickle deserialization (VesselType enum)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "phase1_vessel_tree"))

from utils.nifti_io import load_nifti, save_nifti
from phase3_mask_construction.pe_placement import (
    compute_full_distance_transform,
    extract_segment_lumen,
    extract_cross_sections,
)
from phase3_mask_construction.occlusion_patterns import apply_occlusion
from phase3_mask_construction.taper_and_smooth import compute_taper_weights, smooth_and_clean
from phase3_mask_construction.merge_with_anatomy import save_combined_segmentation

# Default paths
PHASE1_OUTPUT = Path(PROJECT_ROOT) / "phase1_vessel_tree" / "output"
PHASE2_OUTPUT = Path(PROJECT_ROOT) / "phase2_counterfactuals" / "output"
PHASE3_OUTPUT = Path(__file__).resolve().parent / "output"


def build_pe_mask_for_config(config: dict,
                              G,
                              vessel_mask: np.ndarray,
                              dt: np.ndarray,
                              spacing: np.ndarray,
                              affine: np.ndarray,
                              header,
                              output_dir: Path,
                              rng: np.random.Generator) -> dict:
    """Build a PE mask for one counterfactual configuration.

    Args:
        config: Config dict from the Phase 2 JSON grid.
        G: NetworkX vessel graph.
        vessel_mask: Binary vessel mask.
        dt: Full distance transform (mm).
        spacing: Voxel spacing (z, y, x) in mm.
        affine: NIfTI affine matrix.
        header: NIfTI header.
        output_dir: Where to save the PE mask.
        rng: Random generator (seeded per config).

    Returns:
        Dict with mask-construction metrics.
    """
    config_id = config['config_id']
    segments = config['segments']

    pe_mask = np.zeros_like(vessel_mask, dtype=np.uint8)
    total_lumen_voxels = 0
    total_filled_voxels = 0

    # Build edge lookup (both directions)
    edge_lookup = {}
    for u, v, data in G.edges(data=True):
        edge_lookup[(u, v)] = data
        edge_lookup[(v, u)] = data

    for seg_idx, seg in enumerate(segments):
        edge_key = tuple(seg['edge_key'])
        occlusion = seg['occlusion']
        acute = seg.get('acute', False)

        edge_data = edge_lookup.get(edge_key)
        if edge_data is None:
            print(f"    WARNING: edge {edge_key} not found in graph - skipping")
            continue

        segment_coords = edge_data['segment_coords']
        if not segment_coords:
            continue

        # 1. Extract lumen voxels
        lumen_voxels = extract_segment_lumen(segment_coords, vessel_mask, dt, spacing)
        total_lumen_voxels += len(lumen_voxels)

        if not lumen_voxels:
            continue

        # 2. Group into cross-sections
        cross_sections = extract_cross_sections(segment_coords, lumen_voxels, spacing)

        # 3. Compute taper weights
        taper_weights = compute_taper_weights(len(segment_coords), taper_length=3)

        # 4. Apply occlusion pattern
        seg_rng = np.random.default_rng(
            seed=(hash(edge_key) % 2**32) ^ (rng.integers(2**31))
        )
        filled_voxels = apply_occlusion(
            cross_sections, segment_coords, lumen_voxels,
            occlusion, spacing, seg_rng, taper_weights,
            acute=acute,
        )
        total_filled_voxels += len(filled_voxels)

        # 5. Write to PE mask
        for vz, vy, vx in filled_voxels:
            pe_mask[vz, vy, vx] = 1

    # 6. Smooth and clean
    pe_mask = smooth_and_clean(pe_mask, vessel_mask, sigma=0.8, threshold=0.3)

    # 7. Compute metrics
    pe_voxels = int(np.sum(pe_mask))
    voxel_vol_ml = float(np.prod(spacing)) / 1000.0  # mm^3 → ml
    pe_volume_ml = pe_voxels * voxel_vol_ml

    # Containment check
    vessel_overlap = int(np.sum(pe_mask & (vessel_mask > 0)))
    containment = vessel_overlap / pe_voxels if pe_voxels > 0 else 1.0

    # 8. Save PE-only mask
    output_path = output_dir / f"{config_id}_pe_mask.nii.gz"
    save_nifti(pe_mask, affine, header, str(output_path))

    # 9. Save combined segmentation (vessels as label 119, PE as label 121)
    patient_id = config['patient_id']
    combined_path = save_combined_segmentation(
        pe_mask, patient_id, config_id, str(output_dir),
        vessel_mask=vessel_mask,
    )

    is_acute = any(s.get('acute', False) for s in segments)
    metrics = {
        'config_id': config_id,
        'location': config['location'],
        'occlusion': config['occlusion'],
        'burden': config['burden'],
        'acute': is_acute,
        'severity_category': config['severity_category'],
        'qanadli_score': config['qanadli_score'],
        'n_segments': len(segments),
        'total_lumen_voxels': total_lumen_voxels,
        'total_filled_pre_smooth': total_filled_voxels,
        'pe_voxels': pe_voxels,
        'pe_volume_ml': round(pe_volume_ml, 4),
        'containment_fraction': round(containment, 4),
        'pe_mask_path': str(output_path),
        'combined_seg_path': combined_path,
    }

    return metrics


def run_phase3_for_patient(patient_id: str,
                            config_ids: Optional[List[str]] = None,
                            output_dir: str = None,
                            seed: int = None,
                            phase1_output_dir: str = None,
                            phase2_output_dir: str = None):
    """Run Phase 3 for all (or selected) configs of a patient.

    Args:
        patient_id: Patient ID (e.g., <STUDY_ID>).
        config_ids: Optional list of config ID suffixes to process
                    (e.g., ['cf001', 'cf005']). None = all configs.
        output_dir: Output directory. Defaults to phase3_mask_construction/output/.
        seed: Random seed. Defaults to hash(patient_id), which changes per process
              unless PYTHONHASHSEED is set.
        phase1_output_dir: Where to find Phase 1 outputs. Defaults to phase1_vessel_tree/output/.
        phase2_output_dir: Where to find Phase 2 outputs. Defaults to phase2_counterfactuals/output/.
    """
    if output_dir is None:
        output_dir = PHASE3_OUTPUT
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p2_dir = Path(phase1_output_dir) if phase1_output_dir else PHASE1_OUTPUT
    p3_dir = Path(phase2_output_dir) if phase2_output_dir else PHASE2_OUTPUT

    # ── Load shared data (once per patient) ──────────────────────────────
    print(f"Loading data for {patient_id}...")

    # Vessel mask
    vessel_path = p2_dir / f"{patient_id}_vessels_cleaned.nii.gz"
    print(f"  Vessel mask: {vessel_path}")
    vessel_mask, affine, header = load_nifti(str(vessel_path))
    vessel_mask = (vessel_mask > 0).astype(np.uint8)
    spacing = np.array(header.get_zooms()[:3], dtype=np.float64)
    print(f"  Shape: {vessel_mask.shape}, Spacing: {spacing} mm")

    # Distance transform
    print("  Computing distance transform...")
    t0 = time.time()
    dt = compute_full_distance_transform(vessel_mask, spacing)
    print(f"  Distance transform computed in {time.time()-t0:.1f}s")

    # Vessel graph
    graph_path = p2_dir / f"{patient_id}_vessel_graph_final.pkl"
    print(f"  Loading graph: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    G = graph_data['graph']
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Counterfactual grid
    grid_path = p3_dir / f"{patient_id}_counterfactual_grid.json"
    print(f"  Loading grid: {grid_path}")
    with open(grid_path) as f:
        grid_data = json.load(f)

    configs = grid_data['configs']

    # Filter configs if requested
    if config_ids:
        configs = [c for c in configs
                   if any(cid in c['config_id'] for cid in config_ids)]
        print(f"  Selected {len(configs)} configs (filter: {config_ids})")
    else:
        print(f"  Processing all {len(configs)} configs")

    # ── Build each config ───────────────────────────────────────────
    if seed is not None:
        base_rng = np.random.default_rng(seed)
    else:
        base_rng = np.random.default_rng(hash(patient_id) % 2**32)

    all_metrics = []

    for i, config in enumerate(configs):
        config_id = config['config_id']
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] Building {config_id}")
        print(f"  Location: {config['location']}, Burden: {config['burden']}, "
              f"Occlusion: {config['occlusion']}")
        print(f"  Segments: {config['n_segments_affected']}, "
              f"Qanadli: {config['qanadli_score']}")

        t0 = time.time()

        # Per-config RNG (deterministic)
        config_rng = np.random.default_rng(
            seed=base_rng.integers(2**31)
        )

        metrics = build_pe_mask_for_config(
            config, G, vessel_mask, dt, spacing, affine, header,
            output_dir, config_rng
        )

        elapsed = time.time() - t0
        print(f"  PE voxels: {metrics['pe_voxels']}, "
              f"Volume: {metrics['pe_volume_ml']:.4f} ml, "
              f"Containment: {metrics['containment_fraction']:.4f}")
        print(f"  Saved PE mask: {metrics['pe_mask_path']}")
        print(f"  Saved combined: {metrics['combined_seg_path']}")
        print(f"  Time: {elapsed:.1f}s")

        all_metrics.append(metrics)

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = output_dir / f"{patient_id}_mask_summary.csv"
    df = pd.DataFrame(all_metrics)
    df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Summary saved: {summary_path}")
    print(f"\nVolume distribution:")
    print(df[['config_id', 'location', 'burden', 'occlusion',
              'pe_voxels', 'pe_volume_ml', 'containment_fraction']].to_string(index=False))

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Build PE masks from counterfactual grid")
    parser.add_argument("patient_id", help="Patient ID (e.g., <STUDY_ID>)")
    parser.add_argument("--config-ids", nargs='+', default=None,
                        help="Specific config ID suffixes (e.g., cf001 cf005)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_phase3_for_patient(args.patient_id, args.config_ids,
                           args.output_dir, args.seed)
