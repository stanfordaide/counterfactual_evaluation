"""Phase 4: Validation - Orchestrator.

Reads Phase 3 mask-construction outputs and runs automated validation checks
plus visual QC for each counterfactual configuration.

Usage:
    python phase4_validation/run_phase4.py <STUDY_ID>
    python phase4_validation/run_phase4.py <STUDY_ID> --config-ids cf001 cf021
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import ndimage

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from utils.nifti_io import load_nifti, DATA_ROOT
from phase4_validation.automated_checks import validate_config
from phase4_validation.visual_qc import render_qc_slices, render_summary_grid

# Default paths
PHASE1_OUTPUT = Path(PROJECT_ROOT) / "phase1_vessel_tree" / "output"
PHASE3_OUTPUT = Path(PROJECT_ROOT) / "phase3_mask_construction" / "output"
PHASE4_OUTPUT = Path(__file__).resolve().parent / "output"

MERGED_SEG_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_all_lung_pe"

BASE_SEG_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_all"

VESSEL_LABEL = 119


def run_phase4_for_patient(patient_id: str,
                            config_ids: Optional[List[str]] = None,
                            output_dir: str = None,
                            phase1_output_dir: str = None,
                            phase3_output_dir: str = None):
    """Run Phase 4 validation for all (or selected) configs of a patient.

    Args:
        patient_id: Patient ID (e.g., <STUDY_ID>).
        config_ids: Optional list of config ID suffixes to process.
        output_dir: Output directory. Defaults to phase4_validation/output/.
        phase1_output_dir: Where to find Phase 1 outputs. Defaults to phase1_vessel_tree/output/.
        phase3_output_dir: Where to find Phase 3 outputs. Defaults to phase3_mask_construction/output/.
    """
    if output_dir is None:
        output_dir = PHASE4_OUTPUT
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p2_dir = Path(phase1_output_dir) if phase1_output_dir else PHASE1_OUTPUT
    p4_dir = Path(phase3_output_dir) if phase3_output_dir else PHASE3_OUTPUT

    # ── Load Phase 3 mask summary ────────────────────────────────────
    summary_path = p4_dir / f"{patient_id}_mask_summary.csv"
    print(f"Loading mask summary: {summary_path}")
    summary_df = pd.read_csv(summary_path)

    if config_ids:
        summary_df = summary_df[
            summary_df['config_id'].apply(
                lambda x: any(cid in x for cid in config_ids))
        ]
        print(f"  Selected {len(summary_df)} configs (filter: {config_ids})")
    else:
        print(f"  Validating all {len(summary_df)} configs")

    # ── Load shared data (once per patient) ───────────────────────────────
    print(f"\nLoading shared data for {patient_id}...")

    # Vessel mask
    vessel_path = p2_dir / f"{patient_id}_vessels_cleaned.nii.gz"
    print(f"  Vessel mask: {vessel_path}")
    vessel_data, _, header = load_nifti(str(vessel_path))
    vessel_mask = (vessel_data > 0).astype(np.uint8)
    spacing = np.array(header.get_zooms()[:3], dtype=np.float64)
    print(f"  Shape: {vessel_mask.shape}, Spacing: {spacing} mm")

    # Original merged segmentation (without PE)
    orig_seg_path = MERGED_SEG_DIR / f"{patient_id}_segmentation_merged.nii.gz"
    print(f"  Original segmentation: {orig_seg_path}")
    original_seg, _, _ = load_nifti(str(orig_seg_path))

    # Clear legacy label 119 to match what merge_with_anatomy does.
    # merge_pe_with_segmentation restores legacy-119 voxels to base anatomy before
    # writing v2 vessels, so validation must use the same reference.
    legacy_119 = original_seg == VESSEL_LABEL
    if np.any(legacy_119):
        base_path = BASE_SEG_DIR / f"{patient_id}_segmentation_merged.nii.gz"
        print(f"  Base segmentation: {base_path}")
        base_seg, _, _ = load_nifti(str(base_path))
        original_seg[legacy_119] = base_seg[legacy_119]
        # Fill spurious holes (legacy-119 voxels that are background
        # in base seg) with nearest non-zero neighbor label
        holes = legacy_119 & (original_seg == 0)
        if np.any(holes):
            non_zero = original_seg > 0
            _, nearest_idx = ndimage.distance_transform_edt(
                ~non_zero, return_distances=True, return_indices=True
            )
            original_seg[holes] = original_seg[
                nearest_idx[0][holes],
                nearest_idx[1][holes],
                nearest_idx[2][holes],
            ]

    # ── Validate each config ──────────────────────────────────────────────
    all_results = []

    for i, row in summary_df.iterrows():
        config_id = row['config_id']
        pe_mask_path = row['pe_mask_path']
        combined_seg_path = row['combined_seg_path']

        print(f"\n{'='*60}")
        print(f"[{len(all_results)+1}/{len(summary_df)}] Validating {config_id}")

        t0 = time.time()

        # Build config metadata from summary row
        config_metadata = row.to_dict()

        # Run automated checks
        result = validate_config(
            pe_mask_path, combined_seg_path,
            vessel_mask, original_seg, spacing,
            config_metadata
        )

        # Also store the combined_seg_path for visual QC
        result['combined_seg_path'] = combined_seg_path

        elapsed = time.time() - t0

        # Print results
        c1 = 'PASS' if result['c1_pass_containment'] else 'FAIL'
        c2 = 'PASS' if result['c2_pass_volume'] else 'FAIL'
        c3 = 'PASS' if result['c3_pass_shape'] else 'FAIL'
        c4 = 'PASS' if result['c4_pass_anatomy'] else 'FAIL'
        overall = 'PASS' if result['all_checks_pass'] else 'FAIL'

        print(f"  Containment: {c1} ({result['c1_containment_fraction']:.4f})")
        print(f"  Volume:      {c2} ({result['c2_pe_volume_ml']:.4f} ml)")
        print(f"  Shape:       {c3} (elong={result.get('c3_elongation', 'N/A')}, "
              f"spher={result.get('c3_sphericity', 'N/A')})")
        print(f"  Anatomy:     {c4}")
        print(f"  Overall:     {overall}")

        # Render QC slices
        qc_path = render_qc_slices(
            combined_seg_path, config_id, str(output_dir),
            config_metadata=result
        )
        result['qc_slices_path'] = qc_path
        print(f"  QC slices:   {qc_path}")
        print(f"  Time: {time.time()-t0:.1f}s")

        all_results.append(result)

    # ── Summary grid ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating summary grid...")
    grid_path = render_summary_grid(all_results, str(output_dir), patient_id)
    print(f"  Grid: {grid_path}")

    # ── Save validation report ────────────────────────────────────────────
    report_df = pd.DataFrame(all_results)

    # Select key columns for the report
    report_cols = [
        'config_id', 'location', 'occlusion', 'burden',
        'c1_containment_fraction', 'c1_pass_containment',
        'c2_pe_volume_ml', 'c2_pass_volume',
        'c3_n_components', 'c3_elongation', 'c3_sphericity', 'c3_pass_shape',
        'c4_pe_label_match', 'c4_airway_overlap_voxels', 'c4_anatomy_preserved',
        'c4_pass_anatomy',
        'all_checks_pass',
    ]
    report_cols = [c for c in report_cols if c in report_df.columns]
    report_df_clean = report_df[report_cols]

    report_path = output_dir / f"{patient_id}_validation_report.csv"
    report_df_clean.to_csv(report_path, index=False)

    # ── Print summary table ───────────────────────────────────────────────
    n_pass = report_df_clean['all_checks_pass'].sum()
    n_total = len(report_df_clean)
    print(f"\n{'='*60}")
    print(f"Validation Summary: {n_pass}/{n_total} configs passed all checks")
    print(f"Report saved: {report_path}")
    print(f"\n{report_df_clean.to_string(index=False)}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Validate synthetic PE masks")
    parser.add_argument("patient_id", help="Patient ID (e.g., <STUDY_ID>)")
    parser.add_argument("--config-ids", nargs='+', default=None,
                        help="Specific config ID suffixes (e.g., cf001 cf021)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_phase4_for_patient(args.patient_id, args.config_ids, args.output_dir)
