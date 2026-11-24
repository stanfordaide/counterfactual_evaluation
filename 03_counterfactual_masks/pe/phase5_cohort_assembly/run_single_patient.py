"""Phase 5 (cohort assembly): run the full Phase 1→4 pipeline + impression generation for one patient.

Usage:
    python phase5_cohort_assembly/run_single_patient.py <STUDY_ID>
    python phase5_cohort_assembly/run_single_patient.py <STUDY_ID> --output-dir phase5_cohort_assembly/output --max-configs 10
    python phase5_cohort_assembly/run_single_patient.py <STUDY_ID> --output-dir phase5_cohort_assembly/output
"""

import argparse
import json
import pickle
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project root and sys.path setup (required for Phase 1 pickle deserialization)
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "phase1_vessel_tree"))

from phase1_vessel_tree.run_phase1 import run_phase1_pipeline
from phase2_counterfactuals.counterfactual_grid import (
    run_phase2_for_patient,
    EdgeCatalog,
    generate_replacement_configs,
    LOCATION_NAMES,
)
from phase3_mask_construction.run_phase3 import run_phase3_for_patient, build_pe_mask_for_config
from phase3_mask_construction.pe_placement import compute_full_distance_transform
from phase4_validation.run_phase4 import run_phase4_for_patient
from phase4_validation.automated_checks import validate_config
from phase4_validation.visual_qc import render_qc_slices
from phase5_cohort_assembly.impression_generator import generate_impressions_for_patient
from utils.nifti_io import load_nifti, DATA_ROOT

# Retry configuration
TARGET_CONFIGS = 16   # 15 severity-stratified + 1 extra severe
MAX_RETRY_ROUNDS = 3

# Merged segmentation directory (same as Phase 4)
MERGED_SEG_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_all_lung_pe"


def _save_log(result: dict, patient_dir: Path):
    """Save pipeline log to per-patient directory."""
    patient_dir.mkdir(parents=True, exist_ok=True)
    log_path = patient_dir / "pipeline_log.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def _load_patient_text_prompt(patient_id: str, output_dir: Path) -> str:
    """Load original text_prompt from the selected patients JSON."""
    json_path = output_dir / "selected_patients_full.json"
    if json_path.exists():
        with open(json_path) as f:
            patients = json.load(f)
        for p in patients:
            if p.get("patient_id") == patient_id:
                return p.get("text_prompt", "")
    return ""


def _load_grid_configs(phase2_dir: Path, patient_id: str) -> list:
    """Load counterfactual grid configs from Phase 2 JSON output."""
    grid_path = phase2_dir / f"{patient_id}_counterfactual_grid.json"
    if grid_path.exists():
        with open(grid_path) as f:
            data = json.load(f)
        return data.get("configs", [])
    return []


def _run_retry_loop(patient_id: str,
                    validation_results: list,
                    phase1_dir: Path,
                    phase2_dir: Path,
                    phase3_dir: Path,
                    phase4_dir: Path,
                    seed: int,
                    target_configs: int = TARGET_CONFIGS) -> list:
    """Retry failed configs to reach target_configs passing.

    Loads shared data once, then iterates: generate replacement configs,
    build masks, validate. Appends results to grid JSON, mask-summary CSV,
    and validation report CSV.

    Returns updated list of ALL validation results (original + retries).
    """
    n_pass = sum(1 for r in validation_results if r.get("all_checks_pass"))
    if n_pass >= target_configs:
        return validation_results

    print(f"\n{'#'*60}")
    print(f"# RETRY: {n_pass}/{target_configs} passed - generating replacements")
    print(f"{'#'*60}")

    # ── Load shared data once ─────────────────────────────────────────────
    # Vessel mask + distance transform (for mask construction)
    vessel_path = phase1_dir / f"{patient_id}_vessels_cleaned.nii.gz"
    vessel_mask, affine, header = load_nifti(str(vessel_path))
    vessel_mask = (vessel_mask > 0).astype(np.uint8)
    spacing = np.array(header.get_zooms()[:3], dtype=np.float64)

    print("  Computing distance transform for retries...")
    t0 = time.time()
    dt = compute_full_distance_transform(vessel_mask, spacing)
    print(f"  Distance transform computed in {time.time()-t0:.1f}s")

    # Vessel graph + catalog (for replacement config generation)
    graph_path = phase1_dir / f"{patient_id}_vessel_graph_final.pkl"
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    G = graph_data['graph']
    catalog = EdgeCatalog(G)

    # Original segmentation (for validation)
    orig_seg_path = MERGED_SEG_DIR / f"{patient_id}_segmentation_merged.nii.gz"
    original_seg, _, _ = load_nifti(str(orig_seg_path))

    # Clear legacy label 119 to match merge_with_anatomy behavior
    from scipy import ndimage as _ndimage
    legacy_119 = original_seg == 119
    if np.any(legacy_119):
        from phase3_mask_construction.merge_with_anatomy import BASE_SEG_DIR
        base_path = BASE_SEG_DIR / f"{patient_id}_segmentation_merged.nii.gz"
        base_seg, _, _ = load_nifti(str(base_path))
        original_seg[legacy_119] = base_seg[legacy_119]
        # Fill spurious holes with nearest non-zero neighbor
        holes = legacy_119 & (original_seg == 0)
        if np.any(holes):
            non_zero = original_seg > 0
            _, nearest_idx = _ndimage.distance_transform_edt(
                ~non_zero, return_distances=True, return_indices=True
            )
            original_seg[holes] = original_seg[
                nearest_idx[0][holes],
                nearest_idx[1][holes],
                nearest_idx[2][holes],
            ]

    retry_rng = np.random.default_rng(seed=(seed + 1000))

    # Track all validation results
    all_results = list(validation_results)

    # Find max config number from existing configs
    config_nums = []
    for r in all_results:
        cid = r.get('config_id', '')
        try:
            config_nums.append(int(cid.split('_cf')[1]))
        except (IndexError, ValueError):
            pass
    next_config_num = max(config_nums) if config_nums else 0

    # Load the grid JSON for appending
    grid_path = phase2_dir / f"{patient_id}_counterfactual_grid.json"
    with open(grid_path) as f:
        grid_data = json.load(f)

    for retry_round in range(MAX_RETRY_ROUNDS):
        failed = [r for r in all_results if not r.get("all_checks_pass")]
        n_pass = len(all_results) - len(failed)
        n_needed = target_configs - n_pass

        if n_needed <= 0:
            break

        # Only replace up to n_needed failed configs
        configs_to_replace = failed[:n_needed]

        print(f"\n  Retry round {retry_round + 1}: need {n_needed} more, "
              f"replacing {len(configs_to_replace)} configs")

        # Find matching config dicts from the grid JSON
        failed_ids = {r['config_id'] for r in configs_to_replace}
        failed_dicts = [c for c in grid_data['configs']
                        if c['config_id'] in failed_ids]

        # Generate replacements
        replacements = generate_replacement_configs(
            failed_dicts, patient_id, G, catalog,
            retry_rng, next_config_num, attempt=retry_round,
        )

        if not replacements:
            print("  No replacements generated - stopping retries")
            break

        next_config_num += len(replacements)

        # Append replacement configs to grid JSON
        for cfg in replacements:
            grid_data['configs'].append(cfg.to_dict())
        grid_data['n_configs'] = len(grid_data['configs'])
        with open(grid_path, 'w') as f:
            json.dump(grid_data, f, indent=2, default=str)

        # Also append to grid CSV
        grid_csv_path = phase2_dir / f"{patient_id}_counterfactual_grid.csv"
        existing_csv = pd.read_csv(grid_csv_path)
        new_rows = []
        for cfg in replacements:
            new_rows.append({
                'config_id': cfg.config_id,
                'patient_id': cfg.patient_id,
                'location': LOCATION_NAMES[cfg.location],
                'occlusion': cfg.occlusion.value,
                'burden': cfg.burden.value,
                'target_lobes': ','.join(cfg.target_lobes),
                'n_segments_affected': cfg.n_segments_affected,
                'qanadli_score': cfg.qanadli_score,
                'qanadli_percent': round(cfg.qanadli_percent, 1),
                'severity_category': cfg.severity_category,
                'laterality': cfg.laterality,
                'estimated_pe_voxels': cfg.estimated_total_pe_voxels,
                'empirically_observed': cfg.empirically_observed,
                'clinical_note': cfg.clinical_note,
                'edge_keys': str([(s.edge_key[0], s.edge_key[1])
                                  for s in cfg.segments]),
            })
        updated_csv = pd.concat([existing_csv, pd.DataFrame(new_rows)],
                                ignore_index=True)
        updated_csv.to_csv(grid_csv_path, index=False)

        # ── Build + validate each replacement ────────────────────────
        base_rng = np.random.default_rng(seed=(seed + 2000 + retry_round))

        new_metrics = []
        new_validations = []

        for cfg in replacements:
            config_dict = cfg.to_dict()
            config_id = cfg.config_id

            print(f"\n    Building {config_id} "
                  f"({LOCATION_NAMES[cfg.location]}, {cfg.burden.value}, "
                  f"{cfg.occlusion.value})")

            config_rng = np.random.default_rng(
                seed=base_rng.integers(2**31))

            metrics = build_pe_mask_for_config(
                config_dict, G, vessel_mask, dt, spacing, affine, header,
                phase3_dir, config_rng
            )
            new_metrics.append(metrics)

            print(f"    PE voxels: {metrics['pe_voxels']}, "
                  f"Volume: {metrics['pe_volume_ml']:.4f} ml")

            # Validate
            val_result = validate_config(
                metrics['pe_mask_path'], metrics['combined_seg_path'],
                vessel_mask, original_seg, spacing,
                metrics
            )

            # Render QC
            qc_path = render_qc_slices(
                metrics['combined_seg_path'], config_id,
                str(phase4_dir), config_metadata=val_result
            )
            val_result['qc_slices_path'] = qc_path

            passed = val_result.get('all_checks_pass', False)
            print(f"    Validation: {'PASS' if passed else 'FAIL'}")

            new_validations.append(val_result)
            all_results.append(val_result)

        # Append to mask summary CSV
        synth_csv_path = phase3_dir / f"{patient_id}_mask_summary.csv"
        existing_synth = pd.read_csv(synth_csv_path)
        updated_synth = pd.concat([existing_synth, pd.DataFrame(new_metrics)],
                                  ignore_index=True)
        updated_synth.to_csv(synth_csv_path, index=False)

        # Append to validation report CSV
        val_csv_path = phase4_dir / f"{patient_id}_validation_report.csv"
        existing_val = pd.read_csv(val_csv_path)
        val_cols = [c for c in existing_val.columns]
        new_val_rows = []
        for vr in new_validations:
            row = {c: vr.get(c) for c in val_cols}
            new_val_rows.append(row)
        updated_val = pd.concat([existing_val, pd.DataFrame(new_val_rows)],
                                ignore_index=True)
        updated_val.to_csv(val_csv_path, index=False)

        n_new_pass = sum(1 for r in new_validations
                         if r.get("all_checks_pass"))
        total_pass = sum(1 for r in all_results
                         if r.get("all_checks_pass"))
        print(f"\n  Round {retry_round + 1}: {n_new_pass}/{len(new_validations)} "
              f"new passed, total {total_pass}/{target_configs} passing")

    final_pass = sum(1 for r in all_results if r.get("all_checks_pass"))
    print(f"\n  Retry complete: {final_pass}/{target_configs} passing configs")
    return all_results


def run_single_patient(patient_id: str,
                       base_output_dir: str = "phase5_cohort_assembly/output",
                       seed: int = None,
                       max_configs: int = 16) -> dict:
    """Run the full pipeline for one patient.

    Returns dict with per-phase status, timing, and error info.
    """
    base_output_dir = Path(base_output_dir)
    patient_dir = base_output_dir / "patients" / patient_id
    phase1_dir = patient_dir / "phase1"
    phase2_dir = patient_dir / "phase2"
    phase3_dir = patient_dir / "phase3"
    phase4_dir = patient_dir / "phase4"

    # Clean stale outputs from previous runs to avoid accumulation
    for d in [phase1_dir, phase2_dir, phase3_dir, phase4_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = hash(patient_id) % (2**32)

    target_configs = TARGET_CONFIGS

    result = {
        "patient_id": patient_id,
        "start_time": datetime.now().isoformat(),
        "seed": seed,
        "max_configs": max_configs,
        "target_configs": target_configs,
        "phases": {},
    }

    pipeline_start = time.time()

    # ── Phase 1: Vessel Tree Extraction ──────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# PHASE 1: Vessel Tree Extraction for {patient_id}")
    print(f"{'#'*60}")
    try:
        t0 = time.time()
        stats = run_phase1_pipeline(patient_id, output_dir=str(phase1_dir))
        result["phases"]["phase1"] = {
            "status": "success",
            "elapsed_seconds": round(time.time() - t0, 1),
        }
    except Exception as e:
        result["phases"]["phase1"] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        result["overall_status"] = "failed_phase1"
        _save_log(result, patient_dir)
        return result

    # ── Phase 2: Counterfactual Grid ─────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# PHASE 2: Counterfactual Grid for {patient_id}")
    print(f"{'#'*60}")
    try:
        t0 = time.time()
        configs = run_phase2_for_patient(
            patient_id,
            output_dir=str(phase2_dir),
            seed=seed,
            max_configs=max_configs,
            phase1_output_dir=str(phase1_dir),
        )
        result["phases"]["phase2"] = {
            "status": "success",
            "elapsed_seconds": round(time.time() - t0, 1),
            "n_configs": len(configs),
        }
    except Exception as e:
        result["phases"]["phase2"] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        result["overall_status"] = "failed_phase2"
        _save_log(result, patient_dir)
        return result

    # ── Phase 3: PE Mask Construction ───────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# PHASE 3: PE Mask Construction for {patient_id}")
    print(f"{'#'*60}")
    try:
        t0 = time.time()
        metrics = run_phase3_for_patient(
            patient_id,
            output_dir=str(phase3_dir),
            seed=seed,
            phase1_output_dir=str(phase1_dir),
            phase2_output_dir=str(phase2_dir),
        )
        result["phases"]["phase3"] = {
            "status": "success",
            "elapsed_seconds": round(time.time() - t0, 1),
            "n_configs_built": len(metrics),
        }
    except Exception as e:
        result["phases"]["phase3"] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        result["overall_status"] = "failed_phase3"
        _save_log(result, patient_dir)
        return result

    # ── Phase 4: Validation ──────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# PHASE 4: Validation for {patient_id}")
    print(f"{'#'*60}")
    try:
        t0 = time.time()
        validation = run_phase4_for_patient(
            patient_id,
            output_dir=str(phase4_dir),
            phase1_output_dir=str(phase1_dir),
            phase3_output_dir=str(phase3_dir),
        )
        n_pass = sum(1 for r in validation if r.get("all_checks_pass"))
        result["phases"]["phase4"] = {
            "status": "success",
            "elapsed_seconds": round(time.time() - t0, 1),
            "n_validated": len(validation),
            "n_pass": n_pass,
        }
    except Exception as e:
        result["phases"]["phase4"] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        result["overall_status"] = "failed_phase4"
        _save_log(result, patient_dir)
        return result

    # ── Retry loop: ensure target_configs passing configs ──────────────
    if n_pass < target_configs:
        try:
            t0 = time.time()
            all_validation = _run_retry_loop(
                patient_id, validation,
                phase1_dir, phase2_dir, phase3_dir, phase4_dir,
                seed, target_configs=target_configs,
            )
            final_pass = sum(1 for r in all_validation
                             if r.get("all_checks_pass"))
            result["phases"]["retry"] = {
                "status": "success",
                "elapsed_seconds": round(time.time() - t0, 1),
                "total_configs": len(all_validation),
                "total_pass": final_pass,
                "retry_configs": len(all_validation) - len(validation),
            }
            # Update Phase 4 counts with final totals
            result["phases"]["phase4"]["n_validated"] = len(all_validation)
            result["phases"]["phase4"]["n_pass"] = final_pass
        except Exception as e:
            print(f"WARNING: Retry loop failed: {e}")
            traceback.print_exc()
            result["phases"]["retry"] = {
                "status": "failed",
                "error": str(e),
            }
    else:
        print(f"\n  All {n_pass}/{target_configs} configs passed - no retries needed")

    # ── Impression Generation ────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# Impression Generation for {patient_id}")
    print(f"{'#'*60}")
    try:
        original_text = _load_patient_text_prompt(patient_id, base_output_dir)
        if original_text:
            grid_configs = _load_grid_configs(phase2_dir, patient_id)
            impressions = generate_impressions_for_patient(original_text, grid_configs)
            result["impressions"] = impressions

            # Save impressions to a separate file for easy access
            imp_path = patient_dir / "impressions.json"
            with open(imp_path, "w") as f:
                json.dump(impressions, f, indent=2)
            print(f"Generated {len(impressions)} impressions")
        else:
            print("WARNING: No original text_prompt found - skipping impression generation")
            result["impressions"] = {}
    except Exception as e:
        print(f"WARNING: Impression generation failed: {e}")
        result["impressions"] = {}

    # ── Summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    result["overall_status"] = "success"
    result["end_time"] = datetime.now().isoformat()
    result["total_elapsed_seconds"] = round(total_elapsed, 1)

    _save_log(result, patient_dir)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE for {patient_id}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Configs: {result['phases']['phase2'].get('n_configs', 0)}")
    print(f"  Validation: {result['phases']['phase4'].get('n_pass', 0)}/"
          f"{result['phases']['phase4'].get('n_validated', 0)} pass")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5 (cohort assembly): run full Phase 1→4 pipeline + impressions for one patient")
    parser.add_argument("patient_id", help="Patient ID (e.g., <STUDY_ID>)")
    parser.add_argument("--output-dir", type=str, default="phase5_cohort_assembly/output",
                        help="Base output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: hash(patient_id); Python string hashes change per process unless PYTHONHASHSEED is set)")
    parser.add_argument("--max-configs", type=int, default=16,
                        help="Max configs per patient (default: 16)")
    args = parser.parse_args()

    result = run_single_patient(
        args.patient_id, args.output_dir, args.seed, args.max_configs,
    )

    status = result.get("overall_status", "unknown")
    if status != "success":
        print(f"\nFAILED: {status}")
        sys.exit(1)
