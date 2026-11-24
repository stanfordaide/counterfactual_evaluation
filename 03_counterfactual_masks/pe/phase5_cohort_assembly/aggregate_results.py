"""Phase 5 (cohort assembly): aggregate all patients into the master CSV + diffusion datalist JSON.

Run after all SLURM array jobs complete.

Usage:
    python phase5_cohort_assembly/aggregate_results.py
    python phase5_cohort_assembly/aggregate_results.py --output-dir phase5_cohort_assembly/output
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase5_cohort_assembly.impression_generator import modify_impression

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def aggregate_results(base_output_dir: str = None):
    """Aggregate all patient results into master metadata and datalist."""
    if base_output_dir is None:
        base_output_dir = OUTPUT_DIR
    else:
        base_output_dir = Path(base_output_dir)

    patients_dir = base_output_dir / "patients"
    if not patients_dir.exists():
        print(f"ERROR: {patients_dir} does not exist")
        return

    # Load selected patients metadata - index by image_id (PE-prefixed, matches
    # directory names) and also keep a fallback on numeric patient_id.
    selected_path = base_output_dir / "selected_patients_full.json"
    patient_meta = {}
    if selected_path.exists():
        with open(selected_path) as f:
            for p in json.load(f):
                patient_meta[p.get("image_id", p["patient_id"])] = p
                # Also index by numeric patient_id as fallback
                if p.get("image_id") != p.get("patient_id"):
                    patient_meta[p["patient_id"]] = p

    # Walk patient directories
    all_rows = []
    failed = []
    datalist_entries = []

    patient_dirs = sorted(patients_dir.iterdir())
    print(f"Processing {len(patient_dirs)} patient directories...")

    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        log_path = patient_dir / "pipeline_log.json"

        if not log_path.exists():
            failed.append({
                "patient_id": patient_id,
                "failed_phase": "unknown",
                "error": "No pipeline_log.json found",
            })
            continue

        with open(log_path) as f:
            log = json.load(f)

        if log.get("overall_status") != "success":
            failed.append({
                "patient_id": patient_id,
                "failed_phase": log.get("overall_status", "unknown"),
                "error": _get_first_error(log),
            })
            continue

        # Load Phase 2 grid
        grid_path = patient_dir / "phase2" / f"{patient_id}_counterfactual_grid.csv"
        if not grid_path.exists():
            continue
        grid_df = pd.read_csv(grid_path)

        # Load Phase 3 mask summary
        synth_path = patient_dir / "phase3" / f"{patient_id}_mask_summary.csv"
        if not synth_path.exists():
            continue
        synth_df = pd.read_csv(synth_path)

        # Load Phase 4 validation report
        val_path = patient_dir / "phase4" / f"{patient_id}_validation_report.csv"
        if not val_path.exists():
            continue
        val_df = pd.read_csv(val_path)

        # Get patient metadata (try image_id first, then patient_id)
        meta = patient_meta.get(patient_id, {})

        # Generate impressions on-the-fly from original text_prompt
        original_text = meta.get("text_prompt", "")
        impressions = log.get("impressions", {})
        if not impressions and original_text:
            # Generate from grid configs + original text
            for _, gr in grid_df.iterrows():
                cfg = gr.to_dict()
                impressions[cfg["config_id"]] = modify_impression(original_text, cfg)
            print(f"  Generated {len(impressions)} impressions for {patient_id}")

        # Join data per config
        for _, grid_row in grid_df.iterrows():
            config_id = grid_row["config_id"]

            # Find matching mask-summary row
            synth_match = synth_df[synth_df["config_id"] == config_id]
            # Find matching validation row
            val_match = val_df[val_df["config_id"] == config_id]

            row = {
                "patient_id": patient_id,
                "gender": meta.get("gender", ""),
                "age": meta.get("age", ""),
                "race": meta.get("race", ""),
                "config_id": config_id,
                "location": grid_row.get("location", ""),
                "occlusion": grid_row.get("occlusion", ""),
                "burden": grid_row.get("burden", ""),
                "target_lobes": grid_row.get("target_lobes", ""),
                "n_segments_affected": grid_row.get("n_segments_affected", 0),
                "qanadli_score": grid_row.get("qanadli_score", 0),
                "qanadli_percent": grid_row.get("qanadli_percent", 0),
                "severity_category": grid_row.get("severity_category", ""),
                "laterality": grid_row.get("laterality", ""),
                "empirically_observed": grid_row.get("empirically_observed", ""),
            }

            if len(synth_match) > 0:
                sr = synth_match.iloc[0]
                row["pe_voxels"] = sr.get("pe_voxels", 0)
                row["pe_volume_ml"] = sr.get("pe_volume_ml", 0)
                row["containment_fraction"] = sr.get("containment_fraction", 0)
                row["pe_mask_path"] = sr.get("pe_mask_path", "")
                row["combined_seg_path"] = sr.get("combined_seg_path", "")

            if len(val_match) > 0:
                vr = val_match.iloc[0]
                row["c1_pass_containment"] = vr.get("c1_pass_containment", False)
                row["c2_pass_volume"] = vr.get("c2_pass_volume", False)
                row["c3_pass_shape"] = vr.get("c3_pass_shape", False)
                row["c4_pass_anatomy"] = vr.get("c4_pass_anatomy", False)
                row["all_checks_pass"] = vr.get("all_checks_pass", False)

            row["text_prompt"] = impressions.get(config_id, "")

            all_rows.append(row)

            # Build datalist entry
            if row.get("all_checks_pass", False) and row.get("text_prompt"):
                datalist_entries.append(
                    _build_datalist_entry(row, meta, config_id)
                )

    # Save master metadata CSV
    master_df = pd.DataFrame(all_rows)
    master_path = base_output_dir / "master_metadata.csv"
    master_df.to_csv(master_path, index=False)
    print(f"\nSaved master metadata: {master_path}")
    print(f"  Total configs: {len(master_df)}")
    if "all_checks_pass" in master_df.columns:
        n_pass = master_df["all_checks_pass"].sum()
        print(f"  Passing validation: {n_pass}/{len(master_df)}")

    # Save failed patients
    if failed:
        failed_df = pd.DataFrame(failed)
        failed_path = base_output_dir / "failed_patients.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"  Failed patients: {len(failed)} (see {failed_path})")
    else:
        print("  No failed patients")

    # Save counterfactual datalist JSON
    datalist = {"test": datalist_entries}
    datalist_path = base_output_dir / "counterfactual_datalist.json"
    with open(datalist_path, "w") as f:
        json.dump(datalist, f, indent=2)
    print(f"\nSaved datalist: {datalist_path}")
    print(f"  Entries: {len(datalist_entries)}")

    # Print summary statistics
    if len(master_df) > 0:
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}")
        print(f"\nBy severity:")
        if "severity_category" in master_df.columns:
            print(master_df["severity_category"].value_counts().to_string())
        print(f"\nBy location:")
        if "location" in master_df.columns:
            print(master_df["location"].value_counts().to_string())
        print(f"\nBy burden:")
        if "burden" in master_df.columns:
            print(master_df["burden"].value_counts().to_string())

    return master_df


def _get_first_error(log: dict) -> str:
    """Extract first error message from pipeline log."""
    for phase_name, phase_data in log.get("phases", {}).items():
        if isinstance(phase_data, dict) and phase_data.get("status") == "failed":
            return phase_data.get("error", "unknown error")
    return "unknown error"


def _build_datalist_entry(row: dict, meta: dict, config_id: str) -> dict:
    """Build a datalist entry in the same format as the input JSON."""
    return {
        "image": meta.get("image", ""),
        "embedding": meta.get("embedding", ""),
        "segmentation": row.get("combined_seg_path", ""),
        "label": 1,
        "dim": meta.get("dim", []),
        "spacing": meta.get("spacing", []),
        "top_region_index": meta.get("top_region_index", []),
        "bottom_region_index": meta.get("bottom_region_index", []),
        "fold": meta.get("fold", 0),
        "pe_status": 1,
        "text_prompt": row.get("text_prompt", ""),
        "image_id": config_id,
        "patient_id": meta.get("patient_id", ""),
        "gender": meta.get("gender", ""),
        "age": meta.get("age", 0),
        "race": meta.get("race", ""),
        "ethnicity": meta.get("ethnicity", ""),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5 (cohort assembly): aggregate results into master CSV and datalist JSON")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory (default: phase5_cohort_assembly/output/)")
    args = parser.parse_args()

    aggregate_results(args.output_dir)
