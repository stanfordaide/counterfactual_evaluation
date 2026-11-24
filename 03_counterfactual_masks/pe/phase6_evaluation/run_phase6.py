"""Phase 6: Evaluation - Compare detected PE in synthetic CTs vs intended PE.

Compares the PE segmentation output from inference on synthetic counterfactual
CTs against the intended PE masks used as conditioning (label 121 in the
label files). Reports spatial overlap, volume fidelity, lobe concordance,
and component-level detection rates.

Usage:
    python phase6_evaluation/run_phase6.py --patient-id <STUDY_ID>
    python phase6_evaluation/run_phase6.py --all
    python phase6_evaluation/run_phase6.py --aggregate-only
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label as ndimage_label

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from utils.nifti_io import load_nifti, DATA_ROOT, generator_slice_map

# ── Paths ──────────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = DATA_ROOT / "pe_training_data_all" / "predictions_controlnet_counterfactuals"
PREDICTIONS_CSV = PREDICTIONS_DIR / "controlnet_predictions_counterfactuals.csv"
GRID_BASE_DIR = Path(PROJECT_ROOT) / "phase5_cohort_assembly" / "output" / "patients"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"

# ── Lobe mapping (from phase2_counterfactuals/axis_definitions.py) ─────────
LABEL_TO_LOBE = {10: "LUL", 11: "LLL", 12: "RUL", 13: "RML", 14: "RLL"}
LOBE_LABELS = set(LABEL_TO_LOBE.keys())

PE_LABEL = 121


# ── Utilities ──────────────────────────────────────────────────────────────────

def build_file_index(predictions_csv: Path,
                     patient_id: Optional[str] = None) -> List[Dict]:
    """Parse predictions CSV and derive label/pe_seg paths for each config.

    Returns list of dicts with keys: image_id, patient_id, config_id,
    prediction_path, label_path, pe_seg_path.
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
            pe_seg_path = pred_path.replace("_image.nii.gz", "_pe_seg.nii.gz")

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

    configs = grid.get("configs", [])
    for cfg in configs:
        if cfg.get("config_id") == config_id:
            return cfg
    return {}


def hausdorff_95(mask_a: np.ndarray, mask_b: np.ndarray,
                 spacing: tuple) -> float:
    """95th-percentile Hausdorff distance (mm) between two binary masks.

    Operates on a cropped ROI around both masks for efficiency.
    """
    if np.sum(mask_a) == 0 or np.sum(mask_b) == 0:
        return float("nan")

    # Crop to bounding box of union of both masks with generous padding
    union_mask = (mask_a > 0) | (mask_b > 0)
    coords = np.where(union_mask)
    pad = 5
    slices = tuple(
        slice(max(0, c.min() - pad), min(mask_a.shape[i], c.max() + pad + 1))
        for i, c in enumerate(coords)
    )
    a_roi = mask_a[slices]
    b_roi = mask_b[slices]

    dt_b = distance_transform_edt(~(b_roi > 0), sampling=spacing)
    distances_a_to_b = dt_b[a_roi > 0]

    dt_a = distance_transform_edt(~(a_roi > 0), sampling=spacing)
    distances_b_to_a = dt_a[b_roi > 0]

    h95_a = np.percentile(distances_a_to_b, 95)
    h95_b = np.percentile(distances_b_to_a, 95)
    return float(max(h95_a, h95_b))


# ── Core evaluation ────────────────────────────────────────────────────────────

def _find_lobes_for_mask(pe_mask: np.ndarray, label_data: np.ndarray,
                         spacing: tuple = (1.0, 1.0, 1.0)) -> set:
    """Find which lung lobes a PE mask overlaps with.

    Two-step approach:
      1. Direct overlap - check lobe labels at PE voxel locations.
      2. Nearest-lobe EDT - for PE voxels not on any lobe, compute per-lobe
         distance transforms in a cropped ROI and assign to nearest lobe.

    Args:
        pe_mask: Binary mask of PE voxels.
        label_data: Full label volume (same shape as pe_mask).
        spacing: Voxel spacing in mm (used for distance_transform_edt).

    Returns:
        Set of lobe name strings (e.g., {"RLL", "LUL"}).
    """
    if not np.any(pe_mask):
        return set()

    lobe_label_set = list(LABEL_TO_LOBE.keys())

    # Step 1: Direct overlap - PE voxels that sit on a lobe label
    labels_at_pe = label_data[pe_mask > 0]
    found_lobes = set()
    for lbl, name in LABEL_TO_LOBE.items():
        if np.any(labels_at_pe == lbl):
            found_lobes.add(name)

    # If every PE voxel is on a lobe label, we're done
    if np.all(np.isin(labels_at_pe, lobe_label_set)):
        return found_lobes

    # Step 2: Nearest-lobe EDT for PE voxels not on any lobe
    coords = np.where(pe_mask)
    pad = 30  # generous padding so EDT can reach nearby lobes
    slices = tuple(
        slice(max(0, c.min() - pad), min(pe_mask.shape[i], c.max() + pad + 1))
        for i, c in enumerate(coords)
    )
    pe_roi = pe_mask[slices]
    label_roi = label_data[slices]

    # Identify PE voxels in the ROI that are NOT on a lobe label
    unassigned = (pe_roi > 0) & ~np.isin(label_roi, lobe_label_set)
    if not np.any(unassigned):
        return found_lobes

    # Compute per-lobe distance transform in the ROI, track nearest lobe
    min_dist = np.full(label_roi.shape, np.inf, dtype=np.float32)
    nearest_lobe = np.zeros(label_roi.shape, dtype=np.int32)

    for lbl in lobe_label_set:
        lobe_mask = (label_roi == lbl)
        if not np.any(lobe_mask):
            continue
        dist = distance_transform_edt(~lobe_mask, sampling=spacing).astype(np.float32)
        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        nearest_lobe[closer] = lbl

    # Assign lobes to the unassigned PE voxels
    assigned_labels = nearest_lobe[unassigned]
    for lbl, name in LABEL_TO_LOBE.items():
        if np.any(assigned_labels == lbl):
            found_lobes.add(name)

    return found_lobes


def evaluate_single_config(label_path: str, pe_seg_path: str,
                           config_meta: Dict) -> Dict:
    """Compute all evaluation metrics for one counterfactual config.

    Args:
        label_path: Path to conditioning label NIfTI (contains label 121 = PE).
        pe_seg_path: Path to detected PE segmentation NIfTI.
        config_meta: Config metadata dict from grid JSON.

    Returns:
        Dict with all metrics.
    """
    result = {}

    # Copy metadata
    for key in ["config_id", "patient_id", "location", "occlusion", "burden",
                "severity_category", "qanadli_score", "n_segments_affected",
                "target_lobes", "laterality"]:
        result[key] = config_meta.get(key, "")

    # Stringify target_lobes list
    tl = result.get("target_lobes", "")
    if isinstance(tl, list):
        result["target_lobes"] = ";".join(tl)

    # Load volumes
    label_data, label_affine, label_header = load_nifti(label_path)
    pe_seg_data, _, _ = load_nifti(pe_seg_path)

    spacing = tuple(float(s) for s in label_header.get_zooms()[:3])

    # ── Shape alignment ────────────────────────────────────────────────────
    # The prediction is on the generated CT grid (z_seg slices). The generator resized
    # the label map from z_label to z_seg slices (nearest neighbour, x/y unchanged), so
    # the label is put on that grid with the same slice map. Sizes on that grid use the
    # label spacing with z scaled by z_label / z_seg.
    # Two kinds of output:
    #   - properties of the intended mask alone (intended_voxels, intended_volume_ml,
    #     n_intended_components, intended lobes): full label grid, label spacing;
    #   - everything that uses the prediction (overlap, detected_*, lobe concordance):
    #     generated grid.
    z_label = label_data.shape[2]
    z_seg = pe_seg_data.shape[2]

    if label_data.shape[:2] != pe_seg_data.shape[:2]:
        result["error"] = f"XY shape mismatch: label={label_data.shape}, seg={pe_seg_data.shape}"
        return result

    if z_label != z_seg:
        result["z_label"] = z_label
        result["z_seg"] = z_seg

    label_full = label_data                                              # label grid
    label_data = label_data[:, :, generator_slice_map(z_label, z_seg)]   # generated grid
    seg_spacing = (spacing[0], spacing[1], spacing[2] * z_label / z_seg)

    # Extract masks: intended and detected on the generated grid, intended also on the label grid
    intended_pe = (label_data == PE_LABEL).astype(np.uint8)
    detected_pe = (pe_seg_data > 0).astype(np.uint8)
    intended_pe_full = (label_full == PE_LABEL).astype(np.uint8)

    # Predicted voxels at slice index >= z_label. This index-based count is kept from the
    # original code. With the slice map every predicted slice is compared, so it is not a
    # padding region.
    fp_padding = 0
    if z_seg > z_label:
        fp_padding = int(np.sum(pe_seg_data[:, :, z_label:] > 0))
    result["fp_padding_voxels"] = fp_padding

    # ── Basic counts ───────────────────────────────────────────────────────
    intended_voxels = int(np.sum(intended_pe))             # generated grid, for the overlap metrics
    detected_voxels = int(np.sum(detected_pe))
    result["intended_voxels"] = int(np.sum(intended_pe_full))   # label grid
    result["detected_voxels"] = detected_voxels
    result["pe_detected"] = detected_voxels > 0

    # Guard: no intended PE
    if intended_voxels == 0:
        result["error"] = "no_intended_pe"
        for m in ["dice", "iou", "sensitivity", "precision", "hausdorff_95_mm",
                   "volume_ratio", "component_detection_rate", "lobe_concordance"]:
            result[m] = float("nan")
        return result

    # ── Spatial overlap ────────────────────────────────────────────────────
    intersection = int(np.sum((intended_pe > 0) & (detected_pe > 0)))
    union = int(np.sum((intended_pe > 0) | (detected_pe > 0)))

    result["true_positive_voxels"] = intersection
    result["false_positive_voxels"] = int(detected_voxels - intersection)
    result["false_negative_voxels"] = int(intended_voxels - intersection)

    denom_dice = intended_voxels + detected_voxels
    result["dice"] = (2.0 * intersection / denom_dice) if denom_dice > 0 else 0.0
    result["iou"] = (intersection / union) if union > 0 else 0.0
    result["sensitivity"] = intersection / intended_voxels
    result["precision"] = (intersection / detected_voxels) if detected_voxels > 0 else float("nan")

    # ── Hausdorff 95 ──────────────────────────────────────────────────────
    result["hausdorff_95_mm"] = hausdorff_95(intended_pe, detected_pe, seg_spacing)

    # ── Volume comparison ──────────────────────────────────────────────────
    # Intended volume on the full label grid (label spacing; same value and rounding as
    # pe_volume_ml of Phase 3); detected volume on the generated grid (its spacing)
    intended_vol = int(np.sum(intended_pe_full)) * float(np.prod(spacing)) / 1000.0
    detected_vol = detected_voxels * float(np.prod(seg_spacing)) / 1000.0
    result["intended_volume_ml"] = round(intended_vol, 4)
    result["detected_volume_ml"] = round(detected_vol, 6)
    result["volume_ratio"] = (detected_vol / intended_vol) if intended_vol > 0 else float("nan")
    result["volume_diff_ml"] = round(detected_vol - intended_vol, 6)

    # ── Connected component analysis ───────────────────────────────────────
    intended_labeled, n_intended = ndimage_label(intended_pe)        # generated grid
    _, n_detected = ndimage_label(detected_pe)
    result["n_intended_components"] = ndimage_label(intended_pe_full)[1]   # label grid
    result["n_detected_components"] = n_detected

    n_intended_detected = 0
    for comp_id in range(1, n_intended + 1):
        if np.any(detected_pe[intended_labeled == comp_id]):
            n_intended_detected += 1

    result["n_intended_detected"] = n_intended_detected
    result["component_detection_rate"] = (
        n_intended_detected / n_intended if n_intended > 0 else float("nan")
    )

    # ── Lobe concordance ───────────────────────────────────────────────────
    # PE voxels have label 121, not lobe labels (10-14). Use config metadata
    # for intended lobes, and nearest-lobe EDT for detected lobes.
    lobe_data = label_data                     # label on the generated grid

    # Intended lobes: use target_lobes from config metadata (ground truth)
    tl_raw = config_meta.get("target_lobes", [])
    if isinstance(tl_raw, str):
        intended_lobes = set(tl_raw.split(";")) if tl_raw else set()
    elif isinstance(tl_raw, list):
        intended_lobes = set(tl_raw)
    else:
        intended_lobes = set()
    # Fallback: if no metadata, derive from nearest-lobe lookup
    if not intended_lobes and intended_voxels > 0:
        intended_lobes = _find_lobes_for_mask(intended_pe_full, label_full, spacing)   # label grid

    # Detected lobes: derive from nearest-lobe lookup
    detected_lobes = _find_lobes_for_mask(detected_pe, lobe_data, seg_spacing) if detected_voxels > 0 else set()

    lobes_correct = intended_lobes & detected_lobes
    lobes_missed = intended_lobes - detected_lobes
    lobes_extra = detected_lobes - intended_lobes

    result["intended_lobes"] = ";".join(sorted(intended_lobes)) if intended_lobes else ""
    result["detected_lobes"] = ";".join(sorted(detected_lobes)) if detected_lobes else ""
    result["lobes_correct"] = ";".join(sorted(lobes_correct)) if lobes_correct else ""
    result["lobes_missed"] = ";".join(sorted(lobes_missed)) if lobes_missed else ""
    result["lobes_extra"] = ";".join(sorted(lobes_extra)) if lobes_extra else ""
    result["n_lobes_intended"] = len(intended_lobes)
    result["n_lobes_detected"] = len(detected_lobes)
    result["n_lobes_correct"] = len(lobes_correct)
    result["n_lobes_missed"] = len(lobes_missed)
    result["n_lobes_extra"] = len(lobes_extra)
    result["lobe_concordance"] = (
        len(lobes_correct) / len(intended_lobes) if intended_lobes else float("nan")
    )

    result["label_path"] = label_path
    result["pe_seg_path"] = pe_seg_path

    return result


# ── Patient-level driver ───────────────────────────────────────────────────────

CSV_COLUMNS = [
    "patient_id", "config_id", "location", "occlusion", "burden",
    "severity_category", "qanadli_score", "n_segments_affected",
    "target_lobes", "laterality",
    "intended_voxels", "detected_voxels", "pe_detected",
    "dice", "iou", "sensitivity", "precision", "hausdorff_95_mm",
    "intended_volume_ml", "detected_volume_ml", "volume_ratio", "volume_diff_ml",
    "n_intended_components", "n_detected_components",
    "n_intended_detected", "component_detection_rate",
    "intended_lobes", "detected_lobes", "lobes_correct", "lobes_missed", "lobes_extra",
    "n_lobes_intended", "n_lobes_detected", "n_lobes_correct",
    "n_lobes_missed", "n_lobes_extra", "lobe_concordance",
    "true_positive_voxels", "false_positive_voxels", "false_negative_voxels",
    "fp_padding_voxels",
    "label_path", "pe_seg_path", "error",
]


def evaluate_patient(patient_id: str, file_index: List[Dict],
                     output_dir: Path) -> List[Dict]:
    """Evaluate all configs for one patient."""
    patient_entries = [e for e in file_index if e["patient_id"] == patient_id]

    if not patient_entries:
        print(f"  No entries found for {patient_id}")
        return []

    print(f"\nEvaluating {patient_id}: {len(patient_entries)} configs", flush=True)
    results = []

    for i, entry in enumerate(patient_entries):
        config_id = entry["config_id"]
        label_path = entry["label_path"]
        pe_seg_path = entry["pe_seg_path"]

        # Check files exist
        if not Path(label_path).exists():
            print(f"  SKIP {config_id}: missing label file", flush=True)
            results.append({"config_id": config_id, "patient_id": patient_id,
                            "error": "missing_label"})
            continue
        if not Path(pe_seg_path).exists():
            print(f"  SKIP {config_id}: missing pe_seg file", flush=True)
            results.append({"config_id": config_id, "patient_id": patient_id,
                            "error": "missing_pe_seg"})
            continue

        config_meta = load_config_metadata(patient_id, config_id)
        if not config_meta:
            print(f"  WARN {config_id}: no grid metadata found", flush=True)
            config_meta = {"config_id": config_id, "patient_id": patient_id}

        try:
            result = evaluate_single_config(label_path, pe_seg_path, config_meta)
            det_str = "YES" if result.get("pe_detected") else "no"
            dice_str = f"{result.get('dice', 0):.3f}" if result.get("dice") is not None else "N/A"
            print(f"  [{i+1}/{len(patient_entries)}] {config_id}: "
                  f"intended={result.get('intended_voxels', '?')}, "
                  f"detected={result.get('detected_voxels', '?')}, "
                  f"pe_found={det_str}, dice={dice_str}", flush=True)
        except Exception as e:
            print(f"  ERROR {config_id}: {e}", flush=True)
            result = {"config_id": config_id, "patient_id": patient_id,
                      "error": str(e)}

        results.append(result)

    # Write per-patient CSV
    per_config_dir = output_dir / "per_config"
    per_config_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_config_dir / f"{patient_id}_evaluation.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"  Wrote {csv_path} ({len(results)} rows)")
    return results


# ── Aggregation ────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "dice", "iou", "sensitivity", "precision", "hausdorff_95_mm",
    "volume_ratio", "component_detection_rate", "lobe_concordance",
]


def compute_aggregates(output_dir: Path):
    """Concatenate per-patient CSVs and compute grouped summaries."""
    per_config_dir = output_dir / "per_config"
    csv_files = sorted(per_config_dir.glob("*_evaluation.csv"))

    if not csv_files:
        print("No per-patient evaluation CSVs found.")
        return

    # Concatenate
    dfs = [pd.read_csv(f) for f in csv_files]
    all_df = pd.concat(dfs, ignore_index=True)
    all_path = output_dir / "evaluation_all_configs.csv"
    all_df.to_csv(all_path, index=False)
    print(f"\nConcatenated {len(all_df)} configs → {all_path}")

    # Filter to configs with intended PE
    valid = all_df[all_df["intended_voxels"] > 0].copy()
    print(f"  Valid configs (intended_voxels > 0): {len(valid)}")

    # Volume bins
    bins = [0, 0.01, 0.1, 1.0, float("inf")]
    labels = ["0-0.01ml", "0.01-0.1ml", "0.1-1.0ml", "1.0+ml"]
    valid["volume_bin"] = pd.cut(valid["intended_volume_ml"], bins=bins, labels=labels)

    # Group by each categorical variable
    group_vars = ["location", "occlusion", "burden", "severity_category", "volume_bin"]
    agg_rows = []

    for gvar in group_vars:
        grouped = valid.groupby(gvar, observed=True)
        for gval, gdf in grouped:
            row = {"group_variable": gvar, "group_value": str(gval), "n_configs": len(gdf)}
            row["detection_rate"] = gdf["pe_detected"].mean()

            for metric in METRIC_COLS:
                col = pd.to_numeric(gdf[metric], errors="coerce")
                row[f"{metric}_mean"] = col.mean()
                row[f"{metric}_median"] = col.median()
                row[f"{metric}_std"] = col.std()

            agg_rows.append(row)

    # Overall row
    overall = {"group_variable": "overall", "group_value": "all", "n_configs": len(valid)}
    overall["detection_rate"] = valid["pe_detected"].mean()
    for metric in METRIC_COLS:
        col = pd.to_numeric(valid[metric], errors="coerce")
        overall[f"{metric}_mean"] = col.mean()
        overall[f"{metric}_median"] = col.median()
        overall[f"{metric}_std"] = col.std()
    agg_rows.insert(0, overall)

    agg_df = pd.DataFrame(agg_rows)
    agg_path = output_dir / "evaluation_aggregate.csv"
    agg_df.to_csv(agg_path, index=False, float_format="%.4f")
    print(f"  Aggregate summary → {agg_path}")

    # Print summary
    print_summary(valid, output_dir)


def print_summary(df: pd.DataFrame, output_dir: Path):
    """Print and save a human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 6 EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total configs evaluated: {len(df)}")
    lines.append(f"Unique patients: {df['patient_id'].nunique()}")
    lines.append("")

    # Detection rates
    det_rate = df["pe_detected"].mean() * 100
    lines.append(f"Overall PE detection rate: {det_rate:.1f}%")
    lines.append("")

    # By location
    lines.append("Detection rate by PE location:")
    for loc, gdf in df.groupby("location", observed=True):
        rate = gdf["pe_detected"].mean() * 100
        lines.append(f"  {loc:15s}  {rate:5.1f}%  (n={len(gdf)})")
    lines.append("")

    # Overlap metrics (only where PE was detected)
    det_df = df[df["pe_detected"]].copy()
    if len(det_df) > 0:
        lines.append(f"Spatial overlap (among {len(det_df)} configs with detected PE):")
        for metric in ["dice", "iou", "sensitivity", "precision"]:
            col = pd.to_numeric(det_df[metric], errors="coerce")
            lines.append(f"  {metric:15s}  mean={col.mean():.3f}  median={col.median():.3f}  std={col.std():.3f}")

        h95 = pd.to_numeric(det_df["hausdorff_95_mm"], errors="coerce").dropna()
        if len(h95) > 0:
            lines.append(f"  {'hausdorff_95mm':15s}  mean={h95.mean():.1f}  median={h95.median():.1f}  std={h95.std():.1f}")
        lines.append("")

        lines.append("Volume comparison (detected cases):")
        vr = pd.to_numeric(det_df["volume_ratio"], errors="coerce").dropna()
        lines.append(f"  Volume ratio (detected/intended):  mean={vr.mean():.2f}  median={vr.median():.2f}")
        lines.append("")

        lines.append("Component detection:")
        cdr = pd.to_numeric(det_df["component_detection_rate"], errors="coerce").dropna()
        lines.append(f"  Component detection rate:  mean={cdr.mean():.2f}  median={cdr.median():.2f}")
        lines.append("")

        lines.append("Lobe concordance (detected cases):")
        lc = pd.to_numeric(det_df["lobe_concordance"], errors="coerce").dropna()
        lines.append(f"  Lobe concordance:  mean={lc.mean():.2f}  median={lc.median():.2f}")
    else:
        lines.append("No PE detected in any config.")

    lines.append("")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved to {summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Evaluate synthetic PE vs intended PE masks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient-id", type=str, help="Evaluate a single patient")
    group.add_argument("--all", action="store_true", help="Evaluate all patients")
    group.add_argument("--aggregate-only", action="store_true",
                       help="Aggregate existing per-patient CSVs (skip evaluation)")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: phase6_evaluation/output/)")
    parser.add_argument("--predictions-csv", type=str, default=None,
                        help="Path to predictions CSV")
    parser.add_argument("--grid-base-dir", type=str, default=None,
                        help="Path to grid metadata base dir (patients/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else PREDICTIONS_CSV

    global GRID_BASE_DIR
    if args.grid_base_dir:
        GRID_BASE_DIR = Path(args.grid_base_dir)

    if args.aggregate_only:
        compute_aggregates(output_dir)
        return

    # Build file index
    file_index = build_file_index(predictions_csv,
                                  patient_id=args.patient_id if args.patient_id else None)
    print(f"File index: {len(file_index)} configs")

    if args.patient_id:
        evaluate_patient(args.patient_id, file_index, output_dir)
    else:
        # All patients
        patient_ids = sorted(set(e["patient_id"] for e in file_index))
        print(f"Processing {len(patient_ids)} patients")
        for pid in patient_ids:
            evaluate_patient(pid, file_index, output_dir)

        # Auto-aggregate
        compute_aggregates(output_dir)


if __name__ == "__main__":
    main()
