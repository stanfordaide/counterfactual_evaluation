"""
Quality control script for MAISI ControlNet generated CT scans.

Detects common artifacts:
1. Sliding window banding/seam artifacts (from autoencoder patch decoding)
2. Abnormal intensity statistics (too flat or too noisy)
3. Excessive air-like voxels inside body mask

Usage:
    python scripts/qc_generated_scans.py \
        --pred-dir /path/to/predictions \
        --output-csv /path/to/qc_results.csv \
        --patch-size 80 --overlap 0.4
"""

import argparse
import csv
import glob
import os
import sys

import nibabel as nib
import numpy as np


def compute_banding_metrics(img_data, patch_size=80, overlap=0.4):
    """
    Detect sliding-window seam artifacts by comparing gradient magnitude
    at expected patch boundaries vs. elsewhere.

    Returns dict with per-axis metrics:
        - mean_grad: average absolute gradient along the axis
        - boundary_grad: average gradient at patch boundary slices
        - non_boundary_grad: average gradient at non-boundary slices
        - ratio: boundary_grad / non_boundary_grad (>1 means banding)
    """
    step = int(patch_size * (1 - overlap))
    metrics = {}

    for axis, name in enumerate(["x", "y", "z"]):
        grad = np.abs(np.diff(img_data, axis=axis))
        other_axes = tuple(i for i in range(3) if i != axis)
        slice_mean_grad = np.mean(grad, axis=other_axes)
        n = len(slice_mean_grad)

        # Boundary slices (+/-1 around expected seams)
        boundary_set = set()
        for b in range(step, n, step):
            for offset in [-1, 0, 1]:
                if 0 <= b + offset < n:
                    boundary_set.add(b + offset)
        non_boundary = [i for i in range(n) if i not in boundary_set]

        if boundary_set and non_boundary:
            bg = np.mean([slice_mean_grad[i] for i in boundary_set])
            nbg = np.mean([slice_mean_grad[i] for i in non_boundary])
            ratio = bg / nbg if nbg > 0 else 0.0
        else:
            bg = nbg = ratio = 0.0

        metrics[name] = {
            "mean_grad": float(np.mean(slice_mean_grad)),
            "boundary_grad": float(bg),
            "non_boundary_grad": float(nbg),
            "ratio": float(ratio),
        }

    return metrics


def compute_intensity_metrics(img_data):
    """
    Compute intensity statistics for body region (voxels > -900 HU).
    """
    body_mask = img_data > -900
    body_frac = body_mask.sum() / img_data.size

    if not body_mask.any():
        return {
            "body_fraction": 0.0,
            "body_mean": None,
            "body_std": None,
            "body_p1": None,
            "body_p99": None,
            "air_in_body_frac": None,
        }

    body_vals = img_data[body_mask]
    return {
        "body_fraction": float(body_frac),
        "body_mean": float(np.mean(body_vals)),
        "body_std": float(np.std(body_vals)),
        "body_p1": float(np.percentile(body_vals, 1)),
        "body_p99": float(np.percentile(body_vals, 99)),
        "air_in_body_frac": float((body_vals < -500).mean()),
    }


def compute_body_region_metrics(label_data):
    """
    Check that key chest anatomy is sufficiently present in the label map.

    Expected labels for chest CT:
        Left lung lobes:  10 (upper), 11 (lower)
        Right lung lobes: 12 (upper), 13 (middle), 14 (lower)
        Heart: 51
        Lung vessels: 120
    """
    LEFT_LUNG = [10, 11]
    RIGHT_LUNG = [12, 13, 14]
    HEART = [51]
    LUNG_VESSELS = [120]

    total = label_data.size
    left_lung_frac = float(np.isin(label_data, LEFT_LUNG).sum() / total)
    right_lung_frac = float(np.isin(label_data, RIGHT_LUNG).sum() / total)
    heart_frac = float(np.isin(label_data, HEART).sum() / total)
    lung_vessels_frac = float(np.isin(label_data, LUNG_VESSELS).sum() / total)

    present_labels = set(np.unique(label_data).astype(int))
    missing_left = sorted(set(LEFT_LUNG) - present_labels)
    missing_right = sorted(set(RIGHT_LUNG) - present_labels)
    missing_heart = 51 not in present_labels
    missing_vessels = 120 not in present_labels

    return {
        "left_lung_frac": left_lung_frac,
        "right_lung_frac": right_lung_frac,
        "heart_frac": heart_frac,
        "lung_vessels_frac": lung_vessels_frac,
        "missing_left_lobes": missing_left,
        "missing_right_lobes": missing_right,
        "missing_heart": missing_heart,
        "missing_vessels": missing_vessels,
    }


def qc_single_scan(
    image_path,
    patch_size=80,
    overlap=0.4,
    banding_ratio_thresh=1.15,
    z_grad_thresh=35.0,
    body_std_low=180.0,
    body_std_high=400.0,
    min_lung_frac=0.005,
    min_heart_frac=0.002,
):
    """
    Run QC checks on a single generated scan.

    Returns:
        dict with QC metrics and flags.
    """
    img = nib.load(image_path)
    img_data = img.get_fdata()

    # Check for label file, shape mismatch, and body region completeness
    label_path = image_path.replace("_image.nii.gz", "_label.nii.gz")
    label_shape_mismatch = False
    body_region = None
    if os.path.exists(label_path):
        lbl = nib.load(label_path)
        lbl_shape = lbl.header.get_data_shape()
        if lbl_shape != img_data.shape:
            label_shape_mismatch = True
        lbl_data = lbl.get_fdata()
        body_region = compute_body_region_metrics(lbl_data)

    banding = compute_banding_metrics(img_data, patch_size, overlap)
    intensity = compute_intensity_metrics(img_data)

    # Flag conditions - only artifact-related issues trigger regeneration
    flags = []
    warnings = []  # informational, do not trigger regeneration

    # 1. Banding detection
    max_ratio = max(banding[ax]["ratio"] for ax in ["x", "y", "z"])
    z_grad = banding["z"]["mean_grad"]

    if z_grad > z_grad_thresh:
        flags.append(f"high_z_gradient({z_grad:.1f})")

    for ax in ["x", "y", "z"]:
        if banding[ax]["ratio"] > banding_ratio_thresh:
            flags.append(f"banding_{ax}(ratio={banding[ax]['ratio']:.3f})")

    # 2. Intensity checks
    if intensity["body_std"] is not None:
        if intensity["body_std"] < body_std_low:
            flags.append(f"low_body_std({intensity['body_std']:.1f})")
        if intensity["body_std"] > body_std_high:
            flags.append(f"high_body_std({intensity['body_std']:.1f})")

    # 3. Body fraction (too small = mostly air)
    if intensity["body_fraction"] < 0.15:
        flags.append(f"low_body_fraction({intensity['body_fraction']:.3f})")

    # 4. Body region completeness (label-based)
    if body_region is not None:
        if body_region["missing_left_lobes"]:
            flags.append(f"missing_left_lobes({body_region['missing_left_lobes']})")
        if body_region["missing_right_lobes"]:
            flags.append(f"missing_right_lobes({body_region['missing_right_lobes']})")
        if body_region["missing_heart"]:
            flags.append("missing_heart")
        if body_region["left_lung_frac"] < min_lung_frac:
            flags.append(f"insufficient_left_lung({body_region['left_lung_frac']:.4f})")
        if body_region["right_lung_frac"] < min_lung_frac:
            flags.append(f"insufficient_right_lung({body_region['right_lung_frac']:.4f})")
        if body_region["heart_frac"] < min_heart_frac:
            flags.append(f"insufficient_heart({body_region['heart_frac']:.4f})")

    # 5. Label-image shape mismatch (informational only - expected when
    #    saved label is at original resolution and image at output_size)
    if label_shape_mismatch:
        warnings.append("label_shape_mismatch")

    return {
        "image_path": image_path,
        "shape": img_data.shape,
        "z_grad": z_grad,
        "z_ratio": banding["z"]["ratio"],
        "x_ratio": banding["x"]["ratio"],
        "y_ratio": banding["y"]["ratio"],
        "max_ratio": max_ratio,
        "body_std": intensity["body_std"],
        "body_mean": intensity["body_mean"],
        "body_fraction": intensity["body_fraction"],
        "air_in_body_frac": intensity["air_in_body_frac"],
        "left_lung_frac": body_region["left_lung_frac"] if body_region else None,
        "right_lung_frac": body_region["right_lung_frac"] if body_region else None,
        "heart_frac": body_region["heart_frac"] if body_region else None,
        "lung_vessels_frac": body_region["lung_vessels_frac"] if body_region else None,
        "flags": flags,
        "warnings": warnings,
        "needs_regeneration": len(flags) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="QC for MAISI ControlNet generated scans")
    parser.add_argument("--pred-dir", required=True, help="Directory containing generated scans")
    parser.add_argument("--output-csv", default=None, help="Path to save QC results CSV")
    parser.add_argument("--patch-size", type=int, default=80, help="Sliding window patch size used during generation")
    parser.add_argument("--overlap", type=float, default=0.4, help="Sliding window overlap used during generation")
    parser.add_argument("--banding-ratio-thresh", type=float, default=1.15,
                        help="Boundary/non-boundary gradient ratio threshold for flagging banding")
    parser.add_argument("--z-grad-thresh", type=float, default=35.0,
                        help="Absolute z-axis gradient threshold for flagging severe banding")
    parser.add_argument("--body-std-low", type=float, default=180.0,
                        help="Minimum body HU std (below = too uniform/flat)")
    parser.add_argument("--body-std-high", type=float, default=400.0,
                        help="Maximum body HU std (above = too noisy)")
    parser.add_argument("--min-lung-frac", type=float, default=0.005,
                        help="Minimum volume fraction for each lung side (below = cut-off lung)")
    parser.add_argument("--min-heart-frac", type=float, default=0.002,
                        help="Minimum volume fraction for heart (below = missing/cut-off heart)")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Start index (inclusive) for processing a subset of scans (for array jobs)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index (exclusive) for processing a subset of scans (for array jobs)")
    parser.add_argument("--case-list", default=None,
                        help="Path to a text file with case IDs (one per line) to restrict QC to only those cases")
    parser.add_argument("--delete-flagged", action="store_true", default=False,
                        help="Delete flagged image/label/pe_seg files so they can be re-generated")
    parser.add_argument("--tally", default=None,
                        help="Path to cumulative tally file tracking flagged cases across QC rounds")
    args = parser.parse_args()

    image_files = sorted(glob.glob(os.path.join(args.pred_dir, "*_image.nii.gz")))
    if not image_files:
        print(f"No *_image.nii.gz files found in {args.pred_dir}")
        sys.exit(1)

    # Filter to specific case IDs if provided
    if args.case_list is not None:
        with open(args.case_list) as f:
            allowed_ids = set(line.strip() for line in f if line.strip())
        image_files = [
            p for p in image_files
            if os.path.basename(p).split("_pred_")[0] in allowed_ids
        ]
        print(f"Filtered to {len(image_files)} scans matching {len(allowed_ids)} case IDs from {args.case_list}")

    # Subset for array jobs
    total = len(image_files)
    if args.start_idx is not None or args.end_idx is not None:
        s = args.start_idx or 0
        e = args.end_idx or total
        image_files = image_files[s:e]
        print(f"Array task: processing indices [{s}:{e}) out of {total} total scans")

    print(f"Running QC on {len(image_files)} scans in {args.pred_dir}")
    print(f"Params: patch_size={args.patch_size}, overlap={args.overlap}, "
          f"banding_ratio_thresh={args.banding_ratio_thresh}, z_grad_thresh={args.z_grad_thresh}")
    print()

    results = []
    flagged = []

    for i, img_path in enumerate(image_files):
        case_id = os.path.basename(img_path).split("_pred_")[0]
        print(f"[{i+1}/{len(image_files)}] {case_id}...", end=" ", flush=True)

        result = qc_single_scan(
            img_path,
            patch_size=args.patch_size,
            overlap=args.overlap,
            banding_ratio_thresh=args.banding_ratio_thresh,
            z_grad_thresh=args.z_grad_thresh,
            body_std_low=args.body_std_low,
            body_std_high=args.body_std_high,
            min_lung_frac=args.min_lung_frac,
            min_heart_frac=args.min_heart_frac,
        )
        result["case_id"] = case_id
        results.append(result)

        if result["needs_regeneration"]:
            flagged.append(result)
            print(f"FLAGGED: {'; '.join(result['flags'])}")
        elif result["warnings"]:
            print(f"OK (warnings: {'; '.join(result['warnings'])})")
        else:
            print("OK")

    # Summary
    print(f"\n{'='*60}")
    print(f"QC Summary: {len(flagged)}/{len(results)} scans flagged for regeneration")
    print(f"{'='*60}")

    if flagged:
        print("\nFlagged scans:")
        for r in flagged:
            print(f"  {r['case_id']}: {'; '.join(r['flags'])}")

        print("\nImage IDs to regenerate:")
        for r in flagged:
            print(f"  {r['case_id']}")

    # Save CSV
    if args.output_csv is None:
        args.output_csv = os.path.join(args.pred_dir, "qc_results.csv")

    fieldnames = [
        "case_id", "shape", "needs_regeneration", "flags", "warnings",
        "z_grad", "z_ratio", "x_ratio", "y_ratio", "max_ratio",
        "body_std", "body_mean", "body_fraction", "air_in_body_frac",
        "left_lung_frac", "right_lung_frac", "heart_frac", "lung_vessels_frac",
        "image_path",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["shape"] = "x".join(str(s) for s in r["shape"])
            row["flags"] = "; ".join(r["flags"]) if r["flags"] else ""
            row["warnings"] = "; ".join(r["warnings"]) if r["warnings"] else ""
            # Round floats
            for k in ["z_grad", "z_ratio", "x_ratio", "y_ratio", "max_ratio",
                       "body_std", "body_mean", "body_fraction", "air_in_body_frac",
                       "left_lung_frac", "right_lung_frac", "heart_frac", "lung_vessels_frac"]:
                if row.get(k) is not None:
                    row[k] = round(row[k], 4)
            writer.writerow(row)

    print(f"\nResults saved to: {args.output_csv}")

    # Also save list of IDs to regenerate
    regen_list_path = os.path.join(os.path.dirname(args.output_csv), "regenerate_list.txt")
    with open(regen_list_path, "w") as f:
        for r in flagged:
            f.write(r["case_id"] + "\n")
    print(f"Regeneration list saved to: {regen_list_path}")

    # Update cumulative tally across QC rounds
    if args.tally is None:
        args.tally = os.path.join(os.path.dirname(args.output_csv), "qc_tally.csv")

    # Load existing tally if present
    tally = {}
    if os.path.exists(args.tally):
        with open(args.tally) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tally[row["case_id"]] = {
                    "times_flagged": int(row["times_flagged"]),
                    "times_checked": int(row["times_checked"]),
                    "last_flags": row["last_flags"],
                }

    # Update tally with current round
    for r in results:
        cid = r["case_id"]
        if cid not in tally:
            tally[cid] = {"times_flagged": 0, "times_checked": 0, "last_flags": ""}
        tally[cid]["times_checked"] += 1
        if r["needs_regeneration"]:
            tally[cid]["times_flagged"] += 1
            tally[cid]["last_flags"] = "; ".join(r["flags"])
        else:
            tally[cid]["last_flags"] = ""

    with open(args.tally, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "times_flagged", "times_checked", "last_flags"])
        writer.writeheader()
        for cid in sorted(tally.keys()):
            writer.writerow({"case_id": cid, **tally[cid]})

    repeat_offenders = {cid: v for cid, v in tally.items() if v["times_flagged"] >= 2}
    if repeat_offenders:
        print(f"\nRepeat offenders (flagged >=2 times):")
        for cid, v in sorted(repeat_offenders.items(), key=lambda x: x[1]["times_flagged"], reverse=True):
            print(f"  {cid}: flagged {v['times_flagged']}/{v['times_checked']} times")

    print(f"Tally updated: {args.tally}")

    # Delete flagged files so --skip-existing won't skip them on re-run
    if args.delete_flagged and flagged:
        deleted = 0
        for r in flagged:
            img_path = r["image_path"]
            # Delete image, label, and pe_seg files with the same prefix
            prefix = img_path.replace("_image.nii.gz", "")
            for suffix in ["_image.nii.gz", "_label.nii.gz", "_pe_seg.nii.gz"]:
                path = prefix + suffix
                if os.path.exists(path):
                    os.remove(path)
                    deleted += 1
        print(f"Deleted {deleted} files for {len(flagged)} flagged cases")


if __name__ == "__main__":
    main()
