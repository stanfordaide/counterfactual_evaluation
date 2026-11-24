"""
Detect interleaved/corrupted CT scans where two series are merged into one volume.

These scans show a periodic oscillation in body cross-section and mean HU between
adjacent slices (e.g., alternating between arms-up and arms-down reconstructions).
The diffusion model learns to reproduce this pattern, causing banding artifacts
in generated images.

Usage:
    python scripts/detect_interleaved_cts.py \
        --ct-dir /path/to/CTPA \
        --output-csv /path/to/interleaved_cts.csv \
        --workers 8
"""

import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np


def detect_interleaving(ct_path, body_thresh=-900, min_oscillations=5, jump_thresh=0.08):
    """
    Detect if a CT scan has interleaved slices from two different series.

    Checks for periodic oscillation in body cross-section fraction along z.
    Interleaved scans show a pattern where body% alternates between two values
    (e.g., 68% and 51%) every N slices.

    Args:
        ct_path: path to NIfTI file
        body_thresh: HU threshold for body vs air (default -900)
        min_oscillations: minimum number of oscillation cycles to flag
        jump_thresh: minimum body fraction jump to count as oscillation (default 0.08 = 8%)

    Returns:
        dict with detection results
    """
    case_id = os.path.basename(ct_path).replace('.nii.gz', '')

    try:
        img = nib.load(ct_path)
        data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        return {"case_id": case_id, "error": str(e)}

    shape = data.shape
    spacing_z = abs(img.affine[2, 2])

    # Compute body fraction per z-slice
    body_frac = (data > body_thresh).sum(axis=(0, 1)) / (shape[0] * shape[1])

    # Detect oscillation: look for alternating large jumps in body fraction
    body_jumps = np.diff(body_frac)
    large_jumps = np.abs(body_jumps) > jump_thresh

    # Count sign-alternating large jumps (up-down-up-down pattern)
    oscillation_count = 0
    last_sign = None
    oscillation_start = None
    oscillation_end = None

    for i in range(len(body_jumps)):
        if not large_jumps[i]:
            continue
        sign = 1 if body_jumps[i] > 0 else -1
        if last_sign is not None and sign != last_sign:
            oscillation_count += 1
            oscillation_end = i
            if oscillation_start is None:
                oscillation_start = i
        last_sign = sign

    # Also compute the mean HU jump magnitude in the oscillating region
    mean_hu_jump = 0.0
    if oscillation_start is not None and oscillation_end is not None:
        region = slice(oscillation_start, oscillation_end + 1)
        # Compute mean HU per slice in the oscillating region
        mean_hu = []
        for z in range(oscillation_start, min(oscillation_end + 2, shape[2])):
            s = data[:, :, z]
            body = s[s > body_thresh]
            mean_hu.append(body.mean() if len(body) > 0 else -1000)
        if len(mean_hu) > 1:
            mean_hu_jump = np.mean(np.abs(np.diff(mean_hu)))

    # Compute body fraction variability (std of body fraction where body exists)
    active_slices = body_frac > 0.05
    body_frac_std = body_frac[active_slices].std() if active_slices.any() else 0

    is_interleaved = oscillation_count >= min_oscillations

    return {
        "case_id": case_id,
        "shape": shape,
        "spacing_z": round(spacing_z, 4),
        "z_coverage_mm": round(shape[2] * spacing_z, 1),
        "oscillation_count": oscillation_count,
        "oscillation_start_z": oscillation_start,
        "oscillation_end_z": oscillation_end,
        "body_frac_std": round(body_frac_std, 4),
        "body_frac_min": round(body_frac[active_slices].min(), 4) if active_slices.any() else 0,
        "body_frac_max": round(body_frac.max(), 4),
        "mean_hu_jump": round(mean_hu_jump, 1),
        "is_interleaved": is_interleaved,
        "error": None,
    }


def process_one(ct_path):
    return detect_interleaving(ct_path)


def main():
    parser = argparse.ArgumentParser(description="Detect interleaved/corrupted CT scans")
    parser.add_argument("--ct-dir", required=True, help="Directory containing source CT NIfTI files")
    parser.add_argument("--output-csv", default=None, help="Path to save results CSV")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--min-oscillations", type=int, default=5,
                        help="Minimum oscillation cycles to flag as interleaved")
    parser.add_argument("--jump-thresh", type=float, default=0.08,
                        help="Minimum body fraction jump (0-1) to count as oscillation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files to process (0 = all)")
    parser.add_argument("--case-list", default=None,
                        help="Optional text file with case IDs to check (one per line)")
    args = parser.parse_args()

    if args.case_list:
        with open(args.case_list) as f:
            case_ids = [line.strip() for line in f if line.strip()]
        ct_files = [os.path.join(args.ct_dir, f"{cid}.nii.gz") for cid in case_ids]
        ct_files = [f for f in ct_files if os.path.exists(f)]
    else:
        ct_files = sorted(glob.glob(os.path.join(args.ct_dir, "*.nii.gz")))

    if args.limit > 0:
        ct_files = ct_files[:args.limit]

    if not ct_files:
        print(f"No .nii.gz files found in {args.ct_dir}")
        sys.exit(1)

    print(f"Scanning {len(ct_files)} CT scans for interleaving artifacts...")
    print(f"Params: min_oscillations={args.min_oscillations}, jump_thresh={args.jump_thresh}")

    results = []
    flagged = []
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, f): f for f in ct_files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            if result.get("error"):
                errors.append(result)
                status = f"ERROR: {result['error']}"
            elif result["is_interleaved"]:
                flagged.append(result)
                status = f"INTERLEAVED (oscillations={result['oscillation_count']}, hu_jump={result['mean_hu_jump']:.0f})"
            else:
                status = "OK"

            if (i + 1) % 100 == 0 or result.get("is_interleaved"):
                print(f"[{i+1}/{len(ct_files)}] {result['case_id']}: {status}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Scan complete: {len(flagged)}/{len(results)} interleaved, {len(errors)} errors")
    print(f"{'='*60}")

    if flagged:
        print(f"\nInterleaved scans:")
        for r in sorted(flagged, key=lambda x: x["oscillation_count"], reverse=True):
            print(f"  {r['case_id']}: {r['oscillation_count']} oscillations, "
                  f"body% range [{r['body_frac_min']:.1%}-{r['body_frac_max']:.1%}], "
                  f"mean HU jump={r['mean_hu_jump']:.0f}")

    # Save CSV
    if args.output_csv is None:
        args.output_csv = os.path.join(args.ct_dir, "interleaved_scan_detection.csv")

    fieldnames = [
        "case_id", "is_interleaved", "oscillation_count", "mean_hu_jump",
        "body_frac_std", "body_frac_min", "body_frac_max",
        "oscillation_start_z", "oscillation_end_z",
        "shape", "spacing_z", "z_coverage_mm", "error",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.get("oscillation_count", 0), reverse=True):
            row = dict(r)
            row["shape"] = "x".join(str(s) for s in r["shape"]) if isinstance(r.get("shape"), tuple) else ""
            writer.writerow(row)

    print(f"\nResults saved to: {args.output_csv}")

    # Save list of interleaved case IDs
    if flagged:
        exclude_path = os.path.join(os.path.dirname(args.output_csv), "interleaved_case_ids.txt")
        with open(exclude_path, "w") as f:
            for r in flagged:
                f.write(r["case_id"] + "\n")
        print(f"Interleaved case IDs saved to: {exclude_path}")


if __name__ == "__main__":
    main()
