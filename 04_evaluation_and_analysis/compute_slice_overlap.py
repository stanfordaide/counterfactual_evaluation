"""Compute axial-slice overlap between intended and detected PE masks.

For each counterfactual config, loads the intended PE mask (label_path,
label==121) and detected PE segmentation (pe_seg_path, value>0), then
checks whether the two share at least one axial slice.  Supports a
slice-margin tolerance: the intended PE slice set is dilated by ±N slices
before checking overlap.

The detector prediction is on the grid of the generated CT. The generator
(01_pe_diffusion/scripts/infer_controlnet.py) resized the label map from nz
to D slices with nearest-neighbour interpolation (x and y unchanged), so
generated slice k shows label slice floor(k * nz / D). The intended mask is
put on the generated grid with this slice map before the slices are compared.
All slice counts and margins are on the generated grid.

Usage:
    # nnU-Net, exact match (margin=0)
    python compute_slice_overlap.py

    # nnU-Net, multiple margins in one pass
    python compute_slice_overlap.py --margins 0 2 5 10

    # APES
    python compute_slice_overlap.py \
        --vessel-csv $CF_RAW/pe_detector_runs/apes/vessel_overlay_all_configs.csv \
        --output-csv output/detector_outputs/apes/slice_overlap.csv \
        --margins 0 2 5 10

The vessel-overlay CSV must carry the local mask paths (label_path, pe_seg_path). The
public copies in CF_DATA/cohort/ have these two columns removed, so this step needs the
conditioning masks (HuggingFace dataset) and your own detector predictions.
The paper uses margin 5 (strict ±5 axial slices).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


RAW = Path(os.environ.get("CF_RAW", "/path/to/private_data"))
OUT = Path(__file__).resolve().parent / "output"
NNUNET_VESSEL_CSV = str(RAW / "pe_detector_runs/nnunet/vessel_overlay_all_configs.csv")
DEFAULT_OUTPUT_CSV = str(OUT / "detector_outputs/nnunet/slice_overlap.csv")

INTENDED_LABEL = 121  # PE label in the conditioning label NIfTI


def generator_slice_map(nz: int, d: int) -> np.ndarray:
    """Label slice index of each generated slice k = 0..d-1.

    Same index rule as torch F.interpolate(mode="nearest"), which the generator used to
    resize the label map from nz to d slices: floor(k * (nz / d)) in float32, at most nz - 1.
    """
    if nz < 1 or d < 1:
        raise ValueError(f"label and prediction grids cannot be related: nz={nz}, D={d}")
    scale = np.float32(nz) / np.float32(d)
    idx = np.floor(np.arange(d, dtype=np.float32) * scale).astype(np.int64)
    return np.minimum(idx, nz - 1)


def compute_one(label_path: str, seg_path: str,
                margins: list[int] | None = None,
                intended_label: int = INTENDED_LABEL) -> dict:
    """Compute axial-slice overlap for a single label/seg NIfTI pair.

    When *margins* is provided, results for each margin are stored under
    suffixed keys (e.g. ``slice_overlap_m0``, ``slice_overlap_m2``).
    When *margins* is None, behaves like margin=0 with unsuffixed keys
    (backward compatible).
    """
    if margins is None:
        margins_list = [0]
        use_suffix = False
    else:
        margins_list = sorted(margins)
        use_suffix = True

    try:
        lab_nii = nib.load(label_path)
        seg_nii = nib.load(seg_path)
    except Exception as e:
        return {"error": f"load_failed: {e}"}

    lab_data = lab_nii.get_fdata()
    seg_data = seg_nii.get_fdata()

    # x/y must match (the generator keeps x and y); only the slice axis was resized
    assert lab_data.ndim == 3 and seg_data.ndim == 3, (
        f"label and prediction grids cannot be related: label={lab_data.shape}, seg={seg_data.shape}")
    assert lab_data.shape[:2] == seg_data.shape[:2], (
        f"label and prediction grids cannot be related (x/y differ): "
        f"label={lab_data.shape}, seg={seg_data.shape}")

    # Put the intended mask on the generated grid (generator slice map), no truncation
    n_z = seg_data.shape[2]
    z_map = generator_slice_map(lab_data.shape[2], n_z)
    intended = (lab_data[:, :, z_map] == intended_label)
    detected = (seg_data > 0)

    # Axial slices with any PE
    intended_slices = set(np.where(intended.any(axis=(0, 1)))[0])
    detected_slices = set(np.where(detected.any(axis=(0, 1)))[0])

    n_intended = len(intended_slices)
    n_detected = len(detected_slices)

    result = {
        "n_intended_slices": n_intended,
        "n_detected_slices": n_detected,
        "error": None,
    }

    for m in margins_list:
        # Dilate intended slices by ±m
        if m == 0:
            dilated = intended_slices
        else:
            dilated = set()
            for s in intended_slices:
                dilated.update(range(max(0, s - m), min(n_z, s + m + 1)))

        overlap = dilated & detected_slices
        n_overlap = len(overlap)

        # Detected voxels only on overlapping slices
        det_voxels_overlap = 0
        if n_overlap > 0:
            overlap_idx = sorted(overlap)
            det_voxels_overlap = int(detected[:, :, overlap_idx].sum())

        n_dilated = len(dilated)
        sfx = f"_m{m}" if use_suffix else ""
        result[f"n_dilated_slices{sfx}"] = n_dilated
        result[f"n_overlap_slices{sfx}"] = n_overlap
        result[f"slice_overlap{sfx}"] = n_overlap > 0
        result[f"slice_recall{sfx}"] = n_overlap / n_dilated if n_dilated > 0 else 0.0
        result[f"detected_voxels_on_overlap_slices{sfx}"] = det_voxels_overlap

    return result


# Must be top-level for pickling
_WORKER_MARGINS = None


def _worker(args):
    """Wrapper for ProcessPoolExecutor."""
    config_id, label_path, seg_path = args
    result = compute_one(label_path, seg_path, margins=_WORKER_MARGINS)
    result["config_id"] = config_id
    return result


def _init_worker(margins):
    """Initializer for worker processes - sets the margins list."""
    global _WORKER_MARGINS
    _WORKER_MARGINS = margins


def compute_all(vessel_csv: str, output_csv: str, n_workers: int = 8,
                margins: list[int] | None = None) -> pd.DataFrame:
    """Process all valid rows in the vessel overlay CSV."""
    df = pd.read_csv(vessel_csv)

    # Keep only rows with both paths available and not error rows
    mask = (
        df["label_path"].notna()
        & df["pe_seg_path"].notna()
        & ~df["error"].isin(["missing_pe_seg", "missing_label"])
    )
    valid = df[mask]
    print(f"Total rows: {len(df)}, valid for slice overlap: {len(valid)}")
    if margins:
        print(f"  Margins: {margins}")

    tasks = [
        (row["config_id"], row["label_path"], row["pe_seg_path"])
        for _, row in valid.iterrows()
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=(margins,)) as pool:
        futures = {pool.submit(_worker, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 200 == 0:
                print(f"  Processed {i}/{len(tasks)}...")

    out_df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    # Report
    ok = out_df[out_df["error"].isna()]
    print(f"\nSaved {len(out_df)} results to {output_csv}")
    print(f"  Errors: {out_df['error'].notna().sum()}")

    if margins:
        for m in sorted(margins):
            col = f"slice_overlap_m{m}"
            det = ok[col].sum()
            print(f"  margin={m:2d}: {det}/{len(ok)} = {ok[col].mean():.1%}")
    else:
        print(f"  Slice overlap (detection): {ok['slice_overlap'].sum()}/{len(ok)} "
              f"= {ok['slice_overlap'].mean():.1%}")
        print(f"  Mean slice recall: {ok['slice_recall'].mean():.3f}")

    print(f"  Mean intended slices: {ok['n_intended_slices'].mean():.1f}")
    print(f"  Mean detected slices: {ok['n_detected_slices'].mean():.1f}")

    return out_df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vessel-csv", default=NNUNET_VESSEL_CSV,
                        help="Vessel overlay CSV with label_path and pe_seg_path columns")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV,
                        help="Output CSV path for slice overlap results")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--margins", type=int, nargs="+", default=None,
                        help="Slice margins to compute (e.g. 0 2 5 10). "
                             "If omitted, computes margin=0 only (backward compatible).")
    args = parser.parse_args()

    compute_all(args.vessel_csv, args.output_csv, args.n_workers, args.margins)


if __name__ == "__main__":
    main()
