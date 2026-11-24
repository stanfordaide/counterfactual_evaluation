"""Run segmentation evaluation with slice-overlap detection criterion.

Instead of counting PE as 'detected' when any voxel is found anywhere in the
volume, this wrapper redefines detection as: the intended and detected PE masks
share at least one axial slice (optionally with a ±N slice margin).

Creates a modified vessel overlay CSV with pe_detected overridden by
slice_overlap, then calls the existing evaluation pipeline unchanged.

Usage (from 04_evaluation_and_analysis/):
    python -m analysis.run_slicematch_evaluation

Inputs (detector runs, not shipped; see README): CF_RAW/pe_detector_runs/nnunet/
    vessel_overlay_all_configs.csv (with mask paths) and nope_baseline_detection.csv.
Output per margin (the paper uses margin 5):
    output/results_slicematch_m{N}/segmentation_model/
    output/figures_slicematch_m{N}/segmentation_model/
"""

import os
from pathlib import Path

import pandas as pd


# ── nnU-Net paths ───────────────────────────────────────────────────────────
RES = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
RAW = Path(os.environ.get("CF_RAW", "/path/to/private_data"))
OUT = Path(__file__).resolve().parents[1] / "output"
NNUNET_VESSEL_CSV = str(RAW / "pe_detector_runs/nnunet/vessel_overlay_all_configs.csv")
NOPE_BASELINE_CSV = str(RAW / "pe_detector_runs/nnunet/nope_baseline_detection.csv")

MASTER_CSV = str(RES / "cohort/master_metadata.csv")

MARGINS = [5]  # paper uses strict +/-5 axial-slice overlap
SLICE_OVERLAP_CSV = str(OUT / "detector_outputs/nnunet/slice_overlap.csv")


def create_slicematch_vessel_csv(
    original_csv: str,
    slice_overlap_csv: str,
    output_csv: str,
    margin: int = 0,
) -> str:
    """Merge slice overlap results into vessel overlay CSV.

    Replaces ``pe_detected`` with the slice-overlap criterion (at the given
    margin) and saves a modified CSV that the existing evaluation pipeline
    can consume directly.
    """
    vessel_df = pd.read_csv(original_csv)
    overlap_df = pd.read_csv(slice_overlap_csv)

    sfx = f"_m{margin}"

    # Keep only the columns we need from the overlap CSV
    overlap_cols = ["config_id", "n_intended_slices", "n_detected_slices"]
    # Add margin-specific columns
    for col in [f"n_overlap_slices{sfx}", f"slice_overlap{sfx}",
                f"slice_recall{sfx}", f"detected_voxels_on_overlap_slices{sfx}"]:
        if col in overlap_df.columns:
            overlap_cols.append(col)

    ok = overlap_df[overlap_df["error"].isna()][overlap_cols].copy()

    # Rename margin-suffixed columns to generic names for the pipeline
    rename_map = {
        f"n_overlap_slices{sfx}": "n_overlap_slices",
        f"slice_overlap{sfx}": "slice_overlap",
        f"slice_recall{sfx}": "slice_recall",
        f"detected_voxels_on_overlap_slices{sfx}": "detected_voxels_on_overlap_slices",
    }
    ok = ok.rename(columns={k: v for k, v in rename_map.items() if k in ok.columns})

    # Merge
    merged = vessel_df.merge(ok, on="config_id", how="left")

    # Preserve original pe_detected, then override
    merged["pe_detected_any"] = merged["pe_detected"]
    merged["pe_detected"] = merged["slice_overlap"].fillna(False).astype(bool)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    n_any = vessel_df["pe_detected"].sum()
    n_slice = merged["pe_detected"].sum()
    print(f"Created slicematch vessel CSV (margin={margin}): {output_csv}")
    print(f"  pe_detected (any voxel): {n_any}")
    print(f"  pe_detected (slice overlap, margin={margin}): {n_slice}")
    print(f"  Reclassified as missed: {n_any - n_slice}")

    return output_csv


def run_one_margin(margin: int, vessel_csv: str, nope_csv: str,
                   slice_csv: str, master_csv: str,
                   results_base: str, figures_base: str,
                   model_prefix: str,
                   xlim_barplots: tuple[float, float] | None = None):
    """Run the full evaluation pipeline for a single margin value."""
    from analysis.run_segmentation_evaluation import run_segmentation_evaluation
    from analysis.pe_drivers_gee import load, run_cluster_robust_model

    output_dir = f"{results_base}_m{margin}/segmentation_model"
    figure_dir = f"{figures_base}_m{margin}/segmentation_model"
    model_name = f"{model_prefix}_m{margin}"

    print("\n" + "#" * 60)
    print(f"# MARGIN = {margin} slices")
    print("#" * 60)

    # Step 1: Create modified vessel overlay CSV
    print("\n" + "=" * 60)
    print(f"Creating slicematch vessel overlay CSV (margin={margin})")
    print("=" * 60)
    modified_csv = f"{output_dir}/vessel_overlay_slicematch.csv"
    create_slicematch_vessel_csv(vessel_csv, slice_csv, modified_csv, margin=margin)

    # Step 2: Segmentation stress-test
    print("\n" + "=" * 60)
    print("Segmentation stress-test evaluation")
    print("=" * 60)
    run_segmentation_evaluation(
        vessel_overlay_csv=modified_csv,
        nope_baseline_csv=nope_csv,
        output_dir=output_dir,
        figure_dir=figure_dir,
        model_name=model_name,
        xlim_barplots=xlim_barplots,
    )

    # Step 3: driver model (cluster-robust GEE) -> glmm_results.csv
    print("\n" + "=" * 60)
    print("Driver model (cluster-robust GEE)")
    print("=" * 60)
    df = load(analysis_csv=f"{output_dir}/analysis_dataframe.csv", master_csv=master_csv)
    run_cluster_robust_model(df, results_dir=output_dir)

    print(f"\n  Results: {output_dir}/")
    print(f"  Figures: {figure_dir}/")


def main():
    from compute_slice_overlap import compute_all

    # Step 0: Compute slice overlaps for all margins (if not cached)
    slice_csv = Path(SLICE_OVERLAP_CSV)
    if slice_csv.exists():
        print(f"Slice overlap CSV already exists: {slice_csv}")
        print("  Delete it to force recomputation.\n")
    else:
        print("=" * 60)
        print(f"Computing slice overlaps (margins={MARGINS})")
        print("=" * 60)
        compute_all(NNUNET_VESSEL_CSV, str(slice_csv), margins=MARGINS)

    # Compute standardized x-axis for bar plots: use the strictest margin
    # (m=0, lowest detection rates) to determine the global lower bound.
    overlap_df = pd.read_csv(str(slice_csv))
    ok = overlap_df[overlap_df["error"].isna()]
    min_margin = min(MARGINS)
    col = f"slice_overlap_m{min_margin}"
    global_det_rate = ok[col].mean()
    # Set xlim to start well below the lowest group rate (use 60% of overall as floor)
    x_lo = max(0, round(global_det_rate * 0.6 * 20) / 20)
    xlim = (x_lo, 1.00)
    print(f"\nStandardized bar plot x-axis: {xlim}")

    # Run evaluation for each margin
    for margin in MARGINS:
        run_one_margin(
            margin=margin,
            vessel_csv=NNUNET_VESSEL_CSV,
            nope_csv=NOPE_BASELINE_CSV,
            slice_csv=str(slice_csv),
            master_csv=MASTER_CSV,
            results_base=str(OUT / "results_slicematch"),
            figures_base=str(OUT / "figures_slicematch"),
            model_prefix="nnunet_pe_seg_slicematch",
            xlim_barplots=xlim,
        )

    print("\n" + "=" * 60)
    print("ALL DONE - nnU-Net slicematch evaluation complete")
    print(f"  Margins evaluated: {MARGINS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
