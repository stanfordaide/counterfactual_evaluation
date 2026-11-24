"""Run APES segmentation evaluation with slice-overlap detection criterion.

Same approach as run_slicematch_evaluation.py but with APES-specific paths.

Usage (from 04_evaluation_and_analysis/):
    python -m analysis.run_apes_slicematch_evaluation

Inputs (detector runs, not shipped; see README): CF_RAW/pe_detector_runs/apes/
    vessel_overlay_all_configs.csv (with mask paths) and nope_baseline_detection.csv.
Output per margin (the paper uses margin 5):
    output/results_apes_slicematch_m{N}/segmentation_model/
    output/figures_apes_slicematch_m{N}/segmentation_model/
"""

import os
from pathlib import Path


# ── APES-specific paths ─────────────────────────────────────────────────────
RES = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
RAW = Path(os.environ.get("CF_RAW", "/path/to/private_data"))
OUT = Path(__file__).resolve().parents[1] / "output"
APES_VESSEL_CSV = str(RAW / "pe_detector_runs/apes/vessel_overlay_all_configs.csv")

NOPE_BASELINE_CSV = str(RAW / "pe_detector_runs/apes/nope_baseline_detection.csv")
NOPE_FALLBACK_CSV = str(RAW / "pe_detector_runs/nnunet/nope_baseline_detection.csv")

MASTER_CSV = str(RES / "cohort/master_metadata.csv")

MARGINS = [5]  # paper uses strict +/-5 axial-slice overlap
SLICE_OVERLAP_CSV = str(OUT / "detector_outputs/apes/slice_overlap.csv")


def main():
    from analysis.run_slicematch_evaluation import run_one_margin
    from compute_slice_overlap import compute_all

    # Pick APES nope baseline if available, else fall back to nnU-Net's
    nope_csv = NOPE_BASELINE_CSV if Path(NOPE_BASELINE_CSV).exists() else NOPE_FALLBACK_CSV
    if nope_csv == NOPE_FALLBACK_CSV:
        print(f"NOTE: Using nnU-Net nope baseline ({NOPE_FALLBACK_CSV})")
        print(f"      Place APES nope results at {NOPE_BASELINE_CSV} to use those instead.\n")

    # Step 0: Compute slice overlaps for all margins (if not cached)
    slice_csv = Path(SLICE_OVERLAP_CSV)
    if slice_csv.exists():
        print(f"Slice overlap CSV already exists: {slice_csv}")
        print("  Delete it to force recomputation.\n")
    else:
        print("=" * 60)
        print(f"Computing slice overlaps - APES (margins={MARGINS})")
        print("=" * 60)
        compute_all(APES_VESSEL_CSV, str(slice_csv), margins=MARGINS)

    # Compute standardized x-axis for bar plots
    import pandas as pd
    overlap_df = pd.read_csv(str(slice_csv))
    ok = overlap_df[overlap_df["error"].isna()]
    min_margin = min(MARGINS)
    col = f"slice_overlap_m{min_margin}"
    global_det_rate = ok[col].mean()
    x_lo = max(0, round(global_det_rate * 0.6 * 20) / 20)
    xlim = (x_lo, 1.00)
    print(f"\nStandardized bar plot x-axis: {xlim}")

    # Run evaluation for each margin
    for margin in MARGINS:
        run_one_margin(
            margin=margin,
            vessel_csv=APES_VESSEL_CSV,
            nope_csv=nope_csv,
            slice_csv=str(slice_csv),
            master_csv=MASTER_CSV,
            results_base=str(OUT / "results_apes_slicematch"),
            figures_base=str(OUT / "figures_apes_slicematch"),
            model_prefix="apes_slicematch",
            xlim_barplots=xlim,
        )

    print("\n" + "=" * 60)
    print("ALL DONE - APES slicematch evaluation complete")
    print(f"  Margins evaluated: {MARGINS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
