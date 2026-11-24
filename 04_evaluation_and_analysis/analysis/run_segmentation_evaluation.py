"""Stress-test evaluation of PE segmentation model on counterfactual data.

Three-level analysis:
  Level 0 - Baseline comparison: No-PE synthetic vs counterfactual (the PE signal is real)
  Level 1 - Binary detection: Did the model find *any* PE? (pe_detected)
  Level 2 - Segmentation quality: For detected cases, how well did it segment?

Uses the vessel overlay CSV (113 columns) for fine-grained morphology features
(occlusion fraction, vessel radius, generation level, shape category, etc.)
and the no-PE baseline results for paired patient comparison.

Usage:
    python -m analysis.run_segmentation_evaluation
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression

from analysis.config import (
    PE_STRATIFICATION_AXES,
    set_nature_style,
    COLORS,
    COLOR_LIST,
    NATURE_SINGLE_COL_IN,
    NATURE_DOUBLE_COL_IN,
)
from analysis.stratified_analysis import run_stratified_analysis, compute_overall_metrics
from analysis.statistical_tests import run_all_tests
from analysis.patient_analysis import (
    compute_patient_level_metrics,
    summarize_patient_variation,
    icc_patient_effect,
)
from analysis.threshold_analysis import run_threshold_analysis
from analysis.plotting import (
    _save_fig,
    plot_failure_rate_heatmap,
    plot_forest,
    plot_sensitivity_bar_chart,
    plot_sensitivity_by_category,
    plot_patient_sensitivity,
    plot_patient_score_heatmap,
    plot_cumulative_sensitivity_curves,
)


# --- Configuration ---

RES = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
RAW = Path(os.environ.get("CF_RAW", "/path/to/private_data"))
OUT = Path(__file__).resolve().parents[1] / "output"
VESSEL_OVERLAY_CSV = str(RES / "cohort/vessel_overlay_all_configs.csv")
NOPE_BASELINE_CSV = str(RAW / "pe_detector_runs/nnunet/nope_baseline_detection.csv")
REAL_SCAN_PE_CSV = str(RAW / "pe_detector_runs/nnunet/real_scan_pe_detection.csv")
OUTPUT_DIR = str(OUT / "detector_outputs/nnunet")
FIGURE_DIR = str(OUT / "figures/segmentation_model")
MODEL_NAME = "nnunet_pe_seg"

# Extended stratification axes including vessel overlay features
VESSEL_STRATIFICATION_AXES = PE_STRATIFICATION_AXES + [
    "intended_shape_category",
    "intended_occlusion_cat",
]

# Fine-grained continuous features from vessel overlay
VESSEL_CONTINUOUS_FEATURES = [
    "intended_voxels",
    "intended_occlusion_frac",
    "intended_radius_mean_mm",
    "intended_compactness",
    "intended_wall_adherence_frac",
    "n_edges_with_intended_pe",
    "intended_gen_subsegmental_frac",
]

# Recall thresholds for multi-level detection analysis
RECALL_THRESHOLDS = [0.10, 0.25, 0.50, 0.75]

# --- Color palettes (consistent with vessel overlay plots) ---

def _shade_palette(base_hex: str, n: int) -> list[str]:
    """Generate n shades from light (35% saturation) to full base color."""
    import matplotlib.colors as mcolors
    base_rgb = mcolors.to_rgb(base_hex)
    white = np.array([1.0, 1.0, 1.0])
    base = np.array(base_rgb)
    fracs = np.linspace(0.35, 1.0, n)
    return [mcolors.to_hex(white * (1 - f) + base * f) for f in fracs]

# Per-axis palettes: maps stratum name → color
_LOC_ORDER = ["central", "lobar", "segmental", "subsegmental"]
_OCC_ORDER = ["complete", "near_complete", "partial"]
_BUR_ORDER = ["extensive", "multi_bilateral", "multi_unilateral", "isolated"]
_SEV_ORDER = ["mild", "moderate", "severe"]

_LAT_ORDER = ["central", "bilateral", "unilateral_left", "unilateral_right"]
_SHAPE_ORDER = ["acute", "indeterminate", "chronic"]
_OCC_CAT_ORDER = ["complete", "near_complete", "partial"]

PALETTE_LOC = dict(zip(_LOC_ORDER, _shade_palette("#2563EB", 4)))
PALETTE_OCC = dict(zip(_OCC_ORDER, _shade_palette("#7C3AED", 3)))
PALETTE_BUR = dict(zip(_BUR_ORDER, _shade_palette("#059669", 4)))
PALETTE_SEV = dict(zip(_SEV_ORDER, _shade_palette("#F97316", 3)))
PALETTE_LAT = dict(zip(_LAT_ORDER, _shade_palette("#0891B2", 4)))  # cyan/teal
PALETTE_SHAPE = dict(zip(_SHAPE_ORDER, _shade_palette("#D946EF", 3)))  # fuchsia
PALETTE_OCC_CAT = dict(zip(_OCC_CAT_ORDER, _shade_palette("#7C3AED", 3)))  # same purple as occlusion

AXIS_PALETTES = {
    "location": PALETTE_LOC,
    "occlusion": PALETTE_OCC,
    "burden": PALETTE_BUR,
    "severity_category": PALETTE_SEV,
    "laterality": PALETTE_LAT,
    "intended_shape_category": PALETTE_SHAPE,
    "intended_occlusion_cat": PALETTE_OCC_CAT,
}

# Detected vs missed colors
COLOR_DETECTED = "#2563EB"  # blue
COLOR_MISSED = "#E11D48"    # red


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(vessel_csv: str, nope_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load vessel overlay CSV and no-PE baseline results."""
    df = pd.read_csv(vessel_csv)
    print(f"Loaded {len(df)} rows from vessel overlay CSV ({len(df.columns)} columns)")

    # Restrict to the FROZEN final cohort (the common set valid in BOTH models)
    # if present, so APES and nnU-Net are evaluated on identical configs.
    from pathlib import Path as _P
    frozen_csv = _P(RES / "cohort/final_cohort.csv")
    if frozen_csv.exists() and "config_id" in df.columns:
        frozen = set(pd.read_csv(frozen_csv)["config_id"].astype(str))
        n0 = len(df)
        df = df[df["config_id"].astype(str).isin(frozen)].copy()
        print(f"Restricted to FROZEN cohort: {len(df)} configs (was {n0})")

    # A few configs (4) carry duplicate rows from an upstream vessel-overlay x
    # slice-overlap join (the detection flag pe_detected is identical across the
    # copies; only fine-grained geometry differs). Collapse to one row/config so
    # no config is multiply-counted in rates, stratified tables, or the GLMM.
    if "config_id" in df.columns:
        n_dup = len(df) - df["config_id"].nunique()
        if n_dup:
            df = df.drop_duplicates(subset="config_id", keep="first").copy()
            print(f"Dropped {n_dup} duplicate config rows -> {len(df)} unique configs")

    # Drop rows where segmentation could not run at all
    n_missing_seg = (df["error"] == "missing_pe_seg").sum()
    df = df[df["error"] != "missing_pe_seg"].copy()
    print(f"Dropped {n_missing_seg} missing_pe_seg errors")

    # Tag analysis groups
    df["analysis_group"] = np.where(
        df["error"] == "no_intended_pe", "no_intended_pe", "valid"
    )
    n_valid = (df["analysis_group"] == "valid").sum()
    n_no_intended = (df["analysis_group"] == "no_intended_pe").sum()
    print(f"  valid: {n_valid}, no_intended_pe: {n_no_intended}")

    # Load no-PE baseline
    nope_df = pd.read_csv(nope_csv)
    print(f"Loaded {len(nope_df)} no-PE baseline results")

    return df, nope_df


def prepare_analysis_df(df: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    """Prepare the valid subset for analysis."""
    df = df[df["analysis_group"] == "valid"].copy()

    df["pe_detected"] = df["pe_detected"].fillna(False).astype(bool)
    df["detected_voxels"] = df["detected_voxels"].fillna(0)

    # Recall: fraction of intended PE volume recovered (capped at 1.0)
    df["recall"] = np.where(
        df["intended_voxels"] > 0,
        np.minimum(df["detected_voxels"] / df["intended_voxels"], 1.0),
        0.0,
    )

    # Columns expected by the existing analysis pipeline
    df["image_id"] = df["config_id"]
    df["target"] = 1.0
    df["predicted_score"] = df["pe_detected"].astype(float)
    df["predicted_label"] = df["pe_detected"].astype(float)
    df["classifier_correct"] = df["pe_detected"].astype(int)
    df["classifier_missed"] = 1 - df["classifier_correct"]
    df["patient_id_metadata"] = df["patient_id"]
    df["classifier_name"] = model_name or MODEL_NAME
    df["threshold"] = 0.5

    # Map metadata columns
    df["pe_volume_ml"] = df.get("intended_volume_ml", pd.NA)
    df["qanadli_percent"] = df["qanadli_score"] / 40.0 * 100.0

    for col in ["gender", "age", "race"]:
        df[f"{col}_metadata"] = df.get(col, pd.NA)

    return df


# ---------------------------------------------------------------------------
# Level 0: Baseline comparison (no-PE vs counterfactual)
# ---------------------------------------------------------------------------

def run_baseline_comparison(df: pd.DataFrame, nope_df: pd.DataFrame,
                            res_dir: Path, fig_dir: Path) -> dict:
    """Compare segmentation model output on no-PE vs counterfactual scans."""
    print("\n" + "=" * 60)
    print("LEVEL 0: BASELINE COMPARISON (no-PE vs counterfactual)")
    print("=" * 60)

    # Per-patient counterfactual summary
    cf_per_patient = df.groupby("patient_id").agg(
        cf_n=("pe_detected", "count"),
        cf_n_detected=("pe_detected", "sum"),
        cf_detection_rate=("pe_detected", "mean"),
        cf_detected_voxels_mean=("detected_voxels", "mean"),
        cf_detected_voxels_median=("detected_voxels", "median"),
        cf_recall_mean=("recall", "mean"),
    ).reset_index()

    n_nope = len(nope_df)
    n_cf = len(cf_per_patient)
    merged = nope_df.merge(cf_per_patient, on="patient_id", how="inner")
    if len(merged) < min(n_nope, n_cf):
        print(f"  WARNING: Inner merge dropped patients - nope={n_nope}, cf={n_cf}, merged={len(merged)}")
    merged["delta_voxels"] = merged["cf_detected_voxels_mean"] - merged["pe_voxels"]
    merged.to_csv(res_dir / "nope_vs_counterfactual.csv", index=False)

    results = {
        "n_patients": len(merged),
        "nope_any_pe_detected": int(merged["pe_detected"].sum()),
        "nope_mean_voxels": float(merged["pe_voxels"].mean()),
        "cf_mean_voxels": float(merged["cf_detected_voxels_mean"].mean()),
        "cf_detection_rate_mean": float(merged["cf_detection_rate"].mean()),
        "delta_voxels_mean": float(merged["delta_voxels"].mean()),
    }

    print(f"  Patients: {results['n_patients']}")
    print(f"  No-PE: {results['nope_any_pe_detected']}/{results['n_patients']} detected any PE, "
          f"mean voxels = {results['nope_mean_voxels']:.0f}")
    print(f"  Counterfactual: mean detection rate = {results['cf_detection_rate_mean']:.1%}, "
          f"mean voxels = {results['cf_mean_voxels']:.0f}")
    print(f"  Delta (CF - noPE): mean = {results['delta_voxels_mean']:.0f} voxels")

    # --- Plot: paired per-patient comparison ---
    _plot_baseline_paired(merged, fig_dir)
    _plot_baseline_waterfall(merged, fig_dir)

    # --- Plot: per-patient score shift (real → synthetic noPE → counterfactual) ---
    real_csv = Path(REAL_SCAN_PE_CSV)
    if real_csv.exists():
        real_df = pd.read_csv(real_csv)
        _plot_per_patient_score_shift(merged, real_df, df, fig_dir, res_dir)
    else:
        print(f"  Skipping score shift plot ({real_csv} not found yet)")

    return results


def _plot_baseline_paired(merged: pd.DataFrame, fig_dir: Path):
    """Paired dot plot: no-PE vs counterfactual detected voxels per patient."""
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE_COL_IN, 2.5))

    # Panel A: paired lines
    ax = axes[0]
    merged_sorted = merged.sort_values("cf_detected_voxels_mean")
    for _, row in merged_sorted.iterrows():
        ax.plot([0, 1], [row["pe_voxels"], row["cf_detected_voxels_mean"]],
                color=COLORS["gray"], alpha=0.3, linewidth=0.5)
    ax.scatter([0] * len(merged), merged_sorted["pe_voxels"],
               s=8, color=COLORS["blue"], alpha=0.7, zorder=5, label="No-PE synthetic")
    ax.scatter([1] * len(merged), merged_sorted["cf_detected_voxels_mean"],
               s=8, color=COLORS["red"], alpha=0.7, zorder=5, label="Counterfactual (mean)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No-PE\nsynthetic", "Counterfactual\n(mean)"])
    ax.set_ylabel("Detected PE voxels")
    ax.legend(fontsize=5, loc="upper left")
    ax.set_title("Per-patient PE detection", fontsize=7)

    # Panel B: histogram of CF detection rate per patient
    ax = axes[1]
    ax.hist(merged["cf_detection_rate"], bins=20, color=COLORS["blue"],
            alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(merged["cf_detection_rate"].mean(), color=COLORS["red"],
               linestyle="--", linewidth=1, label=f"mean={merged['cf_detection_rate'].mean():.1%}")
    ax.set_xlabel("Counterfactual detection rate")
    ax.set_ylabel("Patients")
    ax.legend(fontsize=5)
    ax.set_title("Per-patient CF detection rate", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "baseline_nope_vs_counterfactual.pdf"))
    plt.close(fig)
    print("  Baseline comparison plots saved.")


def _plot_baseline_waterfall(merged: pd.DataFrame, fig_dir: Path):
    """Waterfall plot: per-patient delta (CF - noPE) sorted by magnitude."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 2.5))

    sorted_df = merged.sort_values("cf_detected_voxels_mean")
    x = range(len(sorted_df))
    colors = [COLORS["red"] if v > 0 else COLORS["blue"]
              for v in sorted_df["cf_detected_voxels_mean"]]
    ax.bar(x, sorted_df["cf_detected_voxels_mean"], color=colors, width=1.0, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Patients (sorted by CF detected voxels)")
    ax.set_ylabel("Mean detected PE voxels (counterfactual)")
    ax.set_title("Per-patient counterfactual PE detection (no-PE baseline = 0 for all)", fontsize=7)
    ax.set_xlim(-1, len(sorted_df))

    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "baseline_waterfall.pdf"))
    plt.close(fig)
    print("  Waterfall plot saved.")


def _plot_per_patient_score_shift(merged: pd.DataFrame, real_df: pd.DataFrame,
                                   cf_df: pd.DataFrame, fig_dir: Path, res_dir: Path):
    """Per-patient PE volume shift: real scan → synthetic noPE → counterfactual.

    Like panel (b) of the classifier baseline comparison, but for detected PE
    volume in ml. Shows four levels per patient:
      - Real scan (original, may have real PE)
      - Synthetic no-PE (should be 0)
      - Counterfactual mean ± std (injected PE)
    """
    set_nature_style()

    # Compute per-patient CF stats in ml
    # Need voxel volume - approximate from intended_voxels and intended_volume_ml
    cf_with_vol = cf_df[cf_df["detected_voxels"] > 0].copy()
    # Use a rough conversion: get median voxel volume from the data
    if "intended_volume_ml" in cf_df.columns and "intended_voxels" in cf_df.columns:
        valid = cf_df[(cf_df["intended_voxels"] > 0) & (cf_df["intended_volume_ml"] > 0)]
        if len(valid) > 0:
            voxel_vol_ml = (valid["intended_volume_ml"] / valid["intended_voxels"]).median()
        else:
            voxel_vol_ml = 0.0005  # fallback
    else:
        voxel_vol_ml = 0.0005

    cf_per_patient = cf_df.groupby("patient_id").agg(
        cf_det_voxels_mean=("detected_voxels", "mean"),
        cf_det_voxels_std=("detected_voxels", "std"),
    ).reset_index()
    cf_per_patient["cf_det_ml_mean"] = cf_per_patient["cf_det_voxels_mean"] * voxel_vol_ml
    cf_per_patient["cf_det_ml_std"] = cf_per_patient["cf_det_voxels_std"].fillna(0) * voxel_vol_ml

    # Merge all sources
    plot_df = cf_per_patient.merge(
        real_df[["patient_id", "pe_volume_ml"]].rename(columns={"pe_volume_ml": "real_pe_ml"}),
        on="patient_id", how="left",
    ).merge(
        merged[["patient_id", "pe_volume_ml"]].rename(columns={"pe_volume_ml": "nope_pe_ml"}),
        on="patient_id", how="left",
    )
    plot_df["nope_pe_ml"] = plot_df["nope_pe_ml"].fillna(0)
    plot_df = plot_df.sort_values("cf_det_ml_mean")
    plot_df.to_csv(res_dir / "per_patient_score_shift.csv", index=False)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 3.5))

    x = np.arange(len(plot_df))

    # Real scan PE (gray dots)
    ax.scatter(x, plot_df["real_pe_ml"], s=12, color="#9CA3AF", alpha=0.8,
               zorder=3, label="Real scan (original)")

    # Synthetic no-PE (all zeros - show as a line at 0)
    ax.axhline(0, color="#9CA3AF", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.scatter(x, plot_df["nope_pe_ml"], s=8, marker="s", color="#059669",
               alpha=0.6, zorder=2, label="Synthetic no-PE")

    # Counterfactual mean ± std (clip lower error bar at 0)
    yerr_upper = plot_df["cf_det_ml_std"].fillna(0)
    yerr_lower = np.minimum(plot_df["cf_det_ml_std"].fillna(0), plot_df["cf_det_ml_mean"])
    ax.errorbar(x, plot_df["cf_det_ml_mean"],
                yerr=[yerr_lower, yerr_upper],
                fmt="s", color=COLOR_DETECTED, markersize=4,
                capsize=1.5, linewidth=0.6, capthick=0.5, alpha=0.8,
                zorder=4, label="Counterfactual (mean ± std)")

    ax.set_xlabel("Patient (sorted by CF detected PE volume)", fontsize=9)
    ax.set_ylabel("Detected PE volume (ml)", fontsize=9)
    ax.set_title("Per-patient detected PE volume across conditions", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(-1, len(plot_df))
    ax.set_ylim(0, 6)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "per_patient_score_shift.pdf"))
    plt.close(fig)
    print("  Per-patient score shift plot saved.")


# ---------------------------------------------------------------------------
# Level 1: Binary detection analysis
# ---------------------------------------------------------------------------

def run_binary_detection_analysis(df: pd.DataFrame, res_dir: Path, fig_dir: Path) -> dict:
    """Analyze binary detection: did the model find any PE at all?"""
    print("\n" + "=" * 60)
    print("LEVEL 1: BINARY DETECTION (pe_detected)")
    print("=" * 60)

    results = {}
    n_total = len(df)
    n_detected = df["pe_detected"].sum()
    n_missed = n_total - n_detected
    det_rate = n_detected / n_total

    print(f"  {n_detected}/{n_total} detected ({det_rate:.1%}), {n_missed} complete misses")
    results["n_total"] = n_total
    results["n_detected"] = int(n_detected)
    results["n_missed"] = int(n_missed)
    results["detection_rate"] = float(det_rate)

    # Overall metrics
    results["overall"] = compute_overall_metrics(df, threshold=0.5)

    # Stratified binary detection (original + vessel overlay axes)
    print("\n--- Stratified Binary Detection ---")
    valid_strat_axes = [a for a in VESSEL_STRATIFICATION_AXES
                        if a in df.columns and not df[a].isna().all()]
    stratified = run_stratified_analysis(df, threshold=0.5, axes=valid_strat_axes)
    results["stratified"] = {k: v.to_dict("records") for k, v in stratified.items()}

    for axis, axis_df in stratified.items():
        axis_df.to_csv(res_dir / f"binary_stratified_{axis}.csv", index=False)

    plot_failure_rate_heatmap(stratified, str(fig_dir / "binary_failure_rate_heatmap.pdf"))
    plot_sensitivity_bar_chart(stratified, str(fig_dir / "binary_sensitivity_bars.pdf"))
    plot_sensitivity_by_category(stratified, str(fig_dir / "binary_sensitivity_by_category.pdf"))
    print("  Binary detection plots saved.")

    # Statistical tests
    test_results = run_all_tests(df, valid_strat_axes)
    if "fdr_correction" in test_results:
        test_results["fdr_correction"].to_csv(res_dir / "binary_fdr_correction.csv", index=False)
    if "logistic_regression" in test_results and len(test_results["logistic_regression"]) > 0:
        test_results["logistic_regression"].to_csv(res_dir / "binary_logistic_regression.csv", index=False)

    for axis in valid_strat_axes:
        if axis in test_results.get("odds_ratios", {}):
            or_df = test_results["odds_ratios"][axis]
            or_df.to_csv(res_dir / f"binary_odds_ratios_{axis}.csv", index=False)
            if len(or_df[~or_df["is_reference"]]) > 1:
                plot_forest(or_df, str(fig_dir / f"binary_forest_{axis}.pdf"),
                            title=f"Binary detection odds ratios: {axis}")

    # Characterize missed cases
    missed = df[df["pe_detected"] == False]
    if len(missed) > 0:
        missed.to_csv(res_dir / "missed_cases.csv", index=False)
        print(f"  Saved {len(missed)} missed cases")

    # Threshold analysis
    print("\n--- Binary Detection Threshold Analysis ---")
    threshold_results = run_threshold_analysis(
        df, target_sensitivities=[0.90, 0.95, 0.99], classifier_threshold=0.5,
    )
    results["detection_thresholds"] = threshold_results["thresholds"]
    for axis, curve_df in threshold_results["curves"].items():
        curve_df.to_csv(res_dir / f"binary_cumulative_{axis}.csv", index=False)
    plot_cumulative_sensitivity_curves(
        threshold_results["curves"],
        str(fig_dir / "binary_cumulative_sensitivity.pdf"),
        threshold_results=threshold_results["thresholds"],
    )

    # Patient-level binary detection
    print("\n--- Patient-Level Binary Detection ---")
    patient_df = compute_patient_level_metrics(df, threshold=0.5)
    patient_df.to_csv(res_dir / "binary_patient_metrics.csv", index=False)
    patient_summary = summarize_patient_variation(patient_df)
    results["patient_summary"] = patient_summary
    results["patient_icc"] = icc_patient_effect(df)

    plot_patient_sensitivity(patient_df, str(fig_dir / "binary_patient_sensitivity.pdf"))
    plot_patient_score_heatmap(df, save_path=str(fig_dir / "binary_patient_heatmap.pdf"))
    print("  Patient-level plots saved.")

    return results


# ---------------------------------------------------------------------------
# Level 2: Segmentation quality analysis
# ---------------------------------------------------------------------------

def run_segmentation_quality_analysis(df: pd.DataFrame, res_dir: Path, fig_dir: Path) -> dict:
    """Analyze segmentation quality for detected cases + detection predictors."""
    print("\n" + "=" * 60)
    print("LEVEL 2: SEGMENTATION QUALITY & DETECTION PREDICTORS")
    print("=" * 60)

    detected = df[df["pe_detected"] == True].copy()
    print(f"  Analyzing {len(detected)} detected cases (+ {len(df) - len(detected)} misses)")

    results = {"n_detected": len(detected)}

    # --- 2a. Segmentation metric distributions ---
    print("\n--- Segmentation Metric Distributions ---")
    metric_summary = {}
    for metric in ["recall", "intended_occlusion_frac", "intended_compactness",
                    "intended_wall_adherence_frac", "intended_radius_mean_mm",
                    "edge_detection_rate", "target_edge_detection_rate"]:
        if metric not in detected.columns:
            continue
        vals = detected[metric].dropna()
        if len(vals) == 0:
            continue
        s = {
            "n": len(vals), "mean": float(vals.mean()), "median": float(vals.median()),
            "std": float(vals.std()), "p25": float(vals.quantile(0.25)),
            "p75": float(vals.quantile(0.75)),
        }
        metric_summary[metric] = s
        print(f"  {metric:40s}  mean={s['mean']:.3f}  median={s['median']:.3f}  "
              f"IQR=[{s['p25']:.3f}, {s['p75']:.3f}]")
    results["metric_summary"] = metric_summary

    # --- 2b. Stratified quality with vessel overlay features ---
    print("\n--- Stratified Segmentation Quality ---")
    quality_stratified = _compute_stratified_quality(detected)
    for axis, axis_df in quality_stratified.items():
        axis_df.to_csv(res_dir / f"quality_stratified_{axis}.csv", index=False)
    results["quality_stratified"] = {k: v.to_dict("records") for k, v in quality_stratified.items()}

    _plot_quality_boxplots(detected, quality_stratified, fig_dir)
    _plot_metric_distributions(detected, fig_dir)

    # --- 2c. Multi-threshold detection rates ---
    print("\n--- Multi-Threshold Detection Rates ---")
    all_df = df.copy()
    all_df["recall"] = all_df["recall"].fillna(0.0)

    threshold_rates = []
    for thresh in RECALL_THRESHOLDS:
        n_above = (all_df["recall"] >= thresh).sum()
        rate = n_above / len(all_df)
        threshold_rates.append({
            "recall_threshold": thresh, "n_above": int(n_above),
            "n_total": len(all_df), "detection_rate": float(rate),
        })
        print(f"  recall >= {thresh:.0%}: {n_above}/{len(all_df)} = {rate:.1%}")
    results["multi_threshold_rates"] = threshold_rates
    pd.DataFrame(threshold_rates).to_csv(res_dir / "multi_threshold_rates.csv", index=False)

    _plot_multi_threshold_heatmap(all_df, fig_dir)

    # --- 2d. Volume-threshold curves ---
    print("\n--- Volume-Threshold Curves ---")
    _plot_recall_vs_volume(all_df, fig_dir, res_dir)

    # --- 2e. Detection predictors (what makes a PE detectable?) ---
    print("\n--- Detection Predictors ---")
    results["detection_predictors"] = _analyze_detection_predictors(df, res_dir, fig_dir)

    # --- 2f. Quality regression ---
    print("\n--- Quality Regression (detected cases) ---")
    results["quality_regression"] = _run_quality_regression(detected, res_dir)

    return results


def _compute_stratified_quality(detected: pd.DataFrame) -> dict:
    """Stratified quality metrics using original + vessel overlay axes."""
    quality_stratified = {}
    all_axes = VESSEL_STRATIFICATION_AXES
    metrics = ["recall", "edge_detection_rate", "intended_occlusion_frac"]

    for axis in all_axes:
        if axis not in detected.columns or detected[axis].isna().all():
            continue
        rows = []
        for stratum, grp in detected.groupby(axis, observed=True):
            row = {"stratum": stratum, "n": len(grp)}
            for metric in metrics:
                if metric in grp.columns:
                    vals = grp[metric].dropna()
                    row[f"{metric}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
                    row[f"{metric}_median"] = float(vals.median()) if len(vals) > 0 else float("nan")
            rows.append(row)
        axis_df = pd.DataFrame(rows)
        quality_stratified[axis] = axis_df

        print(f"  {axis}:")
        for _, r in axis_df.iterrows():
            recall_str = f"recall={r.get('recall_mean', float('nan')):.3f}"
            edr_str = f"edge_det={r.get('edge_detection_rate_mean', float('nan')):.3f}"
            print(f"    {r['stratum']:25s} n={r['n']:4.0f}  {recall_str}  {edr_str}")

    return quality_stratified


def _analyze_detection_predictors(df: pd.DataFrame, res_dir: Path, fig_dir: Path) -> dict:
    """Logistic regression + univariate analysis: what predicts binary detection?"""
    df = df.copy()
    df["detected"] = df["pe_detected"].astype(int)

    # Univariate: each feature vs detection
    features = VESSEL_CONTINUOUS_FEATURES
    available = [f for f in features if f in df.columns and df[f].notna().sum() > 50]

    univariate = []
    for feat in available:
        sub = df[[feat, "detected"]].dropna()
        det = sub[sub["detected"] == 1][feat]
        miss = sub[sub["detected"] == 0][feat]
        if len(miss) < 5:
            continue
        stat, pval = stats.mannwhitneyu(det, miss, alternative="two-sided")
        univariate.append({
            "feature": feat,
            "detected_mean": float(det.mean()), "detected_median": float(det.median()),
            "missed_mean": float(miss.mean()), "missed_median": float(miss.median()),
            "mann_whitney_U": float(stat), "p_value": float(pval),
        })
    univariate_df = pd.DataFrame(univariate).sort_values("p_value")
    univariate_df.to_csv(res_dir / "detection_univariate_tests.csv", index=False)

    print("  Univariate feature comparison (detected vs missed):")
    for _, r in univariate_df.iterrows():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"    {r['feature']:45s}  det={r['detected_mean']:.3f}  miss={r['missed_mean']:.3f}  "
              f"p={r['p_value']:.4f} {sig}")

    # Multivariate logistic regression
    reg_features = [f for f in available if df[f].std() > 0]
    reg_df = df[reg_features + ["detected"]].dropna()

    results = {"univariate": univariate_df.to_dict("records")}

    if len(reg_df) > 100 and len(reg_features) > 2:
        X = reg_df[reg_features].copy()
        # Standardize for comparable coefficients
        means = X.mean()
        stds = X.std()
        # Drop features with zero std after NaN removal
        nonzero_std = stds[stds > 0].index.tolist()
        X = (X[nonzero_std] - means[nonzero_std]) / stds[nonzero_std]
        reg_features = nonzero_std
        y = reg_df["detected"]

        model = LogisticRegression(max_iter=1000, penalty=None)
        model.fit(X, y)

        coef_df = pd.DataFrame({
            "feature": reg_features,
            "coefficient": model.coef_[0],
            "odds_ratio": np.exp(model.coef_[0]),
        }).sort_values("coefficient", key=abs, ascending=False)
        coef_df.to_csv(res_dir / "detection_logistic_regression.csv", index=False)

        print(f"\n  Logistic regression (standardized, n={len(reg_df)}):")
        for _, r in coef_df.iterrows():
            print(f"    {r['feature']:45s}  β={r['coefficient']:+.3f}  OR={r['odds_ratio']:.2f}")
        results["logistic_regression"] = coef_df.to_dict("records")

    # --- Detection probability plots ---
    _plot_detection_probability(df, available, fig_dir)

    return results


def _plot_detection_probability(df: pd.DataFrame, features: list, fig_dir: Path):
    """Plot detection probability as a function of key PE characteristics."""
    set_nature_style()

    # Select top features for plotting
    plot_features = [f for f in [
        "intended_voxels", "intended_radius_mean_mm", "intended_occlusion_frac",
        "intended_compactness", "n_edges_with_intended_pe", "intended_wall_adherence_frac",
    ] if f in features]

    n = len(plot_features)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(NATURE_DOUBLE_COL_IN, 4.0))
    axes = axes.flatten()

    df = df.copy()
    df["detected"] = df["pe_detected"].astype(int)

    for i, feat in enumerate(plot_features):
        ax = axes[i]
        sub = df[[feat, "detected"]].dropna()

        # Bin into deciles and compute detection rate per bin
        sub["bin"] = pd.qcut(sub[feat], q=10, duplicates="drop")
        binned = sub.groupby("bin", observed=True).agg(
            det_rate=("detected", "mean"),
            n=("detected", "count"),
            feat_median=(feat, "median"),
        )

        ax.bar(range(len(binned)), binned["det_rate"], color=COLORS["blue"], alpha=0.7, width=0.8)
        ax.set_ylim(0.75, 1.01)
        ax.set_ylabel("Detection rate" if i % 3 == 0 else "")
        ax.set_xlabel(feat.replace("intended_", "").replace("_", " "))

        # Add sample size annotation
        for j, (_, row) in enumerate(binned.iterrows()):
            ax.text(j, row["det_rate"] + 0.005, f"n={row['n']:.0f}", ha="center",
                    fontsize=4, color=COLORS["gray"])

        ax.set_xticks(range(len(binned)))
        ax.set_xticklabels([f"{row['feat_median']:.1f}" for _, row in binned.iterrows()],
                           rotation=45, fontsize=5)

    # Hide unused axes
    for i in range(len(plot_features), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Binary detection rate by PE characteristic (decile bins)", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "detection_probability_by_feature.pdf"))
    plt.close(fig)
    print("  Detection probability plots saved.")


# ---------------------------------------------------------------------------
# Per-patient deep analysis
# ---------------------------------------------------------------------------

def run_per_patient_analysis(df: pd.DataFrame, nope_df: pd.DataFrame,
                             res_dir: Path, fig_dir: Path) -> dict:
    """Per-patient analysis: what makes some patients' PEs harder to detect?"""
    print("\n" + "=" * 60)
    print("PER-PATIENT ANALYSIS")
    print("=" * 60)

    # Aggregate per patient
    agg_cols = {
        "pe_detected": ["count", "sum", "mean"],
        "recall": ["mean", "median", "std"],
        "detected_voxels": ["mean", "sum"],
        "intended_voxels": ["mean", "sum"],
    }
    # Add vessel overlay features
    for feat in ["intended_radius_mean_mm", "intended_occlusion_frac",
                  "intended_compactness", "intended_wall_adherence_frac",
                  "n_edges_with_intended_pe", "intended_gen_subsegmental_frac"]:
        if feat in df.columns:
            agg_cols[feat] = "mean"

    patient_df = df.groupby("patient_id").agg(**{
        f"{col}_{fn}" if isinstance(fn, str) else f"{col}_{fn.__name__}" if callable(fn) else f"{col}_{fn}": (col, fn)
        for col, fns in agg_cols.items()
        for fn in (fns if isinstance(fns, list) else [fns])
    }).reset_index()

    # Rename for clarity
    patient_df.rename(columns={
        "pe_detected_count": "n_counterfactuals",
        "pe_detected_sum": "n_detected",
        "pe_detected_mean": "detection_rate",
        "recall_mean": "mean_recall",
        "recall_median": "median_recall",
    }, inplace=True)

    # Merge no-PE baseline
    patient_df = patient_df.merge(
        nope_df[["patient_id", "pe_voxels", "pe_volume_ml"]].rename(
            columns={"pe_voxels": "nope_pe_voxels", "pe_volume_ml": "nope_pe_volume_ml"}
        ),
        on="patient_id", how="left",
    )

    patient_df.to_csv(res_dir / "per_patient_full.csv", index=False)
    print(f"  {len(patient_df)} patients")

    # Detection rate distribution
    print(f"  Detection rate: mean={patient_df['detection_rate'].mean():.3f}, "
          f"median={patient_df['detection_rate'].median():.3f}")
    print(f"  Perfect detection (100%): {(patient_df['detection_rate'] == 1.0).sum()}")
    print(f"  Below 80%: {(patient_df['detection_rate'] < 0.8).sum()}")

    # What patient-level features correlate with detection rate?
    print("\n--- Patient-Level Feature Correlations ---")
    patient_features = [c for c in patient_df.columns
                        if c not in ["patient_id", "detection_rate", "n_detected",
                                     "n_counterfactuals", "nope_pe_voxels", "nope_pe_volume_ml"]
                        and patient_df[c].dtype in ["float64", "int64"]
                        and patient_df[c].notna().sum() > 10]

    correlations = []
    for feat in patient_features:
        corr = patient_df[["detection_rate", feat]].dropna().corr().iloc[0, 1]
        if not np.isnan(corr):
            correlations.append({"feature": feat, "corr_with_detection_rate": corr})
    corr_df = pd.DataFrame(correlations).sort_values("corr_with_detection_rate", key=abs, ascending=False)
    corr_df.to_csv(res_dir / "per_patient_correlations.csv", index=False)

    for _, r in corr_df.head(10).iterrows():
        print(f"    {r['feature']:45s}  r={r['corr_with_detection_rate']:+.3f}")

    # --- Per-patient plot ---
    _plot_per_patient_detection(patient_df, fig_dir)

    return {
        "n_patients": len(patient_df),
        "mean_detection_rate": float(patient_df["detection_rate"].mean()),
        "perfect_detection": int((patient_df["detection_rate"] == 1.0).sum()),
        "below_80": int((patient_df["detection_rate"] < 0.8).sum()),
        "correlations": corr_df.head(10).to_dict("records"),
    }


def _plot_per_patient_detection(patient_df: pd.DataFrame, fig_dir: Path):
    """Per-patient detection rate with feature annotations."""
    set_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(NATURE_DOUBLE_COL_IN, 3.0))

    sorted_df = patient_df.sort_values("detection_rate")

    # Panel A: detection rate per patient
    ax = axes[0]
    colors = [COLORS["red"] if r < 0.8 else COLORS["blue"]
              for r in sorted_df["detection_rate"]]
    ax.bar(range(len(sorted_df)), sorted_df["detection_rate"],
           color=colors, width=1.0, alpha=0.8)
    overall_rate = patient_df["detection_rate"].mean()
    ax.axhline(overall_rate, color="black", linestyle="--", linewidth=0.5,
               label=f"Overall ({overall_rate:.1%})")
    ax.set_xlabel("Patients (sorted)")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5)
    ax.set_title("Per-patient detection rate", fontsize=7)

    # Panel B: detection rate vs mean intended voxels
    ax = axes[1]
    if "intended_voxels_mean" in patient_df.columns:
        ax.scatter(patient_df["intended_voxels_mean"], patient_df["detection_rate"],
                   s=10, alpha=0.6, color=COLORS["blue"])
        ax.set_xlabel("Mean intended voxels")
        ax.set_ylabel("Detection rate")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_title("Detection rate vs PE size", fontsize=7)

    # Panel C: detection rate vs mean number of PE edges
    ax = axes[2]
    edge_col = "n_edges_with_intended_pe_mean"
    if edge_col in patient_df.columns:
        ax.scatter(patient_df[edge_col], patient_df["detection_rate"],
                   s=10, alpha=0.6, color=COLORS["green"])
        ax.set_xlabel("Mean PE vessel edges")
        ax.set_ylabel("Detection rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("Detection rate vs PE spread", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "per_patient_detection.pdf"))
    plt.close(fig)
    print("  Per-patient detection plots saved.")


# ---------------------------------------------------------------------------
# Segmentation quality plots (shared)
# ---------------------------------------------------------------------------

def _plot_quality_boxplots(detected: pd.DataFrame, quality_stratified: dict, fig_dir: Path):
    """Box plots of recall stratified by PE characteristics."""
    set_nature_style()

    for metric in ["recall", "edge_detection_rate"]:
        if metric not in detected.columns:
            continue

        axes_to_plot = {k: v for k, v in quality_stratified.items()
                        if any(metric in c for c in v.columns)}
        n_axes = len(axes_to_plot)
        if n_axes == 0:
            continue

        fig, axes = plt.subplots(1, min(n_axes, 5), figsize=(NATURE_DOUBLE_COL_IN, 2.5))
        if min(n_axes, 5) == 1:
            axes = [axes]

        for i, (axis, _) in enumerate(list(axes_to_plot.items())[:5]):
            ax = axes[i]
            strata = detected[axis].dropna().unique()
            data = [detected.loc[detected[axis] == s, metric].dropna().values for s in strata]
            bp = ax.boxplot(data, tick_labels=strata, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black", linewidth=1))
            for patch, color in zip(bp["boxes"], COLOR_LIST):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel(metric if i == 0 else "")
            ax.set_xlabel(axis.replace("intended_", ""))
            ax.tick_params(axis="x", rotation=45)
            if metric != "edge_detection_rate":
                ax.set_ylim(-0.05, 1.05)

        fig.suptitle(f"{metric} by PE characteristics", fontsize=8)
        fig.tight_layout()
        _save_fig(fig, str(fig_dir / f"quality_boxplot_{metric}.pdf"))
        plt.close(fig)

    print("  Quality box plots saved.")


def _plot_metric_distributions(detected: pd.DataFrame, fig_dir: Path):
    """Histograms of key segmentation metrics."""
    set_nature_style()

    metrics = [m for m in ["recall", "intended_occlusion_frac",
                            "intended_compactness", "edge_detection_rate"]
               if m in detected.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(NATURE_DOUBLE_COL_IN, 2.0))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = detected[metric].dropna()
        ax.hist(vals, bins=50, color=COLORS["blue"], alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(metric.replace("intended_", ""))
        ax.set_ylabel("Count" if ax == axes[0] else "")
        ax.axvline(vals.median(), color=COLORS["red"], linestyle="--", linewidth=0.8,
                    label=f"median={vals.median():.2f}")
        ax.legend(fontsize=5)

    fig.suptitle("Metric distributions (detected cases)", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "quality_metric_distributions.pdf"))
    plt.close(fig)
    print("  Metric distributions saved.")


def _plot_multi_threshold_heatmap(df: pd.DataFrame, fig_dir: Path):
    """Heatmap: detection rate at multiple recall thresholds × PE characteristics."""
    set_nature_style()

    axes_to_plot = [a for a in VESSEL_STRATIFICATION_AXES
                    if a in df.columns and not df[a].isna().all()]

    strata_labels = []
    rate_matrix = []

    for thresh in RECALL_THRESHOLDS:
        rates = []
        for axis in axes_to_plot:
            for stratum in sorted(df[axis].dropna().unique()):
                grp = df[df[axis] == stratum]
                rate = (grp["recall"] >= thresh).mean()
                rates.append(rate)
                if thresh == RECALL_THRESHOLDS[0]:
                    strata_labels.append(f"{stratum}\n({axis[:3]})")
        rate_matrix.append(rates)

    rate_matrix = np.array(rate_matrix)

    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 2.2))
    im = ax.imshow(rate_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(RECALL_THRESHOLDS)))
    ax.set_yticklabels([f"recall≥{t:.0%}" for t in RECALL_THRESHOLDS])
    ax.set_xticks(range(len(strata_labels)))
    ax.set_xticklabels(strata_labels, rotation=90, fontsize=4)

    for i in range(rate_matrix.shape[0]):
        for j in range(rate_matrix.shape[1]):
            val = rate_matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=3, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Detection rate")
    fig.suptitle("Detection rate at multiple recall thresholds", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "multi_threshold_heatmap.pdf"))
    plt.close(fig)
    print("  Multi-threshold heatmap saved.")


def _plot_recall_vs_volume(df: pd.DataFrame, fig_dir: Path, res_dir: Path):
    """Plot recall vs intended PE volume."""
    set_nature_style()

    plot_df = df[df["intended_voxels"].notna() & (df["intended_voxels"] > 0)].copy()
    # Convert to volume if available
    vol_col = "intended_voxels"  # use voxels as universal metric

    fig, axes = plt.subplots(1, 3, figsize=(NATURE_DOUBLE_COL_IN, 2.5))

    # Panel A: scatter
    ax = axes[0]
    ax.scatter(plot_df[vol_col], plot_df["recall"],
               alpha=0.15, s=4, color=COLORS["blue"], rasterized=True)
    ax.set_xlabel("Intended PE voxels")
    ax.set_ylabel("Recall")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Recall vs PE size", fontsize=7)

    # Binned median
    bins = np.logspace(np.log10(max(plot_df[vol_col].min(), 1)),
                       np.log10(plot_df[vol_col].max()), 20)
    plot_df["vol_bin"] = pd.cut(plot_df[vol_col], bins=bins)
    binned = plot_df.groupby("vol_bin", observed=True)["recall"].agg(["median", "count"])
    bin_centers = [interval.mid for interval in binned.index]
    ax.plot(bin_centers, binned["median"], color=COLORS["red"], linewidth=1.5,
            label="Median", zorder=5)
    ax.legend(fontsize=5)

    # Panel B: cumulative detection rate by volume ceiling
    ax = axes[1]
    cutoffs = np.logspace(np.log10(max(plot_df[vol_col].min(), 1)),
                          np.log10(plot_df[vol_col].max()), 100)
    for thresh, color in zip(RECALL_THRESHOLDS, COLOR_LIST):
        rates = []
        for c in cutoffs:
            subset = plot_df[plot_df[vol_col] <= c]
            if len(subset) < 10:
                rates.append(float("nan"))
            else:
                rates.append((subset["recall"] >= thresh).mean())
        ax.plot(cutoffs, rates, label=f"recall≥{thresh:.0%}", color=color, linewidth=1)
    ax.set_xlabel("PE voxels ≤ X")
    ax.set_ylabel("Fraction achieving threshold")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=5)
    ax.set_title("Recall vs volume ceiling", fontsize=7)

    # Panel C: detected vs intended voxels
    ax = axes[2]
    det_df = plot_df[plot_df["detected_voxels"] > 0]
    ax.scatter(det_df["intended_voxels"], det_df["detected_voxels"],
               alpha=0.15, s=4, color=COLORS["green"], rasterized=True)
    max_val = max(det_df["intended_voxels"].max(), det_df["detected_voxels"].max())
    ax.plot([1, max_val], [1, max_val], "k--", linewidth=0.5, alpha=0.5, label="y=x")
    ax.set_xlabel("Intended voxels")
    ax.set_ylabel("Detected voxels")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Detected vs intended", fontsize=7)
    ax.legend(fontsize=5)

    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "recall_vs_volume.pdf"))
    plt.close(fig)
    print("  Recall vs volume plots saved.")


def _run_quality_regression(detected: pd.DataFrame, res_dir: Path) -> dict:
    """Linear regression: what predicts segmentation recall?"""
    features = ["intended_voxels", "intended_occlusion_frac", "intended_radius_mean_mm",
                "intended_compactness", "intended_wall_adherence_frac",
                "n_edges_with_intended_pe", "intended_gen_subsegmental_frac"]
    available = [f for f in features if f in detected.columns]

    reg_df = detected[["recall"] + available].dropna()
    if len(reg_df) < 50:
        print("  Too few samples for regression")
        return {}

    # Log-transform skewed features
    if "intended_voxels" in reg_df.columns:
        reg_df["log_intended_voxels"] = np.log1p(reg_df["intended_voxels"])
        available = [f if f != "intended_voxels" else "log_intended_voxels" for f in available]

    X = reg_df[available]
    y = reg_df["recall"]

    model = LinearRegression()
    model.fit(X, y)

    coef_df = pd.DataFrame({
        "feature": available,
        "coefficient": model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv(res_dir / "quality_regression_coefficients.csv", index=False)

    r_squared = model.score(X, y)
    print(f"  R² = {r_squared:.3f}")
    for _, row in coef_df.head(5).iterrows():
        print(f"    {row['feature']:45s}  β={row['coefficient']:+.4f}")

    return {
        "r_squared": float(r_squared),
        "coefficients": coef_df.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Deep detection analysis: what PEs get missed and why?
# ---------------------------------------------------------------------------

def run_deep_detection_analysis(df: pd.DataFrame, res_dir: Path, fig_dir: Path,
                                xlim_barplots: tuple[float, float] | None = None) -> dict:
    """Deep analysis of detected vs missed PEs with interaction effects."""
    print("\n" + "=" * 60)
    print("DEEP DETECTION ANALYSIS")
    print("=" * 60)

    df = df.copy()
    df["detected"] = df["pe_detected"].astype(int)
    deep_dir = fig_dir / "deep_detection"
    deep_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- 1. Interaction heatmaps ---
    print("\n--- Interaction Heatmaps ---")
    results["interactions"] = _compute_and_plot_interaction_heatmaps(df, res_dir, deep_dir)

    # --- 2. Within-patient paired comparison ---
    print("\n--- Within-Patient Paired Analysis ---")
    results["within_patient"] = _within_patient_analysis(df, res_dir, deep_dir)

    # --- 3. Hard-PE profile ---
    print("\n--- Hard-to-Detect PE Profile ---")
    results["hard_pe_profile"] = _hard_pe_profile(df, res_dir, deep_dir,
                                                      xlim_barplots=xlim_barplots)

    # --- 4. Worst patient case studies ---
    print("\n--- Worst Patient Case Studies ---")
    results["worst_patients"] = _worst_patient_analysis(df, res_dir, deep_dir)

    return results


def _compute_and_plot_interaction_heatmaps(df: pd.DataFrame, res_dir: Path, fig_dir: Path) -> dict:
    """Compute and plot miss rate for two-way feature interactions."""
    set_nature_style()
    results = {}

    interactions = [
        ("location", "intended_shape_category", "Location × Shape"),
        ("location", "severity_category", "Location × Severity"),
        ("intended_shape_category", "burden", "Shape × Burden"),
    ]

    # Volume bins for location × volume interaction
    df["volume_bin"] = pd.cut(
        df["intended_voxels"],
        bins=[0, 200, 500, 1000, 5000, 100000],
        labels=["<200", "200–500", "500–1k", "1k–5k", ">5k"],
    )
    # Wall adherence bins
    df["wall_adh_bin"] = pd.cut(
        df["intended_wall_adherence_frac"],
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=["<0.2", "0.2–0.4", "0.4–0.6", ">0.6"],
    )

    interactions += [
        ("location", "volume_bin", "Location × PE Volume"),
        ("intended_shape_category", "wall_adh_bin", "Shape × Wall Adherence"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(NATURE_DOUBLE_COL_IN, 5.0))
    axes = axes.flatten()

    for idx, (row_var, col_var, title) in enumerate(interactions):
        if row_var not in df.columns or col_var not in df.columns:
            continue

        # Compute miss rate per cell
        ct = df.groupby([row_var, col_var], observed=True).agg(
            n=("detected", "count"),
            n_det=("detected", "sum"),
        ).reset_index()
        ct["miss_rate"] = 1 - ct["n_det"] / ct["n"]

        # Pivot for heatmap
        pivot = ct.pivot_table(index=row_var, columns=col_var, values="miss_rate", fill_value=float("nan"))
        pivot_n = ct.pivot_table(index=row_var, columns=col_var, values="n", fill_value=0)

        # Save table
        table_name = f"interaction_{row_var}_x_{col_var}"
        ct.to_csv(res_dir / f"{table_name}.csv", index=False)
        results[table_name] = ct.to_dict("records")

        ax = axes[idx]
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.25)

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=5)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=5, rotation=45, ha="right")
        ax.set_title(title, fontsize=7)

        # Annotate with miss rate and n
        for i in range(pivot.values.shape[0]):
            for j in range(pivot.values.shape[1]):
                val = pivot.values[i, j]
                n = pivot_n.values[i, j]
                if np.isnan(val) or n < 5:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=4, color="gray")
                else:
                    color = "white" if val > 0.12 else "black"
                    ax.text(j, i, f"{val:.0%}\nn={n:.0f}", ha="center", va="center",
                            fontsize=4, color=color)

    # Hide unused
    for i in range(len(interactions), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Miss rate by PE characteristic interactions", fontsize=8, y=1.01)
    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "interaction_heatmaps.pdf"))
    plt.close(fig)
    print("  Interaction heatmaps saved.")

    return results


def _within_patient_analysis(df: pd.DataFrame, res_dir: Path, fig_dir: Path) -> dict:
    """Paired within-patient comparison of detected vs missed counterfactuals."""
    # Identify mixed patients
    patient_stats = df.groupby("patient_id")["detected"].agg(["sum", "count"])
    patient_stats["n_missed"] = patient_stats["count"] - patient_stats["sum"]
    mixed_pids = patient_stats[
        (patient_stats["sum"] > 0) & (patient_stats["n_missed"] > 0)
    ].index
    mixed_df = df[df["patient_id"].isin(mixed_pids)].copy()

    print(f"  Mixed patients (some detected, some missed): {len(mixed_pids)}")
    print(f"  CFs from mixed patients: {len(mixed_df)} ({mixed_df['detected'].sum()} det, "
          f"{(~mixed_df['pe_detected']).sum()} miss)")

    # Continuous features: paired Wilcoxon within patients
    features = [
        ("intended_voxels", "PE volume (voxels)"),
        ("intended_radius_mean_mm", "Vessel radius (mm)"),
        ("intended_wall_adherence_frac", "Wall adherence"),
        ("intended_compactness", "Compactness"),
        ("n_edges_with_intended_pe", "Vessel edges with PE"),
        ("qanadli_score", "Qanadli score"),
    ]

    # Per-patient deltas
    paired_results = []
    for feat, label in features:
        if feat not in mixed_df.columns:
            continue
        deltas = []
        for pid in mixed_pids:
            grp = mixed_df[mixed_df["patient_id"] == pid]
            det_med = grp.loc[grp["detected"] == 1, feat].median()
            miss_med = grp.loc[grp["detected"] == 0, feat].median()
            if not np.isnan(det_med) and not np.isnan(miss_med):
                deltas.append(det_med - miss_med)
        if len(deltas) < 10:
            continue
        deltas = np.array(deltas)
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(deltas)
        paired_results.append({
            "feature": feat, "label": label,
            "mean_delta": float(np.mean(deltas)), "median_delta": float(np.median(deltas)),
            "n_positive": int((deltas > 0).sum()), "n_negative": int((deltas < 0).sum()),
            "wilcoxon_p": float(pval),
        })
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    {label:30s}  Δ={np.mean(deltas):+.2f}  +/-={int((deltas>0).sum())}/{int((deltas<0).sum())}  "
              f"p={pval:.4f} {sig}")

    paired_df = pd.DataFrame(paired_results).sort_values("wilcoxon_p")
    paired_df.to_csv(res_dir / "within_patient_paired_tests.csv", index=False)

    # --- Plot: within-patient feature comparison (notched boxplots) ---
    set_nature_style()
    sig_features = [(f, l) for f, l in features if f in mixed_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(NATURE_DOUBLE_COL_IN, 4.5))
    axes = axes.flatten()

    for i, (feat, label) in enumerate(sig_features):
        if i >= len(axes):
            break
        ax = axes[i]
        det_vals = mixed_df.loc[mixed_df["detected"] == 1, feat].dropna()
        miss_vals = mixed_df.loc[mixed_df["detected"] == 0, feat].dropna()

        bp = ax.boxplot(
            [det_vals, miss_vals], positions=[0, 1], widths=0.5,
            notch=True, patch_artist=True,
            medianprops=dict(color="black", linewidth=1),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        bp["boxes"][0].set_facecolor(COLOR_DETECTED)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(COLOR_MISSED)
        bp["boxes"][1].set_alpha(0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Detected", "Missed"], fontsize=6)
        ax.set_ylabel(label, fontsize=6)

        # Add p-value
        p_row = paired_df[paired_df["feature"] == feat]
        if len(p_row) > 0:
            pval = p_row.iloc[0]["wilcoxon_p"]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_title(f"{label}\n(p={pval:.3f} {sig})", fontsize=6)

    for i in range(len(sig_features), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Within-patient: detected vs missed CF characteristics\n"
                 "(82 patients with mixed detection)", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "within_patient_boxplot.pdf"))
    plt.close(fig)

    # --- Categorical within-patient breakdown ---
    cat_results = {}
    for feat in ["location", "intended_shape_category", "severity_category", "laterality"]:
        if feat not in mixed_df.columns:
            continue
        ct = pd.crosstab(mixed_df[feat], mixed_df["detected"])
        ct.columns = ["missed", "detected"]
        ct["total"] = ct["detected"] + ct["missed"]
        ct["det_rate"] = ct["detected"] / ct["total"]
        ct = ct.sort_values("det_rate")
        ct.to_csv(res_dir / f"within_patient_crosstab_{feat}.csv")
        cat_results[feat] = ct.reset_index().to_dict("records")

    # Plot categorical breakdown
    cat_features = [f for f in ["location", "intended_shape_category", "severity_category"]
                    if f in mixed_df.columns]
    if cat_features:
        fig, axes = plt.subplots(1, len(cat_features), figsize=(NATURE_DOUBLE_COL_IN, 2.5))
        if len(cat_features) == 1:
            axes = [axes]

        for ax, feat in zip(axes, cat_features):
            ct = pd.crosstab(mixed_df[feat], mixed_df["detected"])
            ct.columns = ["missed", "detected"]
            ct["total"] = ct["detected"] + ct["missed"]
            ct["miss_rate"] = ct["missed"] / ct["total"]
            ct = ct.sort_values("miss_rate", ascending=False)

            ax.barh(range(len(ct)), ct["miss_rate"], color=COLORS["red"], alpha=0.7)
            ax.set_yticks(range(len(ct)))
            ax.set_yticklabels(ct.index, fontsize=6)
            ax.set_xlabel("Miss rate")
            ax.set_title(feat.replace("intended_", ""), fontsize=7)
            ax.set_xlim(0, 0.25)

            # Annotate with n
            for j, (idx, row) in enumerate(ct.iterrows()):
                ax.text(row["miss_rate"] + 0.005, j, f"n={row['total']:.0f}",
                        va="center", fontsize=5, color=COLORS["gray"])

        fig.suptitle("Miss rate by category (within mixed patients)", fontsize=8)
        fig.tight_layout()
        _save_fig(fig, str(fig_dir / "within_patient_categorical.pdf"))
        plt.close(fig)

    print("  Within-patient plots saved.")

    return {
        "n_mixed_patients": len(mixed_pids),
        "paired_tests": paired_df.to_dict("records"),
        "categorical": cat_results,
    }


def _hard_pe_profile(df: pd.DataFrame, res_dir: Path, fig_dir: Path,
                     xlim_barplots: tuple[float, float] | None = None) -> dict:
    """Characterize the profile of hard-to-detect PEs."""
    set_nature_style()

    missed = df[df["detected"] == 0]
    detected = df[df["detected"] == 1]

    # Build profile table
    profile_rows = []
    for feat, label, fmt in [
        ("intended_voxels", "Volume (voxels)", ".0f"),
        ("intended_radius_mean_mm", "Vessel radius (mm)", ".2f"),
        ("intended_wall_adherence_frac", "Wall adherence", ".3f"),
        ("intended_compactness", "Compactness", ".3f"),
        ("intended_centrality", "Centrality", ".3f"),
        ("intended_eccentricity", "Eccentricity", ".3f"),
        ("intended_occlusion_frac", "Occlusion fraction", ".3f"),
        ("n_edges_with_intended_pe", "Vessel edges with PE", ".0f"),
        ("qanadli_score", "Qanadli score", ".1f"),
        ("n_segments_affected", "Segments affected", ".1f"),
    ]:
        if feat not in df.columns:
            continue
        det_vals = detected[feat].dropna()
        miss_vals = missed[feat].dropna()
        if len(miss_vals) < 5:
            continue
        profile_rows.append({
            "feature": label,
            "detected_median": float(det_vals.median()),
            "detected_mean": float(det_vals.mean()),
            "missed_median": float(miss_vals.median()),
            "missed_mean": float(miss_vals.mean()),
            "direction": "↑ harder" if miss_vals.median() > det_vals.median() else "↓ harder",
        })

    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(res_dir / "hard_pe_profile.csv", index=False)

    print("  Hard-PE profile (missed vs detected medians):")
    for _, r in profile_df.iterrows():
        print(f"    {r['feature']:30s}  det={r['detected_median']:.3f}  "
              f"miss={r['missed_median']:.3f}  {r['direction']}")

    # --- Profile radar/bar chart ---
    # Normalize features to [0,1] range for comparison
    norm_features = []
    for _, r in profile_df.iterrows():
        all_vals = df[[c for c in df.columns if r["feature"].lower().replace(" ", "_").replace("(", "").replace(")", "") in c.lower().replace(" ", "_")]].iloc[:, 0] if False else None
        norm_features.append(r["feature"])

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN * 1.5, 3.0))

    y_pos = range(len(profile_df))
    # Plot detected vs missed medians as paired horizontal bars
    bar_h = 0.35
    ax.barh([y - bar_h / 2 for y in y_pos], profile_df["detected_median"],
            height=bar_h, color=COLOR_DETECTED, alpha=0.7, label="Detected (median)")
    ax.barh([y + bar_h / 2 for y in y_pos], profile_df["missed_median"],
            height=bar_h, color=COLOR_MISSED, alpha=0.7, label="Missed (median)")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(profile_df["feature"], fontsize=6)
    ax.set_xlabel("Value")
    ax.legend(fontsize=6)
    ax.set_title("Hard-to-Detect PE Profile: Detected vs Missed", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "hard_pe_profile.pdf"))
    plt.close(fig)

    # --- Categorical profile ---
    cat_profile = {}
    for feat in ["location", "intended_shape_category", "severity_category",
                  "occlusion", "burden", "laterality"]:
        if feat not in df.columns:
            continue
        total_ct = df[feat].value_counts()
        missed_ct = missed[feat].value_counts()
        rows = []
        for cat in total_ct.index:
            n_miss = missed_ct.get(cat, 0)
            n_total = total_ct[cat]
            rows.append({
                "category": cat, "n_total": n_total, "n_missed": n_miss,
                "miss_rate": n_miss / n_total,
            })
        cat_df = pd.DataFrame(rows).sort_values("miss_rate", ascending=False)
        cat_df.to_csv(res_dir / f"miss_rate_by_{feat}.csv", index=False)
        cat_profile[feat] = cat_df.to_dict("records")

    # Individual detection rate plots per axis (for PowerPoint)
    plot_feats = [f for f in ["location", "intended_shape_category", "severity_category",
                               "occlusion", "burden", "laterality"]
                  if f in cat_profile]

    overall_det_rate = float(df["pe_detected"].mean())

    from statsmodels.stats.proportion import proportion_confint

    for feat in plot_feats:
        set_nature_style()
        cat_df = pd.DataFrame(cat_profile[feat]).copy()
        cat_df["det_rate"] = 1 - cat_df["miss_rate"]
        cat_df["n_detected"] = cat_df["n_total"] - cat_df["n_missed"]
        cat_df = cat_df.sort_values("det_rate")

        # Clopper-Pearson 95% CI for each proportion
        ci_lo, ci_hi = proportion_confint(
            cat_df["n_detected"], cat_df["n_total"],
            alpha=0.05, method="beta",
        )
        cat_df["ci_lo"] = ci_lo
        cat_df["ci_hi"] = ci_hi
        cat_df["err_lo"] = cat_df["det_rate"] - ci_lo
        cat_df["err_hi"] = ci_hi - cat_df["det_rate"]

        n_bars = len(cat_df)
        fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN * 1.3, 2.8))

        palette = AXIS_PALETTES.get(feat, {})
        bar_colors = [palette.get(cat, COLORS["gray"]) for cat in cat_df["category"]]
        ax.barh(
            range(n_bars), cat_df["det_rate"], color=bar_colors, alpha=0.85, height=0.6,
            xerr=[cat_df["err_lo"].values, cat_df["err_hi"].values],
            ecolor="black", capsize=2, error_kw={"linewidth": 0.8, "capthick": 0.8},
        )
        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(cat_df["category"], fontsize=10)
        ax.set_xlabel("Detection rate", fontsize=10)
        ax.set_title(feat.replace("intended_", "").replace("_", " ").title(), fontsize=11)
        ax.axvline(overall_det_rate, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        # X-axis: use override if provided, else dynamic
        if xlim_barplots is not None:
            ax.set_xlim(xlim_barplots)
        else:
            x_lo = max(0, cat_df["ci_lo"].min() - 0.05)
            x_lo = round(x_lo * 20) / 20  # round down to nearest 0.05
            ax.set_xlim(x_lo, 1.00)
        ax.tick_params(axis="x", labelsize=9)

        for j, (_, row) in enumerate(cat_df.iterrows()):
            ax.text(row["ci_hi"] + 0.01, j,
                    f"{row['det_rate']:.1%} (n={row['n_total']})",
                    va="center", fontsize=8)

        fig.tight_layout()
        clean_name = feat.replace("intended_", "")
        _save_fig(fig, str(fig_dir / f"detection_rate_{clean_name}.pdf"))
        plt.close(fig)
    plt.close(fig)

    print("  Hard-PE profile plots saved.")
    return {"continuous": profile_df.to_dict("records"), "categorical": cat_profile}


def _worst_patient_analysis(df: pd.DataFrame, res_dir: Path, fig_dir: Path) -> dict:
    """Deep dive into patients with lowest detection rates."""
    set_nature_style()

    patient_stats = df.groupby("patient_id").agg(
        n=("detected", "count"),
        n_det=("detected", "sum"),
        det_rate=("detected", "mean"),
        mean_voxels=("intended_voxels", "mean"),
        mean_radius=("intended_radius_mean_mm", "mean"),
        mean_wall_adh=("intended_wall_adherence_frac", "mean"),
        mean_compactness=("intended_compactness", "mean"),
        mean_qanadli=("qanadli_score", "mean"),
    ).reset_index()
    patient_stats["n_missed"] = patient_stats["n"] - patient_stats["n_det"]
    patient_stats = patient_stats.sort_values("det_rate")

    # Save full table
    patient_stats.to_csv(res_dir / "patient_detection_summary.csv", index=False)

    # Worst 20 patients
    worst = patient_stats.head(20).copy()
    worst.to_csv(res_dir / "worst_patients.csv", index=False)

    print(f"  Worst 20 patients:")
    for _, row in worst.iterrows():
        print(f"    {row['patient_id']:15s}  det={row['n_det']:.0f}/{row['n']:.0f} ({row['det_rate']:.0%})"
              f"  voxels={row['mean_voxels']:.0f}  radius={row['mean_radius']:.1f}mm"
              f"  wall_adh={row['mean_wall_adh']:.2f}")

    # --- Plot: worst patients with feature breakdown ---
    fig, axes = plt.subplots(1, 3, figsize=(NATURE_DOUBLE_COL_IN, 4.0))

    # Panel A: detection rate sorted bar
    ax = axes[0]
    all_sorted = patient_stats.sort_values("det_rate")
    colors = [COLORS["red"] if r < 0.8 else COLORS["yellow"] if r < 0.9
              else COLORS["blue"] for r in all_sorted["det_rate"]]
    ax.barh(range(len(all_sorted)), all_sorted["det_rate"],
            color=colors, alpha=0.8, height=1.0)
    ax.axvline(0.926, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Detection rate")
    ax.set_ylabel("Patients (sorted)")
    ax.set_title("Per-patient detection rate", fontsize=7)
    ax.set_xlim(0.4, 1.02)

    # Panel B: for worst 20, show what their missed CFs look like
    ax = axes[1]
    worst_pids = set(worst["patient_id"])
    worst_df = df[df["patient_id"].isin(worst_pids)]
    worst_missed = worst_df[worst_df["detected"] == 0]
    worst_detected = worst_df[worst_df["detected"] == 1]

    # Location distribution of missed CFs from worst patients
    if len(worst_missed) > 0:
        loc_counts = worst_missed["location"].value_counts()
        ax.barh(range(len(loc_counts)), loc_counts.values, color=COLORS["red"], alpha=0.7)
        ax.set_yticks(range(len(loc_counts)))
        ax.set_yticklabels(loc_counts.index, fontsize=6)
        ax.set_xlabel("Count")
        ax.set_title("Missed CFs from worst 20 patients:\nby location", fontsize=7)

    # Panel C: shape category of missed CFs
    ax = axes[2]
    if len(worst_missed) > 0 and "intended_shape_category" in worst_missed.columns:
        shape_counts = worst_missed["intended_shape_category"].value_counts()
        ax.barh(range(len(shape_counts)), shape_counts.values, color=COLORS["red"], alpha=0.7)
        ax.set_yticks(range(len(shape_counts)))
        ax.set_yticklabels(shape_counts.index, fontsize=6)
        ax.set_xlabel("Count")
        ax.set_title("Missed CFs from worst 20 patients:\nby shape", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "worst_patients.pdf"))
    plt.close(fig)

    # --- Per-patient detail plot: top 10 worst patients, each CF as a dot ---
    top10 = patient_stats.head(10)
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 3.0))

    for i, (_, pat) in enumerate(top10.iterrows()):
        pid = pat["patient_id"]
        pat_cfs = df[df["patient_id"] == pid]
        det_cfs = pat_cfs[pat_cfs["detected"] == 1]
        miss_cfs = pat_cfs[pat_cfs["detected"] == 0]

        # Scatter: x=intended_voxels, different markers for det/miss
        ax.scatter(det_cfs["intended_voxels"], [i] * len(det_cfs),
                   marker="o", s=15, alpha=0.6, color=COLORS["blue"], zorder=3)
        ax.scatter(miss_cfs["intended_voxels"], [i] * len(miss_cfs),
                   marker="x", s=20, alpha=0.9, color=COLORS["red"], linewidths=1, zorder=4)

    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([f"{r['patient_id'][:10]}.. ({r['n_det']:.0f}/{r['n']:.0f})"
                        for _, r in top10.iterrows()], fontsize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Intended PE voxels")
    ax.set_title("Top 10 worst patients: each CF as a dot (o=detected, x=missed)", fontsize=7)
    # Legend
    ax.scatter([], [], marker="o", color=COLORS["blue"], s=15, label="Detected")
    ax.scatter([], [], marker="x", color=COLORS["red"], s=20, linewidths=1, label="Missed")
    ax.legend(fontsize=5, loc="lower right")

    fig.tight_layout()
    _save_fig(fig, str(fig_dir / "worst_patients_detail.pdf"))
    plt.close(fig)

    print("  Worst patient plots saved.")

    return {
        "worst_20": worst.to_dict("records"),
        "n_below_80": int((patient_stats["det_rate"] < 0.8).sum()),
        "n_below_90": int((patient_stats["det_rate"] < 0.9).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_segmentation_evaluation(
    vessel_overlay_csv: str | None = None,
    nope_baseline_csv: str | None = None,
    output_dir: str | None = None,
    figure_dir: str | None = None,
    model_name: str | None = None,
    xlim_barplots: tuple[float, float] | None = None,
):
    """Run the full three-level segmentation stress-test."""

    _model = model_name or MODEL_NAME
    res_dir = Path(output_dir or OUTPUT_DIR)
    fig_dir = Path(figure_dir or FIGURE_DIR)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SEGMENTATION MODEL STRESS-TEST: {_model}")
    print("=" * 60)

    # --- Load data ---
    raw_df, nope_df = load_data(vessel_overlay_csv or VESSEL_OVERLAY_CSV,
                                nope_baseline_csv or NOPE_BASELINE_CSV)
    df = prepare_analysis_df(raw_df, model_name=_model)
    df.to_csv(res_dir / "analysis_dataframe.csv", index=False)
    print(f"\nAnalysis dataframe: {len(df)} samples, {df['patient_id'].nunique()} patients")
    print(f"Binary detection rate: {df['pe_detected'].mean():.3f}")
    print(f"Recall distribution: mean={df['recall'].mean():.3f}, median={df['recall'].median():.3f}")

    results = {}

    # --- Level 0: Baseline comparison ---
    results["baseline"] = run_baseline_comparison(df, nope_df, res_dir, fig_dir)

    # --- Level 1: Binary detection ---
    results["binary_detection"] = run_binary_detection_analysis(df, res_dir, fig_dir)

    # --- Level 2: Segmentation quality + detection predictors ---
    results["segmentation_quality"] = run_segmentation_quality_analysis(df, res_dir, fig_dir)

    # --- Per-patient deep analysis ---
    results["per_patient"] = run_per_patient_analysis(df, nope_df, res_dir, fig_dir)

    # --- Deep detection analysis ---
    results["deep_detection"] = run_deep_detection_analysis(df, res_dir, fig_dir,
                                                              xlim_barplots=xlim_barplots)

    # --- False positive analysis (no_intended_pe cases) ---
    fp_df = raw_df[raw_df["analysis_group"] == "no_intended_pe"]
    if len(fp_df) > 0:
        print(f"\n--- Text-Conditioned False Positives ({len(fp_df)} cases) ---")
        print(f"  Scans with PE text conditioning but empty mask - model detected PE in all {len(fp_df)}")
        fp_df.to_csv(res_dir / "text_conditioned_false_positives.csv", index=False)
        results["text_conditioned_fp"] = {
            "n": len(fp_df),
            "detected_voxels_mean": float(fp_df["detected_voxels"].mean()),
        }

    # --- Summary ---
    summary = {
        "model": _model,
        "n_valid_samples": results["binary_detection"]["n_total"],
        "n_patients": int(df["patient_id"].nunique()),
        "nope_baseline_any_pe": results["baseline"]["nope_any_pe_detected"],
        "binary_detection_rate": results["binary_detection"]["detection_rate"],
        "n_complete_misses": results["binary_detection"]["n_missed"],
        "multi_threshold_rates": results["segmentation_quality"].get("multi_threshold_rates", []),
        "quality_regression_r2": results["segmentation_quality"].get("quality_regression", {}).get("r_squared"),
        "patient_icc": results["binary_detection"].get("patient_icc"),
        "per_patient_mean_detection": results["per_patient"]["mean_detection_rate"],
    }

    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"DONE. Results in {res_dir}/, figures in {fig_dir}/")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    run_segmentation_evaluation()
