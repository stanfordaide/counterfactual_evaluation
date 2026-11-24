"""Stratified failure analysis for counterfactual PE evaluation.

Groups predictions by PE characteristics and computes per-stratum metrics,
failure rates, and flags small strata.
"""

import warnings

import numpy as np
import pandas as pd

from analysis.config import PE_STRATIFICATION_AXES
from analysis.metrics import (
    compute_metrics,
    bootstrap_ci,
    sensitivity_at_threshold,
)


MIN_STRATUM_SIZE = 20  # Flag strata below this


def compute_stratum_metrics(
    df: pd.DataFrame,
    group_col: str,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute metrics for each stratum of a categorical PE characteristic.

    Args:
        df: Analysis DataFrame with predicted_score, target, and group_col.
        group_col: Column name to stratify by (e.g., 'location').
        threshold: Decision threshold.
        n_bootstrap: Bootstrap iterations for CIs.

    Returns:
        DataFrame with one row per stratum, containing metrics + CIs.
    """
    if group_col not in df.columns:
        warnings.warn(f"Column '{group_col}' not found in DataFrame. Skipping.", stacklevel=2)
        return pd.DataFrame()

    if df["predicted_score"].isna().all():
        warnings.warn("No prediction scores available. Run inference first.", stacklevel=2)
        return pd.DataFrame()

    rows = []
    for stratum, group in df.groupby(group_col):
        targets = group["target"].values
        scores = group["predicted_score"].values

        # Drop NaN scores
        valid = ~np.isnan(scores)
        targets = targets[valid]
        scores = scores[valid]

        if len(targets) == 0:
            continue

        m = compute_metrics(targets, scores, threshold=threshold)

        # Bootstrap CI for sensitivity
        def _sens(t, s):
            return sensitivity_at_threshold(t, s, threshold)

        _, sens_lo, sens_hi = bootstrap_ci(targets, scores, _sens, n_bootstrap=n_bootstrap)

        row = {
            "stratum": stratum,
            "characteristic": group_col,
            "n": m.n_samples,
            "n_positive": m.n_positive,
            "sensitivity": m.sensitivity,
            "sensitivity_ci_lower": sens_lo,
            "sensitivity_ci_upper": sens_hi,
            "miss_rate": m.miss_rate,
            "f1": m.f1,
            "balanced_accuracy": m.balanced_accuracy,
            "mean_score": m.mean_score,
            "median_score": m.median_score,
            "small_stratum": m.n_samples < MIN_STRATUM_SIZE,
        }

        # AUC only if both classes present
        if m.auc_roc is not None:
            row["auc_roc"] = m.auc_roc
            row["auc_pr"] = m.auc_pr

        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values("miss_rate", ascending=False)
    return result


def run_stratified_analysis(
    df: pd.DataFrame,
    threshold: float = 0.5,
    axes: list[str] | None = None,
    n_bootstrap: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Run stratified analysis across all PE characteristic axes.

    Args:
        df: Analysis DataFrame from build_analysis_dataframe().
        threshold: Decision threshold.
        axes: List of columns to stratify by. Defaults to PE_STRATIFICATION_AXES.
        n_bootstrap: Bootstrap iterations for CIs.

    Returns:
        Dict mapping characteristic name -> per-stratum metrics DataFrame.
    """
    if axes is None:
        axes = PE_STRATIFICATION_AXES

    results = {}
    for axis in axes:
        print(f"\n  Stratifying by: {axis}")
        stratum_df = compute_stratum_metrics(df, axis, threshold=threshold, n_bootstrap=n_bootstrap)
        if len(stratum_df) > 0:
            results[axis] = stratum_df
            # Print summary
            for _, row in stratum_df.iterrows():
                flag = " [!]" if row.get("small_stratum", False) else ""
                print(
                    f"    {row['stratum']:25s}  n={row['n']:4d}  "
                    f"sensitivity={row['sensitivity']:.3f} "
                    f"[{row['sensitivity_ci_lower']:.3f}, {row['sensitivity_ci_upper']:.3f}]  "
                    f"miss_rate={row['miss_rate']:.3f}{flag}"
                )
    return results


def compute_overall_metrics(
    df: pd.DataFrame,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
) -> dict:
    """Compute overall (unstratified) metrics with bootstrap CIs.

    Args:
        df: Analysis DataFrame.
        threshold: Decision threshold.
        n_bootstrap: Bootstrap iterations.

    Returns:
        Dict with overall metrics and CIs.
    """
    targets = df["target"].values
    scores = df["predicted_score"].values

    valid = ~np.isnan(scores)
    targets = targets[valid]
    scores = scores[valid]

    m = compute_metrics(targets, scores, threshold=threshold)

    def _sens(t, s):
        return sensitivity_at_threshold(t, s, threshold)

    _, sens_lo, sens_hi = bootstrap_ci(targets, scores, _sens, n_bootstrap=n_bootstrap)

    result = m.to_dict()
    result["sensitivity_ci_lower"] = sens_lo
    result["sensitivity_ci_upper"] = sens_hi

    print(f"\n  Overall ({m.n_samples} samples):")
    print(f"    Sensitivity @{threshold}: {m.sensitivity:.3f} [{sens_lo:.3f}, {sens_hi:.3f}]")
    print(f"    Miss rate: {m.miss_rate:.3f}")
    print(f"    Mean score: {m.mean_score:.3f}")
    if m.auc_roc is not None:
        print(f"    AUC-ROC: {m.auc_roc:.3f}")
    return result
