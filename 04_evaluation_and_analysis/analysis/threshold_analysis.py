"""Continuous variable threshold analysis: at what PE size/burden is detection reliable?

For each continuous PE characteristic (volume, Qanadli score, number of segments),
finds the threshold above which the classifier achieves a target sensitivity
(e.g., 95%). Produces dose-response style curves.
"""

import pandas as pd

from analysis.config import PE_CONTINUOUS_AXES


def compute_cumulative_sensitivity(
    df: pd.DataFrame,
    continuous_col: str,
    threshold: float = 0.5,
    direction: str = "above",
    min_samples: int = 10,
) -> pd.DataFrame:
    """Compute sensitivity for samples above (or below) increasing thresholds of a continuous variable.

    For example: "What is the sensitivity for all PEs with volume > X?"

    Args:
        df: Analysis DataFrame with predicted_score and the continuous column.
        continuous_col: Continuous PE characteristic column.
        threshold: Classifier decision threshold.
        direction: 'above' means we compute sensitivity for samples >= cutoff (increasing).
                   'below' means sensitivity for samples <= cutoff (decreasing).
        min_samples: Minimum samples required to report sensitivity at a cutoff.

    Returns:
        DataFrame with cutoff, sensitivity, n_samples, n_detected for each cutoff.
    """
    valid = df[[continuous_col, "predicted_score", "target"]].dropna()
    if len(valid) == 0:
        return pd.DataFrame()

    values = sorted(valid[continuous_col].unique())
    preds = (valid["predicted_score"].values >= threshold).astype(int)

    rows = []
    for cutoff in values:
        if direction == "above":
            mask = valid[continuous_col].values >= cutoff
        else:
            mask = valid[continuous_col].values <= cutoff

        n = int(mask.sum())
        if n < min_samples:
            continue

        n_detected = int(preds[mask].sum())
        sensitivity = n_detected / n

        rows.append({
            "cutoff": cutoff,
            "n_samples": n,
            "n_detected": n_detected,
            "sensitivity": sensitivity,
            "miss_rate": 1.0 - sensitivity,
        })

    return pd.DataFrame(rows)


def find_sensitivity_threshold(
    df: pd.DataFrame,
    continuous_col: str,
    target_sensitivity: float = 0.95,
    classifier_threshold: float = 0.5,
    direction: str = "above",
    min_samples: int = 10,
) -> dict:
    """Find the cutoff of a continuous PE variable above which sensitivity reaches a target.

    Example: "PE volume above X ml is detected in >95% of cases."

    Args:
        df: Analysis DataFrame.
        continuous_col: Continuous PE characteristic.
        target_sensitivity: Target sensitivity level (e.g., 0.95).
        classifier_threshold: Classifier decision threshold.
        direction: 'above' or 'below'.
        min_samples: Minimum samples for a valid cutoff.

    Returns:
        Dict with the threshold, achieved sensitivity, and sample count.
        Returns None values if the target is never reached.
    """
    curve = compute_cumulative_sensitivity(
        df, continuous_col,
        threshold=classifier_threshold,
        direction=direction,
        min_samples=min_samples,
    )

    if len(curve) == 0:
        return {
            "continuous_var": continuous_col,
            "target_sensitivity": target_sensitivity,
            "threshold_value": None,
            "achieved_sensitivity": None,
            "n_samples_at_threshold": None,
            "direction": direction,
            "reached": False,
        }

    # Find first cutoff where sensitivity >= target
    above_target = curve[curve["sensitivity"] >= target_sensitivity]

    if len(above_target) > 0:
        best = above_target.iloc[0]  # first (lowest cutoff) that meets the target
        return {
            "continuous_var": continuous_col,
            "target_sensitivity": target_sensitivity,
            "threshold_value": float(best["cutoff"]),
            "achieved_sensitivity": float(best["sensitivity"]),
            "n_samples_at_threshold": int(best["n_samples"]),
            "direction": direction,
            "reached": True,
        }
    else:
        # Target never reached; report best achieved
        best_idx = curve["sensitivity"].idxmax()
        best = curve.loc[best_idx]
        return {
            "continuous_var": continuous_col,
            "target_sensitivity": target_sensitivity,
            "threshold_value": None,
            "best_achieved_sensitivity": float(best["sensitivity"]),
            "best_at_cutoff": float(best["cutoff"]),
            "n_samples_at_best": int(best["n_samples"]),
            "direction": direction,
            "reached": False,
        }


def run_threshold_analysis(
    df: pd.DataFrame,
    axes: list[str] | None = None,
    target_sensitivities: list[float] | None = None,
    classifier_threshold: float = 0.5,
    min_samples: int = 10,
) -> dict:
    """Run threshold analysis across all continuous PE variables.

    Args:
        df: Analysis DataFrame.
        axes: Continuous columns to analyze. Defaults to PE_CONTINUOUS_AXES.
        target_sensitivities: Sensitivity targets to find thresholds for.
        classifier_threshold: Classifier decision threshold.
        min_samples: Minimum samples for valid cutoffs.

    Returns:
        Dict with:
          - 'curves': {axis_name -> cumulative sensitivity DataFrame}
          - 'thresholds': list of threshold results for each (axis, target) combo
    """
    if axes is None:
        axes = PE_CONTINUOUS_AXES
    if target_sensitivities is None:
        target_sensitivities = [0.80, 0.90, 0.95]

    results = {"curves": {}, "thresholds": []}

    for axis in axes:
        if axis not in df.columns or df[axis].isna().all():
            continue

        print(f"\n  Analyzing: {axis}")
        print(f"    Range: [{df[axis].min():.4f}, {df[axis].max():.4f}], "
              f"median: {df[axis].median():.4f}")

        # Cumulative sensitivity curve
        curve = compute_cumulative_sensitivity(
            df, axis,
            threshold=classifier_threshold,
            direction="above",
            min_samples=min_samples,
        )
        results["curves"][axis] = curve

        # Find thresholds for each target sensitivity
        for target in target_sensitivities:
            result = find_sensitivity_threshold(
                df, axis,
                target_sensitivity=target,
                classifier_threshold=classifier_threshold,
                direction="above",
                min_samples=min_samples,
            )
            results["thresholds"].append(result)

            if result["reached"]:
                print(f"    {axis} >= {result['threshold_value']:.4f} -> "
                      f"sensitivity {result['achieved_sensitivity']:.1%} "
                      f"(n={result['n_samples_at_threshold']})")
            else:
                best_sens = result.get("best_achieved_sensitivity", "N/A")
                print(f"    Target {target:.0%} NOT reached (best: {best_sens})")

    return results
