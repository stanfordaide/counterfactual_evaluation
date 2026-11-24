"""Patient-level analysis: are some patients systematically harder?

Separates patient-driven failures from PE-characteristic-driven failures.
"""

import numpy as np
import pandas as pd



def compute_patient_level_metrics(
    df: pd.DataFrame,
    patient_col: str = "patient_id_metadata",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute per-patient sensitivity and score statistics.

    Args:
        df: Analysis DataFrame.
        patient_col: Column identifying patients.
        threshold: Decision threshold.

    Returns:
        DataFrame with one row per patient: sensitivity, miss_rate, mean_score,
        n_counterfactuals, and patient demographics.
    """
    rows = []
    for pid, group in df.groupby(patient_col):
        targets = group["target"].values
        scores = group["predicted_score"].values
        preds = (scores >= threshold).astype(int)

        n = len(group)
        n_detected = int(preds.sum())
        sensitivity = n_detected / n

        row = {
            "patient_id": pid,
            "n_counterfactuals": n,
            "n_detected": n_detected,
            "n_missed": n - n_detected,
            "sensitivity": sensitivity,
            "miss_rate": 1.0 - sensitivity,
            "mean_score": float(scores.mean()),
            "median_score": float(np.median(scores)),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "score_std": float(scores.std()),
        }

        # Add demographics from first row
        for col in ["gender_metadata", "age_metadata", "race_metadata"]:
            if col in group.columns:
                row[col.replace("_metadata", "")] = group[col].iloc[0]

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("sensitivity", ascending=True)
    return result


def summarize_patient_variation(patient_df: pd.DataFrame) -> dict:
    """Summarize how much variation in performance is driven by patient identity.

    Args:
        patient_df: Output of compute_patient_level_metrics().

    Returns:
        Dict with summary statistics about patient-level variation.
    """
    sens = patient_df["sensitivity"]
    summary = {
        "n_patients": len(patient_df),
        "mean_sensitivity": float(sens.mean()),
        "std_sensitivity": float(sens.std()),
        "min_sensitivity": float(sens.min()),
        "max_sensitivity": float(sens.max()),
        "iqr_sensitivity": float(sens.quantile(0.75) - sens.quantile(0.25)),
        "n_patients_perfect": int((sens == 1.0).sum()),
        "n_patients_zero": int((sens == 0.0).sum()),
        "n_patients_below_50pct": int((sens < 0.5).sum()),
        "n_patients_above_90pct": int((sens >= 0.9).sum()),
    }

    print(f"\n  Patient-level variation ({summary['n_patients']} patients):")
    print(f"    Sensitivity: {summary['mean_sensitivity']:.3f} +/- {summary['std_sensitivity']:.3f}")
    print(f"    Range: [{summary['min_sensitivity']:.3f}, {summary['max_sensitivity']:.3f}]")
    print(f"    IQR: {summary['iqr_sensitivity']:.3f}")
    print(f"    Perfect detection (100%): {summary['n_patients_perfect']} patients")
    print(f"    Zero detection (0%): {summary['n_patients_zero']} patients")
    print(f"    Below 50% sensitivity: {summary['n_patients_below_50pct']} patients")
    print(f"    Above 90% sensitivity: {summary['n_patients_above_90pct']} patients")

    return summary


def icc_patient_effect(
    df: pd.DataFrame,
    patient_col: str = "patient_id_metadata",
    outcome_col: str = "classifier_missed",
) -> float | None:
    """Compute intraclass correlation (ICC) to quantify patient-level clustering of failures.

    A high ICC means failures cluster within patients (patient anatomy matters).
    A low ICC means failures are spread uniformly (PE characteristics or randomness).

    Uses one-way random effects ANOVA-based ICC(1,1).

    Args:
        df: Analysis DataFrame.
        patient_col: Column identifying patients.
        outcome_col: Binary outcome column.

    Returns:
        ICC value (0 to 1), or None if computation fails.
    """
    groups = [group[outcome_col].values for _, group in df.groupby(patient_col)]
    k = len(groups)
    if k < 2:
        return None

    # One-way ANOVA decomposition
    ns = np.array([len(g) for g in groups])
    n_total = ns.sum()
    grand_mean = df[outcome_col].mean()

    # Between-group sum of squares
    group_means = np.array([g.mean() for g in groups])
    ss_between = np.sum(ns * (group_means - grand_mean) ** 2)

    # Within-group sum of squares
    ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)

    # Average group size (harmonic-ish for unequal groups)
    n0 = (n_total - np.sum(ns ** 2) / n_total) / (k - 1)

    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    icc = max(0.0, icc)  # ICC can't be negative in practice

    print(f"\n  ICC(1,1) for patient clustering: {icc:.3f}")
    if icc < 0.05:
        print("    -> Negligible patient effect (failures are uniformly distributed)")
    elif icc < 0.15:
        print("    -> Small patient effect")
    elif icc < 0.30:
        print("    -> Moderate patient effect (some patients are systematically harder)")
    else:
        print("    -> Large patient effect (patient anatomy strongly drives failures)")

    return icc
