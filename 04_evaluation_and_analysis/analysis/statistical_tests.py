"""Hypothesis tests and effect sizes for stratified failure analysis.

Tests whether failure rates differ significantly across PE characteristics,
computes odds ratios, and applies multiple comparisons correction.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def chi_squared_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str = "classifier_missed",
) -> dict:
    """Chi-squared test for association between a PE characteristic and classifier failure.

    Args:
        df: Analysis DataFrame.
        group_col: Categorical column to test (e.g., 'location').
        outcome_col: Binary outcome column (1 = failure).

    Returns:
        Dict with chi2 statistic, p-value, degrees of freedom, and contingency table.
    """
    ct = pd.crosstab(df[group_col], df[outcome_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": float("nan"), "p_value": float("nan"), "dof": 0, "contingency_table": ct}

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    # Check expected cell counts
    low_expected = (expected < 5).sum()
    if low_expected > 0:
        warnings.warn(
            f"Chi-squared test for '{group_col}': {low_expected} cells have expected count < 5. "
            "Consider Fisher's exact test.",
            stacklevel=2,
        )

    return {"chi2": chi2, "p_value": p, "dof": dof, "contingency_table": ct}


def fisher_exact_pairwise(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str = "classifier_missed",
) -> pd.DataFrame:
    """Pairwise Fisher's exact tests between all strata of a PE characteristic.

    Args:
        df: Analysis DataFrame.
        group_col: Categorical column.
        outcome_col: Binary outcome column.

    Returns:
        DataFrame with columns: stratum_a, stratum_b, odds_ratio, p_value.
    """
    strata = sorted(df[group_col].dropna().unique())
    rows = []

    for i, a in enumerate(strata):
        for b in strata[i + 1:]:
            grp_a = df[df[group_col] == a][outcome_col]
            grp_b = df[df[group_col] == b][outcome_col]

            table = np.array([
                [int((grp_a == 0).sum()), int((grp_a == 1).sum())],
                [int((grp_b == 0).sum()), int((grp_b == 1).sum())],
            ])

            if table.min() < 0 or table.sum() == 0:
                continue

            oddsratio, p = stats.fisher_exact(table)
            rows.append({
                "stratum_a": a,
                "stratum_b": b,
                "n_a": len(grp_a),
                "n_b": len(grp_b),
                "failure_rate_a": grp_a.mean(),
                "failure_rate_b": grp_b.mean(),
                "odds_ratio": oddsratio,
                "p_value": p,
            })

    return pd.DataFrame(rows)


def odds_ratios_vs_reference(
    df: pd.DataFrame,
    group_col: str,
    reference_stratum: str | None = None,
    outcome_col: str = "classifier_missed",
) -> pd.DataFrame:
    """Compute odds ratios for classifier failure vs. a reference stratum.

    Args:
        df: Analysis DataFrame.
        group_col: Categorical column.
        reference_stratum: Reference category. If None, uses the stratum with the lowest failure rate.
        outcome_col: Binary outcome column.

    Returns:
        DataFrame with odds_ratio, ci_lower, ci_upper, p_value per stratum.
    """
    strata = sorted(df[group_col].dropna().unique())

    if reference_stratum is None:
        failure_rates = df.groupby(group_col)[outcome_col].mean()
        reference_stratum = failure_rates.idxmin()

    ref = df[df[group_col] == reference_stratum][outcome_col]
    ref_table = [int((ref == 0).sum()), int((ref == 1).sum())]

    rows = []
    for stratum in strata:
        grp = df[df[group_col] == stratum][outcome_col]
        table = np.array([
            ref_table,
            [int((grp == 0).sum()), int((grp == 1).sum())],
        ])

        oddsratio, p = stats.fisher_exact(table)

        # Log-odds CI (Woolf's method)
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        if all(x > 0 for x in [a, b, c, d]):
            log_or = np.log(oddsratio)
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_lower = np.exp(log_or - 1.96 * se)
            ci_upper = np.exp(log_or + 1.96 * se)
        else:
            ci_lower = ci_upper = float("nan")

        rows.append({
            "stratum": stratum,
            "is_reference": stratum == reference_stratum,
            "n": len(grp),
            "failure_rate": grp.mean(),
            "odds_ratio": oddsratio,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p,
        })

    result = pd.DataFrame(rows)
    return result


def logistic_regression_failure(
    df: pd.DataFrame,
    predictors: list[str],
    outcome_col: str = "classifier_missed",
) -> pd.DataFrame:
    """Logistic regression predicting classifier failure from PE characteristics.

    Args:
        df: Analysis DataFrame.
        predictors: List of categorical/continuous column names.
        outcome_col: Binary outcome column.

    Returns:
        DataFrame with coefficient, odds_ratio, ci_lower, ci_upper, p_value per predictor.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        warnings.warn("statsmodels not installed. Skipping logistic regression.", stacklevel=2)
        return pd.DataFrame()

    df_model = df[predictors + [outcome_col]].dropna()
    if len(df_model) < 10:
        warnings.warn("Too few samples for logistic regression.", stacklevel=2)
        return pd.DataFrame()

    # One-hot encode categoricals
    X = pd.get_dummies(df_model[predictors], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    y = df_model[outcome_col].astype(float)

    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception as e:
        warnings.warn(f"Logistic regression failed: {e}", stacklevel=2)
        return pd.DataFrame()

    results = pd.DataFrame({
        "predictor": model.params.index,
        "coefficient": model.params.values,
        "odds_ratio": np.exp(model.params.values),
        "ci_lower": np.exp(model.conf_int()[0].values),
        "ci_upper": np.exp(model.conf_int()[1].values),
        "p_value": model.pvalues.values,
    })
    return results


def apply_fdr_correction(
    p_values: np.ndarray | list[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply multiple comparisons correction.

    Args:
        p_values: Array of p-values.
        method: Correction method ('fdr_bh' for Benjamini-Hochberg).
        alpha: Significance level.

    Returns:
        Tuple of (reject_array, corrected_p_values).
    """
    p_values = np.asarray(p_values)
    valid = ~np.isnan(p_values)
    if valid.sum() == 0:
        return np.full_like(p_values, False, dtype=bool), p_values

    reject = np.full_like(p_values, False, dtype=bool)
    corrected = np.full_like(p_values, np.nan)

    reject[valid], corrected[valid], _, _ = multipletests(
        p_values[valid], alpha=alpha, method=method
    )
    return reject, corrected


def run_all_tests(
    df: pd.DataFrame,
    axes: list[str],
    outcome_col: str = "classifier_missed",
) -> dict:
    """Run the full suite of statistical tests across all stratification axes.

    Args:
        df: Analysis DataFrame.
        axes: List of categorical columns to test.
        outcome_col: Binary outcome column.

    Returns:
        Dict with keys: chi_squared, fisher_pairwise, odds_ratios, logistic_regression.
        Each maps axis name -> results DataFrame/dict.
    """
    if df[outcome_col].isna().all():
        warnings.warn("No outcome data available. Run inference first.", stacklevel=2)
        return {}

    results = {
        "chi_squared": {},
        "fisher_pairwise": {},
        "odds_ratios": {},
    }

    all_p_values = []
    p_value_labels = []

    for axis in axes:
        if axis not in df.columns or df[axis].isna().all():
            continue

        print(f"\n  Testing: {axis}")

        # Chi-squared
        chi2_result = chi_squared_test(df, axis, outcome_col)
        results["chi_squared"][axis] = chi2_result
        print(f"    Chi-squared: chi2={chi2_result['chi2']:.3f}, p={chi2_result['p_value']:.4f}")
        all_p_values.append(chi2_result["p_value"])
        p_value_labels.append(f"chi2_{axis}")

        # Fisher pairwise
        fisher_df = fisher_exact_pairwise(df, axis, outcome_col)
        results["fisher_pairwise"][axis] = fisher_df
        for _, row in fisher_df.iterrows():
            all_p_values.append(row["p_value"])
            p_value_labels.append(f"fisher_{axis}_{row['stratum_a']}_vs_{row['stratum_b']}")

        # Odds ratios vs reference
        or_df = odds_ratios_vs_reference(df, axis, outcome_col=outcome_col)
        results["odds_ratios"][axis] = or_df

    # FDR correction across all tests
    if all_p_values:
        reject, corrected = apply_fdr_correction(all_p_values)
        results["fdr_correction"] = pd.DataFrame({
            "test": p_value_labels,
            "p_value": all_p_values,
            "p_corrected": corrected,
            "significant": reject,
        })
        n_sig = int(reject.sum())
        print(f"\n  FDR correction: {n_sig}/{len(reject)} tests significant after BH correction")

    # Logistic regression with all axes
    valid_axes = [a for a in axes if a in df.columns and not df[a].isna().all()]
    if valid_axes:
        print("\n  Logistic regression (all axes as predictors):")
        lr_df = logistic_regression_failure(df, valid_axes, outcome_col)
        results["logistic_regression"] = lr_df
        if len(lr_df) > 0:
            for _, row in lr_df.iterrows():
                sig = "*" if row["p_value"] < 0.05 else ""
                print(f"    {row['predictor']:30s}  OR={row['odds_ratio']:.2f}  p={row['p_value']:.4f}{sig}")

    return results
