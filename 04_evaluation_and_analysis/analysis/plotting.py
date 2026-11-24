"""Nature Biomedical Engineering quality plots for PE classifier evaluation.

All functions produce publication-ready figures. Each saves both PDF and PNG.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from analysis.config import (
    set_nature_style,
    COLORS,
    NATURE_SINGLE_COL_IN,
    NATURE_DOUBLE_COL_IN,
)


def _save_fig(fig: plt.Figure, path: str) -> None:
    """Save figure as both PDF and PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    # Also save the other format
    if path.endswith(".pdf"):
        fig.savefig(path.replace(".pdf", ".png"), dpi=300)
    elif path.endswith(".png"):
        fig.savefig(path.replace(".png", ".pdf"))


def plot_score_distribution(
    df: pd.DataFrame,
    save_path: str,
    title: str = "Classifier Score Distribution",
    threshold: float = 0.5,
) -> plt.Figure:
    """Histogram + KDE of classifier scores for counterfactual data.

    Since all counterfactuals are PE-positive, scores should ideally be high.
    Low scores indicate missed PEs.

    Args:
        df: Analysis DataFrame with predicted_score column.
        save_path: Path to save figure (without extension - saves both .pdf and .png).
        title: Figure title.
        threshold: Decision threshold to mark on plot.
    """
    set_nature_style()
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN, 2.5))

    scores = df["predicted_score"].dropna().values

    ax.hist(scores, bins=40, color=COLORS["blue"], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(threshold, color=COLORS["red"], linestyle="--", linewidth=0.8, label=f"Threshold = {threshold}")

    miss_rate = (scores < threshold).mean()
    ax.text(
        0.02, 0.95,
        f"Miss rate: {miss_rate:.1%}\nn = {len(scores)}",
        transform=ax.transAxes, va="top", fontsize=6,
    )

    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


DEFAULT_BOX_COLOR = "#4477AA"  # steel blue fallback


def _shade_palette(base_hex: str, n: int) -> list[str]:
    """Generate n shades from light (35% saturation) to full base color."""
    import matplotlib.colors as mcolors
    base_rgb = mcolors.to_rgb(base_hex)
    white = np.array([1.0, 1.0, 1.0])
    base = np.array(base_rgb)
    fracs = np.linspace(0.35, 1.0, n)
    return [mcolors.to_hex(white * (1 - f) + base * f) for f in fracs]


# Per-axis color palettes (consistent with vessel overlay plots)
AXIS_PALETTES = {
    "location": dict(zip(
        ["central", "lobar", "segmental", "subsegmental"],
        _shade_palette("#2563EB", 4))),
    "occlusion": dict(zip(
        ["complete", "near_complete", "partial"],
        _shade_palette("#7C3AED", 3))),
    "burden": dict(zip(
        ["extensive", "multi_bilateral", "multi_unilateral", "isolated"],
        _shade_palette("#059669", 4))),
    "severity_category": dict(zip(
        ["mild", "moderate", "severe"],
        _shade_palette("#F97316", 3))),
    "laterality": dict(zip(
        ["central", "bilateral", "unilateral_left", "unilateral_right"],
        _shade_palette("#0891B2", 4))),
    "intended_shape_category": dict(zip(
        ["acute", "indeterminate", "chronic"],
        _shade_palette("#D946EF", 3))),
    "intended_occlusion_cat": dict(zip(
        ["complete", "near_complete", "partial"],
        _shade_palette("#7C3AED", 3))),
}


def _stratum_color(group_col: str, stratum: str) -> str:
    """Return the bar/box color for a stratum using axis-specific palettes."""
    palette = AXIS_PALETTES.get(group_col, {})
    return palette.get(stratum, DEFAULT_BOX_COLOR)


def plot_score_distributions_by_stratum(
    df: pd.DataFrame,
    group_col: str,
    save_path: str,
    threshold: float = 0.5,
) -> plt.Figure:
    """Notched boxplots of score distributions stratified by a PE characteristic.

    Args:
        df: Analysis DataFrame.
        group_col: Column to stratify by.
        save_path: Path to save figure.
        threshold: Decision threshold to mark.
    """
    set_nature_style()

    strata = sorted(df[group_col].dropna().unique())
    n_strata = len(strata)

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN, max(2.0, 0.6 * n_strata)))

    data = [df[df[group_col] == s]["predicted_score"].dropna().values for s in strata]
    positions = list(range(n_strata))

    bp = ax.boxplot(
        data, positions=positions, vert=False, notch=True, patch_artist=True,
        widths=0.5,
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.8),
        flierprops=dict(marker="o", markersize=2, alpha=0.4, markerfacecolor="gray",
                        markeredgecolor="none"),
    )
    for i, (patch, s) in enumerate(zip(bp["boxes"], strata)):
        color = _stratum_color(group_col, s)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points if n < 50
    for i, (d, s) in enumerate(zip(data, strata)):
        if len(d) < 50:
            jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(d))
            ax.scatter(d, i + jitter, s=3, color="black", alpha=0.4, zorder=3)

    ax.axvline(threshold, color=COLORS["red"], linestyle="--", linewidth=0.6, alpha=0.7)

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{s}\n(n={len(d)})" for s, d in zip(strata, data)])
    ax.set_xlabel("Predicted score")
    ax.set_title(f"Score distribution by {group_col}")

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_failure_rate_heatmap(
    stratified_results: dict[str, pd.DataFrame],
    save_path: str,
) -> plt.Figure:
    """Heatmap of failure rates across PE characteristics.

    Args:
        stratified_results: Dict from run_stratified_analysis(), mapping characteristic -> DataFrame.
        save_path: Path to save figure.
    """
    set_nature_style()

    # Build matrix: rows = strata, columns = characteristics
    all_rows = []
    for char_name, char_df in stratified_results.items():
        for _, row in char_df.iterrows():
            all_rows.append({
                "characteristic": char_name,
                "stratum": row["stratum"],
                "miss_rate": row["miss_rate"],
                "n": row["n"],
            })

    if not all_rows:
        return plt.figure()

    heatmap_df = pd.DataFrame(all_rows)
    pivot = heatmap_df.pivot(index="stratum", columns="characteristic", values="miss_rate")
    pivot_n = heatmap_df.pivot(index="stratum", columns="characteristic", values="n")

    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, max(2.5, 0.35 * len(pivot))))

    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Miss rate")
    cbar.ax.tick_params(labelsize=6)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            n = pivot_n.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n(n={int(n)})", ha="center", va="center", fontsize=5, color=color)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Miss rate by PE characteristic")

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_forest(
    odds_ratios_df: pd.DataFrame,
    save_path: str,
    title: str = "Odds ratios for classifier failure",
) -> plt.Figure:
    """Forest plot of odds ratios with 95% CIs.

    Args:
        odds_ratios_df: DataFrame with stratum, odds_ratio, ci_lower, ci_upper columns.
        save_path: Path to save figure.
        title: Figure title.
    """
    set_nature_style()

    df = odds_ratios_df[~odds_ratios_df["is_reference"]].copy()
    df = df.sort_values("odds_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN, max(2.0, 0.4 * len(df))))

    y_pos = range(len(df))

    ax.errorbar(
        df["odds_ratio"], y_pos,
        xerr=[df["odds_ratio"] - df["ci_lower"], df["ci_upper"] - df["odds_ratio"]],
        fmt="o", color=COLORS["blue"], markersize=4, capsize=2, linewidth=0.8,
    )
    ax.axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=0.6)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{row['stratum']} (n={row['n']})" for _, row in df.iterrows()])
    ax.set_xlabel("Odds ratio (vs. reference)")
    ax.set_title(title)

    # Log scale if range is large
    or_range = df["ci_upper"].max() / max(df["ci_lower"].min(), 0.01)
    if or_range > 10:
        ax.set_xscale("log")

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_roc_curve(
    df: pd.DataFrame,
    save_path: str,
    title: str = "ROC Curve",
    n_bootstrap: int = 1000,
) -> plt.Figure:
    """ROC curve with 95% CI band. Requires both classes in data.

    Args:
        df: Analysis DataFrame with target and predicted_score.
        save_path: Path to save figure.
        title: Figure title.
        n_bootstrap: Bootstrap iterations for CI band.
    """
    set_nature_style()

    targets = df["target"].values
    scores = df["predicted_score"].dropna().values
    if len(np.unique(targets)) < 2:
        print("  ROC curve requires both classes. Skipping.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN, NATURE_SINGLE_COL_IN))

    fpr, tpr, _ = roc_curve(targets, scores)
    auc_val = roc_auc_score(targets, scores)
    ax.plot(fpr, tpr, color=COLORS["blue"], linewidth=1.0, label=f"AUC = {auc_val:.3f}")

    # Bootstrap CI band
    rng = np.random.RandomState(42)
    tpr_interp = np.linspace(0, 1, 100)
    boot_fprs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(targets), len(targets))
        if len(np.unique(targets[idx])) < 2:
            continue
        b_fpr, b_tpr, _ = roc_curve(targets[idx], scores[idx])
        boot_fprs.append(np.interp(tpr_interp, b_tpr, b_fpr))

    if boot_fprs:
        boot_fprs = np.array(boot_fprs)
        ax.fill_betweenx(
            tpr_interp,
            np.percentile(boot_fprs, 2.5, axis=0),
            np.percentile(boot_fprs, 97.5, axis=0),
            alpha=0.15, color=COLORS["blue"],
        )

    ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_roc_comparison(
    datasets: list[tuple[pd.DataFrame, str, str]],
    save_path: str,
    title: str = "ROC Curve Comparison",
    n_bootstrap: int = 1000,
) -> plt.Figure:
    """Overlay ROC curves for multiple datasets on the same axes.

    Args:
        datasets: List of (DataFrame, label, color_key) tuples.
            Each DataFrame must have target and predicted_score columns.
        save_path: Path to save figure.
        title: Figure title.
        n_bootstrap: Bootstrap iterations for CI bands.
    """
    set_nature_style()

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN, NATURE_SINGLE_COL_IN))

    rng = np.random.RandomState(42)
    mean_fpr = np.linspace(0, 1, 100)

    for df, label, color in datasets:
        targets = df["target"].values
        scores = df["predicted_score"].dropna().values
        if len(np.unique(targets)) < 2:
            continue

        fpr, tpr, _ = roc_curve(targets, scores)

        # Bootstrap AUC distribution and CI band
        boot_tprs = []
        boot_aucs = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(targets), len(targets))
            if len(np.unique(targets[idx])) < 2:
                continue
            b_fpr, b_tpr, _ = roc_curve(targets[idx], scores[idx])
            boot_tprs.append(np.interp(mean_fpr, b_fpr, b_tpr))
            boot_aucs.append(roc_auc_score(targets[idx], scores[idx]))

        auc_mean = float(np.mean(boot_aucs)) if boot_aucs else roc_auc_score(targets, scores)
        auc_std = float(np.std(boot_aucs)) if boot_aucs else 0.0

        ax.plot(fpr, tpr, color=color, linewidth=1.0,
                label=f"{label} (AUC = {auc_mean:.3f} \u00b1 {auc_std:.3f})")

        if boot_tprs:
            boot_tprs = np.array(boot_tprs)
            ax.fill_between(
                mean_fpr,
                np.percentile(boot_tprs, 2.5, axis=0),
                np.percentile(boot_tprs, 97.5, axis=0),
                alpha=0.12, color=color,
            )

    ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_balanced_accuracy_comparison(
    datasets: list[tuple[pd.DataFrame, str, str]],
    save_path: str,
    title: str = "Balanced Accuracy Comparison",
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
) -> plt.Figure:
    """Bar plot comparing balanced accuracy across datasets with bootstrap CIs and p-value.

    Args:
        datasets: List of (DataFrame, label, color_key) tuples.
        save_path: Path to save figure.
        title: Figure title.
        threshold: Classification threshold.
        n_bootstrap: Bootstrap iterations.
    """
    from sklearn.metrics import balanced_accuracy_score
    from scipy.stats import mannwhitneyu

    set_nature_style()

    def _boot_bacc(targets, scores, rng, n=2000):
        baccs = []
        preds = (scores >= threshold).astype(int)
        for _ in range(n):
            idx = rng.randint(0, len(targets), len(targets))
            if len(np.unique(targets[idx])) < 2:
                continue
            baccs.append(balanced_accuracy_score(targets[idx], preds[idx]))
        return np.array(baccs)

    rng = np.random.RandomState(42)
    labels, means, stds, colors_list = [], [], [], []
    boot_distributions = []

    for df, label, color in datasets:
        targets = df["target"].values
        scores = df["predicted_score"].dropna().values
        preds = (scores >= threshold).astype(int)
        bacc = balanced_accuracy_score(targets, preds)
        boot = _boot_bacc(targets, scores, rng, n=n_bootstrap)

        labels.append(label)
        means.append(bacc)
        stds.append(boot.std())
        colors_list.append(color)
        boot_distributions.append(boot)

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN, 2.8))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors_list,
                  edgecolor="white", linewidth=0.5, width=0.5,
                  error_kw={"linewidth": 0.8, "capthick": 0.6}, alpha=0.85)

    # Annotate values on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={len(datasets[i][0])})" for i, l in enumerate(labels)])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(title)
    ax.set_ylim(max(0, min(means) - max(stds) - 0.08), min(1.02, max(means) + max(stds) + 0.08))

    # p-value between the two distributions (if exactly 2 datasets)
    if len(boot_distributions) == 2:
        stat, p = mannwhitneyu(boot_distributions[0], boot_distributions[1], alternative="two-sided")
        # Draw bracket
        y_max = max(m + s for m, s in zip(means, stds)) + 0.03
        ax.plot([0, 0, 1, 1], [y_max, y_max + 0.01, y_max + 0.01, y_max],
                color="black", linewidth=0.6)
        p_str = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.5, y_max + 0.012, p_str, ha="center", fontsize=6)

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_precision_recall_curve(
    df: pd.DataFrame,
    save_path: str,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """Precision-recall curve. Requires both classes.

    Args:
        df: Analysis DataFrame.
        save_path: Path to save figure.
        title: Figure title.
    """
    set_nature_style()

    targets = df["target"].values
    scores = df["predicted_score"].dropna().values
    if len(np.unique(targets)) < 2:
        print("  PR curve requires both classes. Skipping.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_COL_IN, NATURE_SINGLE_COL_IN))

    precision, recall, _ = precision_recall_curve(targets, scores)
    ap = average_precision_score(targets, scores)

    ax.plot(recall, precision, color=COLORS["blue"], linewidth=1.0, label=f"AP = {ap:.3f}")

    prevalence = targets.mean()
    ax.axhline(prevalence, color=COLORS["gray"], linestyle="--", linewidth=0.5, label=f"Prevalence = {prevalence:.2f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_patient_sensitivity(
    patient_df: pd.DataFrame,
    save_path: str,
    title: str = "Per-patient sensitivity",
) -> plt.Figure:
    """Bar chart of per-patient sensitivity, sorted, with overall mean line.

    Args:
        patient_df: Output of compute_patient_level_metrics().
        save_path: Path to save figure.
        title: Figure title.
    """
    set_nature_style()

    df = patient_df.sort_values("sensitivity", ascending=False).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 2.5))

    colors = [COLORS["red"] if s < 0.5 else COLORS["blue"] for s in df["sensitivity"]]
    ax.bar(range(n), df["sensitivity"], color=colors, width=0.8, edgecolor="white", linewidth=0.2)

    mean_sens = df["sensitivity"].mean()
    ax.axhline(mean_sens, color="black", linestyle="--", linewidth=0.6,
               label=f"Mean = {mean_sens:.2f}")

    ax.set_xlabel("Patient (sorted)")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, n - 0.5)
    ax.legend(frameon=False, fontsize=6)

    # Remove individual x-tick labels (too many patients)
    ax.set_xticks([0, n // 4, n // 2, 3 * n // 4, n - 1])
    ax.set_xticklabels([1, n // 4 + 1, n // 2 + 1, 3 * n // 4 + 1, n])

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_patient_score_heatmap(
    df: pd.DataFrame,
    patient_col: str = "patient_id_metadata",
    save_path: str = "",
    title: str = "Scores per patient x counterfactual",
) -> plt.Figure:
    """Heatmap: patients (rows) vs. counterfactuals (cols), colored by score.

    Patients sorted by mean score. Reveals patient-level clustering.

    Args:
        df: Analysis DataFrame.
        patient_col: Patient ID column.
        save_path: Path to save figure.
        title: Figure title.
    """
    set_nature_style()

    # Pivot: patient x counterfactual index
    patient_means = df.groupby(patient_col)["predicted_score"].mean().sort_values()
    patient_order = patient_means.index.tolist()

    # Build matrix
    max_cf = df.groupby(patient_col).size().max()
    matrix = np.full((len(patient_order), max_cf), np.nan)
    for i, pid in enumerate(patient_order):
        scores = df[df[patient_col] == pid]["predicted_score"].values
        matrix[i, :len(scores)] = sorted(scores)

    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, max(3.0, 0.12 * len(patient_order))))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label="Predicted score")
    cbar.ax.tick_params(labelsize=6)

    # Mark threshold
    ax.set_xlabel("Counterfactual index (sorted by score)")
    ax.set_ylabel("Patient (sorted by mean score)")
    ax.set_title(title)

    # Simplify y-ticks
    if len(patient_order) > 20:
        step = max(1, len(patient_order) // 10)
        ax.set_yticks(range(0, len(patient_order), step))
        ax.set_yticklabels([patient_order[i] for i in range(0, len(patient_order), step)], fontsize=5)
    else:
        ax.set_yticks(range(len(patient_order)))
        ax.set_yticklabels(patient_order, fontsize=5)

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_cumulative_sensitivity_curves(
    curves: dict[str, pd.DataFrame],
    save_path: str,
    threshold_results: list[dict] | None = None,
    title: str = "Sensitivity vs. PE characteristic threshold",
) -> plt.Figure:
    """Plot cumulative sensitivity as a function of continuous PE variables.

    Shows: "above this PE volume/Qanadli score, sensitivity is X%."

    Args:
        curves: Dict mapping variable name -> cumulative sensitivity DataFrame.
        save_path: Path to save figure.
        threshold_results: Optional list of threshold dicts to annotate.
        title: Figure title.
    """
    set_nature_style()

    n_vars = len(curves)
    if n_vars == 0:
        return plt.figure()

    fig, axes = plt.subplots(1, n_vars, figsize=(NATURE_DOUBLE_COL_IN, 2.5))
    if n_vars == 1:
        axes = [axes]

    for ax, (var_name, curve_df) in zip(axes, curves.items()):
        if len(curve_df) == 0:
            continue

        ax.plot(
            curve_df["cutoff"], curve_df["sensitivity"],
            color=COLORS["blue"], linewidth=1.0,
        )
        fill_base = max(0, curve_df["sensitivity"].min() - 0.05) if len(curve_df) else 0
        ax.fill_between(
            curve_df["cutoff"], curve_df["sensitivity"], fill_base,
            alpha=0.1, color=COLORS["blue"],
        )

        # Mark target sensitivity levels
        for target, color in [(0.80, COLORS["yellow"]), (0.90, COLORS["green"]), (0.95, COLORS["red"])]:
            ax.axhline(target, color=color, linestyle=":", linewidth=0.5, alpha=0.7)
            ax.text(ax.get_xlim()[1], target, f" {target:.0%}", va="center", fontsize=5, color=color)

        # Annotate found thresholds
        if threshold_results:
            for tr in threshold_results:
                if tr["continuous_var"] == var_name and tr["reached"]:
                    ax.axvline(
                        tr["threshold_value"], color=COLORS["red"],
                        linestyle="--", linewidth=0.6, alpha=0.7,
                    )

        ax.set_xlabel(var_name.replace("_", " "))
        ax.set_ylabel("Sensitivity")
        # Dynamic y-axis: 5% below the minimum data point, capped at 0
        y_min = max(0, curve_df["sensitivity"].min() - 0.05) if len(curve_df) else 0
        ax.set_ylim(y_min, 1.02)
        ax.set_title(var_name.replace("_", " "))

    fig.suptitle(title, fontsize=8, y=1.02)
    fig.tight_layout()

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


_CATEGORY_ORDER = {
    "burden": ["isolated", "multi_unilateral", "multi_bilateral", "extensive"],
    "location": ["subsegmental", "segmental", "lobar"],
    "severity_category": ["mild", "moderate", "severe"],
    "occlusion": ["partial", "near_complete", "complete"],
    "laterality": ["unilateral_left", "unilateral_right", "bilateral"],
}


def plot_sensitivity_by_category(
    stratified_results: dict[str, pd.DataFrame],
    save_path: str,
    title: str = "Sensitivity by PE characteristic",
) -> plt.Figure:
    """Sensitivity dot-and-error-bar plots for categorical PE variables.

    One subplot per categorical axis, with categories on the x-axis and
    sensitivity (with 95% CI) on the y-axis. Uses the same clean style
    as the cumulative sensitivity curves.

    Args:
        stratified_results: Dict from run_stratified_analysis(),
            mapping characteristic name -> per-stratum metrics DataFrame.
        save_path: Path to save figure.
        title: Figure suptitle.
    """
    set_nature_style()

    n_vars = len(stratified_results)
    if n_vars == 0:
        return plt.figure()

    ncols = min(3, n_vars)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(NATURE_DOUBLE_COL_IN, 2.8 * nrows),
    )
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()

    for idx, (var_name, strat_df) in enumerate(stratified_results.items()):
        ax = flat_axes[idx]
        if len(strat_df) == 0:
            continue

        # Sort categories in a clinically meaningful order if available
        order = _CATEGORY_ORDER.get(var_name)
        if order:
            strat_df = strat_df.copy()
            strat_df["_sort"] = strat_df["stratum"].apply(
                lambda s: order.index(s) if s in order else len(order)
            )
            strat_df = strat_df.sort_values("_sort")
        else:
            strat_df = strat_df.sort_values("stratum")

        x = np.arange(len(strat_df))
        yerr_lo = strat_df["sensitivity"].values - strat_df["sensitivity_ci_lower"].values
        yerr_hi = strat_df["sensitivity_ci_upper"].values - strat_df["sensitivity"].values

        bar_colors = [_stratum_color(var_name, s) for s in strat_df["stratum"]]
        for xi, yi, yl, yh, c in zip(x, strat_df["sensitivity"], yerr_lo, yerr_hi, bar_colors):
            ax.errorbar(
                xi, yi, yerr=[[yl], [yh]],
                fmt="o", color=c, markersize=5,
                capsize=3, linewidth=0.8, capthick=0.6,
            )

        # Reference lines
        for target, color in [(0.80, COLORS["yellow"]), (0.90, COLORS["green"]), (0.95, COLORS["red"])]:
            ax.axhline(target, color=color, linestyle=":", linewidth=0.5, alpha=0.7)

        # Labels with sample size
        labels = [
            f"{_prettify_label(s)}\n(n={n})"
            for s, n in zip(strat_df["stratum"], strat_df["n"])
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.5, rotation=35, ha="right")
        ax.set_ylabel("Sensitivity")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(_prettify_label(var_name))

    # Hide unused subplots
    for idx in range(n_vars, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=8, y=1.02)
    fig.tight_layout()

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_baseline_comparison(
    comparison_df: pd.DataFrame,
    save_path: str,
    title: str = "Real (no PE) vs. counterfactual (with PE) scores",
) -> plt.Figure:
    """Two-panel figure: scatter of real vs CF scores + paired delta by patient group.

    Args:
        comparison_df: Output of build_baseline_comparison().
        save_path: Path to save figure.
        title: Figure title.
    """
    set_nature_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 3.0),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )

    df = comparison_df
    group_colors = {
        "zero_detection": COLORS["red"],
        "partial": COLORS["yellow"],
        "perfect_detection": COLORS["blue"],
    }
    group_counts = {g: int((df["patient_group"] == g).sum()) for g in group_colors}
    group_labels = {
        "zero_detection": f"Zero detection (n={group_counts['zero_detection']})",
        "partial": f"Partial (n={group_counts['partial']})",
        "perfect_detection": f"Perfect detection (n={group_counts['perfect_detection']})",
    }

    # --- Panel (a): Scatter plot real vs CF score with CI bars ---
    has_std = "cf_std_score" in df.columns and df["cf_std_score"].notna().any()
    for group, color in group_colors.items():
        mask = df["patient_group"] == group
        gdf = df.loc[mask]
        if has_std:
            ax1.errorbar(
                gdf["real_score"], gdf["cf_mean_score"],
                yerr=gdf["cf_std_score"].fillna(0),
                fmt="o", color=color, markersize=4, alpha=0.8,
                elinewidth=0.5, capsize=0, ecolor=color,
                markeredgecolor="white", markeredgewidth=0.3,
                label=group_labels[group], zorder=3,
            )
        else:
            ax1.scatter(
                gdf["real_score"], gdf["cf_mean_score"],
                c=color, s=20, alpha=0.8, label=group_labels[group],
                edgecolors="white", linewidths=0.3, zorder=3,
            )

    # Identity line
    lims = [0.2, 0.9]
    ax1.plot(lims, lims, color=COLORS["gray"], linestyle="--", linewidth=0.5, zorder=1)

    # Threshold lines
    ax1.axhline(0.5, color=COLORS["gray"], linestyle=":", linewidth=0.4, alpha=0.5)
    ax1.axvline(0.5, color=COLORS["gray"], linestyle=":", linewidth=0.4, alpha=0.5)

    from scipy.stats import pearsonr
    r, p = pearsonr(df["real_score"], df["cf_mean_score"])
    ax1.text(
        0.05, 0.95, f"r = {r:.2f}\np = {p:.1e}",
        transform=ax1.transAxes, va="top", fontsize=6,
    )

    ax1.set_xlabel("Real score (no PE)")
    ax1.set_ylabel("Counterfactual score (with PE)")
    ax1.set_title("(a) Score correlation")
    ax1.legend(frameon=False, fontsize=5, loc="lower right")
    ax1.set_xlim(0.2, 0.9)
    ax1.set_ylim(0.2, 0.9)

    # --- Panel (b): Paired lines per patient, grouped ---
    # Sort patients by real score within each group
    group_order = ["zero_detection", "partial", "perfect_detection"]
    sorted_df = pd.concat([
        df[df["patient_group"] == g].sort_values("real_score")
        for g in group_order
    ]).reset_index(drop=True)

    x = np.arange(len(sorted_df))

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        color = group_colors[row["patient_group"]]
        ax2.plot(
            [i, i], [row["real_score"], row["cf_mean_score"]],
            color=COLORS["gray"], linewidth=0.3, alpha=0.2,
        )
        ax2.scatter(i, row["real_score"], color=COLORS["gray"], s=8, zorder=3, marker="o")
        # CF mean with std error bar
        cf_std = row.get("cf_std_score", 0) if has_std else 0
        cf_std = cf_std if not np.isnan(cf_std) else 0
        ax2.errorbar(
            i, row["cf_mean_score"], yerr=cf_std,
            fmt="s", color=color, markersize=3,
            elinewidth=0.5, capsize=1.5, capthick=0.4,
            ecolor=color, zorder=4,
        )

    ax2.axhline(0.5, color=COLORS["gray"], linestyle=":", linewidth=0.4, alpha=0.5)

    # Group boundaries - place labels below x-axis
    group_short = {
        "zero_detection": "Zero detection",
        "partial": "Partial",
        "perfect_detection": "Perfect detection",
    }
    cumulative = 0
    for g in group_order:
        n = (sorted_df["patient_group"] == g).sum()
        if cumulative > 0:
            ax2.axvline(cumulative - 0.5, color="black", linestyle="-", linewidth=0.3, alpha=0.3)
        mid = cumulative + n / 2
        ax2.text(
            mid, -0.08, f"{group_short[g]}\n(n={n})",
            ha="center", fontsize=5, style="italic",
            transform=ax2.get_xaxis_transform(),
        )
        cumulative += n

    # Legend entries
    ax2.scatter([], [], color=COLORS["gray"], s=8, marker="o", label="Real (no PE)")
    ax2.scatter([], [], color=COLORS["blue"], s=8, marker="s", label="CF (with PE)")
    ax2.legend(frameon=False, fontsize=5, loc="upper left")

    ax2.set_xlabel("Patient (sorted by group + real score)", labelpad=24)
    ax2.set_ylabel("Predicted score")
    ax2.set_title("(b) Per-patient score shift")
    ax2.set_ylim(0.2, 0.9)
    ax2.set_xticks([])

    fig.suptitle(title, fontsize=8, y=1.02)
    fig.tight_layout()

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_three_way_comparison(
    comparison_df: pd.DataFrame,
    synthetic_predictions_csv: str,
    save_path: str,
    title: str = "Real vs. synthetic vs. counterfactual scores",
) -> plt.Figure:
    """Three-way per-patient comparison: Real no-PE, Synthetic no-PE, CF with-PE.

    Args:
        comparison_df: Output of build_baseline_comparison() (has real_score, cf_mean_score).
        synthetic_predictions_csv: Path to synthetic no-PE predictions CSV.
        save_path: Path to save figure.
        title: Figure title.
    """
    set_nature_style()

    df = comparison_df.copy()

    # Load synthetic scores and merge
    synth = pd.read_csv(synthetic_predictions_csv)
    synth = synth.rename(columns={"predicted_score": "synth_score"})
    # Handle potential duplicates by taking the mean
    synth_dedup = synth.groupby("image_id")["synth_score"].mean().reset_index()
    df = df.merge(synth_dedup, left_on="image_id", right_on="image_id", how="left")

    # Drop patients without synthetic scores
    n_before = len(df)
    df = df.dropna(subset=["synth_score"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Three-way plot: {n_before - len(df)} patients without synthetic scores dropped")

    # Sort by real score
    df = df.sort_values("real_score").reset_index(drop=True)
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 3.5),
        gridspec_kw={"width_ratios": [1.8, 1]},
    )

    # --- Panel (a): Per-patient lines with 3 conditions ---
    for i, (_, row) in enumerate(df.iterrows()):
        # Vertical span line
        vals = [row["real_score"], row["synth_score"], row["cf_mean_score"]]
        ax1.plot(
            [i, i], [min(vals), max(vals)],
            color=COLORS["gray"], linewidth=0.3, alpha=0.3,
        )
        ax1.scatter(i, row["real_score"], color=COLORS["blue"], s=10, zorder=3, marker="o")
        ax1.scatter(i, row["synth_score"], color=COLORS["green"], s=10, zorder=3, marker="D")
        cf_std = row.get("cf_std_score", 0)
        cf_std = cf_std if not pd.isna(cf_std) else 0
        ax1.errorbar(
            i, row["cf_mean_score"], yerr=cf_std,
            fmt="s", color=COLORS["red"], markersize=3,
            elinewidth=0.5, capsize=1, capthick=0.3,
            ecolor=COLORS["red"], zorder=4,
        )

    ax1.axhline(0.5, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    # Legend
    ax1.scatter([], [], color=COLORS["blue"], s=12, marker="o", label="Real (no PE)")
    ax1.scatter([], [], color=COLORS["green"], s=12, marker="D", label="Synthetic (no PE)")
    ax1.scatter([], [], color=COLORS["red"], s=12, marker="s", label="Counterfactual (with PE)")
    ax1.legend(frameon=False, fontsize=5.5, loc="upper left")

    ax1.set_xlabel("Patient (sorted by real score)")
    ax1.set_ylabel("Predicted score")
    ax1.set_title("(a) Per-patient scores across conditions")
    y_min = max(0, df[["real_score", "synth_score", "cf_mean_score"]].min().min() - 0.05)
    y_max = min(1.0, df[["real_score", "synth_score", "cf_mean_score"]].max().max() + 0.05)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xticks([])

    # --- Panel (b): Grouped box/violin plot ---
    data_for_box = [
        df["real_score"].values,
        df["synth_score"].values,
        df["cf_mean_score"].values,
    ]
    box_labels = ["Real\n(no PE)", "Synthetic\n(no PE)", "Counterfactual\n(with PE)"]
    box_colors = [COLORS["blue"], COLORS["green"], COLORS["red"]]

    parts = ax2.violinplot(data_for_box, positions=[0, 1, 2], showmeans=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(box_colors[i])
        pc.set_alpha(0.3)
        pc.set_edgecolor(box_colors[i])

    bp = ax2.boxplot(
        data_for_box, positions=[0, 1, 2], widths=0.3,
        patch_artist=True, showfliers=False,
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.8),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(box_colors[i])
        patch.set_alpha(0.6)

    # Annotate means
    for i, d in enumerate(data_for_box):
        mean_val = np.mean(d)
        ax2.text(i, mean_val + 0.01, f"{mean_val:.3f}", ha="center", fontsize=6, fontweight="bold")

    ax2.axhline(0.5, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(box_labels, fontsize=6)
    ax2.set_ylabel("Predicted score")
    ax2.set_title("(b) Score distribution by condition")
    ax2.set_ylim(y_min, y_max)

    fig.suptitle(title, fontsize=8, y=1.02)
    fig.tight_layout()

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def _prettify_label(s: str) -> str:
    """Convert snake_case stratum names to readable labels."""
    return s.replace("_", " ").title()


def plot_sensitivity_bar_chart(
    stratified_results: dict[str, pd.DataFrame],
    save_path: str,
    title: str = "Sensitivity by PE characteristic",
) -> plt.Figure:
    """Compact grid of horizontal bar charts showing sensitivity per stratum.

    Uses a 2x3 (or appropriate) grid layout so the figure fits on a single slide.

    Args:
        stratified_results: Dict from run_stratified_analysis().
        save_path: Path to save figure.
        title: Figure title.
    """
    set_nature_style()

    n_axes = len(stratified_results)
    if n_axes == 0:
        return plt.figure()

    ncols = min(3, n_axes)
    nrows = (n_axes + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(NATURE_DOUBLE_COL_IN, 2.2 * nrows),
    )
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()

    for idx, (char_name, char_df) in enumerate(stratified_results.items()):
        ax = flat_axes[idx]
        char_df = char_df.sort_values("sensitivity", ascending=True)
        y = range(len(char_df))

        xerr_lo = char_df["sensitivity"].values - char_df["sensitivity_ci_lower"].values
        xerr_hi = char_df["sensitivity_ci_upper"].values - char_df["sensitivity"].values

        bar_colors = [
            _stratum_color(char_name, s) for s in char_df["stratum"]
        ]

        ax.barh(
            y, char_df["sensitivity"],
            xerr=[xerr_lo, xerr_hi],
            capsize=2, color=bar_colors,
            edgecolor="white", linewidth=0.3, error_kw={"linewidth": 0.6},
            height=0.6,
        )

        labels = [
            f"{_prettify_label(s)}  (n={n})"
            for s, n in zip(char_df["stratum"], char_df["n"])
        ]
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(_prettify_label(char_name), fontsize=7, fontstyle="italic")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Sensitivity", fontsize=6)

    # Hide unused subplots
    for idx in range(n_axes, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=8)
    fig.tight_layout()

    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Demographic analysis plots
# ---------------------------------------------------------------------------

SEX_COLORS = {"Female": "#AA3377", "Male": "#4477AA"}  # purple / blue


def _build_patient_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-patient sensitivity with demographics from analysis df."""
    pat = df.groupby("patient_id_metadata").agg(
        gender=("gender_metadata", "first"),
        age=("age_metadata", "first"),
        n_cf=("image_id", "count"),
        sensitivity=("classifier_correct", "mean"),
        mean_score=("predicted_score", "mean"),
    ).reset_index()
    return pat


def plot_sex_analysis(
    df: pd.DataFrame,
    save_path: str,
) -> plt.Figure:
    """Two-panel figure: (a) score distributions by sex, (b) per-patient sensitivity by sex.

    Args:
        df: Full analysis DataFrame (volume-level).
        save_path: Path to save figure.
    """
    set_nature_style()
    pat = _build_patient_demographics(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(NATURE_DOUBLE_COL_IN, 3.0))

    # --- Panel (a): Score distributions by sex (notched boxplots) ---
    sexes = ["Female", "Male"]
    score_data = [df[df["gender_metadata"] == s]["predicted_score"].dropna().values for s in sexes]

    bp = ax1.boxplot(
        score_data, vert=True, notch=True, patch_artist=True, widths=0.5,
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.8),
        flierprops=dict(marker="o", markersize=2, alpha=0.4,
                        markerfacecolor="gray", markeredgecolor="none"),
    )
    for patch, sex in zip(bp["boxes"], sexes):
        patch.set_facecolor(SEX_COLORS[sex])
        patch.set_alpha(0.7)

    ax1.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.set_ylabel("Predicted score")
    ax1.set_title("(a) Score distribution by sex")

    # Annotate sensitivity inside the x-tick labels to avoid overlap
    sens_labels = []
    for s, d in zip(sexes, score_data):
        sens = (df[df["gender_metadata"] == s]["classifier_correct"]).mean()
        sens_labels.append(f"{s}\n(n={len(d)})\nSens: {sens:.1%}")
    ax1.set_xticklabels(sens_labels)

    # --- Panel (b): Per-patient sensitivity by sex (strip + boxplot) ---
    pat_data = [pat[pat["gender"] == s]["sensitivity"].values for s in sexes]

    bp2 = ax2.boxplot(
        pat_data, vert=True, notch=True, patch_artist=True, widths=0.45,
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.8),
        flierprops=dict(markersize=0),  # hide default fliers, we overlay strip
    )
    for patch, sex in zip(bp2["boxes"], sexes):
        patch.set_facecolor(SEX_COLORS[sex])
        patch.set_alpha(0.35)

    # Overlay individual patient points
    rng = np.random.RandomState(42)
    for i, (d, sex) in enumerate(zip(pat_data, sexes)):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax2.scatter(
            i + 1 + jitter, d, s=12, color=SEX_COLORS[sex],
            alpha=0.7, edgecolors="white", linewidths=0.3, zorder=3,
        )

    ax2.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_xticklabels([f"{s}\n(n={len(d)})" for s, d in zip(sexes, pat_data)])
    ax2.set_ylabel("Per-patient sensitivity")
    ax2.set_title("(b) Patient-level sensitivity by sex")
    ax2.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_age_analysis(
    df: pd.DataFrame,
    save_path: str,
    age_bins: list[int] | None = None,
    age_labels: list[str] | None = None,
) -> plt.Figure:
    """Two-panel figure: (a) age vs sensitivity scatter, (b) sensitivity by age bin.

    Args:
        df: Full analysis DataFrame (volume-level).
        save_path: Path to save figure.
        age_bins: Bin edges for age groups.
        age_labels: Labels for age bins.
    """
    set_nature_style()
    pat = _build_patient_demographics(df)

    if age_bins is None:
        age_bins = [0, 50, 60, 70, 80, 100]
    if age_labels is None:
        age_labels = ["<50", "50-60", "60-70", "70-80", ">80"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(NATURE_DOUBLE_COL_IN, 3.0))

    # --- Panel (a): Scatter of age vs sensitivity with trend line ---
    for sex in ["Female", "Male"]:
        mask = pat["gender"] == sex
        ax1.scatter(
            pat.loc[mask, "age"], pat.loc[mask, "sensitivity"],
            s=18, color=SEX_COLORS[sex], alpha=0.7,
            edgecolors="white", linewidths=0.3, label=sex, zorder=3,
        )

    # Linear trend line (all patients)
    from numpy.polynomial.polynomial import polyfit
    valid = pat.dropna(subset=["age", "sensitivity"])
    coeffs = polyfit(valid["age"].values, valid["sensitivity"].values, 1)
    x_fit = np.linspace(pat["age"].min(), pat["age"].max(), 100)
    y_fit = coeffs[0] + coeffs[1] * x_fit
    ax1.plot(x_fit, y_fit, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # Pearson correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(valid["age"], valid["sensitivity"])
    ax1.text(
        0.05, 0.05, f"r = {r:.2f}, p = {p:.3f}",
        transform=ax1.transAxes, fontsize=6,
    )

    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("Per-patient sensitivity")
    ax1.set_title("(a) Sensitivity vs. age")
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(frameon=False, fontsize=6, loc="upper left")

    # --- Panel (b): Boxplot by age bin ---
    pat["age_bin"] = pd.cut(pat["age"], bins=age_bins, labels=age_labels)
    bin_order = [lbl for lbl in age_labels if lbl in pat["age_bin"].values]
    bin_data = [pat[pat["age_bin"] == lbl]["sensitivity"].values for lbl in bin_order]
    bin_counts = [len(d) for d in bin_data]

    bp = ax2.boxplot(
        bin_data, vert=True, notch=True, patch_artist=True, widths=0.5,
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.8),
        flierprops=dict(markersize=0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["cyan"])
        patch.set_alpha(0.55)

    # Overlay individual points
    rng = np.random.RandomState(42)
    for i, d in enumerate(bin_data):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax2.scatter(
            i + 1 + jitter, d, s=10, color=COLORS["blue"],
            alpha=0.6, edgecolors="white", linewidths=0.3, zorder=3,
        )

    ax2.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(bin_order, bin_counts)])
    ax2.set_xlabel("Age group")
    ax2.set_ylabel("Per-patient sensitivity")
    ax2.set_title("(b) Sensitivity by age group")
    ax2.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_score_shift_by_sex(
    comparison_df: pd.DataFrame,
    pat_demographics: pd.DataFrame,
    save_path: str,
) -> plt.Figure:
    """Per-patient score shift (real vs CF) colored by sex.

    Args:
        comparison_df: Output of build_baseline_comparison().
        pat_demographics: DataFrame with patient_id_metadata and gender columns.
        save_path: Path to save figure.
    """
    set_nature_style()

    # Merge sex info into comparison
    demo = pat_demographics[["patient_id_metadata", "gender"]].drop_duplicates()
    merged = comparison_df.merge(demo, left_on="image_id", right_on="patient_id_metadata", how="left")

    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COL_IN, 3.0))

    # Sort by sex then real score
    sex_order = ["Female", "Male"]
    sorted_df = pd.concat([
        merged[merged["gender"] == s].sort_values("real_score")
        for s in sex_order
    ]).reset_index(drop=True)

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        color = SEX_COLORS.get(row["gender"], COLORS["gray"])
        ax.plot(
            [i, i], [row["real_score"], row["cf_mean_score"]],
            color=color, linewidth=0.6, alpha=0.7,
        )
        ax.scatter(i, row["real_score"], color=COLORS["gray"], s=8, zorder=3, marker="o")
        ax.scatter(i, row["cf_mean_score"], color=color, s=8, zorder=3, marker="s")

    ax.axhline(0.5, color=COLORS["gray"], linestyle=":", linewidth=0.4, alpha=0.5)

    # Group boundary
    n_female = (sorted_df["gender"] == "Female").sum()
    if 0 < n_female < len(sorted_df):
        ax.axvline(n_female - 0.5, color="black", linestyle="-", linewidth=0.3, alpha=0.3)
        ax.text(n_female / 2, 0.22, "Female", ha="center", fontsize=6, style="italic")
        ax.text(n_female + (len(sorted_df) - n_female) / 2, 0.22, "Male",
                ha="center", fontsize=6, style="italic")

    # Legend
    ax.scatter([], [], color=COLORS["gray"], s=8, marker="o", label="Real (no PE)")
    ax.scatter([], [], color=SEX_COLORS["Female"], s=8, marker="s", label="CF Female")
    ax.scatter([], [], color=SEX_COLORS["Male"], s=8, marker="s", label="CF Male")
    ax.legend(frameon=False, fontsize=5, loc="upper left")

    ax.set_xlabel("Patient (sorted by sex + real score)")
    ax.set_ylabel("Predicted score")
    ax.set_title("Per-patient score shift by sex")
    ax.set_ylim(0.2, 0.9)
    ax.set_xticks([])

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)
    return fig


def plot_age_score_analysis(
    df: pd.DataFrame,
    save_path: str,
    age_bins: list[int] | None = None,
    age_labels: list[str] | None = None,
) -> plt.Figure:
    """Two-panel figure: (a) predicted score vs age scatter, (b) score by age bin boxplots.

    Args:
        df: Full analysis DataFrame (volume-level).
        save_path: Path to save figure.
        age_bins: Bin edges for age groups.
        age_labels: Labels for age bins.
    """
    set_nature_style()
    pat = _build_patient_demographics(df)

    if age_bins is None:
        age_bins = [0, 50, 60, 70, 80, 100]
    if age_labels is None:
        age_labels = ["<50", "50-60", "60-70", "70-80", ">80"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(NATURE_DOUBLE_COL_IN, 3.0))

    # --- Panel (a): Scatter of age vs mean predicted score ---
    for sex in ["Female", "Male"]:
        mask = pat["gender"] == sex
        ax1.scatter(
            pat.loc[mask, "age"], pat.loc[mask, "mean_score"],
            s=18, color=SEX_COLORS[sex], alpha=0.7,
            edgecolors="white", linewidths=0.3, label=sex, zorder=3,
        )

    # Linear trend line
    from numpy.polynomial.polynomial import polyfit
    valid = pat.dropna(subset=["age", "mean_score"])
    coeffs = polyfit(valid["age"].values, valid["mean_score"].values, 1)
    x_fit = np.linspace(pat["age"].min(), pat["age"].max(), 100)
    y_fit = coeffs[0] + coeffs[1] * x_fit
    ax1.plot(x_fit, y_fit, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # Pearson correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(valid["age"], valid["mean_score"])
    ax1.text(
        0.05, 0.05, f"r = {r:.2f}, p = {p:.3f}",
        transform=ax1.transAxes, fontsize=6,
    )

    ax1.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("Mean predicted score")
    ax1.set_title("(a) Predicted score vs. age")
    ax1.set_ylim(0.2, 0.95)
    ax1.legend(frameon=False, fontsize=6, loc="upper left")

    # --- Panel (b): Boxplot of predicted score by age bin ---
    pat["age_bin"] = pd.cut(pat["age"], bins=age_bins, labels=age_labels)
    bin_order = [lbl for lbl in age_labels if lbl in pat["age_bin"].values]
    bin_data = [pat[pat["age_bin"] == lbl]["mean_score"].values for lbl in bin_order]
    bin_counts = [len(d) for d in bin_data]

    bp = ax2.boxplot(
        bin_data, vert=True, notch=True, patch_artist=True, widths=0.5,
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.8),
        flierprops=dict(markersize=0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["cyan"])
        patch.set_alpha(0.55)

    # Overlay individual points
    rng = np.random.RandomState(42)
    for i, d in enumerate(bin_data):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax2.scatter(
            i + 1 + jitter, d, s=10, color=COLORS["blue"],
            alpha=0.6, edgecolors="white", linewidths=0.3, zorder=3,
        )

    ax2.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(bin_order, bin_counts)])
    ax2.set_xlabel("Age group")
    ax2.set_ylabel("Mean predicted score")
    ax2.set_title("(b) Predicted score by age group")
    ax2.set_ylim(0.2, 0.95)

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)
    return fig
