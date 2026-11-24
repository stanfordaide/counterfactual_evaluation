"""Configuration for counterfactual PE detector evaluation.

Stratification axes (present in ``cohort/master_metadata.csv``) and the
Nature Biomedical Engineering figure style. Input data paths live with the
scripts that read them, rooted at the ``CF_DATA`` resources folder.
"""

import matplotlib as mpl


# --- PE characteristic definitions ---

# Stratification axes present in master_metadata.csv
PE_STRATIFICATION_AXES = [
    "location",           # segmental, subsegmental, lobar
    "occlusion",          # partial, near_complete, complete
    "burden",             # isolated, multi_unilateral, multi_bilateral, extensive
    "laterality",         # unilateral_left, unilateral_right, bilateral
    "severity_category",  # mild, moderate, severe
]

PE_CONTINUOUS_AXES = [
    "n_segments_affected",
    "qanadli_score",
    "qanadli_percent",
    "pe_volume_ml",
]


# --- Visualization constants (Nature BME) ---

NATURE_SINGLE_COL_IN = 3.46   # 88mm
NATURE_DOUBLE_COL_IN = 7.09   # 180mm

COLORS = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "purple": "#AA3377",
    "cyan": "#66CCEE",
    "gray": "#BBBBBB",
}

COLOR_LIST = list(COLORS.values())


def set_nature_style() -> None:
    """Apply Nature Biomedical Engineering figure style globally."""
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
