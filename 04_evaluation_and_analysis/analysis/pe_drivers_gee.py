"""PE-driver model (Fig. 5a): the non-plotting part of the original plot_detection_limits.py.

`load` and `run_cluster_robust_model` are copied verbatim from the original module (only the
default paths are now resolved from CF_DATA and the output folder). The original module also
drew the detection-limit figures and fitted a variational-Bayes reference model; both are not
needed for any reported number and are not included.

Used by analysis/refit_pe_drivers.py and analysis/run_slicematch_evaluation.py.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

RES = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
OUT = Path(__file__).resolve().parents[1] / "output"

# ── paths ────────────────────────────────────────────────────────────────────
ANALYSIS_CSV = str(RES / "detector_outputs/nnunet/analysis_dataframe.csv")
MASTER_CSV = str(RES / "cohort/master_metadata.csv")


# ── data ─────────────────────────────────────────────────────────────────────
def load(analysis_csv: str | None = None, master_csv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(analysis_csv or ANALYSIS_CSV, engine="python")
    # Drop rows with missing seg or no intended PE (3 QC-failed patients)
    df = df[df["error"] != "missing_pe_seg"].copy()
    df = df[df["error"] != "no_intended_pe"].copy()
    df = df.dropna(subset=["intended_voxels"])

    # Merge intended PE volume in ml from master metadata
    master = pd.read_csv(master_csv or MASTER_CSV)
    df = df.merge(
        master[["config_id", "pe_volume_ml"]],
        on="config_id",
        how="left",
        suffixes=("_old", ""),
    )
    # Drop the old empty column if present
    if "pe_volume_ml_old" in df.columns:
        df.drop(columns=["pe_volume_ml_old"], inplace=True)

    df["pe_detected_int"] = df["pe_detected"].astype(int)
    df["log_volume_ml"] = np.log1p(df["pe_volume_ml"])
    print(f"Loaded {len(df)} configs, {df.patient_id.nunique()} patients")
    return df


def run_cluster_robust_model(df: pd.DataFrame, results_dir: str | None = None) -> None:
    """PRIMARY PE-driver estimator (Fig. 5a): population-average cluster-robust
    logistic regression - GEE with an independence working correlation and sandwich
    standard errors clustered on patient - adjusted for z-scored natural-log volume
    and the categorical lesion properties, with Benjamini-Hochberg FDR over the
    categorical contrasts (volume gets a raw p). Lesion properties detected in
    >=~95% of cases quasi-separate; those terms are flagged `separated` (not stably
    estimable) and blanked. Identical estimator to the ICH driver + PE co-morbidity
    models (independence working correlation). Writes glmm_results.csv."""
    from statsmodels.stats.multitest import multipletests
    SEP_THR = 8.0
    m = df.copy()
    m = m[m["pe_volume_ml"] > 0].copy()
    lv = np.log(m["pe_volume_ml"])
    m["log_vol_z"] = (lv - lv.mean()) / lv.std()
    cat_vars = {
        "intended_shape_category": "indeterminate", "location": "lobar",
        "occlusion": "complete", "burden": "extensive", "severity_category": "mild",
    }
    for col, ref in cat_vars.items():
        m[col] = pd.Categorical(m[col])
        m[col] = m[col].cat.reorder_categories(
            [ref] + [c for c in m[col].cat.categories if c != ref])
    dummies = pd.get_dummies(m[list(cat_vars)], drop_first=True, dtype=float)
    x_cols = ["log_vol_z"] + list(dummies.columns)
    m = pd.concat([m, dummies], axis=1)
    X = sm.add_constant(m[x_cols])
    res = sm.GEE(m["pe_detected_int"], X, groups=m["patient_id"],
                 family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Independence()).fit(maxiter=200)
    pretty = {
        "log_vol_z": "Log volume (z-scored)",
        "intended_shape_category_acute": "Shape: acute",
        "intended_shape_category_chronic": "Shape: chronic",
        "location_central": "Location: central", "location_segmental": "Location: segmental",
        "location_subsegmental": "Location: subsegmental",
        "occlusion_near_complete": "Occlusion: near-complete", "occlusion_partial": "Occlusion: partial",
        "burden_isolated": "Burden: isolated", "burden_multi_bilateral": "Burden: multi-bilateral",
        "burden_multi_unilateral": "Burden: multi-unilateral",
        "severity_category_moderate": "Severity: moderate", "severity_category_severe": "Severity: severe",
    }
    rows = []
    for nm in x_cols:
        b, se, p = float(res.params[nm]), float(res.bse[nm]), float(res.pvalues[nm])
        sep = bool(abs(b) > SEP_THR or se > SEP_THR)
        rows.append({"predictor": pretty.get(nm, nm), "OR": np.exp(b),
                     "CI_lo": np.exp(b - 1.96 * se), "CI_hi": np.exp(b + 1.96 * se),
                     "coef": b, "se": se, "p": p, "separated": sep,
                     "_cat": nm != "log_vol_z"})
    t = pd.DataFrame(rows)
    fmask = (t["_cat"] & ~t["separated"]).to_numpy()
    q = np.full(len(t), np.nan)
    if fmask.any():
        q[fmask] = multipletests(t.loc[fmask, "p"], method="fdr_bh")[1]
    t["q_fdr"] = q
    for c in ("OR", "CI_lo", "CI_hi", "p", "q_fdr"):
        t.loc[t["separated"], c] = np.nan
    t = t.drop(columns="_cat")
    glmm_dir = results_dir or str(OUT / "detector_outputs/nnunet")
    Path(glmm_dir).mkdir(parents=True, exist_ok=True)
    out = f"{glmm_dir}/glmm_results.csv"
    t.to_csv(out, index=False)
    print(f"Saved cluster-robust GLMM table to {out}")
    print(t[["predictor", "OR", "CI_lo", "CI_hi", "p", "q_fdr", "separated"]].to_string(index=False))
