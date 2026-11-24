#!/usr/bin/env python
"""Formal detector x pleural-effusion INTERACTION test (Results, Fig. 6a; Methods).

Pleural effusion lowers PE detection for PE-Detector-2 (APES; cluster-robust OR 0.54)
but not for PE-Detector-1 (nnUNet3D-ResEncXL; OR 0.90). "Significant in one detector,
non-significant in the other" is not evidence that the two effects differ; that requires
a formal interaction test (Gelman & Stern, 2006).

This script fits a POOLED joint population-average GEE (logistic, sandwich/robust SE
clustered on patient) on BOTH detectors' per-embolus detection stacked into one table:

    det ~ detector * pleural_effusion + log_vol_z + C(location)          (PRIMARY)

The detector x pleural_effusion term is the formal test. Because both detectors score
the IDENTICAL counterfactual configs (a paired design), clustering on patient absorbs
the within-config/within-patient correlation, and the interaction is estimated with
more power than comparing two independent confidence intervals.

Reference (per-model) pipeline: analysis/comorbidity_gee.py :: glmm(). We
reproduce its exact per-model fits (all co-morbidities with n>=10 + log_vol_z +
location, GEE Independence, cluster = patient) to confirm the 0.54 / 0.90 headline,
then add the pooled interaction model in the SAME statistical family.

Adjustment sets:
  * PRIMARY pooled model  : effusion + volume + location.
  * ROBUSTNESS pooled model: additionally carries every other co-morbidity (n>=10) as a
    shared main effect, so the effusion adjustment set matches the paper's 0.54/0.90.
Reporting the interaction under both shows the conclusion is not an artifact of the
adjustment set.

Run (any CWD):
  python analysis/effusion_interaction.py
Output (output/, gitignored): effusion_interaction.csv
"""
from __future__ import annotations
import os

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
CODE_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = CODE_ROOT / "analysis_data"
sys.path.insert(0, str(CODE_ROOT / "analysis_data" / "lib"))
import data  # noqa: E402  (unified figure data layer)

COMORB_CSV = REPO / "comorbidity" / "comorbidity_llm.csv"

# Co-morbidity flags + min-patient threshold: identical to comorbidity_gee.py so the
# per-model reproduction matches the paper exactly.
FLAGS = ["pleural_effusion", "atelectasis", "consolidation", "pneumonia",
         "pulmonary_edema", "mass_nodule", "malignancy", "lymphadenopathy",
         "emphysema_copd", "fibrosis_ild", "pneumothorax", "pericardial_effusion",
         "cardiomegaly", "ground_glass"]
MIN_N = 10
LOC_ORDER = ["lobar", "central", "segmental", "subsegmental"]   # reference = lobar
MODEL_LABEL = {"nnunet": "PE-Detector-1 (nnUNet3D-ResEncXL)",
               "apes": "PE-Detector-2 (APES)"}

# The two names patsy assigns the effusion interaction depending on how detector
# is coded; we look up whichever is present.
INTERACTION_KEYS = [
    "C(detector, Treatment('nnunet'))[T.apes]:pleural_effusion",
    "pleural_effusion:C(detector, Treatment('nnunet'))[T.apes]",
]


# --------------------------------------------------------------------- data --
def _comorbidities() -> pd.DataFrame:
    cm = pd.read_csv(COMORB_CSV)
    for f in FLAGS:
        cm[f] = pd.to_numeric(cm[f], errors="coerce").fillna(0).astype(int)
    return cm[["patient_id"] + FLAGS]


def _use_flags(cm: pd.DataFrame) -> list:
    """Co-morbidities modeled by comorbidity_gee.glmm(): flag present in >=MIN_N
    PATIENTS (patient-level count, NOT embolus-level). This exact set reproduces the
    paper's 0.54 / 0.90 effusion ORs."""
    return [f for f in FLAGS if int(cm[f].sum()) >= MIN_N]


def _one_model(model: str, cm: pd.DataFrame) -> pd.DataFrame:
    """Per-embolus detection for one detector, merged with co-morbidity flags,
    with log-volume z-scored and location as an ordered categorical.
    Mirrors comorbidity_gee.glmm() preprocessing (volume_ml > 0)."""
    pe = data.load_pe_seg(model).copy()
    pe = pe[pe.volume_ml > 0].copy()
    pe = pe.merge(cm, on="patient_id", how="left")
    pe[FLAGS] = pe[FLAGS].fillna(0).astype(float)
    pe["det_int"] = pe.detected.astype(int)
    pe["detector"] = model
    return pe


def build_pooled(cm: pd.DataFrame):
    """Stack both detectors; z-score log-volume on the POOLED table.

    Both detectors score the identical frozen configs, so log-volume mean/SD are the
    same pooled or per-model -- z-scoring on the pool keeps the per-model and pooled
    coefficients on one common scale."""
    nn = _one_model("nnunet", cm)
    ap = _one_model("apes", cm)
    pool = pd.concat([nn, ap], ignore_index=True)
    lv = np.log(pool.volume_ml)
    pool["log_vol_z"] = (lv - lv.mean()) / lv.std()
    pool["location"] = pd.Categorical(pool.location, categories=LOC_ORDER)
    pool["detector"] = pd.Categorical(pool.detector, categories=["nnunet", "apes"])
    # propagate the identical z-score back onto per-model frames for reduced fits
    per = {}
    for m, sub in (("nnunet", nn), ("apes", ap)):
        s = pool[pool.detector == m].copy()
        per[m] = s
    return pool, per


# --------------------------------------------------------------------- fits --
def _gee(formula: str, df: pd.DataFrame):
    return smf.gee(formula, groups="patient_id", data=df,
                   family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Independence()).fit(maxiter=200)


def _row(fit, key: str) -> dict:
    b, se, p = fit.params[key], fit.bse[key], fit.pvalues[key]
    return {"term": key, "coef": float(b), "se": float(se),
            "OR": float(np.exp(b)), "CI_lo": float(np.exp(b - 1.96 * se)),
            "CI_hi": float(np.exp(b + 1.96 * se)), "p": float(p)}


def per_model_full(per: dict, use: list) -> dict:
    """Reproduce comorbidity_gee per-model effusion OR (FULL adjustment: all
    co-morbidities n>=MIN_N patients + volume + location). Confirms 0.54 / 0.90."""
    out = {}
    for m, df in per.items():
        formula = "det_int ~ log_vol_z + C(location) + " + " + ".join(use)
        fit = _gee(formula, df)
        out[m] = _row(fit, "pleural_effusion")
    return out


def per_model_reduced(per: dict) -> dict:
    """Per-model effusion OR under the REDUCED adjustment (effusion + volume +
    location only) -- the exact terms the pooled PRIMARY model decomposes into."""
    out = {}
    for m, df in per.items():
        fit = _gee("det_int ~ log_vol_z + C(location) + pleural_effusion", df)
        out[m] = _row(fit, "pleural_effusion")
    return out


def pooled_interaction(pool: pd.DataFrame, extra_flags=None):
    """Pooled joint GEE with detector x pleural_effusion.

    extra_flags: other co-morbidities (n>=10) carried as shared main effects; None =
    PRIMARY (effusion, volume and location terms only)."""
    rhs = ["C(detector, Treatment('nnunet')) * pleural_effusion",
           "log_vol_z", "C(location)"]
    if extra_flags:
        rhs += list(extra_flags)
    fit = _gee("det_int ~ " + " + ".join(rhs), pool)
    ikey = next((k for k in INTERACTION_KEYS if k in fit.params.index), None)
    if ikey is None:  # fallback: any param containing both tokens
        ikey = next(k for k in fit.params.index
                    if "pleural_effusion" in k and "T.apes" in k)
    dkey = "C(detector, Treatment('nnunet'))[T.apes]"
    return fit, {
        "interaction": _row(fit, ikey),
        "effusion_main": _row(fit, "pleural_effusion"),   # = effect in nnUNet (ref)
        "detector_main": _row(fit, dkey),                 # APES vs nnUNet at effusion=0
    }


def _implied_apes(res: dict) -> dict:
    """APES effusion effect implied by the pooled model = main + interaction."""
    b = res["effusion_main"]["coef"] + res["interaction"]["coef"]
    return {"coef": b, "OR": float(np.exp(b))}


# --------------------------------------------------------------------- main --
def fmt(r: dict) -> str:
    return (f"OR {r['OR']:.3f} [{r['CI_lo']:.3f}, {r['CI_hi']:.3f}]  "
            f"coef {r['coef']:+.4f}  se {r['se']:.4f}  p {r['p']:.4g}")


def main():
    cm = _comorbidities()
    use = _use_flags(cm)                              # patient-level n>=MIN_N (paper set)
    pool, per = build_pooled(cm)

    n_pat = int(pool.patient_id.nunique())
    n_emb = int(len(per["nnunet"]))                 # emboli per detector
    n_eff_pat = int((cm.set_index("patient_id").loc[per["nnunet"].patient_id.unique(),
                                                    "pleural_effusion"] == 1).sum())
    n_eff_emb = int((per["nnunet"].pleural_effusion == 1).sum())

    print("=" * 78)
    print("Detector x pleural-effusion INTERACTION TEST")
    print("=" * 78)
    print(f"Cohort (frozen, volume_ml>0): {n_pat} patients, {n_emb} emboli PER detector, "
          f"{2 * n_emb} pooled rows.")
    print(f"Pleural effusion present in {n_eff_pat}/{n_pat} patients "
          f"({n_eff_emb}/{n_emb} emboli per detector).")

    # --- 1. reproduce paper 0.54 / 0.90 (full adjustment) -------------------
    full = per_model_full(per, use)
    print(f"\n--- Per-model effusion OR, FULL adjustment ({len(use)} co-morbidities, "
          "reproduces paper) ---")
    for m in ("nnunet", "apes"):
        print(f"  {MODEL_LABEL[m]:38s}: {fmt(full[m])}")

    # --- 2. PRIMARY pooled interaction (effusion, volume, location) ---------
    fit_p, res_p = pooled_interaction(pool, extra_flags=None)
    print("\n--- PRIMARY pooled GEE: det ~ detector*effusion + log_vol_z + C(location) ---")
    print(f"  detector main (APES vs nnUNet, effusion=0): {fmt(res_p['detector_main'])}")
    print(f"  pleural_effusion main (effect in nnUNet ref): {fmt(res_p['effusion_main'])}")
    print(f"  INTERACTION detector[APES] x effusion       : {fmt(res_p['interaction'])}")
    imp = _implied_apes(res_p)
    print(f"  -> implied APES effusion OR (main x interaction) = {imp['OR']:.3f}")

    # --- 3. per-model REDUCED (what the pooled primary decomposes into) -----
    red = per_model_reduced(per)
    print("\n--- Per-model effusion OR, REDUCED adjustment (effusion+vol+loc only) ---")
    for m in ("nnunet", "apes"):
        print(f"  {MODEL_LABEL[m]:38s}: {fmt(red[m])}")
    print(f"  ratio APES/nnUNet reduced OR = {red['apes']['OR'] / red['nnunet']['OR']:.3f} "
          f"(= interaction OR {res_p['interaction']['OR']:.3f}, internal check)")

    # --- 4. ROBUSTNESS pooled interaction (paper full adjustment set) -------
    extra = [f for f in use if f != "pleural_effusion"]
    fit_r, res_r = pooled_interaction(pool, extra_flags=extra)
    print("\n--- ROBUSTNESS pooled GEE: + all co-morbidities (n>=10) as shared mains ---")
    print(f"  pleural_effusion main (nnUNet ref): {fmt(res_r['effusion_main'])}")
    print(f"  INTERACTION detector[APES] x effusion: {fmt(res_r['interaction'])}")

    # --- verdict ------------------------------------------------------------
    ip, ir = res_p["interaction"], res_r["interaction"]
    sig_p, sig_r = ip["p"] < 0.05, ir["p"] < 0.05
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"PRIMARY    interaction OR {ip['OR']:.3f}, p={ip['p']:.4g} -> "
          f"{'SIGNIFICANT' if sig_p else 'NOT significant'} at 0.05")
    print(f"ROBUSTNESS interaction OR {ir['OR']:.3f}, p={ir['p']:.4g} -> "
          f"{'SIGNIFICANT' if sig_r else 'NOT significant'} at 0.05")
    verdict = ("SUPPORTS" if (sig_p and sig_r) else
               "does NOT support" if not (sig_p or sig_r) else
               "PARTIALLY supports (adjustment-set sensitive)")
    print(f"\nThe formal interaction test {verdict} the 'model-specific blind spot' claim.")

    # --- persist a tidy table ----------------------------------------------
    rows = []
    for tag, d in [("per_model_full_nnunet", full["nnunet"]),
                   ("per_model_full_apes", full["apes"]),
                   ("per_model_reduced_nnunet", red["nnunet"]),
                   ("per_model_reduced_apes", red["apes"]),
                   ("pooled_primary_effusion_main", res_p["effusion_main"]),
                   ("pooled_primary_detector_main", res_p["detector_main"]),
                   ("pooled_primary_interaction", res_p["interaction"]),
                   ("pooled_robust_effusion_main", res_r["effusion_main"]),
                   ("pooled_robust_interaction", res_r["interaction"])]:
        rows.append({"quantity": tag, **{k: d[k] for k in
                     ["OR", "CI_lo", "CI_hi", "coef", "se", "p"]}})
    out_csv = CODE_ROOT / "output" / "effusion_interaction.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
