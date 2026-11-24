"""Co-occurring findings and PE detection (Fig. 6a): cluster-robust GEE per PE detector.

FLAGS, PRETTY, MIN_N, MODEL, _bh_fdr and glmm are copied verbatim from the original figure
module (paper_figures/fig_comorbidity.py in the development repository); the figure
functions are not included. glmm() fits

    detected ~ log_vol_z + C(location) + co-occurring findings present in >= MIN_N patients

as a population-average logistic GEE (independence working correlation, sandwich SE
clustered on patient) with BH-FDR across the modeled findings. A variational-Bayes fit is
kept as reference columns (OR_vb, CI_lo_vb, CI_hi_vb) only; it is not used in the paper.

The co-occurring findings are the LLM extraction from each source patient's report
impression (comorbidity/comorbidity_llm.csv; Methods).

Reads  CF_DATA/comorbidity/comorbidity_llm.csv, CF_DATA/detector_outputs/{apes,nnunet}/
Writes output/comorbidity/comorbidity_glmm_{apes,nnunet}.csv
Run:   python analysis/comorbidity_gee.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_data" / "lib"))
import data  # noqa: E402

RES = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
OUTPUT = Path(__file__).resolve().parents[1] / "output" / "comorbidity"

FLAGS = ["pleural_effusion", "atelectasis", "consolidation", "pneumonia",
         "pulmonary_edema", "mass_nodule", "malignancy", "lymphadenopathy",
         "emphysema_copd", "fibrosis_ild", "pneumothorax", "pericardial_effusion",
         "cardiomegaly", "ground_glass"]
PRETTY = {"pleural_effusion": "Pleural effusion", "atelectasis": "Atelectasis/collapse",
          "consolidation": "Consolidation", "pneumonia": "Pneumonia",
          "pulmonary_edema": "Pulmonary edema", "mass_nodule": "Mass/nodule",
          "malignancy": "Malignancy/mets", "lymphadenopathy": "Lymphadenopathy",
          "emphysema_copd": "Emphysema/COPD", "fibrosis_ild": "Fibrosis/ILD",
          "pneumothorax": "Pneumothorax", "pericardial_effusion": "Pericardial effusion",
          "cardiomegaly": "Cardiomegaly", "ground_glass": "Ground-glass"}
MIN_N = 10   # min patients with the flag to model it
MODEL = "apes"   # set by main(); seg model whose detection we analyze


# ------------------------------------------------------------- stats helpers -
def _bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values (same order as input)."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[idx] * n / r)
        q[idx] = prev
    return q


# ----------------------------------------------------------------- GLMM/GEE --
def glmm(cm: pd.DataFrame) -> pd.DataFrame:
    """PRIMARY = cluster-robust GEE (population-average logistic, independence
    working correlation, sandwich SE clustered on patient). SECONDARY = the old
    VB-GLMM (subject-specific), kept only as reference columns to show its SE was
    understated. BH-FDR across the modeled co-morbidities.
    detected ~ comorbidities(n>=MIN_N) + log_vol_z + location, cluster = patient."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    pe = data.load_pe_seg(MODEL).copy()
    pe = pe[pe.volume_ml > 0].copy()
    pe["log_vol_z"] = ((np.log(pe.volume_ml) - np.log(pe.volume_ml).mean())
                       / np.log(pe.volume_ml).std())
    pe["loccat"] = pd.Categorical(pe.location, categories=["lobar", "central",
                                                           "segmental", "subsegmental"])
    use = [f for f in FLAGS if int(cm[f].sum()) >= MIN_N]
    pe = pe.merge(cm[["patient_id"] + use], on="patient_id", how="left")
    pe[use] = pe[use].fillna(0).astype(float)
    pe["det_int"] = pe.detected.astype(int)

    # --- primary: cluster-robust GEE ---
    formula = "det_int ~ log_vol_z + C(loccat) + " + " + ".join(use)
    gee = smf.gee(formula, groups="patient_id", data=pe,
                  family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Independence()).fit(maxiter=200)

    # --- secondary: VB-GLMM (reference only) ---
    loc_d = pd.get_dummies(pe["loccat"], prefix="loc", drop_first=True, dtype=float)
    Xvb = pd.concat([pe[["log_vol_z"]].reset_index(drop=True),
                     loc_d.reset_index(drop=True),
                     pe[use].reset_index(drop=True)], axis=1)
    pid = pe.patient_id.astype("category").cat.codes.to_numpy()
    npat = int(pid.max() + 1)
    exog_re = np.zeros((len(pe), npat)); exog_re[np.arange(len(pe)), pid] = 1.0
    vb = BinomialBayesMixedGLM(pe.det_int.to_numpy(), Xvb.to_numpy(),
                               exog_vc=exog_re, ident=[0] * npat).fit_vb()
    vbm = {c: (float(np.exp(m)), float(np.exp(m - 1.96 * s)), float(np.exp(m + 1.96 * s)))
           for c, m, s in zip(Xvb.columns, np.asarray(vb.fe_mean), np.asarray(vb.fe_sd))}

    def _raw(nm: str) -> str:
        if nm.startswith("C(loccat)[T."):
            return "loc_" + nm[len("C(loccat)[T."):-1]
        return nm
    name = {"Intercept": "Intercept", "log_vol_z": "Volume (per SD, log)",
            "loc_central": "Location: central", "loc_segmental": "Location: segmental",
            "loc_subsegmental": "Location: subsegmental"}
    name.update({f: PRETTY[f] for f in use})
    rows = []
    for nm in gee.params.index:
        raw = _raw(nm)
        b, se, pv = gee.params[nm], gee.bse[nm], gee.pvalues[nm]
        vv = vbm.get(raw, (np.nan, np.nan, np.nan))
        rows.append({"predictor": name.get(raw, raw), "raw": raw,
                     "OR": float(np.exp(b)), "CI_lo": float(np.exp(b - 1.96 * se)),
                     "CI_hi": float(np.exp(b + 1.96 * se)), "coef": float(b),
                     "se": float(se), "p": float(pv),
                     "OR_vb": vv[0], "CI_lo_vb": vv[1], "CI_hi_vb": vv[2]})
    t = pd.DataFrame(rows)
    is_com = t.raw.isin(use)
    t["q_fdr"] = np.nan
    t.loc[is_com, "q_fdr"] = _bh_fdr(t.loc[is_com, "p"].to_numpy())
    t.to_csv(OUTPUT / f"comorbidity_glmm_{MODEL}.csv", index=False)
    print(f"GEE cluster-robust [{MODEL}]: {len(pe)} configs, {npat} patients, "
          f"{len(use)} comorbidities (n>={MIN_N})")
    print(t[t.raw.isin(use)][["predictor", "OR", "CI_lo", "CI_hi", "p", "q_fdr"]]
          .to_string(index=False))
    return t


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cm = pd.read_csv(RES / "comorbidity" / "comorbidity_llm.csv")
    for MODEL in ["apes", "nnunet"]:      # glmm() reads the module-level MODEL
        glmm(cm)
