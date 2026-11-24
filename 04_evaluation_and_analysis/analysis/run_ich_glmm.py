"""Fit the ICH detection driver models (Fig. 5b).

Primary inference = cluster-robust GEE (population-average logistic, INDEPENDENCE
working correlation, sandwich SE clustered on patient) + Benjamini-Hochberg FDR on
the subtype family, the same estimator as the PE driver and PE co-morbidity models
(independence is separation-tolerant, unlike exchangeable). A variational-Bayes
BinomialBayesMixedGLM fit is kept only as reference columns (OR_vb, CI_lo_vb, CI_hi_vb);
it is not used in the paper.

Two outcomes, one table each (same fixed structure):
  * segmentation detection (`detected`, ICH_SEG-selected co-localization; default
    ICH_SEG=viola = ICH-Detector-1)              -> ich_glmm_results{_viola}.csv
  * 3D ResNet-101 classifier detection (emm_prob >= 0.5; ICH-Detector-2)
                                                 -> ich_glmm_results_emm.csv
Fixed: log(ich_volume) z-scored + subtype_grp (5 single subtypes SDH/EDH/SAH/IPH/IVH
shown individually + combination-count levels 2/3/4 subtypes; ref = IPH), so each OR is
volume-adjusted. FDR across the 7 non-reference subtype-group contrasts; volume raw p.

A secondary single-subtype model (det ~ log_vol_z + C(scope), extra- vs intra-axial) is
written to ich_scope_or{_viola}.csv; it is not reported in the paper.

Reads  CF_DATA/ich/cf_2640_samples_recovered.csv (via analysis_data/lib/data.py)
Writes output/ich/
Run:   python analysis/run_ich_glmm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_data" / "lib"))
from data import load_ich, _ICH_SUBTYPE_FRAGMENT, ICH_SEG_SUFFIX, ICH_SEG_LABEL  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "output" / "ich"
EXTRA_AXIAL = ["SDH", "EDH", "SAH"]   # extra-axial (outside the TotalSegmentator cerebral_bleed scope)
INTRA_AXIAL = ["IPH", "IVH"]          # in scope

# ICH subtype predictor: single subtypes shown individually, and a lumped 'Mixed' level
# replaced by combination-count levels (2/3/4 co-occurring subtypes, any combination).
# Reference = IPH (a common intra-axial single subtype); the 2/3/4-subtype rows show
# whether multiplicity adds detectability. One 8-level categorical.
ICH_SUBTYPE_REF = "IPH"
SINGLE_SUBTYPES = ["SDH", "EDH", "SAH", "IPH", "IVH"]
COMBO_LEVELS = ["2 subtypes", "3 subtypes", "4 subtypes"]
SUBTYPE_GRP_LEVELS = SINGLE_SUBTYPES + COMBO_LEVELS

PRETTY = {
    "log_vol_z": "Log volume (z-scored)",
    "sg_SDH": "Subtype: SDH", "sg_EDH": "Subtype: EDH", "sg_SAH": "Subtype: SAH",
    "sg_IPH": "Subtype: IPH", "sg_IVH": "Subtype: IVH",
    "sg_2_subtypes": "2 subtypes", "sg_3_subtypes": "3 subtypes",
    "sg_4_subtypes": "4 subtypes",
}
# FDR family = all non-reference subtype-group contrasts (4 singles + 3 combinations)
SUBTYPE_RAW = [f"sg_{l.replace(' ', '_')}" for l in SUBTYPE_GRP_LEVELS
               if l != ICH_SUBTYPE_REF]


def _subtype_grp(image_path: str) -> str:
    """Single-subtype code if exactly one subtype is tagged, else 'N subtypes'
    (combination count, any combination). 'none' if no recognized subtype."""
    stem = str(image_path).split("/")[-1]
    codes = list(dict.fromkeys(_ICH_SUBTYPE_FRAGMENT.findall(stem)))
    n = len(codes)
    if n == 1:
        return codes[0]
    if 2 <= n <= 4:
        return f"{n} subtypes"
    return "none"

OUTCOMES = [
    (lambda d: d["detected"].astype(int), f"ich_glmm_results{ICH_SEG_SUFFIX}.csv", ICH_SEG_LABEL),
    (lambda d: (d["emm_prob"] >= 0.5).astype(int), "ich_glmm_results_emm.csv", "EMM (emm_prob>=0.5)"),
]


def _bh_fdr(pvals) -> np.ndarray:
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


def _vb_reference(df, X, X_cols, exog_re, n_pat, y) -> dict:
    """Old variational-Bayes fit -> {raw_col: (OR, CI_lo, CI_hi)} reference only."""
    res = BinomialBayesMixedGLM(endog=y, exog=X.to_numpy(), exog_vc=exog_re,
                                ident=[0] * n_pat).fit_vb()
    m, s = np.asarray(res.fe_mean), np.asarray(res.fe_sd)
    return {c: (float(np.exp(mi)), float(np.exp(mi - 1.96 * si)),
                float(np.exp(mi + 1.96 * si)))
            for c, mi, si in zip(X_cols, m, s)}


def _gee_table(df, outcome_fn, desc) -> pd.DataFrame:
    d = df.copy()
    d["det_int"] = outcome_fn(d).to_numpy()
    # GEE (primary): additive, volume-adjusted, cluster-robust sandwich SE
    formula = ("det_int ~ log_vol_z + C(subtype_grp, Treatment(reference='%s'))"
               % ICH_SUBTYPE_REF)
    gee = smf.gee(formula, groups="patient_id", data=d,
                  family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Independence()).fit(maxiter=200)

    def _raw(nm: str) -> str:
        if nm.startswith("C(subtype_grp"):
            return "sg_" + nm.split("T.")[1].rstrip("]").replace(" ", "_")
        return nm
    rows = []
    for nm in gee.params.index:
        if nm == "Intercept":
            continue
        raw = _raw(nm)
        b, se, p = gee.params[nm], gee.bse[nm], gee.pvalues[nm]
        rows.append({"predictor": PRETTY.get(raw, raw), "raw": raw,
                     "OR": float(np.exp(b)), "CI_lo": float(np.exp(b - 1.96 * se)),
                     "CI_hi": float(np.exp(b + 1.96 * se)), "coef": float(b),
                     "se": float(se), "p": float(p)})
    t = pd.DataFrame(rows)
    is_sub = t.raw.isin(SUBTYPE_RAW)
    t["q_fdr"] = np.nan
    t.loc[is_sub, "q_fdr"] = _bh_fdr(t.loc[is_sub, "p"].to_numpy())
    return t, d


def _scope_or(df) -> dict:
    """Volume-adjusted extra-vs-intra(ref) scope OR per detector, single-subtype configs."""
    s = df[df.subtype.isin(EXTRA_AXIAL + INTRA_AXIAL)].copy()
    s["scope"] = np.where(s.subtype.isin(EXTRA_AXIAL), "extra", "intra")
    out = {}
    for det, yfn in [(ICH_SEG_LABEL, lambda x: x.detected.astype(int)),
                     ("EMM", lambda x: (x.emm_prob >= 0.5).astype(int))]:
        s["y"] = yfn(s)
        m = smf.gee("y ~ log_vol_z + C(scope, Treatment('intra'))", groups="patient_id",
                    data=s, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Independence()).fit(maxiter=200)
        nm = [x for x in m.params.index if "scope" in x][0]
        b, se, p = m.params[nm], m.bse[nm], m.pvalues[nm]
        out[det] = {"OR": float(np.exp(b)), "CI_lo": float(np.exp(b - 1.96 * se)),
                    "CI_hi": float(np.exp(b + 1.96 * se)), "p": float(p),
                    "n_extra": int((s.scope == "extra").sum()),
                    "n_intra": int((s.scope == "intra").sum())}
    return out


def main() -> None:
    df = load_ich().copy()
    df = df[df["volume_ml"] > 0].copy()
    df["log_vol_z"] = ((np.log(df["volume_ml"]) - np.log(df["volume_ml"]).mean())
                       / np.log(df["volume_ml"]).std())
    df["subtype"] = df["subtype"].astype(str)
    df["subtype_grp"] = df["image"].apply(_subtype_grp)
    df = df[df["subtype_grp"] != "none"].copy()

    # VB reference design (shared): patient random intercept + same fixed effects
    cats = [ICH_SUBTYPE_REF] + [c for c in SUBTYPE_GRP_LEVELS if c != ICH_SUBTYPE_REF]
    sub_cat = pd.Categorical(df["subtype_grp"], categories=cats)
    sub_d = pd.get_dummies(sub_cat, prefix="sg", drop_first=True, dtype=float)
    sub_d.columns = [c.replace(" ", "_") for c in sub_d.columns]
    Xvb = pd.concat([df[["log_vol_z"]].reset_index(drop=True),
                     sub_d.reset_index(drop=True)], axis=1)
    Xcols = list(Xvb.columns)
    pid = df["patient_id"].astype("category").cat.codes.to_numpy()
    n_pat = int(pid.max() + 1)
    exog_re = np.zeros((len(df), n_pat)); exog_re[np.arange(len(df)), pid] = 1.0

    RESULTS.mkdir(parents=True, exist_ok=True)
    for outcome_fn, fname, desc in OUTCOMES:
        print(f"\n=== {desc} : cluster-robust GEE + FDR ===")
        t, _ = _gee_table(df, outcome_fn, desc)
        vb = _vb_reference(df, Xvb, Xcols, exog_re, n_pat, outcome_fn(df).to_numpy())
        t["OR_vb"] = t.raw.map(lambda r: vb.get(r, (np.nan,) * 3)[0])
        t["CI_lo_vb"] = t.raw.map(lambda r: vb.get(r, (np.nan,) * 3)[1])
        t["CI_hi_vb"] = t.raw.map(lambda r: vb.get(r, (np.nan,) * 3)[2])
        out = RESULTS / fname
        t.to_csv(out, index=False)
        print(t[["predictor", "OR", "CI_lo", "CI_hi", "p", "q_fdr", "OR_vb"]].to_string(index=False))
        print(f"Wrote {out}")

    # Secondary scope OR (extra vs intra, volume-adjusted; not in the paper)
    scope = _scope_or(df)
    sc = pd.DataFrame(scope).T.reset_index().rename(columns={"index": "detector"})
    sc.to_csv(RESULTS / f"ich_scope_or{ICH_SEG_SUFFIX}.csv", index=False)
    print("\n=== SCOPE (extra-axial vs intra-axial, volume-adjusted) ===")
    print(sc.to_string(index=False))
    print(f"Wrote {RESULTS / f'ich_scope_or{ICH_SEG_SUFFIX}.csv'}")


if __name__ == "__main__":
    main()
