"""Patient-level (cluster) statistics for the paper figures and Results numbers.

Unit = the cluster: source patient for the synthetic counterfactuals and the radiologist review,
scan (case) for the real external PE datasets, patient for PhysioNet CT-ICH (one scan each).

  ci / ci_arrays   95% percentile interval of a detection rate from resampling whole clusters
                   (B = 2,000, seed 0). If every observation is detected (or none), the
                   bootstrap has no spread; the interval is then an approximation: Clopper-Pearson
                   on the NUMBER OF CLUSTERS (conservative). The log marks these cases.
  gee_trend_p      Wald p for a linear trend across ordered bins: logistic GEE of detection on the
                   bin score (0..K-1), independence working correlation, robust (sandwich) SE
                   clustered on patient. Clustered analogue of the Cochran-Armitage test.
  gee_group_p      Wald p for a two-group difference from the same GEE with a group indicator.

Each call is logged next to the original statistic (Clopper-Pearson CI, Cochran-Armitage or
Fisher p) so the two can be compared.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "patient_bootstrap"
B = 2000
SEED = 0
CONTEXT = {"figure": ""}
LOG_CI: list[dict] = []
LOG_P: list[dict] = []


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy import stats
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def _cluster_sums(y, cl):
    s = (pd.DataFrame({"y": np.asarray(y, float), "c": np.asarray(cl)})
         .groupby("c", sort=True)["y"].agg(["sum", "count"]))
    return s["sum"].to_numpy(float), s["count"].to_numpy(float)


def _boot_rates(k, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), (B, len(k)))
    return k[idx].sum(1) / n[idx].sum(1)


def _p_from(d):
    p = 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
    return float(min(1.0, max(p, 1.0 / B)))


# ------------------------------------------------------------------ intervals
def ci_arrays(y, cl, label=""):
    k, n = _cluster_sums(y, cl)
    K, N, C = int(k.sum()), int(n.sum()), len(k)
    if N == 0:
        return 0.0, 0.0
    cp = clopper_pearson(K, N)
    if K in (0, N):
        lo, hi = clopper_pearson(0 if K == 0 else C, C)
        method = "approximation: Clopper-Pearson on clusters (no bootstrap spread)"
    else:
        lo, hi = np.percentile(_boot_rates(k, n, SEED), [2.5, 97.5])
        method = "cluster_bootstrap"
    LOG_CI.append(dict(figure=CONTEXT["figure"], label=label, k=K, n=N, n_clusters=C,
                       rate_pct=100 * K / N, cp_lo=100 * cp[0], cp_hi=100 * cp[1],
                       boot_lo=100 * lo, boot_hi=100 * hi, method=method))
    return float(lo), float(hi)


def ci(df, outcome, cluster, label=""):
    return ci_arrays(df[outcome].astype(float).to_numpy(), df[cluster].to_numpy(), label)


# ------------------------------------------------------------------ clustered GEE tests
def _gee_wald(y, x, groups):
    import statsmodels.api as sm
    X = sm.add_constant(np.asarray(x, float))
    r = sm.GEE(np.asarray(y, float), X, groups=np.asarray(groups), family=sm.families.Binomial(),
               cov_struct=sm.cov_struct.Independence()).fit()
    return float(r.params[1]), float(r.pvalues[1])


def gee_group_p(dfa, dfb, outcome, cluster, label=""):
    y = np.r_[dfa[outcome].astype(float).to_numpy(), dfb[outcome].astype(float).to_numpy()]
    x = np.r_[np.ones(len(dfa)), np.zeros(len(dfb))]
    g = np.r_[dfa[cluster].to_numpy(), dfb[cluster].to_numpy()]
    beta, p = _gee_wald(y, x, g)
    Ka, Na, Kb, Nb = int(dfa[outcome].sum()), len(dfa), int(dfb[outcome].sum()), len(dfb)
    _, pf = stats.fisher_exact([[Ka, Na - Ka], [Kb, Nb - Kb]])
    LOG_P.append(dict(figure=CONTEXT["figure"], label=label, test="difference of rates",
                      a=f"{Ka}/{Na}", b=f"{Kb}/{Nb}", clusters=len(np.unique(g)),
                      p_original=pf, original="Fisher exact", p_clustered=p,
                      method="logistic GEE, robust SE clustered on patient", estimate=f"OR {np.exp(beta):.2f}"))
    return p


# ------------------------------------------------------------------ ordered trend
def gee_trend_p(g, outcome, bin_col, bin_order, cluster, min_n=0, label=""):
    for b in bin_order:                                   # same eligibility rule as the original
        if int((g[bin_col] == b).sum()) < min_n:
            return np.nan
    gg = g[g[bin_col].isin(bin_order)]
    score = gg[bin_col].map({b: i for i, b in enumerate(bin_order)}).astype(float).to_numpy()
    beta, p = _gee_wald(gg[outcome].astype(float).to_numpy(), score, gg[cluster].to_numpy())
    det = [int(gg.loc[gg[bin_col] == b, outcome].sum()) for b in bin_order]
    nn = [int((gg[bin_col] == b).sum()) for b in bin_order]
    try:
        pca = float(Table(np.array([det, [a - b for a, b in zip(nn, det)]])).test_ordinal_association().pvalue)
    except Exception:  # noqa: BLE001
        pca = np.nan
    LOG_P.append(dict(figure=CONTEXT["figure"], label=label, test="trend across ordered bins",
                      a=" ".join(f"{k}/{n}" for k, n in zip(det, nn)), b="",
                      clusters=gg[cluster].nunique(), p_original=pca, original="Cochran-Armitage",
                      p_clustered=p, method="logistic GEE on bin score, robust SE clustered on patient",
                      estimate=f"OR per bin {np.exp(beta):.2f}"))
    return p


def write_logs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(LOG_CI).round(2).to_csv(OUT_DIR / "ci_comparison.csv", index=False)
    pd.DataFrame(LOG_P).to_csv(OUT_DIR / "pvalue_comparison.csv", index=False)
