"""Numbers shown in Figs. 4, 6b and 7 (bars, 95% CIs and tests), without drawing the figures.

Same statistics, bins and settings as the figure code (boot.py: cluster bootstrap B = 2,000,
seed 0; logistic GEE clustered on patient):
  Fig 4a  PE detection by location, severity and volume bin, both PE detectors.
  Fig 4b  ICH detection by volume bin, number of subtypes and subtype, both ICH detectors.
  Fig 6b  ICH detection by volume bin within each single subtype (bins <5, 5-15, 15-30,
          30-50 mL, left-closed; >50 mL not shown; strata with n < 10 not shown),
          within-subtype trend (boot.gee_trend_p on the ordered bin score, BH-FDR across
          subtypes), logistic trend per log-mL (reference), and extra- vs intra-axial at
          30-50 mL (rates + boot.gee_group_p).
  Fig 7a  PE detection, synthetic vs FUMPE / CAD-PE / READ, in the most-reliable (>= 2.5 mL)
          and least-reliable (< 0.5 mL) volume ranges. Cluster = patient (synthetic) or
          scan (real). Plus the number of scans and emboli per external dataset.
  Fig 7b  ICH detection, synthetic vs PhysioNet CT-ICH (all 36 scans), bins < 5, 5-30 and
          >= 30 mL. Cluster = patient.

Writes output/patient_bootstrap/: ci_comparison.csv (every bar: k, n, clusters, rate,
Clopper-Pearson and bootstrap CI), pvalue_comparison.csv (every test), ich_volume_subtype_
{catrend,trend,extraVintra}.csv, external_counts.csv.
Run:  python patient_bootstrap/figure_numbers.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["ICH_SEG"] = "viola"                   # ICH-Detector-1 = Viola-UNet
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("CF_DATA", HERE.parents[1] / "resources"))
sys.path[:0] = [str(HERE), str(HERE.parent / "analysis_data" / "lib")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

import boot as BOOT  # noqa: E402
import data  # noqa: E402


def _bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0:
        return p
    order = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * n / (n - rank)); q[idx] = prev
    return q


# ------------------------------------------------------------------ Fig 4a (PE)
PE_MODELS = [("nnunet", "PE-Detector-1"), ("apes", "PE-Detector-2")]
PE_AXES = [("location", ["central", "lobar", "segmental", "subsegmental"]),
           ("severity_category", ["mild", "moderate", "severe"])]
PE_VOL_BINS = [0, 0.5, 1, 2.5, 5, 1e6]
PE_VOL_LABELS = ["<0.5 mL", "0.5-1 mL", "1-2.5 mL", "2.5-5 mL", ">5 mL"]


def fig4a():
    BOOT.CONTEXT["figure"] = "Fig4a"
    dsets = {}
    for m, _ in PE_MODELS:
        d = data.load_pe_seg(m); d = d[d.volume_ml > 0].copy()
        d["volume_bin"] = pd.cut(d.volume_ml, PE_VOL_BINS, labels=PE_VOL_LABELS)
        dsets[m] = d
    for axis, order in PE_AXES + [("volume_bin", PE_VOL_LABELS)]:
        present = [c for c in order if any((dsets[m][axis] == c).any() for m, _ in PE_MODELS)]
        for m, lab in PE_MODELS:
            d = dsets[m]
            for c in present:
                g = d[d[axis] == c]
                if len(g):
                    BOOT.ci(g, "detected", "patient_id", label=f"{lab}|{axis}={c}")


# ------------------------------------------------------------------ Fig 4b (ICH)
SUB_ORDER_4B = ["SDH", "EDH", "SAH", "IPH", "IVH"]


def _ich():
    d = data.load_ich()
    d = d[d.volume_ml > 0].copy()
    d[data.ICH_SEG_LABEL] = d.detected.astype(bool)
    d[data.ICH_CLF_LABEL] = d.emm_prob >= data.ICH_CLF_THRESH
    return d


def fig4b():
    BOOT.CONTEXT["figure"] = "Fig4b"
    d = _ich()
    dets = [data.ICH_SEG_LABEL, data.ICH_CLF_LABEL]
    for axis, order in [("volume_bin", data.ICH_VOLUME_LABELS), ("combo", data.ICH_COMBO_ORDER)]:
        present = [c for c in order if (d[axis] == c).any()]
        for det in dets:
            for c in present:
                g = d[d[axis] == c]
                if len(g):
                    BOOT.ci(g, det, "patient_id", label=f"{det}|{axis}={c}")
    order = SUB_ORDER_4B if data.ICH_SEG_SCOPED else sorted(
        SUB_ORDER_4B, key=lambda s: d.loc[d.subtype == s, data.ICH_SEG_LABEL].mean())
    for det in dets:
        for s in order:
            BOOT.ci(d[d.subtype == s], det, "patient_id", label=f"{det}|subtype={s}")


# ------------------------------------------------------------------ Fig 6b (ICH volume x subtype)
EXTRA = ["SDH", "EDH", "SAH"]
INTRA = ["IPH", "IVH"]
SUB_ORDER = EXTRA + INTRA
VBINS = [-0.001, 5, 15, 30, 50]
VLAB = ["<5 mL", "5-15 mL", "15-30 mL", "30-50 mL"]
MIN_N = 10
DETECTORS = [("detected", data.ICH_SEG_LABEL), ("clf_detected", data.ICH_CLF_LABEL)]


def _trend(g, outcome):
    """Within-subtype volume trend: logistic detection ~ log(volume). OR per log-mL + p."""
    gg = g[g.volume_ml > 0]
    y = gg[outcome].astype(int).values
    if len(gg) < 20 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    X = sm.add_constant(np.log(gg.volume_ml.values))
    try:
        r = sm.Logit(y, X).fit(disp=0)
        return float(np.exp(r.params[1])), float(r.pvalues[1])
    except Exception:
        return np.nan, np.nan


def fig6b():
    BOOT.CONTEXT["figure"] = "Fig6b"
    d = data.load_ich()
    single = d[(d.n_subtypes == 1) & (d.volume_ml > 0)].copy()
    single["vb"] = pd.cut(single.volume_ml, VBINS, labels=VLAB, right=False)
    vrows, trows = [], []
    for outcome, mlab in DETECTORS:
        subs, praw = [], []
        for s in SUB_ORDER:
            g = single[single.subtype == s]
            p = BOOT.gee_trend_p(g, outcome, "vb", VLAB, "patient_id", min_n=MIN_N, label=f"{mlab}|{s}")
            subs.append(s); praw.append(p)
            orr, tp = _trend(g, outcome)
            trows.append({"model": mlab, "subtype": s, "trend_OR_per_logmL": orr, "trend_p": tp})
        ok = [i for i, p in enumerate(praw) if not np.isnan(p)]
        q = _bh_fdr([praw[i] for i in ok])
        qmap = {ok[j]: q[j] for j in range(len(ok))}
        for i, s in enumerate(subs):
            vrows.append({"model": mlab, "subtype": s, "gee_trend_p": praw[i], "q_fdr": qmap.get(i, np.nan)})
    big = single[(single.volume_ml > 30) & (single.volume_ml <= 50)]
    frows = []
    for outcome, mlab in DETECTORS:
        e = big[big.subtype.isin(EXTRA)]; i = big[big.subtype.isin(INTRA)]
        ek, en = int(e[outcome].sum()), len(e); ik, in_ = int(i[outcome].sum()), len(i)
        p = BOOT.gee_group_p(e, i, outcome, "patient_id", label=f"{mlab}|extra vs intra 30-50 mL")
        frows.append({"model": mlab, "extra_det": ek / en, "intra_det": ik / in_, "gee_p": p})
    BOOT.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(vrows).to_csv(BOOT.OUT_DIR / "ich_volume_subtype_catrend.csv", index=False)
    pd.DataFrame(trows).to_csv(BOOT.OUT_DIR / "ich_volume_subtype_trend.csv", index=False)
    pd.DataFrame(frows).to_csv(BOOT.OUT_DIR / "ich_volume_subtype_extraVintra.csv", index=False)
    print("=== Fig 6b: within-subtype volume trend (logistic GEE on bin score), BH-FDR ===")
    print(pd.DataFrame(vrows).to_string(index=False))
    print("\n=== Fig 6b: extra- vs intra-axial at 30-50 mL ===")
    print(pd.DataFrame(frows).to_string(index=False))
    # bars (same order as the figure: subtypes sorted by ICH-Detector-1 detection)
    sub_order = SUB_ORDER
    if not data.ICH_SEG_SCOPED:
        seg_det = single.groupby("subtype")["detected"].mean()
        sub_order = sorted(SUB_ORDER, key=lambda s: seg_det.get(s, 0.0))
    for outcome, _ in DETECTORS:
        for s in sub_order:
            g = single[single.subtype == s]
            for vl in VLAB:
                gg = g[g.vb == vl]
                if len(gg) < MIN_N:
                    continue
                BOOT.ci(gg, outcome, "patient_id", label=f"{outcome}|{s}|{vl}")


# ------------------------------------------------------------------ Fig 7a (external PE)
REGIMES_PE = [("Most reliable", lambda v: v >= 2.5), ("Least reliable", lambda v: v < 0.5)]
LESIONS = {"FUMPE": RES / "external_validation/fumpe/fumpe_lesions.csv",
           "CAD-PE": RES / "external_validation/cadpe/cadpe_lesions.csv",
           "READ": RES / "external_validation/read/read_lesions.csv"}
SOURCES = ["Synthetic", "FUMPE", "CAD-PE", "READ"]


def fig7a():
    BOOT.CONTEXT["figure"] = "Fig7a"
    for model, _ in PE_MODELS:
        for lab, pred in REGIMES_PE:
            for source in SOURCES:
                if source == "Synthetic":
                    d = data.load_pe_seg(model); d = d[d.volume_ml > 0]
                    m = pred(d.volume_ml.to_numpy()); det = d.detected.to_numpy()[m]
                    cl = d.patient_id.to_numpy()[m]                       # cluster = source patient
                else:
                    les = pd.read_csv(LESIONS[source]); les = les[les.model == model]
                    m = pred(les.vol_ml.to_numpy()); det = les.det_m5.to_numpy()[m]
                    cl = les.case.to_numpy()[m]                           # cluster = scan
                BOOT.ci_arrays(det, cl, label=f"{model}|{lab}|{source}")


# ------------------------------------------------------------------ Fig 7b (PhysioNet ICH)
REGIMES_ICH = [("< 5 mL", lambda v: v < 5),
               ("5–30 mL", lambda v: (v >= 5) & (v < 30)),
               ("≥ 30 mL", lambda v: v >= 30)]
PHYSIONET = RES / "external_validation/physionet/physionet_detection.csv"
REAL_SEG_COL = {"viola": "detected_viola_coloc", "totalseg": "detected_ts_coloc"}[data.ICH_SEG]


def _rate(vol, det, pred, cl, tag):
    m = pred(np.asarray(vol, float)); n = int(m.sum())
    y = np.asarray(det, float)[m]
    c = np.asarray(cl)[m]
    lab = [r[0] for r in REGIMES_ICH if r[1] is pred][0]
    if n:
        BOOT.ci_arrays(y, c, label=f"{tag}|{lab}")


def fig7b():
    BOOT.CONTEXT["figure"] = "Fig7b"
    syn = data.load_ich(); syn = syn[syn.volume_ml > 0].copy()
    syn["seg"] = syn.detected
    syn["emm"] = syn.emm_prob >= data.ICH_CLF_THRESH
    real = pd.read_csv(PHYSIONET)                  # all 36 scans (primary cohort of Fig. 7b)
    for sc, rc, title in [("seg", REAL_SEG_COL, data.ICH_SEG_LABEL),
                          ("emm", "detected_emm", data.ICH_CLF_LABEL)]:
        for _lab, pred in REGIMES_ICH:
            _rate(syn.volume_ml, syn[sc], pred, syn.patient_id, tag=f"{title}|synthetic")
            rmask = real[rc].notna()
            _rate(real.gt_volume_ml[rmask], real[rc][rmask].astype(bool), pred, real.pid[rmask],
                  tag=f"{title}|real")


def external_counts():
    rows = []
    for name, csv in LESIONS.items():
        les = pd.read_csv(csv)
        for model in sorted(les.model.unique()):
            g = les[les.model == model]
            per_scan = g.groupby("case").size()
            rows.append({"dataset": name, "model": model, "scans": g.case.nunique(),
                         "emboli": len(g), "emboli_per_scan_median": float(per_scan.median()),
                         "emboli_per_scan_max": int(per_scan.max())})
    real = pd.read_csv(PHYSIONET)
    rows.append({"dataset": "PhysioNet CT-ICH", "model": "", "scans": real.pid.nunique(),
                 "emboli": np.nan, "emboli_per_scan_median": np.nan, "emboli_per_scan_max": np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(BOOT.OUT_DIR / "external_counts.csv", index=False)
    print("\n=== External datasets: scans and annotated emboli ===")
    print(t.to_string(index=False))


def main():
    BOOT.OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig4a()
    fig4b()
    fig6b()
    fig7a()
    fig7b()
    BOOT.write_logs()
    ci = pd.DataFrame(BOOT.LOG_CI).round(2)
    pd.set_option("display.width", 220)
    print("\n=== Bars (k/n, rate %, patient/scan bootstrap 95% CI) ===")
    print(ci[["figure", "label", "k", "n", "n_clusters", "rate_pct", "boot_lo", "boot_hi"]]
          .to_string(index=False))
    external_counts()
    print(f"\nwrote {BOOT.OUT_DIR}")


if __name__ == "__main__":
    main()
