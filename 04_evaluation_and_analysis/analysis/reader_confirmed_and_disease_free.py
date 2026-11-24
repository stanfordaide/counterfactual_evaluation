"""Extended Data Tables 2 and 3 and the LLM concordance grades.

(1) Detection on radiologist-confirmed counterfactuals. For each disease: detection rate of both
    detectors on (a) the full cohort, (b) all counterfactuals in the radiologist review, and
    (c) the reviewed counterfactuals in which the radiologist marked the intended lesion present
    (q4 = 1). 95% CIs: percentile bootstrap over patients (B = 2,000, seed 0); the full-cohort rows
    use the same patient bootstrap.
(2) PE detector output on disease-free scans, restricted to the frozen analysis cohort
    (cohort/final_cohort.csv): predicted embolus volume per scan on PE counterfactuals,
    disease-free synthetic scans and real disease-free scans of the same patients. "Any prediction"
    = at least one predicted embolus voxel in the scan. 95% CIs: percentile bootstrap (B = 2,000,
    seed 0) over scans for the negatives and over patients for the counterfactuals.
(3) Distribution of the LLM concordance grades (0 none / 1 partial / 2 full).

Detection = the manuscript criterion (PE: +-5-slice overlap; ICH: Viola-UNet co-localization,
3D ResNet-101 probability >= 0.5). Read-only, deterministic. Run from 04_evaluation_and_analysis/:
    python analysis/reader_confirmed_and_disease_free.py
Writes output/reader_confirmed_detection.csv, pe_disease_free_scans.csv,
concordance_grades.csv. The disease-free scans are read through
analysis/disease_free_predictions.py (predicted masks under CF_RAW if present, else the
per-scan table in CF_DATA/detector_outputs/).
"""
import os
import sys
from pathlib import Path

os.environ["ICH_SEG"] = "viola"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_data" / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import disease_free_predictions as DF  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)
B = 2000


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def cluster_boot(df, col, seed=0):
    """Detection rate with a percentile bootstrap over patients."""
    rng = np.random.default_rng(seed)
    g = df.groupby("patient_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = rng.integers(0, len(g), (B, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return 100 * s.sum() / c.sum(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def row(disease, subset, df, cols):
    r = {"disease": disease, "subset": subset, "n_scans": len(df), "n_patients": df.patient_id.nunique()}
    for name, col in cols:
        d = df.dropna(subset=[col])
        assert len(d) == len(df), f"{disease}/{subset}: {len(df) - len(d)} rows without detection"
        p, lo, hi = cluster_boot(d.assign(**{col: d[col].astype(float)}), col)
        r[f"{name}_pct"], r[f"{name}_ci"] = round(p, 1), f"{lo:.1f}-{hi:.1f}"
    return r


rows = []
# ------------------------------------------------------------------ PE
pe_rat_all = pd.read_csv(REPO / "radiologist/pe/ratings_per_scan.csv")
man = pd.read_csv(REPO / "radiologist/pe/manifest.csv")
blind = pd.read_csv(REPO / "radiologist/pe/blinding_map.csv")   # baselines under display ids
pe_rat = pe_rat_all[pe_rat_all.scan_id.isin(man.scan_id)].copy()          # the 190 counterfactuals
pe_base = pe_rat_all.merge(blind[["display_scan_id", "scan_type"]], left_on="scan_id", right_on="display_scan_id")
assert len(pe_rat) == 190 and len(pe_base) == 60
pe_rat["config_id"] = pe_rat["scan_id"]
nn = data.load_pe_seg("nnunet")[["config_id", "patient_id", "detected"]].rename(columns={"detected": "d1"})
ap = data.load_pe_seg("apes")[["config_id", "detected"]].rename(columns={"detected": "d2"})
pe_all = nn.merge(ap, on="config_id", how="inner")
assert set(pe_all.config_id) == data.frozen_configs(), len(pe_all)   # both tables hold the whole cohort
pe_rev = pe_rat[["config_id", "q4_pe_present"]].merge(pe_all, on="config_id", how="left")
assert len(pe_rev) == 190 and pe_rev.d1.notna().all()
pe_cols = [("PE-Detector-1", "d1"), ("PE-Detector-2", "d2")]
rows += [row("PE", "Full cohort", pe_all, pe_cols),
         row("PE", "Radiologist review, all", pe_rev, pe_cols),
         row("PE", "Radiologist review, lesion confirmed", pe_rev[pe_rev.q4_pe_present == 1], pe_cols)]

# ------------------------------------------------------------------ ICH
ich_rat = pd.read_csv(REPO / "radiologist/ich/ratings_per_scan.csv")
rd = pd.read_csv(REPO / "radiologist/ich/radiologist_data.csv")[["scan_id", "patient_id", "ich_volume_ml", "is_baseline"]]
ich = ich_rat.drop(columns=["patient_id"]).merge(rd, on="scan_id", how="left")
ich_cf = ich[ich.is_baseline == False].copy()  # noqa: E712
assert len(ich_cf) == 240
di = data.load_ich()
di = di[di.volume_ml > 0][["patient_id", "volume_ml", "detected", "clf_detected"]].copy()
assert len(di) == 2601, len(di)
di["vkey"] = di.volume_ml.round(2)
ich_cf["vkey"] = ich_cf.ich_volume_ml.round(2)
dk = di.drop_duplicates(["patient_id", "vkey"])
amb = di.groupby(["patient_id", "vkey"]).filter(lambda g: g[["detected", "clf_detected"]].nunique().max() > 1)
ich_rev = ich_cf.merge(dk[["patient_id", "vkey", "detected", "clf_detected"]], on=["patient_id", "vkey"], how="left")
n_amb = ich_rev.merge(amb[["patient_id", "vkey"]].drop_duplicates(), on=["patient_id", "vkey"]).shape[0]
print(f"ICH reviewed CFs linked by (patient, volume) whose key is ambiguous in detection: {n_amb}")
ich_rev = ich_rev.rename(columns={"detected": "d1", "clf_detected": "d2"})
assert ich_rev.d1.notna().all(), "unlinked ICH reviewed CF"
ich_all = di.rename(columns={"detected": "d1", "clf_detected": "d2"})
ich_cols = [("ICH-Detector-1", "d1"), ("ICH-Detector-2", "d2")]
rows += [row("ICH", "Full cohort", ich_all, ich_cols),
         row("ICH", "Radiologist review, all", ich_rev, ich_cols),
         row("ICH", "Radiologist review, lesion confirmed", ich_rev[ich_rev.q4_ich_present == 1], ich_cols)]
t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "reader_confirmed_detection.csv", index=False)
print(t1.to_string(index=False))

k = int((ich_cf.q4_ich_present == 1).sum())
lo, hi = cp(k, 240)
print(f"\nICH lesion confirmed: {k}/240 = {100*k/240:.1f}% (CP 95% CI {100*lo:.1f}-{100*hi:.1f})")
nodesc = ich_cf[(ich_cf.q4_ich_present == 1) & ~ich_cf.q5_described.astype(bool)]   # q5_described = q5 free text present (text not released)
print(f"ICH confirmed without a free-text diagnosis: {len(nodesc)}")
k = int((pe_rat.q4_pe_present == 1).sum())
print(f"PE lesion confirmed: {k}/190")

# ------------------------------------------------------------------ reader ratings: references
print("\nPE reader ratings by scan type (anatomical quality q1 0-2, realism q8 0-2):")
for name, d in [("counterfactual", pe_rat), *[(t, pe_base[pe_base.scan_type == t]) for t in ["real", "nope_synthetic"]]]:
    q1, q8 = d.q1_anatomical_quality, d.q8_overall_realism
    print(f"  {name:15s} n={len(d)}  q1>=1 {int((q1>=1).sum())}/{q1.notna().sum()}  q1 mean {q1.mean():.2f}+-{q1.std():.2f}"
          f"  q8>=1 {int((q8>=1).sum())}/{q8.notna().sum()}  q8==2 {int((q8==2).sum())}  q8 mean {q8.mean():.2f}+-{q8.std():.2f}")
q8 = ich_cf.q8_overall_realism
for thr in [1, 2, 3]:
    k = int((q8 >= thr).sum()); lo, hi = cp(k, int(q8.notna().sum()))
    print(f"ICH counterfactual realism >= {thr}: {k}/{q8.notna().sum()} = {100*k/q8.notna().sum():.1f}% (CP {100*lo:.1f}-{100*hi:.1f})")
k = int((pe_rat.q8_overall_realism == 2).sum()); lo, hi = cp(k, 190)
print(f"PE counterfactual realism == 2: {k}/190 = {100*k/190:.1f}% (CP {100*lo:.1f}-{100*hi:.1f})")

# ------------------------------------------------------------------ concordance grades
C = pd.read_csv(REPO / "radiologist/concordance_judged.csv")
g = C.groupby("modality")["concordance"].value_counts().unstack(fill_value=0)
g["n"] = g.sum(1)
for v in g.columns[:-1]:
    g[f"pct_{v}"] = (100 * g[v] / g["n"]).round(1)
g.to_csv(OUT / "concordance_grades.csv")
print("\nLLM concordance grades (0 none / 1 partial / 2 full):\n", g.to_string())

# ------------------------------------------------------------------ PE negatives
pids = sorted(pe_all.patient_id.unique())
neg_scans, _ = DF.load(pids)     # one row per disease-free scan, original scan order


def vol_ml(r):
    """Total predicted volume (mL) of one scan; same arithmetic as the original mask read:
    float(mask.sum()) * |det(affine[:3, :3])| / 1000."""
    return float(r.n_voxels) * r.voxel_mm3 / 1000.0


cfv = {}
for model, csv in [("PE-Detector-1", REPO / "cohort/evaluation_all_configs.csv"),
                   ("PE-Detector-2", REPO / "cohort/evaluation_all_configs_apes.csv")]:
    e = pd.read_csv(csv).drop_duplicates("config_id")
    e = e[e.config_id.isin(pe_all.config_id)][["config_id", "detected_volume_ml"]]
    e["patient_id"] = e.config_id.str.split("_cf").str[0]
    cfv[model] = e


def summarize(x, groups=None, seed=0):
    """Rates and median with percentile bootstrap; resample groups (patients) if given."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if groups is None:
        idx = rng.integers(0, len(x), (B, len(x)))
        samples = [x[i] for i in idx]
    else:
        gs = pd.Series(x).groupby(np.asarray(groups)).apply(np.asarray).tolist()
        samples = [np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))]) for _ in range(B)]
    out = {}
    for key, fn in [("any_pct", lambda a: 100 * (a > 0).mean()),
                    ("ge05_pct", lambda a: 100 * (a >= 0.5).mean()),
                    ("median_ml", np.median)]:
        bs = np.array([fn(s) for s in samples])
        out[key] = round(float(fn(x)), 3 if key == "median_ml" else 1)
        out[key + "_ci"] = (f"{np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f}" if key == "median_ml"
                            else f"{np.percentile(bs, 2.5):.1f}-{np.percentile(bs, 97.5):.1f}")
    return out


rows = []
for model in ["PE-Detector-1", "PE-Detector-2"]:
    e = cfv[model].dropna(subset=["detected_volume_ml"])
    rows.append({"detector": model, "condition": "Synthetic, PE counterfactuals", "n_scans": len(e),
                 **summarize(e.detected_volume_ml, e.patient_id)})
    for cond in ["Synthetic, disease-free", "Real, disease-free"]:
        g = neg_scans[(neg_scans.detector == model) & (neg_scans.scans == cond)]
        v = [vol_ml(r) for r in g.itertuples()]
        rows.append({"detector": model, "condition": cond, "n_scans": len(v), **summarize(v)})
        print(f"{model} {cond}: {len(v)} scans", flush=True)
t2 = pd.DataFrame(rows)
t2.to_csv(OUT / "pe_disease_free_scans.csv", index=False)
print("\n", t2.to_string(index=False))
