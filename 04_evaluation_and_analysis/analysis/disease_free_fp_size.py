"""PE detectors on disease-free scans: specificity and size of false-positive predictions.

Extends section (2) of reader_confirmed_and_disease_free.py (Extended Data Table 2) with the
same inputs: the frozen cohort (cohort/final_cohort.csv), the disease-free synthetic
scan of each patient, and the real disease-free CTPA of each patient (PE-Detector-2 output for 9 of
these comes from a separate run). The disease-free scans are read through
analysis/disease_free_predictions.py (predicted masks under CF_RAW if present, else the
per-scan and per-region tables in CF_DATA/detector_outputs/).

Per detector and scan type:
  - specificity: % of disease-free scans with no predicted embolus voxel (= 100 - "any prediction")
  - % of scans with total predicted volume < 0.5 mL
  - in scans with a false positive: mean, median, IQR and maximum of the total predicted volume,
    and % of scans below 0.5 mL
  - per predicted region (3D connected component): number, median volume, % below 0.5 mL
95% CIs: percentile bootstrap over scans (B = 2,000, seed 0).

For the PE counterfactual rows, only the median predicted volume in scans with a prediction is
given (bootstrap over patients), for the same Extended Data Table 2 column.

Writes output/pe_disease_free_fp_size.csv.
Run:  python analysis/disease_free_fp_size.py
"""
import os
import sys
from pathlib import Path

os.environ["ICH_SEG"] = "viola"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_data" / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import disease_free_predictions as DF  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)
B = 2000

nn = data.load_pe_seg("nnunet")[["config_id", "patient_id"]]
ap = data.load_pe_seg("apes")[["config_id"]]
pe_all = nn.merge(ap, on="config_id", how="inner")
assert set(pe_all.config_id) == data.frozen_configs(), len(pe_all)   # both tables hold the whole cohort
pids = sorted(pe_all.patient_id.unique())

neg_scans, neg_regions = DF.load(pids)     # per-scan and per-region tables, original scan order

def boot(x, fn):
    """Percentile bootstrap over scans; same resampling as summarize() in
    reader_confirmed_and_disease_free.py (fresh seed 0 per call), so specificity is exactly
    100 minus its "any prediction" estimate and CI."""
    x = np.asarray(x, float)
    idx = np.random.default_rng(0).integers(0, len(x), (B, len(x)))
    bs = [fn(x[i]) for i in idx]
    return fn(x), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(t, d=1):
    return f"{t[0]:.{d}f} ({t[1]:.{d}f}-{t[2]:.{d}f})"


rows = []
for (model, cond), g in neg_scans.groupby(["detector", "scans"], sort=False):
    # same arithmetic as the original mask read: voxel_ml = |det(affine[:3, :3])| / 1000,
    # total = mask.sum() * voxel_ml, regions = connected-component sizes * voxel_ml
    vols, regions = [], []
    for r in g.itertuples():
        voxel_ml = r.voxel_mm3 / 1000.0
        vols.append(np.int64(r.n_voxels) * voxel_ml)
        rv = neg_regions.loc[(neg_regions.detector == model) & (neg_regions.scans == cond)
                             & (neg_regions.patient_id == r.patient_id), "region_voxels"].to_numpy()
        regions += list(rv * voxel_ml)
    v, regions = np.array(vols), np.array(regions)
    fp = v[v > 0]
    rows.append({
        "detector": model, "scans": cond, "n_scans": len(v), "n_scans_fp": len(fp),
        "specificity_pct": fmt(boot(v, lambda a: 100 * (a == 0).mean())),
        "total_lt05_pct": fmt(boot(v, lambda a: 100 * (a < 0.5).mean())),
        "fp_mean_ml": fmt(boot(fp, np.mean), 2),
        "fp_median_ml": fmt(boot(fp, np.median), 3),
        "fp_iqr_ml": f"{np.percentile(fp, 25):.3f}-{np.percentile(fp, 75):.3f}",
        "fp_max_ml": round(float(fp.max()), 2),
        "fp_lt05_pct": round(100 * (fp < 0.5).mean(), 1),
        "n_regions": len(regions),
        "region_median_ml": round(float(np.median(regions)), 3),
        "region_lt05_pct": round(100 * (regions < 0.5).mean(), 1),
    })
    print(f"{model} {cond}: {len(v)} scans", flush=True)

# PE counterfactuals: median predicted volume in scans with a prediction (for the same table
# column), bootstrap over patients. Specificity does not apply.
rng_cf = np.random.default_rng(0)
for model, csv in [("PE-Detector-1", REPO / "cohort/evaluation_all_configs.csv"),
                   ("PE-Detector-2", REPO / "cohort/evaluation_all_configs_apes.csv")]:
    e = pd.read_csv(csv).drop_duplicates("config_id")
    e = e[e.config_id.isin(pe_all.config_id)].dropna(subset=["detected_volume_ml"])
    e = e[e.detected_volume_ml > 0].assign(patient_id=lambda d: d.config_id.str.split("_cf").str[0])
    groups = [g.detected_volume_ml.to_numpy() for _, g in e.groupby("patient_id")]
    bs = [np.median(np.concatenate([groups[j] for j in rng_cf.integers(0, len(groups), len(groups))]))
          for _ in range(B)]
    rows.append({
        "detector": model, "scans": "Synthetic, PE counterfactuals", "n_scans_fp": len(e),
        "fp_median_ml": f"{np.median(e.detected_volume_ml):.3f} "
                        f"({np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f})",
        "fp_iqr_ml": f"{e.detected_volume_ml.quantile(.25):.3f}-{e.detected_volume_ml.quantile(.75):.3f}",
    })

t = pd.DataFrame(rows)
t.to_csv(OUT / "pe_disease_free_fp_size.csv", index=False)
print("\n", t.T.to_string())
