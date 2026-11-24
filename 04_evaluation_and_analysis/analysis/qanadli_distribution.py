"""Qanadli obstruction-index distribution across the INCLUDED PE counterfactuals.

Distribution of the per-configuration Qanadli scores (Supplementary Table 1), on the
EXACT analysis cohort: the frozen PE counterfactuals
in cohort/final_cohort.csv - the same set every PE figure /
table runs on (built + documented by analysis_data/make_final_cohort.py). The
Qanadli score is the per-configuration arterial-obstruction index carried through
generation (0-40 points; percent = score / 40 x 100).

Traceability: reads ONLY the frozen cohort CSV, so the number cannot drift from
the rest of the paper. Re-run to regenerate:

    python analysis/qanadli_distribution.py

Output (output/, gitignored):
  qanadli_distribution.csv   per-score and per-band counts + percentages
"""
from __future__ import annotations
import os

from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
COHORT = REPO / "cohort/final_cohort.csv"
OUTDIR = Path(__file__).resolve().parents[1] / "output"
QANADLI_MAX = 40  # full Qanadli obstruction index (points); percent = score/40*100

# Interpretable raw-score bands (points). Qanadli % in brackets for the caption.
BANDS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 40)]
BAND_LABELS = ["1-5", "6-10", "11-15", "16-20", "21-40"]


def _pct(x: float) -> float:
    return x / QANADLI_MAX * 100.0


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(COHORT)
    q = df["qanadli_score"].dropna()
    n, n_pat = len(q), df["patient_id"].nunique()
    per_pat = df.groupby("patient_id").size()

    # ---- summary statistics -------------------------------------------------
    stats = {
        "n_configs": n,
        "n_patients": n_pat,
        "configs_per_patient_mean": per_pat.mean(),
        "configs_per_patient_median": per_pat.median(),
        "configs_per_patient_min": per_pat.min(),
        "configs_per_patient_max": per_pat.max(),
        "qanadli_mean": q.mean(),
        "qanadli_sd": q.std(),
        "qanadli_median": q.median(),
        "qanadli_q1": q.quantile(0.25),
        "qanadli_q3": q.quantile(0.75),
        "qanadli_min": q.min(),
        "qanadli_max": q.max(),
        "qanadli_pct_mean": _pct(q.mean()),
        "qanadli_pct_median": _pct(q.median()),
        "qanadli_pct_max": _pct(q.max()),
    }

    # ---- per-band distribution ---------------------------------------------
    rows = []
    for (lo, hi), lab in zip(BANDS, BAND_LABELS):
        m = (q >= lo) & (q <= hi)
        rows.append({
            "band_points": lab,
            "band_percent": f"{_pct(lo):.0f}-{_pct(hi):.0f}%",
            "n": int(m.sum()),
            "pct_of_cohort": 100.0 * m.sum() / n,
        })
    band_tbl = pd.DataFrame(rows)

    # ---- per-score counts (full traceable table) ---------------------------
    score_tbl = (q.value_counts().sort_index().rename_axis("qanadli_score")
                 .reset_index(name="n"))
    score_tbl["pct_of_cohort"] = 100.0 * score_tbl["n"] / n
    score_tbl["qanadli_percent"] = score_tbl["qanadli_score"].map(_pct)

    # persist a single tidy CSV (bands then per-score, tagged)
    band_out = band_tbl.rename(columns={"band_points": "level"})
    band_out.insert(0, "kind", "band")
    score_out = score_tbl.rename(columns={"qanadli_score": "level"})
    score_out.insert(0, "kind", "score")
    combined = pd.concat([band_out, score_out], ignore_index=True)
    combined.to_csv(OUTDIR / "qanadli_distribution.csv", index=False)

    # ---- report -------------------------------------------------------------
    print(f"Frozen PE cohort: {n:,} counterfactuals / {n_pat} patients "
          f"({stats['configs_per_patient_mean']:.1f} configs/patient, "
          f"range {stats['configs_per_patient_min']}-{stats['configs_per_patient_max']})")
    print(f"Qanadli score: mean {stats['qanadli_mean']:.1f} +/- {stats['qanadli_sd']:.1f}, "
          f"median {stats['qanadli_median']:.0f} "
          f"(IQR {stats['qanadli_q1']:.0f}-{stats['qanadli_q3']:.0f}), "
          f"range {stats['qanadli_min']:.0f}-{stats['qanadli_max']:.0f} points")
    print(f"As % obstruction: median {stats['qanadli_pct_median']:.0f}%, "
          f"max {stats['qanadli_pct_max']:.0f}%")
    print("\nPer-band distribution:")
    print(band_tbl.to_string(index=False,
          formatters={"pct_of_cohort": lambda v: f"{v:5.1f}%"}))
    print(f"\nWrote {OUTDIR/'qanadli_distribution.csv'}")


if __name__ == "__main__":
    main()
