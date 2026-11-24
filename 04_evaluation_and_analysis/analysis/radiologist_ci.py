"""Clopper-Pearson (exact binomial) 95% CIs for the radiologist-assessment paragraph.

The paper uses these exact intervals for the per-patient within-patient resemblance
(one observation per patient). The other proportions in the paragraph are reported with
the patient-bootstrap CIs of patient_bootstrap/text_numbers.py; their Clopper-Pearson
intervals here are a reference only.

Every proportion below is the exact count stated in the manuscript paragraph
(two radiologists, 430 disease-positive counterfactuals: PE n=190, ICH n=240;
within-patient resemblance is per-patient, 30 PE + 30 ICH). This script recomputes
each percentage and its exact (Clopper-Pearson / Beta) two-sided 95% CI so the
numbers are reproducible and traceable.

The k/N counts are hard-coded on purpose: they are the frozen review results the
paragraph reports (radiologist ratings are manual, not regenerated from a pipeline).
They can be checked against the rating CSVs:
  PE  ratings : radiologist/pe/ratings_per_scan.csv, radiologist/pe/ratings_per_patient.csv
  ICH ratings : radiologist/ich/ratings_per_scan.csv, radiologist/ich/ratings_per_patient.csv
"acceptable" = score >= 1 on the 0/1/2 quality & realism scales (ICH realism is
0-3; >= 1 is still "acceptable"). "strong-near-identical resemblance" = Q2 within-
patient consistency score >= 4 on the 1-5 scale.

Re-run:
    python analysis/radiologist_ci.py

Output (output/, gitignored): radiologist_ci.csv
"""
from __future__ import annotations
import os

from pathlib import Path

import pandas as pd
from scipy.stats import beta

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
OUTDIR = Path(__file__).resolve().parents[1] / "output"

# (label, group, metric, k successes, N total) - exactly as reported in the paragraph.
ROWS = [
    ("Anatomical quality acceptable (score>=1)", "PE",  "anat_quality",   177, 190),
    ("Anatomical quality acceptable (score>=1)", "ICH", "anat_quality",   236, 240),
    ("Overall realism acceptable (score>=1)",    "PE",  "overall_realism",173, 190),
    ("Overall realism acceptable (score>=1)",    "ICH", "overall_realism",231, 240),
    ("Radiologist-confirmed disease (blinded)",  "PE",  "reader_detect",  119, 190),
    ("Radiologist-confirmed disease (blinded)",  "ICH", "reader_detect",  177, 240),
    ("Location/subtype match among confirmed",   "PE",  "impression_match", 77, 119),
    ("Location/subtype match among confirmed",   "ICH", "impression_match",148, 174),
    ("Strong-near-identical resemblance (Q2>=4)","PE",  "resemblance",     22,  30),
    ("Strong-near-identical resemblance (Q2>=4)","ICH", "resemblance",     29,  30),
]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact binomial (Clopper-Pearson) CI, in percent."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return 100.0 * lo, 100.0 * hi


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = []
    print(f"{'metric':44s} {'grp':4s} {'k/N':>8s} {'%':>6s}  95% CI (Clopper-Pearson)")
    for label, grp, metric, k, n in ROWS:
        pct = 100.0 * k / n
        lo, hi = clopper_pearson(k, n)
        out.append({"metric": metric, "label": label, "group": grp,
                    "k": k, "n": n, "pct": round(pct, 1),
                    "ci_low": round(lo, 1), "ci_high": round(hi, 1)})
        print(f"{label:44s} {grp:4s} {k:3d}/{n:<4d} {pct:5.1f}  ({lo:.1f}-{hi:.1f})")
    df = pd.DataFrame(out)
    df.to_csv(OUTDIR / "radiologist_ci.csv", index=False)
    print(f"\nWrote {OUTDIR / 'radiologist_ci.csv'}")


if __name__ == "__main__":
    main()
