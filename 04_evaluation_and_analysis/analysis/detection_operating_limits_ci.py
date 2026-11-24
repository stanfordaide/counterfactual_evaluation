"""Per-model cumulative-90% volume thresholds (and Clopper-Pearson 95% CIs) for the
"Detection operating limits across lesion properties" paragraph.

The paper reports the cumulative-90% thresholds from this script (PE about 1.6 mL for both
detectors; ICH 43 mL for ICH-Detector-1 and 50 mL for ICH-Detector-2). The detection-rate
CIs in the paper are the patient-bootstrap CIs of patient_bootstrap/text_numbers.py; the
Clopper-Pearson CIs printed here are a reference only.

Definitions (identical to the Fig. 4 figure code):
  * sensitivity in a stratum = detected.mean() over the counterfactuals in it.
  * CUMULATIVE sensitivity at volume v = detection rate among ALL lesions with
    volume_ml >= v (a right-tail / survival sensitivity, NOT the per-bin rate).
  * cumulative-90% volume threshold = smallest v with >= MIN_N (20) lesions in the
    tail such that cumulative sensitivity >= 90%. This is the "reached 90% only for
    lesions larger than ~v" number.

Detectors: PE nnUNet3D-ResEncXL + APES (data.load_pe_seg); ICH Viola-UNet
(ICH_SEG=viola, `detected`) + 3D ResNet-101 (`clf_detected`, emm_prob>=0.5).
Cohorts are the frozen ones (PE: cohort/final_cohort.csv; ICH: volume_ml > 0).

Re-run:
    python analysis/detection_operating_limits_ci.py

Output (output/, gitignored): detection_operating_limits_ci.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ICH_SEG", "viola")   # ICH-Detector-1 = Viola-UNet

import numpy as np
import pandas as pd
from scipy.stats import beta

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_data" / "lib"))
import data  # noqa: E402

OUTDIR = Path(__file__).resolve().parents[1] / "output"
MIN_N = 20


def cp(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return 100 * lo, 100 * hi


def cum_threshold(vol: np.ndarray, det: np.ndarray, target: float = 0.90) -> float:
    """Smallest v with >= MIN_N lesions >= v and cumulative sensitivity >= target."""
    order = np.argsort(vol)
    v_s, d_s = vol[order], det[order].astype(float)
    for i, v in enumerate(v_s):
        tail = d_s[i:]
        if len(tail) >= MIN_N and tail.mean() >= target:
            return float(v)
    return float("nan")


rows = []


def emit(kind, group, stratum, mask, det, label):
    k, n = int(det[mask].sum()), int(mask.sum())
    if n == 0:
        return
    pct = 100 * k / n
    lo, hi = cp(k, n)
    rows.append({"kind": kind, "model": label, "stratum": f"{group}:{stratum}",
                 "k": k, "n": n, "pct": round(pct, 1),
                 "ci_low": round(lo, 1), "ci_high": round(hi, 1)})
    print(f"  {label:18s} {group}:{stratum:16s} {k:5d}/{n:<5d} {pct:5.1f}%  "
          f"[{lo:4.1f}-{hi:4.1f}]")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # =============================== PE ================================
    print("\n================= PE =================")
    for m, label in [("nnunet", "PE-Detector-1"), ("apes", "PE-Detector-2")]:
        d = data.load_pe_seg(m)
        d = d[d.volume_ml > 0]
        det = d.detected.values.astype(bool)
        vol = d.volume_ml.values
        print(f"\n{label} ({data.__dict__.get('_', '')}n={len(d)})")
        emit("pe", "overall", "all", np.ones(len(d), bool), det, label)
        # volume extremes quoted in the paragraph
        emit("pe", "volume", "<0.5mL", vol < 0.5, det, label)
        emit("pe", "volume", ">5mL", vol > 5, det, label)
        # location
        for loc in ["central", "lobar", "segmental", "subsegmental"]:
            emit("pe", "location", loc, (d.location == loc).values, det, label)
        # severity
        for sev in ["mild", "moderate", "severe"]:
            emit("pe", "severity", sev, (d.severity_category == sev).values, det, label)
        # per-model cumulative-90% threshold
        t90 = cum_threshold(vol, det, 0.90)
        t80, t95 = cum_threshold(vol, det, 0.80), cum_threshold(vol, det, 0.95)
        print(f"  --> cumulative-90% threshold = {t90:.2f} mL "
              f"(80%: {t80:.2f} mL, 95%: {t95:.2f} mL)")
        rows.append({"kind": "pe", "model": label, "stratum": "cum_threshold_p90_mL",
                     "k": "", "n": len(d), "pct": round(t90, 2),
                     "ci_low": round(t80, 2), "ci_high": round(t95, 2)})

    # =============================== ICH ===============================
    print("\n================= ICH =================")
    d = data.load_ich()
    d = d[d.volume_ml > 0].copy()
    vol = d.volume_ml.values
    DETS = [("detected", "ICH-Detector-1"), ("clf_detected", "ICH-Detector-2")]
    for col, label in DETS:
        det = d[col].values.astype(bool)
        print(f"\n{label} (n={len(d)})")
        emit("ich", "overall", "all", np.ones(len(d), bool), det, label)
        for sub in ["EDH", "SDH", "SAH", "IPH", "IVH"]:
            emit("ich", "subtype", sub, (d.subtype == sub).values, det, label)
        emit("ich", "volume", "<5mL", vol < 5, det, label)
        emit("ich", "volume", ">50mL", vol > 50, det, label)
        for k_sub in [1, 2, 3, 4]:
            emit("ich", "n_subtypes", str(k_sub), (d.n_subtypes == k_sub).values, det, label)
        t90 = cum_threshold(vol, det, 0.90)
        t80, t95 = cum_threshold(vol, det, 0.80), cum_threshold(vol, det, 0.95)
        print(f"  --> cumulative-90% threshold = {t90:.1f} mL "
              f"(80%: {t80:.1f} mL, 95%: {t95:.1f} mL)")
        rows.append({"kind": "ich", "model": label, "stratum": "cum_threshold_p90_mL",
                     "k": "", "n": len(d), "pct": round(t90, 1),
                     "ci_low": round(t80, 1), "ci_high": round(t95, 1)})

    pd.DataFrame(rows).to_csv(OUTDIR / "detection_operating_limits_ci.csv", index=False)
    print(f"\nWrote {OUTDIR / 'detection_operating_limits_ci.csv'}")


if __name__ == "__main__":
    main()
