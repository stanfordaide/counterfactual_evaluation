"""Combine the per-config FID bootstraps into the significance table (the FID analog of the
CLIPScore test). For each ablation, delta_b = FID_b[config] - FID_b[final] is PAIRED (bootstrap b
resamples the same images for both), so the difference CI is tight. Reports:
  - fidelity_fid_table.csv : observed FID + 95% percentile CI per config (overlapping set)
  - fidelity_fid_test.csv  : final vs each ablation - observed delta, 95% CI of the paired delta,
    two-sided bootstrap p (Holm-corrected over the 3 comparisons)

Run (after the 4 workers finish):  python fidelity/fid_bootstrap_combine.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

OUT = str(Path(__file__).resolve().parent)
CFGS = ["text", "text_demo", "text_mask", "text_demo_mask"]
LABEL = {"text": "text", "text_demo": "text+demo", "text_mask": "text+mask",
         "text_demo_mask": "text+demo+mask"}
FINAL = "text_demo_mask"


def main():
    d = {c: np.load(f"{OUT}/fid_boot_{c}.npz") for c in CFGS}
    obs = {c: float(d[c]["observed"]) for c in CFGS}
    boot = {c: d[c]["boot"] for c in CFGS}
    B = len(boot[FINAL])

    tbl = []
    for c in CFGS:
        lo, hi = np.percentile(boot[c], [2.5, 97.5])
        tbl.append({"config": LABEL[c], "fid": round(obs[c], 3),
                    "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3)})
    tbl = pd.DataFrame(tbl)
    print(f"2.5D FID on the overlapping set (n images matches the CLIPScore test), B={B} bootstraps\n")
    print(tbl.to_string(index=False))
    tbl.to_csv(f"{OUT}/fidelity_fid_table.csv", index=False)

    rows = []
    for c in CFGS:
        if c == FINAL:
            continue
        delta = boot[c] - boot[FINAL]                 # paired; >0 => final better (lower FID)
        lo, hi = np.percentile(delta, [2.5, 97.5])
        p = min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))
        rows.append({"comparison": f"{LABEL[FINAL]} vs {LABEL[c]}",
                     "delta_fid_obs": round(obs[c] - obs[FINAL], 3),
                     "delta_ci_lo": round(float(lo), 3), "delta_ci_hi": round(float(hi), 3),
                     "boot_p": p, "p_resolution": round(2 / B, 4)})
    R = pd.DataFrame(rows)
    R["holm_p"] = multipletests(R.boot_p.clip(lower=1 / B), method="holm")[1]
    print(f"\nPaired bootstrap, final = {LABEL[FINAL]} (delta>0 => final has the lower/better FID):")
    print(R.to_string(index=False))
    R.to_csv(f"{OUT}/fidelity_fid_test.csv", index=False)

    print(f"\nwrote fidelity_fid_table.csv, fidelity_fid_test.csv -> {OUT}")


if __name__ == "__main__":
    main()
