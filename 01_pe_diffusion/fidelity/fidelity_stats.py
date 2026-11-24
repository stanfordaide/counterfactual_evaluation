"""Fidelity statistics for the synthetic-CT ablation table (PE): is the final configuration
(text + demographics + segmentation mask) significantly better than each ablation?

CLIPScore test
--------------
CLIPScore is per-image (BiomedCLIP similarity between each generated image and its conditioning
impression), so it supports a PAIRED test: for every impression, each config produces a score, and
we compare the final config to each ablation on the SAME images. We use a paired Wilcoxon
signed-rank test (rank-based; does not assume normality), Holm-corrected across the three
comparisons, with a paired t-test as a parametric cross-check.

All statistics are computed on the COMMON image set present in ALL configs (a paired test must
score every config on the same images). The final config is the v2_clean model actually used to
generate the counterfactuals.

This script also writes `fidelity_common_image_ids.csv` - the overlapping image_ids - so the FID
score can be recomputed on the SAME overlapping set for every config (real reference unchanged).

Run:  python fidelity/fidelity_stats.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = str(Path(__file__).resolve().parents[1])  # the 01_pe_diffusion folder
# config -> per-image CLIPScore CSV (columns: image_id, clip_score[, error])
CONFIGS = {
    "text": f"{BASE}/fidelity/clipscore_text.csv",
    "text+demo": f"{BASE}/fidelity/clipscore_text_demo.csv",
    "text+mask": f"{BASE}/fidelity/clipscore_segmentation_text.csv",
    # FINAL = the v2_clean controlnet model actually used to generate the counterfactuals
    "text+demo+mask": f"{BASE}/biomedclip_scores_controlnet_best_demo_pe_all_v2_clean/clipscore_merged.csv",
}
FINAL = "text+demo+mask"
OUT_DIR = f"{BASE}/fidelity"
# QC manifest of the deployed config: image_ids whose generation passed QC (artifact-free).
QC_MANIFEST = os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "pe_training_data_all",
                           "predictions_controlnet_best_demo_pe_all_v2_clean/controlnet_predictions_qc.csv")


def load() -> dict:
    """image_id -> mean clip_score per config (dedup duplicate image_ids by mean)."""
    return {k: pd.read_csv(f).dropna(subset=["clip_score"]).groupby("image_id")["clip_score"].mean()
            for k, f in CONFIGS.items()}


def main() -> None:
    d = load()
    common = sorted(set.intersection(*[set(s.index) for s in d.values()]))
    # QC exclusion: keep only generations that passed QC (present in the deployed
    # config's QC manifest), so fidelity is scored on artifact-free, kept images only.
    qc_ids = set(pd.read_csv(QC_MANIFEST)["image_id"])
    n0 = len(common)
    common = [i for i in common if i in qc_ids]
    print(f"QC filter: kept {len(common)}/{n0} common images (dropped {n0 - len(common)} non-QC)")
    pd.Series(common, name="image_id").to_csv(f"{OUT_DIR}/fidelity_common_image_ids.csv", index=False)

    # ---- table: mean +/- std (sample, ddof=1) on the common overlapping set ----
    print(f"CLIPScore on the OVERLAPPING set (n={len(common)} images present in all {len(CONFIGS)} configs)\n")
    tbl = []
    for k in CONFIGS:
        v = d[k].reindex(common).values
        m, sd, nn = float(v.mean()), float(v.std(ddof=1)), len(v)
        se = sd / (nn ** 0.5)
        tbl.append({"config": k, "n": nn, "clipscore_mean": round(m, 3),
                    "clipscore_std": round(sd, 3),
                    "clip_ci_lo": round(m - 1.96 * se, 3), "clip_ci_hi": round(m + 1.96 * se, 3)})
    tbl = pd.DataFrame(tbl)
    print(tbl.to_string(index=False))
    tbl.to_csv(f"{OUT_DIR}/fidelity_clip_table.csv", index=False)

    # ---- paired Wilcoxon (final vs each) + Holm + paired t-test ----
    final = d[FINAL].reindex(common).values
    rows = []
    for k in [c for c in CONFIGS if c != FINAL]:
        o = d[k].reindex(common).values
        diff = final - o
        w = stats.wilcoxon(final, o)
        t = stats.ttest_rel(final, o)
        rows.append({"comparison": f"{FINAL} vs {k}", "delta_mean": round(float(diff.mean()), 3),
                     "median_delta": round(float(np.median(diff)), 3),
                     "pct_final_higher": round(100 * float((diff > 0).mean()), 1),
                     "wilcoxon_p": w.pvalue, "ttest_p": t.pvalue})
    R = pd.DataFrame(rows)
    R["holm_p"] = multipletests(R.wilcoxon_p, method="holm")[1]
    R = R[["comparison", "delta_mean", "median_delta", "pct_final_higher", "wilcoxon_p", "holm_p", "ttest_p"]]
    print(f"\nPaired Wilcoxon signed-rank (final = {FINAL}), Holm-corrected over 3 comparisons:")
    print(R.to_string(index=False))
    R.to_csv(f"{OUT_DIR}/fidelity_clip_test.csv", index=False)
    print(f"\nwrote fidelity_clip_table.csv, fidelity_clip_test.csv, fidelity_common_image_ids.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
