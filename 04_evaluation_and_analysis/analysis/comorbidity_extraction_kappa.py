#!/usr/bin/env python
"""Co-finding (co-morbidity) extraction reliability: Cohen's kappa (Supplementary Table 2).

Reproduces, deterministically from two ALREADY-SAVED extraction passes (no LLM/API call),
the agreement between the two PE co-morbidity extractions:

    kappa ~ 0.91 overall, 1.00 for pleural effusion,
    median across modeled findings ~ 0.95, minimum ~ 0.60 (pneumonia),

and the per-finding prevalence (patients with the finding) reported in Supplementary
Table 2 and the Methods.

Data provenance (the two passes compared for kappa)
---------------------------------------------------
The kappa is computed over the SYNTHETIC PE counterfactual conditioning impressions
(one impression per source patient; each conditions that patient's ~15 PE
counterfactuals), for the patients that have both passes (pass 2 covers 147 of the 150).
Fourteen boolean thoracic findings, negation-aware:
  Pass 1  = comorbidity/comorbidity_llm.csv   (primary extraction; Methods)
  Pass 2  = comorbidity/comorbidity_pass2.csv (independent second extraction with a
            second large language model; Methods)
Both files are in the resources folder, so kappa is fully recomputable with no API calls.
The same 14 flags and the same n>=10 "modeled" subset as analysis/comorbidity_gee.py are
used, plus the pooled cell-level "overall" kappa and the co-finding prevalence.

Run (any CWD):
  python analysis/comorbidity_extraction_kappa.py
"""
from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))
OUTPUT = REPO / "comorbidity"

# 14 boolean findings + the n>=10 "modeled" threshold: identical to
# analysis/comorbidity_gee.py.
FLAGS = ["pleural_effusion", "atelectasis", "consolidation", "pneumonia",
         "pulmonary_edema", "mass_nodule", "malignancy", "lymphadenopathy",
         "emphysema_copd", "fibrosis_ild", "pneumothorax", "pericardial_effusion",
         "cardiomegaly", "ground_glass"]
MIN_N = 10

# reported values (assert against these)
T_OVERALL = 0.91
T_EFFUSION = 1.00
T_MEDIAN = 0.95
T_MIN = 0.60
TOL = 0.01


def cohen_kappa(a, b):
    """Cohen's kappa for two binary raters. Returns (kappa, pct_agree).
    """
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    po = float((a == b).mean())
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else np.nan
    return kappa, po


def per_finding_kappa():
    p1 = pd.read_csv(OUTPUT / "comorbidity_llm.csv")
    p2 = pd.read_csv(OUTPUT / "comorbidity_pass2.csv")
    m = p1[["patient_id"] + FLAGS].merge(
        p2[["patient_id"] + FLAGS], on="patient_id", suffixes=("_1", "_2"))
    rows, a_stack, b_stack = [], [], []
    for f in FLAGS:
        a = m[f"{f}_1"].astype(int)
        b = m[f"{f}_2"].astype(int)
        k, po = cohen_kappa(a, b)
        rows.append({"flag": f, "n_pass1": int(a.sum()), "n_pass2": int(b.sum()),
                     "kappa": k, "pct_agree": po, "modelled": int(a.sum()) >= MIN_N})
        a_stack.append(a.values)
        b_stack.append(b.values)
    r = pd.DataFrame(rows).sort_values("kappa", ascending=False, na_position="last")
    # pooled cell-level ("overall") kappa: every (patient x finding) cell as one obs
    pooled_k, pooled_agree = cohen_kappa(np.concatenate(a_stack), np.concatenate(b_stack))
    return p1, m, r, pooled_k, pooled_agree, len(np.concatenate(a_stack))


def pe_cofinding_prevalence(p1: pd.DataFrame):
    n_flag = p1[FLAGS].sum(axis=1)
    pat_any = float((n_flag >= 1).mean())
    cf_any = float(p1.loc[n_flag >= 1, "n_cf"].sum() / p1["n_cf"].sum())
    return pat_any, cf_any, int(p1["n_cf"].sum())


def main():
    print("=" * 80)
    print("PE co-morbidity extraction reliability (Cohen's kappa)")
    print("  pass 1 = comorbidity/comorbidity_llm.csv")
    print("  pass 2 = comorbidity/comorbidity_pass2.csv")
    print("=" * 80)

    p1, m, r, pooled_k, pooled_agree, ncells = per_finding_kappa()
    print(f"\n{len(m)} patients matched across both passes; {len(FLAGS)} boolean findings\n")
    print(f"{'finding':22s} {'n1':>3s} {'n2':>3s} {'kappa':>7s} {'%agree':>7s}  modeled")
    for _, row in r.iterrows():
        print(f"{row.flag:22s} {row.n_pass1:3d} {row.n_pass2:3d} {row.kappa:7.4f} "
              f"{row.pct_agree*100:6.2f}%  {'yes' if row.modelled else 'no'}")

    modelled = r[r.modelled]
    med_mod = float(modelled.kappa.median())
    min_mod = float(modelled.kappa.min())
    min_flag = modelled.sort_values("kappa").iloc[0]["flag"]
    eff_k = float(r.loc[r.flag == "pleural_effusion", "kappa"].iloc[0])

    print("\n" + "-" * 80)
    print("OVERALL / SUMMARY kappa")
    print("-" * 80)
    print(f"  pooled cell-level kappa (all {len(FLAGS)}x{len(m)} = {ncells} cells) = "
          f"{pooled_k:.4f}   (%agree {pooled_agree*100:.2f})   <- 'overall'")
    print(f"  pleural effusion kappa                                     = {eff_k:.4f}")
    print(f"  median kappa across MODELED findings (n>=%d)              = {med_mod:.4f}"
          % MIN_N)
    print(f"  minimum kappa across MODELED findings                     = {min_mod:.4f}"
          f"  ({min_flag})")
    print(f"  mean kappa across all {len(FLAGS)} findings                        = "
          f"{float(r.kappa.mean()):.4f}   (reference, not the headline)")

    print("\n" + "-" * 80)
    print("CO-FINDING PREVALENCE")
    print("-" * 80)
    pat_any, cf_any, tot_cf = pe_cofinding_prevalence(p1)
    print("  PE synthetic co-finding prevalence (the extraction this kappa validates):")
    print(f"    patients with >=1 of 14 findings : {pat_any*100:.1f}%  ({int((p1[FLAGS].sum(axis=1)>=1).sum())}/{len(p1)})")
    print(f"    PE counterfactuals (n_cf-weighted): {cf_any*100:.1f}%  of {tot_cf} CFs")

    print("\n" + "=" * 80)
    print("ASSERTIONS vs reported values")
    print("=" * 80)
    checks = [
        ("overall (pooled) kappa ~ 0.91", pooled_k, T_OVERALL),
        ("pleural effusion kappa = 1.00", eff_k, T_EFFUSION),
        ("median (modeled) kappa ~ 0.95", med_mod, T_MEDIAN),
        ("minimum (modeled) kappa ~ 0.60", min_mod, T_MIN),
    ]
    for label, got, target in checks:
        ok = abs(got - target) <= TOL
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:42s} got {got:.4f} (target {target:.2f}, tol {TOL})")
        assert ok, f"{label}: {got:.4f} not within {TOL} of {target}"
    assert min_flag == "pneumonia", f"min modeled kappa flag is {min_flag}, expected pneumonia"
    print("  [PASS] minimum-kappa finding is pneumonia")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
