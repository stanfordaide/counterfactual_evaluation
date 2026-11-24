"""ICH subtype composition + final-cohort n for the synthetic dataset (Results).

Final ICH cohort = the master recovered table (cf_2640_samples_recovered.csv,
2,640 rows) restricted to volume_ml > 0 - the exact filter every ICH analysis
applies (analysis/run_ich_glmm.py, patient_bootstrap/text_numbers.py,
patient_bootstrap/figure_numbers.py). The 39 excluded rows are zero-volume generations
(no bleed placed). Result: 2,601 CFs / 150 patients.

Subtypes are parsed from the conditioning-segmentation filename (lib.data
_ich_codes): five single subtypes plus multi-subtype "Mixed" (2/3/4 combined).

Traceability: goes through lib.data.load_ich() so subtype parsing / volume are
identical to the other analyses. Re-run:

    python analysis/ich_subtype_distribution.py

Output (output/, gitignored): ich_subtype_distribution.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_data"))
from lib import data  # noqa: E402

OUTDIR = Path(__file__).resolve().parents[1] / "output"

# Full subtype names for the write-up.
SUBTYPE_FULL = {
    "EDH": "Epidural hemorrhage",
    "SDH": "Subdural hemorrhage",
    "SAH": "Subarachnoid hemorrhage",
    "IPH": "Intraparenchymal hemorrhage",
    "IVH": "Intraventricular hemorrhage",
}
SINGLE_ORDER = ["EDH", "SDH", "SAH", "IPH", "IVH"]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = data.load_ich()
    n_all = len(d)
    d = d[d["volume_ml"] > 0].copy()            # final analysis cohort filter
    n, n_pat = len(d), d["patient_id"].nunique()

    print(f"Recovered master table: {n_all} rows")
    print(f"Final ICH synthetic dataset (volume > 0): {n:,} CFs / {n_pat} patients "
          f"({n_all - n} zero-volume rows excluded)\n")

    # ---- collapsed view: 5 singles + Mixed ---------------------------------
    print("=== subtype (single subtypes + Mixed) ===")
    coll = (d["subtype"].value_counts()
            .reindex(SINGLE_ORDER + ["Mixed"]).dropna().astype(int))
    rows = []
    for name, cnt in coll.items():
        full = SUBTYPE_FULL.get(name, "Multiple subtypes (2-4 combined)")
        rows.append({"level": name, "full_name": full, "n": int(cnt),
                     "pct": 100.0 * cnt / n})
        print(f"  {name:5s} {full:35s} {cnt:5d}  ({100.0*cnt/n:4.1f}%)")
    coll_tbl = pd.DataFrame(rows); coll_tbl.insert(0, "kind", "subtype")

    # ---- single vs multi (combination count) -------------------------------
    print("\n=== combination count (how many subtypes co-occur) ===")
    combo = d["n_subtypes"].value_counts().sort_index()
    combo_rows = []
    for k, cnt in combo.items():
        lab = f"{int(k)} subtype" + ("s" if k > 1 else "")
        combo_rows.append({"level": lab, "full_name": "", "n": int(cnt),
                           "pct": 100.0 * cnt / n})
        print(f"  {lab:12s} {cnt:5d}  ({100.0*cnt/n:4.1f}%)")
    combo_tbl = pd.DataFrame(combo_rows); combo_tbl.insert(0, "kind", "combo")

    # ---- per-subtype PRESENCE (counts every CF the subtype appears in) ------
    # A Mixed CF contributes to each of its constituent subtypes; presence sums
    # exceed n because multi-subtype CFs are counted once per component.
    import re
    codes = d["image"].astype(str).apply(
        lambda s: set(re.findall(r"(SDH|EDH|SAH|IPH|IVH)", s.split("/")[-1])))
    print("\n=== per-subtype PRESENCE (single + within Mixed) ===")
    pres_rows = []
    for name in SINGLE_ORDER:
        cnt = int(codes.apply(lambda c: name in c).sum())
        pres_rows.append({"level": name, "full_name": SUBTYPE_FULL[name],
                          "n": cnt, "pct": 100.0 * cnt / n})
        print(f"  {name:5s} {SUBTYPE_FULL[name]:35s} {cnt:5d}  "
              f"({100.0*cnt/n:4.1f}% of CFs contain it)")
    pres_tbl = pd.DataFrame(pres_rows); pres_tbl.insert(0, "kind", "presence")

    out = pd.concat([coll_tbl, combo_tbl, pres_tbl], ignore_index=True)
    out.insert(0, "cohort_n", n)
    out.to_csv(OUTDIR / "ich_subtype_distribution.csv", index=False)
    print(f"\nWrote {OUTDIR / 'ich_subtype_distribution.csv'}")


if __name__ == "__main__":
    main()
