#!/usr/bin/env python
"""Radiologist-study numbers for the "Radiologist assessment of the counterfactuals" section.

Standalone, DETERMINISTIC, read-only. Recomputes and prints EVERY radiologist-study
number that goes into the paper's radiologist section, and ASSERTS each against
the reported value. No LLM / API calls are made -- concordance numbers are read from the
already-saved LLM-judge outputs (concordance_judged.csv). Nothing is written; no
manuscript or figure is touched.

Blocks
------
  1. Counterfactual (CF) cohort sizes                 : PE 190 | ICH 240
  2. Intended lesion confirmed present (q4 >= 1), CF  : PE 119/190 62.6% | ICH 177/240 73.8%
  3. Baseline (lesion-free) present rate (control)    : ICH 2/30 6.7%
  4. Ordinal quality scores (mean +/- SD), CF
       PE  anatomical 1.75 +/- 0.57 | PE  realism 1.63 +/- 0.64
       ICH anatomical 1.89 +/- 0.36 | ICH realism 2.26 +/- 0.91
  5. % rated acceptable (rating >= 1), CF             : reported (no fixed target)
  6. Free-text vs intended-impression concordance     : from concordance_judged.csv (LLM judge
       already run + saved); among present-and-described CFs and unconditionally over all CF.

Data joins
--------------------------------------------
  PE : radiologist/pe/ratings_per_scan.csv
        + scan_type from radiologist/pe/manifest.csv (join on scan_id).
        PE baseline ids were re-randomized for blinding, so they do NOT join to a scan_type
        (scan_type -> NaN) -- that drops the ~60 non-CF rows and leaves the 190 CF. CF ==
        scan_type == 'counterfactual'.
  ICH: radiologist/ich/ratings_per_scan.csv
        + is_baseline from radiologist/ich/radiologist_data.csv (join on scan_id).
        CF == is_baseline == False (240) ; baseline == is_baseline == True (30).
  Concordance (both modalities): radiologist/concordance_judged.csv
        (analysis_data/concordance/aggregate.py writes BOTH PE and ICH rows).
        Rows are already restricted to present-and-described CFs (119 PE + 174 ICH).

Run:
  python analysis/radiologist_scores.py
"""
import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("CF_DATA", Path(__file__).resolve().parents[2] / "resources"))

PE_RATINGS = REPO / "radiologist/pe/ratings_per_scan.csv"
PE_MANIFEST = REPO / "radiologist/pe/manifest.csv"
ICH_RATINGS = REPO / "radiologist/ich/ratings_per_scan.csv"
ICH_DATA = REPO / "radiologist/ich/radiologist_data.csv"
CONCORDANCE = REPO / "radiologist/concordance_judged.csv"


# --------------------------------------------------------------------------- #
# Small check framework: print every check, collect failures, fail loudly once.
# --------------------------------------------------------------------------- #
class Checks:
    def __init__(self):
        self.fails = []

    def eq(self, label, got, target):
        """Exact-integer check (counts, denominators, cohort sizes)."""
        ok = int(got) == int(target)
        self._emit(ok, label, f"{int(got)}", f"{int(target)}")
        return ok

    def approx(self, label, got, target, nd, unit=""):
        """Numeric check tolerant to the target's stated rounding precision.

        nd = number of decimals the target is quoted to (0 -> whole %, 1 -> one
        decimal %, 2 -> mean/SD). Passes iff |got - target| < 0.5 * 10**-nd.
        """
        tol = 0.5 * 10 ** (-nd) + 1e-9
        ok = abs(float(got) - float(target)) < tol
        disp = max(nd, 2)
        self._emit(ok, label, f"{got:.{disp}f}{unit}", f"{target:g}{unit}")
        return ok

    def _emit(self, ok, label, got, target):
        tag = "PASS" if ok else "FAIL"
        print(f"    [{tag}] {label:<48} got={got:<10} target={target}")
        if not ok:
            self.fails.append((label, got, target))


def msd(x):
    """n, mean, sample-SD (ddof=1) over numeric-coerced, NaN-dropped values."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    return len(x), float(x.mean()), float(x.std(ddof=1))


def to_bool(s):
    """Robustly coerce a saved boolean column (bool / 0-1 / 'True'/'False') -> bool."""
    if s.dtype == bool:
        return s
    if s.dtype == object:
        return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).astype(bool)


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


# --------------------------------------------------------------------------- #
def main():
    ck = Checks()
    print("=" * 90)
    print("RADIOLOGIST SECTION NUMBERS (deterministic recompute + assertions)")
    print("=" * 90)

    # ----- load -----
    for p in (PE_RATINGS, PE_MANIFEST, ICH_RATINGS, ICH_DATA, CONCORDANCE):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing input: {p}\n"
                "  Provide the radiologist CSVs in your CF_DATA resources folder under "
                "radiologist/pe/ and radiologist/ich/ (see 04 README).")

    pe = pd.read_csv(PE_RATINGS)
    man = pd.read_csv(PE_MANIFEST)[["scan_id", "scan_type"]]
    ich = pd.read_csv(ICH_RATINGS)
    ichd = pd.read_csv(ICH_DATA)[["scan_id", "is_baseline"]]
    con = pd.read_csv(CONCORDANCE)

    # ================================================================= #
    # PE ratings
    # ================================================================= #
    print("\n" + "-" * 90)
    print("BLOCK 1-5 (PE)  radiologist_ratings_per_scan.csv  x  manifest.scan_type")
    print("-" * 90)
    pej = pe.merge(man, on="scan_id", how="left")
    st = pej["scan_type"].value_counts(dropna=False).to_dict()
    print(f"  PE ratings rows                 : {len(pej)}")
    print(f"  scan_type value counts (w/ NaN) : {st}")
    print("  (NaN scan_type = the 60 blinding-randomized baselines that do not join -- expected)")
    pe_cf = pej[pej["scan_type"] == "counterfactual"].copy()

    print("\n  [1] CF cohort size")
    ck.eq("PE N counterfactuals", len(pe_cf), 190)

    q4 = pd.to_numeric(pe_cf["q4_pe_present"], errors="coerce")
    n_pres = int((q4 >= 1).sum())
    print("\n  [2] Intended lesion confirmed present (q4_pe_present >= 1), CF only")
    print(f"      {n_pres}/{q4.notna().sum()} = {pct(n_pres, q4.notna().sum()):.1f}%")
    ck.eq("PE present count", n_pres, 119)
    ck.eq("PE present denom (CF n)", int(q4.notna().sum()), 190)
    ck.approx("PE present rate %", pct(n_pres, int(q4.notna().sum())), 62.6, 1, "%")

    print("\n  [4] Ordinal quality (mean +/- SD)")
    pe_q1 = msd(pe_cf["q1_anatomical_quality"])
    pe_q8 = msd(pe_cf["q8_overall_realism"])
    print(f"      anatomical q1 (0-2): n={pe_q1[0]}  {pe_q1[1]:.2f} +/- {pe_q1[2]:.2f}")
    print(f"      realism    q8 (0-2): n={pe_q8[0]}  {pe_q8[1]:.2f} +/- {pe_q8[2]:.2f}")
    ck.approx("PE anatomical mean", pe_q1[1], 1.75, 2)
    ck.approx("PE anatomical SD", pe_q1[2], 0.57, 2)
    ck.approx("PE realism mean", pe_q8[1], 1.63, 2)
    ck.approx("PE realism SD", pe_q8[2], 0.64, 2)

    print("\n  [5] % rated acceptable (rating >= 1) -- reported, no fixed target")
    pe_acc_q1 = 100 * (pd.to_numeric(pe_cf["q1_anatomical_quality"], errors="coerce") >= 1).mean()
    pe_acc_q8 = 100 * (pd.to_numeric(pe_cf["q8_overall_realism"], errors="coerce") >= 1).mean()
    print(f"      PE anatomical acceptable : {pe_acc_q1:.1f}%")
    print(f"      PE realism    acceptable : {pe_acc_q8:.1f}%")

    # ================================================================= #
    # ICH ratings
    # ================================================================= #
    print("\n" + "-" * 90)
    print("BLOCK 1-5 (ICH)  radiologist_ratings_per_scan.csv  x  radiologist_data.is_baseline")
    print("-" * 90)
    ichj = ich.merge(ichd, on="scan_id", how="left")
    print(f"  ICH ratings rows                : {len(ichj)}")
    print(f"  is_baseline value counts (w/NaN): {ichj['is_baseline'].value_counts(dropna=False).to_dict()}")
    ib = to_bool(ichj["is_baseline"])
    ich_cf = ichj[~ib].copy()
    ich_base = ichj[ib].copy()

    print("\n  [1] CF cohort size")
    ck.eq("ICH N counterfactuals", len(ich_cf), 240)

    q4i = pd.to_numeric(ich_cf["q4_ich_present"], errors="coerce")
    ni_pres = int((q4i >= 1).sum())
    print("\n  [2] Intended lesion confirmed present (q4_ich_present >= 1), CF only")
    print(f"      {ni_pres}/{q4i.notna().sum()} = {pct(ni_pres, q4i.notna().sum()):.1f}%")
    ck.eq("ICH present count", ni_pres, 177)
    ck.eq("ICH present denom (CF n)", int(q4i.notna().sum()), 240)
    ck.approx("ICH present rate %", pct(ni_pres, int(q4i.notna().sum())), 73.8, 1, "%")

    q4b = pd.to_numeric(ich_base["q4_ich_present"], errors="coerce")
    nb_pres = int((q4b >= 1).sum())
    print("\n  [3] Baseline (lesion-free) present rate -- discrimination control")
    print(f"      {nb_pres}/{q4b.notna().sum()} = {pct(nb_pres, q4b.notna().sum()):.1f}%")
    ck.eq("ICH baseline present count", nb_pres, 2)
    ck.eq("ICH baseline denom", int(q4b.notna().sum()), 30)
    ck.approx("ICH baseline present rate %", pct(nb_pres, int(q4b.notna().sum())), 6.7, 1, "%")

    print("\n  [4] Ordinal quality (mean +/- SD)")
    ich_q1 = msd(ich_cf["q1_anatomical_quality"])
    ich_q8 = msd(ich_cf["q8_overall_realism"])
    print(f"      anatomical q1 (0-2): n={ich_q1[0]}  {ich_q1[1]:.2f} +/- {ich_q1[2]:.2f}")
    print(f"      realism    q8 (0-3): n={ich_q8[0]}  {ich_q8[1]:.2f} +/- {ich_q8[2]:.2f}")
    ck.approx("ICH anatomical mean", ich_q1[1], 1.89, 2)
    ck.approx("ICH anatomical SD", ich_q1[2], 0.36, 2)
    ck.approx("ICH realism mean", ich_q8[1], 2.26, 2)
    ck.approx("ICH realism SD", ich_q8[2], 0.91, 2)

    print("\n  [5] % rated acceptable (rating >= 1) -- reported, no fixed target")
    print("      (ICH realism q8 is a 0-3 scale; 'acceptable' still means rating >= 1)")
    ich_acc_q1 = 100 * (pd.to_numeric(ich_cf["q1_anatomical_quality"], errors="coerce") >= 1).mean()
    ich_acc_q8 = 100 * (pd.to_numeric(ich_cf["q8_overall_realism"], errors="coerce") >= 1).mean()
    print(f"      ICH anatomical acceptable : {ich_acc_q1:.1f}%")
    print(f"      ICH realism    acceptable : {ich_acc_q8:.1f}%")

    # ================================================================= #
    # BLOCK 6 -- free-text vs intended-impression concordance
    # ================================================================= #
    print("\n" + "-" * 90)
    print("BLOCK 6  Free-text vs intended-impression concordance  (concordance_judged.csv)")
    print("-" * 90)
    print("  tier-set 'overlap' = reader named >=1 intended component; grade Full=2/Partial=1/None=0.")
    print("  LLM-judge 'concordance' 0/1/2 ; 'type_match' = intended pathology TYPE named.")
    print(f"  rows: {len(con)}  |  by modality: {con['modality'].value_counts().to_dict()}")

    # denominators used for the "unconditional over all CF" rate
    cf_n = {"PE": 190, "ICH": 240}
    con_targets = {
        "PE": dict(n=119, overlap=77, ov_pct=65, full=33, partial=32, none=35,
                   llm_ge1=81, typematch=98, uncond_pct=41),
        "ICH": dict(n=174, overlap=148, ov_pct=85, full=66, partial=19, none=15,
                    llm_ge1=86, typematch=84, uncond_pct=62),
    }

    for mod in ("PE", "ICH"):
        g = con[con["modality"] == mod].copy()
        n = len(g)
        ov = int(to_bool(g["overlap"]).sum())
        full = int((pd.to_numeric(g["tierset_grade"], errors="coerce") == 2).sum())
        part = int((pd.to_numeric(g["tierset_grade"], errors="coerce") == 1).sum())
        none = int((pd.to_numeric(g["tierset_grade"], errors="coerce") == 0).sum())
        conc = pd.to_numeric(g["concordance"], errors="coerce")
        llm_ge1 = int((conc >= 1).sum())
        llm_full = int((conc == 2).sum())
        tm = int(to_bool(g["type_match"]).sum())
        t = con_targets[mod]

        print(f"\n  === {mod} concordance (present-and-described CFs) ===")
        print(f"      n described                     : {n}")
        print(f"      tier-set named >=1 intended     : {ov}/{n} = {pct(ov, n):.1f}%")
        print(f"      tier-set grade  Full/Partial/None: "
              f"{full}/{part}/{none}  =  {pct(full,n):.0f}% / {pct(part,n):.0f}% / {pct(none,n):.0f}%")
        print(f"      LLM-judge >= partial (conc>=1)  : {llm_ge1}/{n} = {pct(llm_ge1,n):.1f}%  "
              f"(full conc==2: {pct(llm_full,n):.1f}%)")
        print(f"      pathology type_match             : {tm}/{n} = {pct(tm,n):.1f}%")
        print(f"      UNCONDITIONAL over all {mod} CF     : {ov}/{cf_n[mod]} = {pct(ov, cf_n[mod]):.1f}%")

        print(f"      -- assertions ({mod}) --")
        ck.eq(f"{mod} concordance n (present+described)", n, t["n"])
        ck.eq(f"{mod} tier-set overlap count", ov, t["overlap"])
        ck.approx(f"{mod} tier-set overlap %", pct(ov, n), t["ov_pct"], 0, "%")
        ck.approx(f"{mod} tier-set Full %", pct(full, n), t["full"], 0, "%")
        ck.approx(f"{mod} tier-set Partial %", pct(part, n), t["partial"], 0, "%")
        ck.approx(f"{mod} tier-set None %", pct(none, n), t["none"], 0, "%")
        ck.approx(f"{mod} LLM-judge >=partial %", pct(llm_ge1, n), t["llm_ge1"], 0, "%")
        ck.approx(f"{mod} pathology type_match %", pct(tm, n), t["typematch"], 0, "%")
        ck.approx(f"{mod} unconditional overlap %", pct(ov, cf_n[mod]), t["uncond_pct"], 0, "%")

    # cross-check: PE described-CF n should equal PE present count (both 119)
    print("\n  cross-check: PE #described CFs == PE #present CFs")
    ck.eq("PE described-n == present-n", len(con[con.modality == "PE"]), n_pres)
    ich_desc_n = len(con[con.modality == "ICH"])
    print(f"  note: ICH described CFs = {ich_desc_n} vs ICH present CFs = {ni_pres} "
          f"(gap {ni_pres - ich_desc_n}: present but no free-text lesion description)")

    # ================================================================= #
    # SUMMARY
    # ================================================================= #
    ov_pe = int(to_bool(con[con.modality == "PE"]["overlap"]).sum())
    ov_ich = int(to_bool(con[con.modality == "ICH"]["overlap"]).sum())
    print("\n" + "=" * 90)
    print("MANUSCRIPT-READY SUMMARY BLOCK")
    print("=" * 90)
    print(f"""
COHORT (counterfactuals): PE n={len(pe_cf)} | ICH n={len(ich_cf)}

INTENDED LESION CONFIRMED PRESENT (reader q4>=1), CF only:
  PE : {n_pres}/190 = {pct(n_pres,190):.1f}%
  ICH: {ni_pres}/240 = {pct(ni_pres,240):.1f}%
DISCRIMINATION CONTROL (lesion-free baseline read as present):
  ICH: {nb_pres}/30 = {pct(nb_pres,30):.1f}%

ORDINAL QUALITY (mean +/- SD):
  PE  anatomical(0-2) {pe_q1[1]:.2f} +/- {pe_q1[2]:.2f} | PE  realism(0-2) {pe_q8[1]:.2f} +/- {pe_q8[2]:.2f}
  ICH anatomical(0-2) {ich_q1[1]:.2f} +/- {ich_q1[2]:.2f} | ICH realism(0-3) {ich_q8[1]:.2f} +/- {ich_q8[2]:.2f}
% RATED ACCEPTABLE (rating >= 1):
  PE  anatomical {pe_acc_q1:.1f}% | PE  realism {pe_acc_q8:.1f}%
  ICH anatomical {ich_acc_q1:.1f}% | ICH realism {ich_acc_q8:.1f}%

FREE-TEXT vs INTENDED-IMPRESSION CONCORDANCE:
  Among present-and-described CFs (reader named >=1 intended component):
    PE : {ov_pe}/119 = {pct(ov_pe,119):.0f}%   ICH: {ov_ich}/174 = {pct(ov_ich,174):.0f}%
  Unconditional over all CF:
    PE : {ov_pe}/190 = {pct(ov_pe,190):.0f}%   ICH: {ov_ich}/240 = {pct(ov_ich,240):.0f}%
""")

    # ----- final verdict -----
    print("=" * 90)
    n_checks = len(ck.fails)
    if not ck.fails:
        print("ALL TARGETS REPRODUCED -- every assertion passed.")
        print("=" * 90)
    else:
        print(f"!!! {n_checks} TARGET(S) DID NOT REPRODUCE -- see below !!!")
        for label, got, target in ck.fails:
            print(f"   MISMATCH  {label}: got={got}  target={target}")
        print("=" * 90)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
