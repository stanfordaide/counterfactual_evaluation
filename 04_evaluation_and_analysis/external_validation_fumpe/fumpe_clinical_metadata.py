"""FUMPE per-patient clinical metadata from Masoudi et al., Sci Data 2018;5:180180
(FUMPE, https://www.nature.com/articles/sdata2018180 / PMC6122162).

The Kaggle mirror ships only CT_scans + GroundTruth masks, so the Qanadli obstruction
score (Table 2) and the main-vs-peripheral PE-ROI counts (Table 1) are transcribed here
from the paper. Verified against the paper's stated Q-score distribution (most frequent
20/7/3 with 5/4/3 subjects; 11 patients with Q>18) - all three checks pass.

Qanadli score = clinical CT obstruction index (0-40; the paper's cohort spans 0-31),
the severity axis we group by (matched to the synthetic qanadli_score). Patients 24 and
32 have Q=0 and 0 PE-ROIs = the two empty-GT scans (no annotated PE).

Patient i maps to PAT<i> = FUMPE_<i:03d> (sorted order).
"""
from pathlib import Path
import pandas as pd

# patient -> (qanadli, roi_main, roi_peripheral)
FUMPE = {
    1: (20, 46, 34), 2: (1, 0, 21), 3: (20, 39, 55), 4: (29, 17, 29), 5: (20, 71, 92),
    6: (3, 0, 15), 7: (31, 95, 147), 8: (7, 23, 36), 9: (4, 0, 53), 10: (7, 0, 8),
    11: (2, 0, 18), 12: (6, 0, 77), 13: (14, 30, 23), 14: (8, 0, 98), 15: (8, 15, 31),
    16: (19, 18, 59), 17: (20, 194, 60), 18: (12, 54, 102), 19: (16, 51, 300),
    20: (5, 19, 48), 21: (10, 51, 0), 22: (3, 0, 22), 23: (17, 0, 116), 24: (0, 0, 0),
    25: (3, 0, 67), 26: (27, 77, 131), 27: (18, 53, 88), 28: (7, 19, 0), 29: (22, 24, 72),
    30: (7, 35, 64), 31: (20, 54, 103), 32: (0, 0, 0), 33: (14, 115, 60), 34: (22, 34, 121),
    35: (9, 0, 154),
}
OUT = Path(__file__).resolve().parents[1] / "output" / "external_validation" / "fumpe" / "fumpe_clinical.csv"


def main():
    rows = [{"patient": f"PAT{ i:03d}".replace(" ", ""), "case": f"FUMPE_{i:03d}",
             "qanadli": q, "roi_main": m, "roi_peripheral": p}
            for i, (q, m, p) in sorted(FUMPE.items())]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT}")
    # verification of the paper's reported distribution
    vc = df.qanadli.value_counts()
    print("Q=20:", int(vc.get(20, 0)), "| Q=7:", int(vc.get(7, 0)), "| Q=3:", int(vc.get(3, 0)),
          "| Q>18:", int((df.qanadli > 18).sum()), "(paper: 5, 4, 3, 11)")


if __name__ == "__main__":
    main()
