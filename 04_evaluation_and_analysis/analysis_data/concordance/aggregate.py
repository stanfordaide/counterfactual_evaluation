"""Aggregate the concordance analysis: radiologist free-text diagnosis vs the intended generation
impression, PE + ICH. Two complementary measures:

  (1) LLM judge (out_*.jsonl) - holistic 0/1/2, already reads the full impression.
  (2) STRUCTURED impression tier-set concordance - compares the reader's pathology-component set
      (PE location tiers; ICH subtypes) to ALL components named in the IMPRESSION (not the single
      manifest label), so a reader tier that matches any intended tier counts. Graded:
        Full   = reader named every intended component (reader superset of intended)
        Partial= reader named some intended components (overlap, incomplete)
        None   = no overlap
      plus `overlap` = named >=1 intended component.

Writes output/concordance_judged.csv (LLM + structured columns) and prints the structured
concordance by modality and by intended primary location / subtype (e.g. the PE central and
subsegmental agreement). The public copy CF_DATA/radiologist/concordance_judged.csv is this
table without the free-text columns (diagnosis, note) and with ICH ids re-keyed.

Code only: reads pairs_all.jsonl and out_*.jsonl from CF_RAW/concordance_judge/ (prep_pairs.py),
which hold free text. The figure of the original script is not included.
"""
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd

HERE = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "concordance_judge"
OUT = Path(__file__).resolve().parents[2] / "output"
PE_ORDER = ["central", "lobar", "segmental", "subsegmental"]
ICH_SUB = {"subdural": "SDH", "epidural": "EDH", "subarachnoid": "SAH",
           "intraparenchymal": "IPH", "intracerebral": "IPH", "intraventricular": "IVH"}


def pe_tiers(t: str) -> frozenset:
    t = str(t).lower()
    s = set()
    if "subsegmental" in t:
        s.add("subsegmental")
    if re.search(r"(?<!sub)segmental", t):
        s.add("segmental")
    if "lobar" in t:
        s.add("lobar")
    if re.search(r"central|main pulmonary|saddle|main pa|main artery", t):
        s.add("central")
    return frozenset(s)


def ich_subs(t: str) -> frozenset:
    t = str(t).lower()
    return frozenset(v for k, v in ICH_SUB.items() if k in t)


def _pe_impression_1(t: str) -> str:
    m = re.search(r"IMPRESSION:\s*1\.?\s*(.*?)(?:\s+2\.|\Z)", str(t), re.S)
    return m.group(1) if m else str(t)


def _grade(intended: frozenset, reader: frozenset) -> int:
    if not intended or not (intended & reader):
        return 0
    return 2 if intended <= reader else 1


def _load_judgments() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(HERE / "out_*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(rows).drop_duplicates("idx")


def main() -> None:
    J = _load_judgments()
    P = pd.DataFrame([json.loads(l) for l in open(HERE / "pairs_all.jsonl")])
    df = P.merge(J, on="idx", how="left")
    print(f"pairs {len(P)} | LLM-judged {J.idx.nunique()} | missing {int(df.concordance.isna().sum())}")

    # ---- structured impression tier-set concordance ----
    def intended_set(r):
        return pe_tiers(_pe_impression_1(r["impression"])) if r.modality == "PE" else ich_subs(r["impression"])

    def reader_set(r):
        return pe_tiers(r["diagnosis"]) if r.modality == "PE" else ich_subs(r["diagnosis"])

    df["intended_set"] = df.apply(intended_set, axis=1)
    df["reader_set"] = df.apply(reader_set, axis=1)
    df["overlap"] = df.apply(lambda r: len(r.intended_set & r.reader_set) > 0, axis=1)
    df["tierset_grade"] = df.apply(lambda r: _grade(r.intended_set, r.reader_set), axis=1)
    df["primary"] = df.apply(
        lambda r: r.intended_location if r.modality == "PE"
        else (list(r.intended_set)[0] if len(r.intended_set) == 1 else ("Mixed" if r.intended_set else "?")),
        axis=1)

    out = df.copy()
    out["intended_set"] = out.intended_set.apply(lambda s: "+".join(sorted(s)))
    out["reader_set"] = out.reader_set.apply(lambda s: "+".join(sorted(s)))
    OUT.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["impression"]).to_csv(OUT / "concordance_judged.csv", index=False)

    def report(g, lab):
        n = len(g)
        print(f"  {lab:20s} n={n:3d} | Full {100*(g.tierset_grade==2).mean():3.0f}% "
              f"Partial {100*(g.tierset_grade==1).mean():3.0f}% None {100*(g.tierset_grade==0).mean():3.0f}% "
              f"| overlap(named any intended) {100*g.overlap.mean():3.0f}% "
              f"| LLM concordant {100*(g.concordance==2).mean():3.0f}% >=partial {100*(g.concordance>=1).mean():3.0f}%")

    print("\n=== impression tier-set concordance (reader components vs ALL intended components) ===")
    for mod in ["PE", "ICH"]:
        report(df[df.modality == mod], mod + " all")
        keys = PE_ORDER if mod == "PE" else ["IPH", "IVH", "SDH", "EDH", "SAH", "Mixed"]
        for k in keys:
            g = df[(df.modality == mod) & (df.primary == k)]
            if len(g):
                report(g, f"  {mod} {k}")


if __name__ == "__main__":
    main()
