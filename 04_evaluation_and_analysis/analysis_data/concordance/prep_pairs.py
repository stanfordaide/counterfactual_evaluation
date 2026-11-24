"""Build LLM-judge concordance pairs: radiologist free-text diagnosis (q5) vs the intended
generation impression, for PE and ICH. Only counterfactual scans where the radiologist marked
the pathology present (q4). Splits the pairs into N batches for parallel LLM grading.

Code only: the inputs hold free text (radiologist diagnoses, conditioning impressions) that is
not released. They are the internal versions of the CF_DATA/radiologist/ tables, under
CF_RAW/radiologist/ (pe/ratings_per_scan.csv, pe/manifest.csv, ich/ratings_per_scan.csv,
ich/radiologist_data.csv). Pairs and batches go to CF_RAW/concordance_judge/.

Grading (no script): a large language model (Methods) read each batch_<k>.jsonl and wrote
out_<k>.jsonl with one JSON object per pair: idx, concordance (0 none / 1 partial / 2 full),
type_match (0/1) and a short note. aggregate.py then builds concordance_judged.csv.
"""
import json
import os
import re

import pandas as pd

RAW = os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "radiologist")
HERE = os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "concordance_judge")
N_BATCHES = 8


def imp_from_prompt(t: str) -> str:
    """Strip leading demographics; keep the IMPRESSION onward (matches the app's impression panel)."""
    m = re.search(r"(IMPRESSION:.*)", str(t), re.S)
    return (m.group(1) if m else str(t)).strip()


def main() -> None:
    os.makedirs(HERE, exist_ok=True)
    pairs = []

    # ---- PE ----
    scan = pd.read_csv(f"{RAW}/pe/ratings_per_scan.csv")
    man = pd.read_csv(f"{RAW}/pe/manifest.csv")
    mg = scan.merge(man[["scan_id", "scan_type", "text_prompt", "location", "severity_category"]],
                    on="scan_id")
    pe = mg[(mg.scan_type == "counterfactual") & (mg.q4_pe_present == 1)
            & (mg.q5_pe_description.astype(str).str.len() > 3)]
    for _, r in pe.iterrows():
        pairs.append({"idx": f"PE_{r.scan_id}", "modality": "PE", "scan_id": r.scan_id,
                      "impression": imp_from_prompt(r.text_prompt),
                      "diagnosis": str(r.q5_pe_description).strip(),
                      "intended_location": r.location, "intended_severity": r.severity_category})

    # ---- ICH ----
    meta = pd.read_csv(f"{RAW}/ich/radiologist_data.csv")
    rat = pd.read_csv(f"{RAW}/ich/ratings_per_scan.csv")
    mi = rat.merge(meta[["scan_id", "impression", "is_baseline"]], on="scan_id", how="left")
    ich = mi[(mi.is_baseline != True) & (mi.q4_ich_present == 1)
             & (mi.q5_ich_description.astype(str).str.len() > 3)
             & (mi.impression.astype(str).str.len() > 3)]
    for _, r in ich.iterrows():
        pairs.append({"idx": f"ICH_{r.scan_id}", "modality": "ICH", "scan_id": r.scan_id,
                      "impression": imp_from_prompt(r.impression),
                      "diagnosis": str(r.q5_ich_description).strip()})

    print(f"total pairs: {len(pairs)}  | PE: {sum(p['modality']=='PE' for p in pairs)}  "
          f"ICH: {sum(p['modality']=='ICH' for p in pairs)}")
    with open(f"{HERE}/pairs_all.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    for k in range(N_BATCHES):
        batch = pairs[k::N_BATCHES]
        with open(f"{HERE}/batch_{k}.jsonl", "w") as f:
            for p in batch:
                f.write(json.dumps(p) + "\n")
        print(f"  batch_{k}: {len(batch)} pairs")


if __name__ == "__main__":
    main()
