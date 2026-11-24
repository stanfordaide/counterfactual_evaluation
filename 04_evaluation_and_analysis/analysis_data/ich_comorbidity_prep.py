"""Prep ICH co-morbidity extraction: gather the UNIQUE conditioning impressions (Methods).

The co-morbidity/artifact signal lives in the impression 'tail' (points 2., 3., ...) of the
EXACT conditioning text (`text_prompt`, identical to the generation datalist). Only 58 unique
tails across 2,640 CFs, so we extract once per unique impression and map back.

Code only: needs the internal ICH counterfactual table with the conditioning text
(output/ich/cf_2640_samples_recovered.csv as written by analysis/recover_cf_decoded_mapping.py;
the public CF_DATA copy has no text). Outputs hold impression text and stay private.

Writes (CF_RAW/ich_comorbidity/):
  ich_impressions_unique.json   [{imp_id, impression}]  -> labeled by a large language model
  ich_config_impression.csv     per-CF: image, imp_id (-1 if no co-morbidity tail)
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
import pandas as pd

ICH_INTERNAL = Path(__file__).resolve().parents[1] / "output" / "ich" / "cf_2640_samples_recovered.csv"
OUT = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "ich_comorbidity"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(ICH_INTERNAL)
df = pd.DataFrame({"image": raw["image"].astype(str),
                   "text_prompt": raw["text_prompt"].astype(str)})

_VOL = re.compile(r"total volume of\s*[\d.]+\s*ml\.?", re.I)
_IMPR = re.compile(r"IMPRESSION:\s*", re.I)


def tail(tp: str) -> str:
    m = _VOL.search(tp)
    return tp[m.end():].strip() if m else ""


def full_impression(tp: str) -> str:
    m = _IMPR.search(tp)
    return (tp[m.end():] if m else tp).strip()


df["tail"] = df["text_prompt"].map(tail)
df["has_cm"] = df["tail"].str.len() > 8              # non-trivial co-morbidity text
# representative full impression per unique tail (context for the labeler)
df["impression"] = df["text_prompt"].map(full_impression)

uniq = (df[df["has_cm"]]
        .drop_duplicates("tail")
        .reset_index(drop=True)[["tail", "impression"]])
uniq.insert(0, "imp_id", range(len(uniq)))
tail2id = dict(zip(uniq["tail"], uniq["imp_id"]))
df["imp_id"] = df["tail"].map(lambda t: tail2id.get(t, -1))

records = [{"imp_id": int(r.imp_id), "impression": r.impression}
           for r in uniq.itertuples()]
(OUT / "ich_impressions_unique.json").write_text(json.dumps(records, indent=1))
df[["image", "imp_id"]].to_csv(OUT / "ich_config_impression.csv", index=False)

print(f"unique co-morbidity impressions: {len(uniq)}")
print(f"CFs with co-morbidity text: {int(df['has_cm'].sum())} / {len(df)}")
print(f"wrote {OUT/'ich_impressions_unique.json'} and {OUT/'ich_config_impression.csv'}")
