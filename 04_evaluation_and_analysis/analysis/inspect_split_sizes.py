#!/usr/bin/env python
"""INSPECT PE generative-model dataset splits (Methods, Training Data).

The PE generative (ControlNet) model was fine-tuned on the INSPECT "training split"
(the datalist reporting 12,367 volumes) and the 150 counterfactual source subjects were
embolism-negative studies from the held-out "test split".

Recomputes, straight from the datalist JSONs:
  (a) number of unique PATIENTS behind the 12,367 training volumes + confirmation the
      split is patient-level (no patient appears in more than one split);
  (b) the TEST split size: number of studies/images AND unique patients.

Code only: needs the INSPECT training datalists under CF_RAW/pe_training_data_all/.
Run: python analysis/inspect_split_sizes.py
"""
import os
import json
from collections import Counter
from pathlib import Path

DL_DIR = Path(os.environ.get("CF_RAW", "/path/to/private_data")) / "pe_training_data_all"
# The "_6_controlnet" datalists are the ones whose training array length == 12,367.
SPLITS = {
    "train": DL_DIR / "pe_datalist_spacing_all_train_with_demo_6_controlnet.json",
    "test":  DL_DIR / "pe_datalist_spacing_all_test_with_demo_6_controlnet.json",
}


def load_entries(path):
    d = json.load(open(path))
    # single top-level key ("training" / "test" / "validation")
    (key,) = [k for k in d if isinstance(d[k], list)]
    return key, d[key]


def main():
    print("=" * 78)
    print("INSPECT PE generative-model splits (patient-level check)")
    print("=" * 78)

    pat_by_split = {}
    for name, path in SPLITS.items():
        key, entries = load_entries(path)
        n_img = len(entries)
        pats = [e.get("patient_id") for e in entries]
        n_missing_pid = sum(p is None for p in pats)
        upats = set(p for p in pats if p is not None)
        pat_by_split[name] = upats
        pe_status = Counter(str(e.get("pe_status")) for e in entries)
        print(f"\n[{name}]  file: {path.name}")
        print(f"  top-level key      : '{key}'")
        print(f"  studies/images (n) : {n_img}")
        print(f"  unique patient_id  : {len(upats)}")
        print(f"  entries w/o pid    : {n_missing_pid}")
        print(f"  pe_status breakdown: {dict(pe_status)}")

    print("\n" + "-" * 78)
    print("PATIENT-LEVEL SPLIT CHECK (overlap of patient_id sets)")
    names = list(pat_by_split)
    any_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = pat_by_split[a] & pat_by_split[b]
            any_overlap = any_overlap or bool(inter)
            print(f"  {a} ∩ {b}: {len(inter)} shared patients"
                  + (f"  e.g. {sorted(inter)[:5]}" if inter else ""))
    print(f"\n  => patient-level (disjoint) split: {'YES' if not any_overlap else 'NO'}")

    print("\n" + "=" * 78)
    print("ANSWERS")
    print("=" * 78)
    print(f"(a) 12,367 training volumes come from {len(pat_by_split['train']):,} unique patients")
    print(f"    Split is patient-level (no patient in >1 split): {'YES' if not any_overlap else 'NO'}")
    print(f"(b) TEST split: {len(load_entries(SPLITS['test'])[1]):,} studies/images "
          f"from {len(pat_by_split['test']):,} unique patients")
    print("\nNote: only 'train' and 'test' datalists exist for the *_6_controlnet series")
    print("(no separate *_6_controlnet validation file); a 'fold' field is present per entry.")


if __name__ == "__main__":
    main()
