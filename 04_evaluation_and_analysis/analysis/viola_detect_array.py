"""Parallel viola_iou merge for a SLURM array. Same seg_IOU + decoded_image join as
viola_detect.py, but each array task computes a STRIDED subset of the master-CSV rows -> a partial
CSV; a final combine step merges the partials back into the collision-recovered master CSV. Turns
the ~1h serial merge (2640 x seg_IOU, each loading two large NIfTIs) into a few minutes across N
tasks.

Modes:
  shard:   viola_detect_array.py shard <viola_glob> <master_csv> <out_partial.csv> <k> <N>
  combine: viola_detect_array.py combine <partials_glob> <master_csv> <out_master.csv>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viola_detect import seg_iou   # noqa: E402  (same criterion as the serial merge)


def _viola_map(viola_glob: str) -> dict:
    return {os.path.basename(m): m for m in glob.glob(viola_glob) if m.endswith((".nii", ".nii.gz"))}


def shard_mode(viola_glob: str, master_csv: str, out_partial: str, k: int, n: int) -> None:
    df = pd.read_csv(master_csv)
    vmap = _viola_map(viola_glob)
    idxs = list(range(k, len(df), n))
    print(f"shard {k}/{n}: {len(idxs)} rows, {len(vmap)} viola masks visible", flush=True)
    rows = []
    for c, i in enumerate(idxs):
        r = df.iloc[i]
        m = vmap.get(str(r.get("decoded_image", "") or ""))
        if m is None:
            rows.append((i, np.nan, False, ""))
        else:
            iou = seg_iou(r["segmentation"], m)
            rows.append((i, iou, iou > 0, os.path.basename(m)))
        if (c + 1) % 25 == 0:
            print(f"shard {k}: {c + 1}/{len(idxs)}", flush=True)
    pd.DataFrame(rows, columns=["idx", "viola_iou", "viola_detected", "viola_mask"]).to_csv(
        out_partial, index=False)
    print(f"shard {k}: wrote {len(rows)} rows -> {out_partial}", flush=True)


def combine_mode(partials_glob: str, master_csv: str, out_master: str) -> None:
    df = pd.read_csv(master_csv)
    parts = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(partials_glob))], ignore_index=True)
    parts = parts.drop_duplicates("idx").set_index("idx").sort_index()
    df["viola_iou"] = np.nan
    df["viola_detected"] = False
    df["viola_mask"] = ""
    df.loc[parts.index, "viola_iou"] = parts["viola_iou"].values
    df.loc[parts.index, "viola_detected"] = parts["viola_detected"].astype(bool).values
    df.loc[parts.index, "viola_mask"] = parts["viola_mask"].fillna("").values
    df.to_csv(out_master, index=False)
    m = df["viola_iou"].notna()
    print(f"combined {len(parts)} partial rows into {out_master}; matched {int(m.sum())}/{len(df)}")
    v = df[df["ich_volume(ml)"] > 0].copy()
    print(f"synthetic Viola detection (vol>0, n={len(v)}): overall {100*(v.viola_iou>0).mean():.1f}%")
    v["vb"] = pd.cut(v["ich_volume(ml)"], [-1e-3, .5, 5, 30, 1e6], labels=["<0.5", "0.5-5", "5-30", ">=30"])
    for b in ["<0.5", "0.5-5", "5-30", ">=30"]:
        g = v[v.vb == b]
        print(f"  {b}: {100*(g.viola_iou>0).mean():.1f}% (n={len(g)})")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "shard":
        shard_mode(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    elif mode == "combine":
        combine_mode(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(f"unknown mode {mode!r} (expected shard|combine)")
