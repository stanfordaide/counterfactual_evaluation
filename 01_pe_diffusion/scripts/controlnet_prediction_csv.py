#!/usr/bin/env python3
import argparse, csv, json, re
from pathlib import Path
from typing import Dict, Iterable, Tuple

# python scripts/controlnet_prediction_csv.py   "/path/to/inspect2/pe_training_data_all/predictions_controlnet/controlnet_text_segmentation_best"   "/path/to/inspect2/pe_training_data_all/pe_datalist_spacing_all_test_3.json"   -o "/path/to/inspect2/pe_training_data_all/predictions_controlnet/controlnet_text_segmentation_best/controlnet_predictions.csv"

IMAGE_PAT = re.compile(r'^(.+?)_pred_.*_image\.nii\.gz$')  # capture everything before '_pred_' as image_id

def load_impressions(json_path: Path) -> Dict[str, str]:
    """
    Load a mapping {image_id -> text_prompt} from a JSON that may have
    top-level lists or dicts of lists (e.g., {'train': [...], 'test': [...]})
    """
    with json_path.open() as f:
        data = json.load(f)

    def rows(iterable: Iterable) -> Iterable[Tuple[str, str]]:
        for item in iterable:
            if not isinstance(item, dict): 
                continue
            iid = item.get("image_id")
            txt = item.get("text_prompt")
            label = item.get("pe_status")
            if iid and txt is not None:
                yield iid, txt, label

    mapping = {}
    if isinstance(data, list):
        for iid, txt, label in rows(data):
            mapping[iid] = txt, label
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                for iid, txt, label in rows(v):
                    mapping[iid] = txt, label
    return mapping

def iter_prediction_files(root: Path) -> Iterable[Path]:
    # search recursively for files ending with _image.nii.gz
    yield from root.rglob("*_image.nii.gz")

def image_id_from_name(p: Path) -> str | None:
    m = IMAGE_PAT.search(p.name)
    return m.group(1) if m else None

def main():
    ap = argparse.ArgumentParser(description="Build CSV mapping predictions to impressions.")
    ap.add_argument("pred_dir", type=Path, help="Root folder to scan for *_image.nii.gz")
    ap.add_argument("json_path", type=Path, help="JSON with image_id and text_prompt")
    ap.add_argument("-o", "--out", type=Path, default=Path("predictions_with_impressions.csv"),
                    help="Output CSV path (default: ./predictions_with_impressions.csv)")
    args = ap.parse_args()

    impressions = load_impressions(args.json_path)
    if not impressions:
        raise SystemExit(f"No (image_id, text_prompt) pairs found in {args.json_path}")

    rows = []
    missing = 0
    for f in iter_prediction_files(args.pred_dir):
        iid = image_id_from_name(f)
        if not iid:
            continue
        imp = impressions.get(iid)
        if imp is None:
            missing += 1
            txt = ""
            label = ""
        else:
            txt = imp[0]
            label = imp[1]
        rows.append((iid, str(f.resolve()), txt, label))

    # sort by image_id for reproducibility
    rows.sort(key=lambda t: t[0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["image_id", "prediction_filename", "impression", "label"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    if missing:
        print(f"Note: {missing} files had no matching impression in the JSON.")

if __name__ == "__main__":
    main()
