"""Build FID filelists on the CLIPScore overlapping image set (fidelity_common_image_ids.csv), so
FID is computed on the SAME images as the CLIPScore test, for every config plus a real filelist.

Each config: image_id -> prediction_filename (from that config's inference manifest) -> path
relative to the config's prediction folder (= --synth_dataset_root for compute_fid_2-5d_ct_old.py).
Real: {image_id}.nii.gz relative to the CTPA root (the real reference is unchanged; this list is
provided for consistency / optional re-extraction).

Outputs -> inspect2/.../fid_overlap_filelists/: real_filelist.txt, filelist_<config>.txt, README.txt
(the README records each config's synth_dataset_root so the FID command just points at it).

Run:  python fidelity/build_fid_filelists.py
"""
import glob
import os

from pathlib import Path

import pandas as pd

R = os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "pe_training_data_all")
CTPA = os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "CTPA")
OUT = f"{R}/fid_overlap_filelists"
COMMON = str(Path(__file__).resolve().parent / "fidelity_common_image_ids.csv")

# config -> (synth_dataset_root, inference manifest with image_id -> prediction_filename)
CONFIGS = {
    "text": (f"{R}/predictions/unet_3d_text_all_l2_loss",
             f"{R}/inference_results_unet_3d_text_all_l2_loss_20250630_205622_all.csv"),
    "text_demo": (f"{R}/predictions/unet_3d_text_all_l2_loss_best_demo",
                  f"{R}/predictions/unet_3d_text_all_l2_loss_best_demo/"
                  f"inference_results_unet_3d_text_all_l2_loss_best_demo_20250730_155945.csv"),
    "text_mask": (f"{R}/predictions_controlnet/controlnet_text_segmentation_best",
                  f"{R}/predictions_controlnet/controlnet_text_segmentation_best/controlnet_predictions.csv"),
    "text_demo_mask": (f"{R}/predictions_controlnet_best_demo_pe_all_v2_clean",
                       f"{R}/predictions_controlnet_best_demo_pe_all_v2_clean/controlnet_predictions.csv"),
}


def resolve(root: str, fn) -> str | None:
    """Path of the file relative to root (basename if at root, else a recursive search), or None."""
    if fn is None or (isinstance(fn, float)):
        return None
    bn = os.path.basename(str(fn))
    if os.path.exists(os.path.join(root, bn)):
        return bn
    hits = glob.glob(f"{root}/**/{bn}", recursive=True)
    return os.path.relpath(hits[0], root) if hits else None


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    common = list(pd.read_csv(COMMON)["image_id"])

    real_ok = [i for i in common if os.path.exists(os.path.join(CTPA, f"{i}.nii.gz"))]
    with open(f"{OUT}/real_filelist.txt", "w") as f:
        f.writelines(f"{i}.nii.gz\n" for i in real_ok)
    print(f"real_filelist.txt: {len(real_ok)}/{len(common)} (root {CTPA})")

    readme = [f"# FID overlapping-set filelists — n_common = {len(common)} (matches the CLIPScore test set)",
              f"# real reference root: {CTPA}   filelist: real_filelist.txt ({len(real_ok)})",
              "# config | --synth_dataset_root | --synth_filelist | n"]
    for cfg, (root, man) in CONFIGS.items():
        m = pd.read_csv(man).drop_duplicates("image_id").set_index("image_id")["prediction_filename"].to_dict()
        rel, miss = [], []
        for i in common:
            r = resolve(root, m.get(i))
            (rel if r else miss).append(r if r else i)
        with open(f"{OUT}/filelist_{cfg}.txt", "w") as f:
            f.writelines(r + "\n" for r in rel)
        print(f"filelist_{cfg}.txt: {len(rel)}/{len(common)} (missing {len(miss)})  root {root}")
        readme.append(f"{cfg} | {root} | filelist_{cfg}.txt | {len(rel)}")

    with open(f"{OUT}/README.txt", "w") as f:
        f.write("\n".join(readme) + "\n")
    print(f"\nwrote real + {len(CONFIGS)} config filelists + README -> {OUT}")


if __name__ == "__main__":
    main()
