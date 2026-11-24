"""Build the open PE example: a conditioning mask in the label space of the PE generator.

Source: one subject of the TotalSegmentator dataset v2.0.1 (Wasserthal et al., Zenodo,
https://doi.org/10.5281/zenodo.10047292, CC BY 4.0). No INSPECT data is used.

Steps:
  1. Labels 1 to 117: the 117 segmentations of the subject (same class names as
     ../../01_pe_diffusion/configs/label_dict_inspect.json).
  2. Label 118 (body): TotalSegmentator `body` task on the subject's CT, where no organ label is set.
  3. Labels 119 (pulmonary vessels) and 120 (airways): TotalSegmentator `lung_vessels` task
     (arteries + veins -> 119, airways -> 120). They replace the labels below them.
  4. Label 121 (embolus): a simple scripted placement, not the vessel-graph pipeline of
     ../../03_counterfactual_masks/pe (that pipeline needs its own vessel graph). Take the
     pulmonary-artery voxels whose distance to the vessel wall is 4 to 8 mm (lobar calibre in
     the paper's rule), choose the one nearest to the centre of mass of the right lower lobe,
     and relabel the connected artery voxels within 10 mm of it.
  5. Resample (nearest neighbour) to 512 x 512 x 256 voxels, which the generator accepts.

Run (the TotalSegmentator calls need the CT of the subject):
  TotalSegmentator -i s0871/ct.nii.gz -o ts_body -ta body
  TotalSegmentator -i s0871/ct.nii.gz -o ts_lung_vessels -ta lung_vessels
  python build_example_mask.py --subject-dir s0871 --body ts_body/body.nii.gz \
      --lung-vessels ts_lung_vessels --out example_pe_mask.nii.gz
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

OUT_SHAPE = (512, 512, 256)
SEED_RADIUS_MM = (4.0, 8.0)
EMBOLUS_RADIUS_MM = 10.0


def load(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj), img.affine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-dir", required=True, help="TotalSegmentator dataset subject folder (segmentations/)")
    ap.add_argument("--body", required=True, help="body.nii.gz of the TotalSegmentator body task")
    ap.add_argument("--lung-vessels", required=True, help="output folder of the TotalSegmentator lung_vessels task")
    ap.add_argument("--out", default="example_pe_mask.nii.gz")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    label_dict = json.load(open(here.parents[1] / "01_pe_diffusion/configs/label_dict_inspect.json"))
    seg_dir = Path(args.subject_dir) / "segmentations"

    # 1. organ labels
    first, affine = load(seg_dir / "spleen.nii.gz")
    label = np.zeros(first.shape, np.uint8)
    for name, value in label_dict.items():
        if value <= 117:
            label[load(seg_dir / f"{name}.nii.gz")[0] > 0] = value
    lobe_rll = load(seg_dir / "lung_lower_lobe_right.nii.gz")[0] > 0

    # 2. body
    body = load(args.body)[0] > 0
    label[body & (label == 0)] = label_dict["body"]

    # 3. vessels and airways
    lv = Path(args.lung_vessels)
    arteries = load(lv / "lung_arteries.nii.gz")[0] > 0
    veins = load(lv / "lung_veins.nii.gz")[0] > 0
    airways = load(lv / "lung_airways.nii.gz")[0] > 0
    label[arteries | veins] = label_dict["lung_vessels"]
    label[airways] = label_dict["lung_trachea_bronchia"]

    # 4. embolus in the artery of lobar calibre nearest to the right lower lobe
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    wall_dist = ndimage.distance_transform_edt(arteries, sampling=spacing)
    cand = np.argwhere(arteries & (wall_dist >= SEED_RADIUS_MM[0]) & (wall_dist <= SEED_RADIUS_MM[1]))
    if len(cand) == 0:
        raise SystemExit("no artery voxel of lobar calibre")
    centre_rll = np.argwhere(lobe_rll).mean(0)
    seed = cand[np.argmin((((cand - centre_rll) * spacing) ** 2).sum(1))]
    grid = np.indices(label.shape).transpose(1, 2, 3, 0)
    near = (((grid - seed) * spacing) ** 2).sum(-1) <= EMBOLUS_RADIUS_MM ** 2
    comp, _ = ndimage.label(arteries & near)
    embolus = comp == comp[tuple(seed)]
    label[embolus] = label_dict["pulmonary_embolism"]
    vol_ml = embolus.sum() * float(np.prod(spacing)) / 1000.0

    # 5. resample to the generator grid (nearest neighbour), keep the field of view
    idx = [np.minimum((np.arange(n) * (s / n)).astype(int), s - 1) for n, s in zip(OUT_SHAPE, label.shape)]
    out = label[np.ix_(*idx)]
    out_affine = affine.copy()
    for i in range(3):
        out_affine[:3, i] *= label.shape[i] / OUT_SHAPE[i]
    nib.save(nib.Nifti1Image(out, out_affine), args.out)
    out_spacing = [round(float(s * a / b), 4) for s, a, b in zip(spacing, label.shape, OUT_SHAPE)]
    print(f"saved {args.out}: shape {out.shape}, spacing {out_spacing} mm, labels {len(np.unique(out)) - 1}")
    print(f"embolus: {int(embolus.sum())} voxels on the source grid, {vol_ml:.2f} mL, seed {seed.tolist()}, "
          f"wall distance at the seed {wall_dist[tuple(seed)]:.1f} mm")


if __name__ == "__main__":
    main()
