# Examples

An input for a first run of the PE generator. The command is in the top-level README
("Generate scans") and in `../01_pe_diffusion/`. A public ICH example will be added.

## PE (`pe/`): open data

| File | Content |
|---|---|
| `example_pe_mask.nii.gz` | Conditioning mask, 512 x 512 x 256 voxels, labels 1 to 121 of the PE generator, with one embolus (label 121, 1.7 mL) in the right lower lobe pulmonary artery |
| `datalist.json` | Datalist entry with a short synthetic prompt |
| `environment.json` | Paths for `infer_controlnet.py` (weights in `../weights/`, output in `pe/output/`) |
| `build_example_mask.py` | The script that made the mask |
| `preview.jpg` | Result of the example run: mask, generated CT, and generated CT with the embolus outline |

Source and license. The mask is derived from subject `s0871` of the TotalSegmentator dataset
v2.0.1 (Wasserthal et al., https://doi.org/10.5281/zenodo.10047292), which is released under
CC BY 4.0. It contains no INSPECT data. Labels 1 to 117 are the dataset's own segmentations.
The body outline (118), the pulmonary vessels (119) and the airways (120) are the output of the
open TotalSegmentator tasks `body` and `lung_vessels` on the CT of that subject. The embolus was
added by `build_example_mask.py` with a simple rule that is described in the script. It was not
placed with the vessel-graph pipeline of `../03_counterfactual_masks/pe/`. The mask is
redistributed here under CC BY 4.0. The age and sex in the prompt are those in the dataset's
metadata table; the dataset has no race field, so the prompt has none.

The preview shows the slice through the middle of the lesion, with the patient's right on the
left of the image.
