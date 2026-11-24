# 03 Counterfactual masks

This folder makes the conditioning masks from which the diffusion models generate the
counterfactual CTs. Each mask keeps the anatomy segmentation of a real disease-free scan and
adds one lesion.

| Folder | Disease | Method |
|--------|---------|--------|
| [`pe/`](pe/) | Pulmonary embolism | Emboli placed in the pulmonary arterial tree of a PE-negative CTPA, on a grid of 16 configurations per patient (location, occlusion, burden, acute or chronic morphology), with four automated checks |
| [`ich/`](ich/) | Intracranial hemorrhage | A donor hemorrhage mask moved into the segmentation of an ICH-negative head CT by rigid skull registration, then clipped to the brain |

What you need: the conditioning masks used in the paper are on HuggingFace (`<HF_URL>`, link to
be added), so this code is needed only to rebuild them or to make new ones. The PE pipeline reads INSPECT scans, segmentations and report impressions
([INSPECT dataset page](https://aimi.stanford.edu/datasets/inspect-Multimodal-Dataset-for-Pulmonary-Embolism-Diagnosis-and-Prognosis)).
The ICH pipeline reads internal head CT data that are not public. All code here runs on a CPU. Each subfolder has its own `README.md` with the commands,
and its own `requirements.txt`.

The PE folder also holds the Phase 6 evaluation scripts. They compare the PE-detector output on
the generated CTs with the intended emboli and write the cohort tables read by
[`../04_evaluation_and_analysis/`](../04_evaluation_and_analysis/).

Next steps: PE masks go to [`../01_pe_diffusion/`](../01_pe_diffusion/), ICH masks to
[`../02_ich_diffusion/`](../02_ich_diffusion/).
