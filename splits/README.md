# INSPECT train and test splits

INSPECT had no official train/test split when this project started. We defined our own split at
the patient level. This folder gives the exact membership of each split.

| File | Studies (CTPA volumes) | Patients |
|------|------------------------|----------|
| `train_study_ids.txt` | 12,367 | 10,463 |
| `test_study_ids.txt`  | 3,527  | 3,010  |

- Each line holds one INSPECT study identifier (`PE` followed by hexadecimal characters).
- No identifier occurs in both files.
- The PE diffusion model (`01_pe_diffusion/`) was trained on the train split only.
  Generation fidelity (FID and CLIPScore) was measured on the test split.
- The 150 source patients of the PE counterfactuals were selected from the test split.
- The evaluated PE detectors are open-weight models from other groups. We did not train them.

The files contain study identifiers only. Access to the INSPECT images and reports is governed
by the INSPECT data use agreement.
