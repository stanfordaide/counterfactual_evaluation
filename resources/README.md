# Resources

Small tables that the analysis code in `../04_evaluation_and_analysis/` reads (`CF_DATA`).
Nothing here needs to be run. The layout, the column descriptions and the id conventions are
in `../04_evaluation_and_analysis/README.md` (section "Resources layout").

| Folder | Content |
|---|---|
| `cohort/` | PE cohort: the frozen list of 2,347 counterfactuals from 150 patients, the table of all generated configurations, and the Phase 6 tables |
| `detector_outputs/` | PE detection tables of both detectors, and their predictions on disease-free scans |
| `ich/` | ICH counterfactual table with the output of both ICH detectors (2,640 rows; 2,601 with an intended volume above 0 mL) |
| `radiologist/` | Radiologist ratings for PE and ICH, without free text |
| `comorbidity/` | Co-occurring findings per PE source patient and per ICH counterfactual |
| `external_validation/` | Detection results on FUMPE, CAD-PE, READ and PhysioNet CT-ICH |

The tables contain no report text, no radiologist free text, no LLM rationales, no
demographics, no file paths and no timestamps. ICH patient ids are release ids (`ICH001` to
`ICH150`). The row order of every table is the order used for the paper, so the bootstrap
results are reproduced exactly.

The CT volumes and masks are in the HuggingFace dataset, not here.
