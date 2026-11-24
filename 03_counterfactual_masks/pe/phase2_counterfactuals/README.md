# Phase 2: counterfactual grid

Builds the 16 counterfactual PE configurations of one patient from the Phase 1 vessel graph.

## Scripts

| Script | Purpose |
|--------|---------|
| `axis_definitions.py` | Enums (`PELocation`, `OcclusionDegree`, `Burden`), dataclasses (`SegmentSpec`, `CounterfactualConfig`), Qanadli weights |
| `counterfactual_grid.py` | `EdgeCatalog`, grid builder (`build_counterfactual_grid`), replacement configurations for retries, Qanadli scoring, CLI |

## Usage

```bash
python phase2_counterfactuals/counterfactual_grid.py <STUDY_ID>
python phase2_counterfactuals/counterfactual_grid.py <STUDY_ID> --seed 42
```

In the cohort run, `phase5_cohort_assembly/run_single_patient.py` calls this phase and writes the outputs to `patients/{id}/phase2/`. Run alone, the script reads the Phase 1 graph from `$CF_OUT/phase1_vessel_tree/output` and writes to `phase2_counterfactuals/output/`.

## Outputs

| File | Description |
|------|-------------|
| `{patient_id}_counterfactual_grid.csv` | One row per configuration: location, occlusion, burden, Qanadli score, laterality, edge keys |
| `{patient_id}_counterfactual_grid.json` | Full detail, including the per-segment specification (`edge_key`, generation, lobe, occlusion, `acute`) |

## Axes

- Location: central (generation 0), lobar (1), segmental (2), subsegmental (3).
- Occlusion: partial (30-50% of the lumen), near-complete (70-90%), complete (100%).
- Burden: isolated (1 segment), multi-unilateral (2-4, one side), multi-bilateral (2-6, both sides), extensive (7 or more).
- Morphology: a per-segment `acute` flag. Acute emboli fill the lumen from the center (concentric). Chronic emboli fill it from the wall (eccentric).

## The 16 configurations

Configurations 1-15 come from `_build_base_grid`, configuration 16 from `build_counterfactual_grid`. The target severity is the design intent; the stored severity is computed from the Qanadli score of the selected edges. A configuration is skipped when the graph has no suitable edges, so the numbering can shift. Edges are drawn from those labeled artery or unknown (veins excluded). If a lobe has no such edge, the filter is relaxed. Lobar edges are drawn without this filter. Central edges (configuration 16) are drawn from artery or unknown edges, and from all edges only if there is none.

| # | Target severity | Location label | Vessels used | Burden | Occlusion | Morphology |
|---|-----------------|----------------|--------------|--------|-----------|------------|
| 1 | Mild | Subsegmental | 1 subsegmental edge | Isolated | Partial | Chronic |
| 2 | Mild | Subsegmental | 1 subsegmental edge | Isolated | Complete | Acute |
| 3 | Mild | Segmental | 1 segmental edge | Isolated | Partial | Acute |
| 4 | Mild | Subsegmental | 3 subsegmental edges, one right lobe | Multi-unilateral | Near-complete | Chronic |
| 5 | Mild | Segmental | 1 segmental edge in a left and in a right lobe | Multi-bilateral | Complete | Acute |
| 6 | Mild | Lobar | 1 lobar edge | Isolated | Complete | Chronic |
| 7 | Mild | Subsegmental | 1 left and 2 right subsegmental edges | Multi-bilateral | Near-complete | Acute |
| 8 | Mild | Segmental | 1 segmental edge | Isolated | Near-complete | Chronic |
| 9 | Mild | Subsegmental | 1 subsegmental edge in each of 3 lobes | Multi-bilateral | Complete | Acute |
| 10 | Moderate | Lobar | Lobar and segmental edges, 3 lobes | Extensive | Complete | Chronic |
| 11 | Moderate | Lobar | Lobar and segmental edges, 4 lobes | Extensive | Complete | Acute |
| 12 | Moderate | Segmental | 1 segmental edge in each of the 5 lobes | Extensive | Complete | Chronic |
| 13 | Moderate | Lobar | Lobar and segmental edges, 4 lobes | Extensive | Near-complete | Acute |
| 14 | Moderate | Lobar | Lobar, segmental and 2-3 subsegmental edges, 3 lobes | Extensive | Complete | Chronic |
| 15 | Severe | Lobar | Lobar, 2 segmental and 3 subsegmental edges per lobe, all 5 lobes and the central region (at least 10 edges) | Extensive | Complete | Acute |
| 16 | Severe (set by rule) | Central | Central edges (partial); in 3 random lobes complete lobar and segmental and near-complete subsegmental edges; in the other 2 lobes partial segmental and subsegmental edges | Extensive | Partial (label of the central part) | Acute |

Configuration 16 is labeled severe regardless of its Qanadli score, because central PE is clinically severe.

If a configuration fails the Phase 4 checks, `generate_replacement_configs` makes a new one (attempt 0: same location, burden and occlusion in a different lobe, larger vessels; attempt 1: also occlusion one level up; attempt 2: also an isolated burden becomes multi-unilateral with 3 edges). Replacement configurations are always chronic: `_make_segment_specs` is called without the `acute` flag, so its default `acute=False` applies. They get the next free number, so some patients have ids above `cf016`.

## Qanadli score

- 20 segments, maximum 40. Edges are grouped by (lobe, generation); subsegmental edges count in the segmental slot of their lobe.
- Weight per slot: partial or near-complete = 1, complete = 2.
- Severity (`severity_from_qanadli`): mild 1-8, moderate 9-16, severe 17-28, massive 29 or more. No configuration of the paper cohort reached the massive range.
