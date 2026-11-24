"""Regenerate the PE-driver tables (Fig. 5a) with the cluster-robust logistic
estimator, for the primary strict ±5-slice (m5) criterion, both segmentation models.

This is the reproducible entry point for glmm_results.csv: it loads the
exact per-lesion dataframe the evaluation used (via pe_drivers_gee.load, which
merges intended volume from master_metadata) and fits the population-average
cluster-robust logistic model (GEE, independence working correlation, sandwich SE
clustered on patient) + BH-FDR - identical estimator to the ICH driver and PE
co-morbidity models.

Reads  CF_DATA/detector_outputs/{nnunet,apes}/analysis_dataframe.csv, CF_DATA/cohort/master_metadata.csv
Writes output/detector_outputs/{nnunet,apes}/glmm_results.csv
Run:  python analysis/refit_pe_drivers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.pe_drivers_gee import load, run_cluster_robust_model, MASTER_CSV, RES, OUT

M5_DIRS = {
    "nnUNet3D-ResEncXL": "detector_outputs/nnunet",
    "APES": "detector_outputs/apes",
}

if __name__ == "__main__":
    for model, d in M5_DIRS.items():
        print(f"\n{'='*70}\n{model}: {d}\n{'='*70}")
        df = load(analysis_csv=str(RES / d / "analysis_dataframe.csv"), master_csv=MASTER_CSV)
        run_cluster_robust_model(df, results_dir=str(OUT / d))
