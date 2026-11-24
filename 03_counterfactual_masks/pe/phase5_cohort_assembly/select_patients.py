"""Phase 5 (cohort assembly): select the PE-negative patients for the counterfactual cohort.

Criteria:
  - PE-negative test-set patients
  - Correctly classified by PE classifier (true negatives)
  - 75 Male + 75 Female, age-matched between sexes
  - Vessel mask + merged segmentation available
  - Lobe coverage QC (>= 4 lobes with >= 2.0 mL vessel volume)

Usage:
    python phase5_cohort_assembly/select_patients.py
    python phase5_cohort_assembly/select_patients.py --run-lobe-qc   # also run lobe coverage QC
    python phase5_cohort_assembly/select_patients.py --finalize       # write output files
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.nifti_io import load_nifti, DATA_ROOT

# Paths
DATALIST_PATH = DATA_ROOT / "pe_training_data_all" / "pe_datalist_spacing_all_test_with_demo_4_controlnet.json"
# PE-classifier predictions on the INSPECT test set (not shipped; columns used:
# image_id, target, predicted_label, predicted_score)
CLASSIFIER_CSV = Path("/path/to/pe_classifier_predictions.csv")
VESSEL_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_lung_vessels"
MERGED_SEG_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_all_lung_pe"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

LOBE_LABELS = {10: "LUL", 11: "LLL", 12: "RUL", 13: "RML", 14: "RLL"}
SEED = 44


def check_vessel_quality(patient_id: str, min_voxels: int = 5000) -> dict:
    """Quick vessel mask quality check."""
    vessel_path = VESSEL_DIR / f"{patient_id}_segmentation_lung_vessels.nii.gz"
    if not vessel_path.exists():
        return {"exists": False, "vessel_voxels": 0, "pass": False}

    data, _, header = load_nifti(str(vessel_path))
    vessel_voxels = int(np.sum(data == 1))
    spacing = np.array(header.get_zooms()[:3], dtype=float)
    vessel_volume_ml = vessel_voxels * float(np.prod(spacing)) / 1000.0

    return {
        "exists": True,
        "vessel_voxels": vessel_voxels,
        "vessel_volume_ml": round(vessel_volume_ml, 2),
        "pass": vessel_voxels >= min_voxels,
    }


def check_lobe_coverage(patient_id: str,
                         min_lobes: int = 4,
                         min_lobe_volume_ml: float = 2.0) -> dict:
    """Check vessel distribution across lung lobes via nearest-lobe EDT."""
    from scipy.ndimage import distance_transform_edt

    vessel_path = VESSEL_DIR / f"{patient_id}_segmentation_lung_vessels.nii.gz"
    seg_path = MERGED_SEG_DIR / f"{patient_id}_segmentation_merged.nii.gz"

    if not vessel_path.exists() or not seg_path.exists():
        return {"pass": False, "n_lobes_covered": 0}

    vessel_data, _, v_header = load_nifti(str(vessel_path))
    seg_data, _, _ = load_nifti(str(seg_path))
    spacing = np.array(v_header.get_zooms()[:3], dtype=float)
    voxel_vol_ml = float(np.prod(spacing)) / 1000.0

    vessel_mask = vessel_data == 1
    if vessel_mask.sum() == 0:
        return {"pass": False, "n_lobes_covered": 0}

    # Bounding box crop for speed
    coords = np.argwhere(vessel_mask)
    pad = 5
    slices = tuple(
        slice(max(0, coords[:, i].min() - pad), min(vessel_mask.shape[i], coords[:, i].max() + pad + 1))
        for i in range(3)
    )
    vessel_crop = vessel_mask[slices]
    seg_crop = seg_data[slices]

    # For each lobe, compute EDT and assign vessel voxels to nearest lobe
    lobe_volumes = {}
    min_dist = np.full(vessel_crop.shape, np.inf)
    nearest_lobe = np.zeros(vessel_crop.shape, dtype=np.int32)

    for label, name in LOBE_LABELS.items():
        lobe_mask = seg_crop == label
        if lobe_mask.sum() == 0:
            lobe_volumes[name] = 0.0
            continue
        dist = distance_transform_edt(~lobe_mask, sampling=spacing)
        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        nearest_lobe[closer] = label

    for label, name in LOBE_LABELS.items():
        voxels_in_lobe = int(((nearest_lobe == label) & vessel_crop).sum())
        lobe_volumes[name] = round(voxels_in_lobe * voxel_vol_ml, 2)

    n_covered = sum(1 for v in lobe_volumes.values() if v >= min_lobe_volume_ml)

    result = {
        "pass": n_covered >= min_lobes,
        "n_lobes_covered": n_covered,
    }
    for name, vol in lobe_volumes.items():
        result[f"lobe_vol_{name}"] = vol
    return result


def get_true_negatives() -> pd.DataFrame:
    """Get PE-negative test patients correctly classified by the PE classifier."""
    # Load classifier results
    clf = pd.read_csv(CLASSIFIER_CSV)

    # Load datalist demographics
    with open(DATALIST_PATH) as f:
        datalist = json.load(f)["test"]
    demo_df = pd.DataFrame([{
        "image_id": d["image_id"],
        "pe_status": d["pe_status"],
        "gender": d["gender"],
        "age": d["age"],
        "race": d.get("race", ""),
        "ethnicity": d.get("ethnicity", ""),
        "text_prompt": d.get("text_prompt", ""),
        "dim": d.get("dim"),
        "spacing": d.get("spacing"),
        "top_region_index": d.get("top_region_index"),
        "bottom_region_index": d.get("bottom_region_index"),
        "fold": d.get("fold"),
        "patient_id_original": d.get("patient_id", ""),
    } for d in datalist])

    # True negatives: target=0, predicted=0
    tn = clf[(clf["target"] == 0.0) & (clf["predicted_label"] == 0)]

    # Merge with demographics (inner join = test set only)
    merged = tn.merge(demo_df, on="image_id", how="inner")

    return merged


def select_age_matched(available: pd.DataFrame, n_per_gender: int = 50) -> pd.DataFrame:
    """Select n_per_gender males and age-matched females."""
    np.random.seed(SEED)

    males = available[available["gender"] == "Male"].copy()
    females = available[available["gender"] == "Female"].copy()

    if len(males) < n_per_gender:
        raise ValueError(f"Only {len(males)} males available, need {n_per_gender}")
    if len(females) < n_per_gender:
        raise ValueError(f"Only {len(females)} females available, need {n_per_gender}")

    # Stratify males by age quintiles, sample proportionally
    males["age_q"] = pd.qcut(males["age"], q=5, labels=False)
    male_counts = males["age_q"].value_counts().sort_index()
    total_m = len(males)
    allocation = (male_counts / total_m * n_per_gender).round().astype(int)
    diff = n_per_gender - allocation.sum()
    if diff != 0:
        allocation.iloc[allocation.argmax()] += diff

    selected_males = []
    for q in range(5):
        pool = males[males["age_q"] == q]
        n = allocation[q]
        selected_males.append(pool.sample(n=min(n, len(pool)), random_state=SEED))
    selected_males = pd.concat(selected_males)

    # Match females by nearest age to each selected male
    females_sorted = females.sort_values("age").copy()
    used = set()
    selected_female_idxs = []
    for _, male_row in selected_males.iterrows():
        target_age = male_row["age"]
        best_idx = None
        best_diff = float("inf")
        for idx, row in females_sorted.iterrows():
            if idx in used:
                continue
            d = abs(row["age"] - target_age)
            if d < best_diff:
                best_diff = d
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            selected_female_idxs.append(best_idx)

    selected_females = females_sorted.loc[selected_female_idxs]

    return pd.concat([selected_males, selected_females]).drop(columns=["age_q"], errors="ignore")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 (cohort assembly): select PE-negative patients for the counterfactual cohort")
    parser.add_argument("--run-lobe-qc", action="store_true",
                        help="Run lobe coverage QC (slow, needs NIfTI loading)")
    parser.add_argument("--finalize", action="store_true",
                        help="Write output files to output/")
    parser.add_argument("--n-per-gender", type=int, default=75)
    args = parser.parse_args()

    print("Loading true negatives from classifier + datalist...")
    available = get_true_negatives()
    print(f"  Available true negatives: {len(available)} "
          f"({(available['gender']=='Male').sum()}M, {(available['gender']=='Female').sum()}F)")

    # Vessel quality check
    print("\nRunning vessel quality check...")
    qc_results = []
    for pid in tqdm(available["image_id"], desc="Vessel QC"):
        qc = check_vessel_quality(pid)
        qc["image_id"] = pid
        qc_results.append(qc)
    qc_df = pd.DataFrame(qc_results)
    passing = qc_df[qc_df["pass"]]["image_id"]
    available = available[available["image_id"].isin(passing)]
    print(f"  Passed vessel QC: {len(available)}")

    if args.run_lobe_qc:
        print("\nRunning lobe coverage QC...")
        lobe_results = []
        for pid in tqdm(available["image_id"], desc="Lobe QC"):
            lc = check_lobe_coverage(pid)
            lc["image_id"] = pid
            lobe_results.append(lc)
        lobe_df = pd.DataFrame(lobe_results)
        lobe_passing = lobe_df[lobe_df["pass"]]["image_id"]
        available = available[available["image_id"].isin(lobe_passing)]
        print(f"  Passed lobe QC: {len(available)}")
        # Save lobe QC results
        if args.finalize:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            lobe_df.to_csv(OUTPUT_DIR / "lobe_qc_results.csv", index=False)

    # Select age-matched cohort
    print(f"\nSelecting {args.n_per_gender}M + {args.n_per_gender}F age-matched...")
    selected = select_age_matched(available, n_per_gender=args.n_per_gender)
    print(f"  Selected: {len(selected)} patients")

    # Stats
    males = selected[selected["gender"] == "Male"]
    females = selected[selected["gender"] == "Female"]
    print(f"\n  Male age:   median={males['age'].median():.1f}, "
          f"mean={males['age'].mean():.1f}, range=[{males['age'].min():.1f}, {males['age'].max():.1f}]")
    print(f"  Female age: median={females['age'].median():.1f}, "
          f"mean={females['age'].mean():.1f}, range=[{females['age'].min():.1f}, {females['age'].max():.1f}]")

    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(males["age"], females["age"])
    print(f"  KS test (M vs F age): stat={ks_stat:.3f}, p={ks_p:.3f}")

    clf_scores = selected["predicted_score"]
    print(f"\n  Classifier score: mean={clf_scores.mean():.3f}, "
          f"median={clf_scores.median():.3f}, range=[{clf_scores.min():.3f}, {clf_scores.max():.3f}]")

    if args.finalize:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Patient IDs
        ids = sorted(selected["image_id"].tolist())
        with open(OUTPUT_DIR / "selected_patient_ids.txt", "w") as f:
            f.write("\n".join(ids) + "\n")
        print(f"\n  Wrote {len(ids)} IDs to {OUTPUT_DIR / 'selected_patient_ids.txt'}")

        # CSV with demographics + classifier score
        csv_cols = ["image_id", "gender", "age", "race", "ethnicity",
                    "predicted_score", "predicted_label"]
        if "vessel_voxels" in selected.columns:
            csv_cols.append("vessel_voxels")
        selected[csv_cols].sort_values("image_id").to_csv(
            OUTPUT_DIR / "selected_patients.csv", index=False)
        print(f"  Wrote {OUTPUT_DIR / 'selected_patients.csv'}")

        # Full JSON with all datalist fields
        with open(DATALIST_PATH) as f:
            all_patients = {d["image_id"]: d for d in json.load(f)["test"]}
        full_records = []
        for pid in ids:
            rec = all_patients[pid].copy()
            row = selected[selected["image_id"] == pid].iloc[0]
            rec["classifier_score"] = float(row["predicted_score"])
            rec["classifier_label"] = int(row["predicted_label"])
            full_records.append(rec)
        with open(OUTPUT_DIR / "selected_patients_full.json", "w") as f:
            json.dump(full_records, f, indent=2)
        print(f"  Wrote {OUTPUT_DIR / 'selected_patients_full.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
