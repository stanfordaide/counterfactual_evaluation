import pandas as pd
import numpy as np


def stratified_train_val_test_split_by_patient(
    df: pd.DataFrame,
    patient_id_col: str,
    stratify_cols: list,
    ratios: tuple = (0.7, 0.1, 0.2),
    age_col: str = "Age",
    age_bin_edges: list = None,
    seed: int = 42,
) -> tuple:
    """
    Split dataframe into train/val/test by patient_id with stratified balance on age, gender, race.
    Same patient never appears in more than one split.
    Returns (train_df, val_df, test_df) with similar distributions of stratify_cols in each split.

    Args:
        df: full dataframe (one row per study; multiple rows per patient allowed).
        patient_id_col: column name for patient identifier.
        stratify_cols: list of column names to balance, e.g. ["Age", "Gender", "Race"].
        ratios: (train_ratio, val_ratio, test_ratio), e.g. (0.7, 0.1, 0.2).
        age_col: column name for age (will be binned for stratification).
        age_bin_edges: bins for age, e.g. [0, 18, 30, 40, 50, 60, 70, 80, 100]. If None, 10-year bins from min to max are used.
        seed: random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "ratios must sum to 1"

    # One row per patient: take first occurrence for Age, Gender, Race
    patient_df = df.groupby(patient_id_col, as_index=False).agg({
        c: "first" for c in stratify_cols
    })

    # Bin age for stratification
    if age_col in stratify_cols and age_bin_edges is None:
        age_min = int(patient_df[age_col].min())
        age_max = int(patient_df[age_col].max()) + 1
        age_bin_edges = list(range(age_min, age_max + 1, 10))
        if age_bin_edges[-1] != age_max:
            age_bin_edges.append(age_max)
    if age_col in stratify_cols and age_bin_edges is not None:
        patient_df["_age_bin"] = pd.cut(
            patient_df[age_col].astype(float),
            bins=age_bin_edges,
            labels=False,
            include_lowest=True,
            right=False,
        ).fillna(-1).astype(int)
        stratify_cols_use = ["_age_bin"] + [c for c in stratify_cols if c != age_col]
    else:
        patient_df["_age_bin"] = patient_df[age_col]
        stratify_cols_use = stratify_cols

    # Assign split per patient: within each stratum, assign proportionally
    pid_to_split = {}
    for _key, grp in patient_df.groupby(stratify_cols_use):
        pid_list = grp[patient_id_col].tolist()
        rng.shuffle(pid_list)
        n = len(pid_list)
        n_train = max(0, int(round(n * train_r)))
        n_val = max(0, int(round(n * val_r)))
        n_test = max(0, n - n_train - n_val)
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        for pid, sp in zip(pid_list, splits):
            pid_to_split[pid] = sp

    # Map back to full dataframe
    df = df.copy()
    df["split"] = df[patient_id_col].map(pid_to_split)

    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df = df[df["split"] == "val"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])

    return train_df, val_df, test_df
