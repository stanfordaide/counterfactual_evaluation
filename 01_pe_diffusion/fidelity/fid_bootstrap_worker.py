"""Per-config FID bootstrap worker (CPU, float64). Computes the observed 2.5D FID of one config against the
fixed real reference on the overlapping image set, plus B bootstrap FIDs that resample IMAGES with
the shared counts from fid_bootstrap_prep.py (so every config is resampled identically -> paired).
Saves fid_boot_<config>.npz (observed scalar + boot vector).

Numerics: every step is float64. For each image and plane the worker keeps the slice count, the
feature sum and the Gram matrix X^T X (float64 sufficient statistics), so the mean and covariance of
each resample are exact weighted sums over images (weights = the shared counts; an all-ones row gives
the observed FID). An earlier float32 version of this worker did not reproduce the float64 FID.
tr[sqrtm(cov_real cov_synth)] = sum sqrt(eigvalsh(half cov_synth half)) with half = cov_real^(1/2)
fixed (from fid_bootstrap_prep.py). FID = mean over XY/YZ/ZX, matching compute_fid_2-5d_ct_old.py.
Memory: (B + 1) x 2048 x 2048 x 8 bytes (34 GB for B = 1,000) plus one chunk of images (peak 46 GB).

  python fidelity/fid_bootstrap_worker.py --config text_mask
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(16)

R = os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "pe_training_data_all")
OUT = str(Path(__file__).resolve().parent)
CHUNK = 128      # images per Gram-matrix chunk
ROWS = 50        # resamples per eigendecomposition batch


def load(f):
    return torch.load(f, weights_only=True, map_location="cpu")


def plane_fid(files, p, W, mu_r, tr_r, half):
    """FID of plane p for every row of W ((R, n) image weights), all float64."""
    n, nrow = len(files), W.shape[0]
    m = torch.zeros(n, dtype=torch.float64)          # slices per image
    S = None                                         # (n, d) feature sum per image
    acc = None                                       # (R, d * d) weighted Gram sums
    with ThreadPoolExecutor(8) as ex:
        for c0 in range(0, n, CHUNK):
            G = None
            for k, t in enumerate(ex.map(load, files[c0:c0 + CHUNK])):
                x = t[p].double()
                if S is None:
                    d = x.shape[1]
                    S = torch.zeros(n, d, dtype=torch.float64)
                    acc = torch.zeros(nrow, d * d, dtype=torch.float64)
                if G is None:
                    G = torch.empty(min(CHUNK, n - c0), d * d, dtype=torch.float64)
                m[c0 + k] = x.shape[0]
                S[c0 + k] = x.sum(0)
                G[k] = (x.T @ x).reshape(-1)
            acc.addmm_(W[:, c0:c0 + G.shape[0]], G)      # in place: no (R, d * d) temporary
    d = S.shape[1]
    out = np.zeros(nrow)
    for a in range(0, nrow, ROWS):
        Wc = W[a:a + ROWS]
        neff = Wc @ m
        mu = (Wc @ S) / neff[:, None]
        cov = (acc[a:a + ROWS] / neff[:, None]).reshape(-1, d, d) - mu[:, :, None] * mu[:, None, :]
        M = half @ cov @ half
        M = 0.5 * (M + M.transpose(1, 2))
        ev = torch.linalg.eigvalsh(M).clamp(min=0)
        dmu = mu_r[None, :] - mu
        out[a:a + len(Wc)] = ((dmu * dmu).sum(1) + tr_r + cov.diagonal(dim1=1, dim2=2).sum(1)
                              - 2 * ev.sqrt().sum(1)).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    config = ap.parse_args().config

    counts = np.load(f"{OUT}/fid_boot_counts.npy")  # (B, n)
    ref = np.load(f"{OUT}/fid_boot_real.npz")
    lines = [l.strip() for l in open(f"{R}/fid_overlap_filelists/filelist_{config}.txt") if l.strip()]
    files = [f"{R}/fid_overlap_{config}/{l.replace('.nii.gz', '.pt')}" for l in lines]
    n = len(files)
    assert counts.shape[1] == n, f"counts n {counts.shape[1]} != images {n}"
    B = counts.shape[0]
    W = torch.from_numpy(np.vstack([np.ones((1, n)), counts]).astype(np.float64))  # row 0 = observed

    per_plane = np.zeros((B + 1, 3))
    for p in range(3):
        per_plane[:, p] = plane_fid(files, p, W, torch.tensor(ref["mu"][p], dtype=torch.float64),
                                    float(np.trace(ref["cov"][p])),
                                    torch.tensor(ref["half"][p], dtype=torch.float64))
    fid = per_plane.mean(1)
    observed, boot = float(fid[0]), fid[1:]
    np.savez(f"{OUT}/fid_boot_{config}.npz", observed=observed, boot=boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{config}: observed FID {observed:.4f} | boot mean {boot.mean():.4f} [{lo:.4f}, {hi:.4f}] (B={B})")


if __name__ == "__main__":
    main()
