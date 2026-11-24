"""Prepare shared inputs for the FID significance bootstrap (run once, before the per-config workers).

  (1) fid_boot_counts.npy  - (B, n) integer image-resample counts, generated with a fixed seed and
      shared across ALL configs, so the FID differences are PAIRED: bootstrap b resamples the SAME
      multiset of overlapping images for every config. n = number of overlapping images (= rows of
      each filelist, which are in the common-image-id order by construction of build_fid_filelists).

  (2) fid_boot_real.npz    - the real reference per plane (XY/YZ/ZX): mu, cov, and half = cov^(1/2)
      (a fixed matrix square root), computed once from the existing fid_real_v2_clean features and
      reused by every worker. Using a fixed half lets each bootstrap get tr[sqrtm(cov_real cov_synth)]
      from sum(sqrt(eigvalsh(half cov_synth half))) - validated to match scipy sqrtm.

Run:  python fidelity/fid_bootstrap_prep.py
"""
import os
from pathlib import Path

import numpy as np
import torch

R = os.path.join(os.environ.get("CF_RAW", "/path/to/private_data"), "pe_training_data_all")
OUT = str(Path(__file__).resolve().parent)
REAL_DIR = f"{R}/fid_real_v2_clean"
REAL_FL = f"{R}/real_file_list_fid_v2_clean.txt"
REF_FILELIST = f"{R}/fid_overlap_filelists/filelist_text_demo_mask.txt"  # defines n + image order
B = 1000
SEED = 1234


def real_reference():
    """Per-plane (mu, cov, half=cov^0.5) from fid_real_v2_clean, incremental float64, single pass."""
    files = [f"{REAL_DIR}/{l.strip().replace('.nii.gz', '.pt')}" for l in open(REAL_FL) if l.strip()]
    s = [None] * 3
    ssq = [None] * 3
    n = [0] * 3
    for f in files:
        t = torch.load(f, weights_only=True, map_location="cpu")
        for pi in range(3):
            x = t[pi].double()
            if s[pi] is None:
                d = x.shape[1]
                s[pi] = torch.zeros(d, dtype=torch.double)
                ssq[pi] = torch.zeros(d, d, dtype=torch.double)
            s[pi] += x.sum(0)
            ssq[pi] += x.T @ x
            n[pi] += x.shape[0]
    mu, cov, half = [], [], []
    for pi in range(3):
        m = s[pi] / n[pi]
        c = ssq[pi] / n[pi] - torch.outer(m, m)
        L, Q = torch.linalg.eigh(c)
        mu.append(m.numpy())
        cov.append(c.numpy())
        half.append(((Q * L.clamp(min=0).sqrt()) @ Q.T).numpy())
    print(f"real reference: {len(files)} scans, {n[0]} slices/plane")
    return np.stack(mu), np.stack(cov), np.stack(half)


def main():
    n = sum(1 for l in open(REF_FILELIST) if l.strip())
    rng = np.random.default_rng(SEED)
    counts = np.zeros((B, n), dtype=np.int32)
    for b in range(B):
        counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    np.save(f"{OUT}/fid_boot_counts.npy", counts)
    print(f"saved fid_boot_counts.npy {counts.shape} (B={B}, n={n}, seed={SEED})")

    mu, cov, half = real_reference()
    np.savez(f"{OUT}/fid_boot_real.npz", mu=mu, cov=cov, half=half)
    print(f"saved fid_boot_real.npz (mu {mu.shape}, cov {cov.shape}, half {half.shape})")


if __name__ == "__main__":
    main()
