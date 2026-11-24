"""Phase 4 - Visual QC for synthetic PE masks.

Generates axial/coronal/sagittal slice renderings at the PE centroid,
plus a summary grid thumbnail for all configs of a patient.
"""

import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)



def _load_canonical(path):
    """Load NIfTI and reorient to RAS canonical orientation.

    Returns (data, spacing) where axes are (R→L, P→A, I→S).
    This guarantees consistent L/R orientation regardless of the
    original image storage order.
    """
    img = nib.load(str(path))
    canonical = nib.as_closest_canonical(img)
    data = np.asarray(canonical.dataobj)
    spacing = np.array(canonical.header.get_zooms()[:3], dtype=np.float64)
    return data, spacing


def _add_lr_labels(ax):
    """Add R/L side markers for radiological convention (R on viewer's left)."""
    ax.text(0.02, 0.5, 'R', transform=ax.transAxes, fontsize=14,
            fontweight='bold', color='white', va='center', ha='left')
    ax.text(0.98, 0.5, 'L', transform=ax.transAxes, fontsize=14,
            fontweight='bold', color='white', va='center', ha='right')

PE_LABEL = 121
VESSEL_LABEL = 119   # lung_vessels (pulmonary vasculature)
AIRWAY_LABEL = 120   # lung_trachea_bronchia (airways)

# Color map for key anatomical structures in the combined segmentation
STRUCTURE_COLORS = {
    0: (0, 0, 0, 0),          # background - transparent
    10: (0.7, 0.85, 1.0, 0.3),  # lung lobes - light blue, semi-transparent
    11: (0.7, 0.85, 1.0, 0.3),
    12: (0.7, 0.85, 1.0, 0.3),
    13: (0.7, 0.85, 1.0, 0.3),
    14: (0.7, 0.85, 1.0, 0.3),
    16: (0.9, 0.9, 0.5, 0.6),  # trachea - yellow
    51: (0.8, 0.2, 0.2, 0.5),  # heart - red
    53: (0.3, 0.3, 0.9, 0.6),  # pulmonary vein - blue
    54: (0.3, 0.3, 0.9, 0.6),  # brachiocephalic trunk - blue
    PE_LABEL: (1.0, 0.15, 0.0, 1.0),  # PE - bright red, fully opaque
}


def _colorize_slice(seg_slice: np.ndarray) -> np.ndarray:
    """Convert a label slice to an RGBA image.

    Args:
        seg_slice: 2D array of integer labels.

    Returns:
        (H, W, 4) float32 RGBA array.
    """
    h, w = seg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    # Background: dark gray
    rgba[seg_slice == 0] = [0.08, 0.08, 0.08, 1.0]

    # Lung lobes
    for lobe_label in [10, 11, 12, 13, 14]:
        mask = seg_slice == lobe_label
        if np.any(mask):
            rgba[mask] = [0.15, 0.18, 0.25, 1.0]

    # Lung vessels (label 119) - blue
    lung_vessel_mask = seg_slice == VESSEL_LABEL
    if np.any(lung_vessel_mask):
        rgba[lung_vessel_mask] = [0.25, 0.35, 0.6, 1.0]

    # Pulmonary vein (label 53) - darker blue
    pv_mask = seg_slice == 53
    if np.any(pv_mask):
        rgba[pv_mask] = [0.2, 0.3, 0.7, 1.0]

    # Brachiocephalic trunk (label 54) - dark red-blue
    bct_mask = seg_slice == 54
    if np.any(bct_mask):
        rgba[bct_mask] = [0.55, 0.15, 0.15, 1.0]

    vessel_mask = lung_vessel_mask | pv_mask | bct_mask

    # Heart
    if np.any(seg_slice == 51):
        rgba[seg_slice == 51] = [0.6, 0.2, 0.2, 1.0]

    # Trachea
    if np.any(seg_slice == 16):
        rgba[seg_slice == 16] = [0.7, 0.7, 0.3, 1.0]

    # Airways (label 120)
    airway_mask = seg_slice == AIRWAY_LABEL
    if np.any(airway_mask):
        rgba[airway_mask] = [0.7, 0.7, 0.3, 1.0]  # same yellow as trachea

    # Other soft tissue (anything else that's non-zero and not already colored)
    other = (seg_slice > 0) & ~np.isin(seg_slice, [10, 11, 12, 13, 14, 16, 51, 53, 54, VESSEL_LABEL, AIRWAY_LABEL, PE_LABEL]) & ~vessel_mask
    if np.any(other):
        rgba[other] = [0.35, 0.30, 0.25, 1.0]

    # PE (on top, bright red)
    pe_mask = seg_slice == PE_LABEL
    if np.any(pe_mask):
        rgba[pe_mask] = [1.0, 0.15, 0.0, 1.0]

    return rgba


def _add_pe_contour(ax, seg_slice: np.ndarray, color='yellow', linewidth=1.0):
    """Overlay PE contour on an axes for visibility."""
    pe_mask = (seg_slice == PE_LABEL).astype(np.float32)
    if np.any(pe_mask):
        ax.contour(pe_mask, levels=[0.5], colors=[color], linewidths=[linewidth])


def render_qc_slices(combined_seg_path: str,
                      config_id: str,
                      output_dir: str,
                      config_metadata: dict = None) -> str:
    """Render 3-view QC slices (axial/coronal/sagittal) centered on PE centroid.

    Args:
        combined_seg_path: Path to combined segmentation NIfTI.
        config_id: Config ID for labeling and filename.
        output_dir: Where to save the PNG.
        config_metadata: Optional dict with location/burden/occlusion for title.

    Returns:
        Path to saved PNG.
    """
    # Load in RAS canonical orientation for consistent L/R display
    seg_data, spacing = _load_canonical(combined_seg_path)
    # RAS axes: 0 = L→R, 1 = P→A, 2 = I→S
    # Flip axis 0 so it goes R→L (high index = Left) for radiological convention
    seg_data = seg_data[::-1, :, :]

    # Find PE centroid
    pe_mask = seg_data == PE_LABEL
    pe_voxels = np.sum(pe_mask)
    if pe_voxels == 0:
        cx, cy, cz = [s // 2 for s in seg_data.shape]
    else:
        coords = np.argwhere(pe_mask)
        cx, cy, cz = coords.mean(axis=0).astype(int)

    # Extract slices in RAS
    axial = seg_data[:, :, cz]       # z (I→S) fixed, shape (nx, ny)
    coronal = seg_data[:, cy, :]     # y (P→A) fixed, shape (nx, nz)
    sagittal = seg_data[cx, :, :]    # x (R→L) fixed, shape (ny, nz)

    # Build title
    title = f"{config_id}"
    if config_metadata:
        loc = config_metadata.get('location', '')
        bur = config_metadata.get('burden', '')
        occ = config_metadata.get('occlusion', '')
        vol = config_metadata.get('c2_pe_volume_ml', config_metadata.get('pe_volume_ml', ''))
        if isinstance(vol, float):
            vol = f"{vol:.3f} ml"
        title += f"\n{loc} | {bur} | {occ} | {vol}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Display with .T + origin='lower' → radiological convention:
    #   Axial:    left=R, right=L, bottom=P, top=A
    #   Coronal:  left=R, right=L, bottom=I, top=S
    #   Sagittal: left=P, right=A, bottom=I, top=S
    for ax, slc, view_name, aspect, show_lr in [
        (axes[0], axial.T, f"Axial (z={cz})", spacing[1] / spacing[0], True),
        (axes[1], coronal.T, f"Coronal (y={cy})", spacing[2] / spacing[0], True),
        (axes[2], sagittal.T, f"Sagittal (x={cx})", spacing[2] / spacing[1], False),
    ]:
        rgba = _colorize_slice(slc)
        ax.imshow(rgba, origin='lower', aspect=aspect)
        _add_pe_contour(ax, slc, color='yellow', linewidth=1.5)
        ax.set_title(view_name, fontsize=11)
        ax.axis('off')
        if show_lr:
            _add_lr_labels(ax)

    plt.tight_layout()
    output_path = Path(output_dir) / f"{config_id}_qc_slices.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='black')
    plt.close(fig)

    return str(output_path)


def render_summary_grid(config_results: List[dict],
                         output_dir: str,
                         patient_id: str) -> str:
    """Render a thumbnail grid showing one axial slice per config.

    Args:
        config_results: List of validation result dicts (must have
            'config_id', 'combined_seg_path', 'all_checks_pass').
        output_dir: Where to save the grid PNG.
        patient_id: Patient ID for the title.

    Returns:
        Path to saved PNG.
    """
    n = len(config_results)
    if n == 0:
        return ""

    ncols = min(7, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.5 * nrows))
    fig.suptitle(f"{patient_id} — QC Overview ({n} configs)",
                 fontsize=14, fontweight='bold', color='white')

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, result in enumerate(config_results):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        config_id = result['config_id']
        combined_path = result.get('combined_seg_path', '')
        passed = result.get('all_checks_pass', False)

        if combined_path and Path(combined_path).exists():
            seg_data, _ = _load_canonical(combined_path)
            # RAS: axis 0 = L→R, axis 2 = I→S; flip axis 0 for radiological
            seg_data = seg_data[::-1, :, :]
            pe_mask = seg_data == PE_LABEL

            if np.any(pe_mask):
                coords = np.argwhere(pe_mask)
                cz = int(coords[:, 2].mean())
            else:
                cz = seg_data.shape[2] // 2

            slc = seg_data[:, :, cz].T  # axial in radiological convention
            rgba = _colorize_slice(slc)
            ax.imshow(rgba, origin='lower', aspect='auto')
            _add_pe_contour(ax, slc, color='yellow', linewidth=0.8)
            _add_lr_labels(ax)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    color='white', fontsize=12, transform=ax.transAxes)

        # Title with pass/fail indicator
        short_id = config_id.split('_')[-1]  # e.g., cf001
        status = 'PASS' if passed else 'FAIL'
        color = '#44bb44' if passed else '#ee4444'
        loc = result.get('location', '')
        ax.set_title(f"{short_id} ({loc})\n{status}", fontsize=9,
                     color=color, fontweight='bold')
        ax.axis('off')

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    plt.tight_layout()
    output_path = Path(output_dir) / f"{patient_id}_qc_grid.png"
    fig.savefig(output_path, dpi=120, bbox_inches='tight',
                facecolor='black')
    plt.close(fig)

    return str(output_path)
