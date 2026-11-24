"""Assign lung lobe labels to vessel segments.

Maps each vessel segment to the lung lobe it primarily resides in,
using the TotalSegmentator anatomy mask.
"""

import os
import numpy as np
import networkx as nx
from typing import Tuple, Dict
from collections import Counter
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pe/ folder (for utils)
from utils.nifti_io import load_nifti


# Lobe label to short name mapping
LOBE_SHORT_NAMES = {
    10: "LUL",  # lung_upper_lobe_left
    11: "LLL",  # lung_lower_lobe_left
    12: "RUL",  # lung_upper_lobe_right
    13: "RML",  # lung_middle_lobe_right
    14: "RLL",  # lung_lower_lobe_right
    0: "central",  # Outside lobes (hilar region)
}


def get_lobe_at_coords(coords: list, anatomy_mask: np.ndarray) -> Dict:
    """
    Determine the primary lobe for a list of coordinates.

    Args:
        coords: List of (z, y, x) voxel coordinates
        anatomy_mask: Anatomy segmentation mask

    Returns:
        Dictionary with lobe statistics
    """
    lobe_labels = [10, 11, 12, 13, 14]  # All lobe labels

    # Get anatomy labels at each coordinate
    labels_at_coords = [anatomy_mask[coord] for coord in coords]

    # Count labels in lobe range
    lobe_counts = Counter()
    for label in labels_at_coords:
        if label in lobe_labels:
            lobe_counts[label] += 1
        else:
            lobe_counts[0] += 1  # Central/outside lobes

    total = len(coords)
    primary_lobe = lobe_counts.most_common(1)[0][0] if lobe_counts else 0

    return {
        'primary_lobe': int(primary_lobe),
        'primary_lobe_name': LOBE_SHORT_NAMES.get(primary_lobe, "unknown"),
        'lobe_fraction': lobe_counts[primary_lobe] / total if total > 0 else 0,
        'lobe_counts': dict(lobe_counts),
    }


def add_lobe_labels_to_graph(G: nx.Graph, anatomy_mask: np.ndarray) -> nx.Graph:
    """
    Add lobe labels to all graph edges.

    Args:
        G: Vessel graph with segment_coords on edges
        anatomy_mask: Anatomy segmentation mask

    Returns:
        Graph with added edge attributes:
        - lobe: Primary lobe label (int)
        - lobe_name: Short name (LUL, RML, etc.)
        - lobe_fraction: Fraction of segment in primary lobe
    """
    print(f"Assigning lobe labels to {G.number_of_edges()} segments...")

    lobe_counts = Counter()

    for u, v, data in G.edges(data=True):
        if 'segment_coords' not in data:
            continue

        lobe_info = get_lobe_at_coords(data['segment_coords'], anatomy_mask)
        lobe_counts[lobe_info['primary_lobe_name']] += 1

        G.edges[u, v]['lobe'] = lobe_info['primary_lobe']
        G.edges[u, v]['lobe_name'] = lobe_info['primary_lobe_name']
        G.edges[u, v]['lobe_fraction'] = lobe_info['lobe_fraction']

    print(f"  Lobe distribution: {dict(lobe_counts)}")

    return G


DEFAULT_OUTPUT_DIR = os.environ.get("CF_OUT", "/path/to/cf_mask_pipeline_output") + "/phase1_vessel_tree/output"


def add_lobe_labels_for_patient(patient_id: str, save_output: bool = True,
                                output_dir: str = None) -> Tuple[nx.Graph, dict]:
    """
    Load graph and anatomy mask, add lobe labels.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Load vessel graph (with hierarchy if available)
    graph_path = f"{output_dir}/{patient_id}_vessel_graph_hierarchy.pkl"
    try:
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        print(f"Loaded graph with hierarchy: {graph_path}")
    except FileNotFoundError:
        # Fall back to scored graph
        graph_path = f"{output_dir}/{patient_id}_vessel_graph_scored.pkl"
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        print(f"Loaded scored graph: {graph_path}")

    G = graph_data['graph']
    spacing = graph_data['spacing']
    affine = graph_data['affine']

    # Load anatomy mask
    from utils.nifti_io import get_anatomy_path
    anatomy_path = get_anatomy_path(patient_id)
    print(f"Loading anatomy mask: {anatomy_path}")
    anatomy_mask, _, _ = load_nifti(anatomy_path)
    anatomy_mask = anatomy_mask.astype(np.int32)

    # Add lobe labels
    G = add_lobe_labels_to_graph(G, anatomy_mask)

    # Compute stats
    stats = {lobe: 0 for lobe in LOBE_SHORT_NAMES.values()}
    for _, _, d in G.edges(data=True):
        lobe_name = d.get('lobe_name', 'unknown')
        if lobe_name in stats:
            stats[lobe_name] += 1

    if save_output:
        output_path = f"{output_dir}/{patient_id}_vessel_graph_final.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({'graph': G, 'spacing': spacing, 'affine': affine}, f)
        print(f"Saved final graph: {output_path}")
        stats['output_path'] = output_path

    return G, stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {os.path.basename(__file__)} <STUDY_ID>")
    patient_id = sys.argv[1]
    G, stats = add_lobe_labels_for_patient(patient_id)
    print(f"\nLobe assignment complete. Stats: {stats}")
