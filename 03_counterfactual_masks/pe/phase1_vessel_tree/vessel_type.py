"""Per-edge artery/vein classification from the TotalSegmentator v2 lung-vessel labels.

Each vessel-graph edge is labeled artery / vein / unknown by majority vote of the v2
segmentation labels (3 = lung_arteries, 4 = lung_veins) along the edge. The counterfactual
grid uses the labels to prefer artery (or unknown) edges when it places emboli.
"""

import os
import numpy as np
import networkx as nx
from typing import Tuple
from enum import Enum
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pe/ folder (for utils)
from utils.nifti_io import load_nifti


class VesselType(Enum):
    """Vessel-edge classification (from v2 arteries/veins labels)."""
    LIKELY_ARTERY = "likely_artery"
    UNKNOWN = "unknown"
    LIKELY_VEIN = "likely_vein"


def classify_edges_from_v2_labels(G: nx.Graph,
                                   v2_data: np.ndarray) -> nx.Graph:
    """Classify vessel edges as artery/vein using v2 segmentation labels.

    v2 labels: 1 = lung_airways, 2 = lung_airways_wall, 3 = lung_arteries, 4 = lung_veins.

    For each edge, computes the fraction of skeleton voxels
    in artery label (3) vs vein label (4). Classification:
      - >50% artery voxels → likely_artery
      - >50% vein voxels   → likely_vein
      - otherwise           → unknown (edge outside both masks)
    """
    print(f"Classifying {G.number_of_edges()} segments using v2 labels...")

    counts = {VesselType.LIKELY_ARTERY: 0, VesselType.UNKNOWN: 0, VesselType.LIKELY_VEIN: 0}

    for u, v, data in G.edges(data=True):
        if 'segment_coords' not in data:
            continue

        coords = data['segment_coords']
        n_total = len(coords)
        if n_total == 0:
            G.edges[u, v]['vessel_type'] = VesselType.UNKNOWN
            G.edges[u, v]['vessel_type_str'] = VesselType.UNKNOWN.value
            counts[VesselType.UNKNOWN] += 1
            continue

        labels_at_coords = [int(v2_data[c]) for c in coords]
        n_artery = sum(1 for l in labels_at_coords if l == 3)
        n_vein = sum(1 for l in labels_at_coords if l == 4)
        frac_artery = n_artery / n_total
        frac_vein = n_vein / n_total

        if frac_artery > frac_vein and frac_artery > 0.0:
            vtype = VesselType.LIKELY_ARTERY
        elif frac_vein > frac_artery and frac_vein > 0.0:
            vtype = VesselType.LIKELY_VEIN
        else:
            vtype = VesselType.UNKNOWN

        counts[vtype] += 1

        G.edges[u, v]['vessel_type'] = vtype
        G.edges[u, v]['vessel_type_str'] = vtype.value
        G.edges[u, v]['v2_frac_artery'] = round(frac_artery, 3)
        G.edges[u, v]['v2_frac_vein'] = round(frac_vein, 3)

    print(f"  Classification: {counts[VesselType.LIKELY_ARTERY]} artery, "
          f"{counts[VesselType.UNKNOWN]} unknown, {counts[VesselType.LIKELY_VEIN]} vein")

    return G


DEFAULT_OUTPUT_DIR = os.environ.get("CF_OUT", "/path/to/cf_mask_pipeline_output") + "/phase1_vessel_tree/output"


def add_v2_vessel_type_for_patient(patient_id: str,
                                    save_output: bool = True,
                                    output_dir: str = None) -> Tuple[nx.Graph, dict]:
    """Classify vessel edges (artery/vein/unknown) from the TotalSegmentator v2
    lung-vessel segmentation labels (3 = arteries, 4 = veins).
    """
    import pickle

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Load vessel graph
    graph_path = f"{output_dir}/{patient_id}_vessel_graph.pkl"
    print(f"Loading vessel graph: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    G = graph_data['graph']
    spacing = graph_data['spacing']
    affine = graph_data['affine']

    # Load v2 segmentation
    from utils.nifti_io import get_vessel_v2_path
    v2_path = get_vessel_v2_path(patient_id)
    print(f"Loading v2 vessel labels: {v2_path}")
    v2_data, _, _ = load_nifti(v2_path)

    # Classify edges
    G = classify_edges_from_v2_labels(G, v2_data)

    # Compute summary stats
    stats = {
        'n_likely_artery': sum(1 for _, _, d in G.edges(data=True)
                               if d.get('vessel_type') == VesselType.LIKELY_ARTERY),
        'n_unknown': sum(1 for _, _, d in G.edges(data=True)
                         if d.get('vessel_type') == VesselType.UNKNOWN),
        'n_likely_vein': sum(1 for _, _, d in G.edges(data=True)
                             if d.get('vessel_type') == VesselType.LIKELY_VEIN),
        'method': 'v2_labels',
    }

    if save_output:
        output_path = f"{output_dir}/{patient_id}_vessel_graph_scored.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({'graph': G, 'spacing': spacing, 'affine': affine}, f)
        print(f"Saved scored graph: {output_path}")
        stats['output_path'] = output_path

    return G, stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {os.path.basename(__file__)} <STUDY_ID>")
    patient_id = sys.argv[1]
    G, stats = add_v2_vessel_type_for_patient(patient_id)
    print(f"\nArtery/vein classification (v2 labels) complete. Stats: {stats}")
