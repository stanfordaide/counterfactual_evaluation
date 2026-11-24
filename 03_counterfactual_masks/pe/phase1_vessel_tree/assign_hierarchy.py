"""Assign vessel hierarchy levels based on radius.

Vessel generation classification:
- Central (>8 mm): Main pulmonary artery / trunk
- Lobar (4-8 mm): Lobar branches (RUL, RML, RLL, LUL, LLL feeders)
- Segmental (2-4 mm): Segmental branches
- Subsegmental (<2 mm): Peripheral/subsegmental vessels
"""

import os
import networkx as nx
from enum import IntEnum
from typing import Tuple
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pe/ folder (for utils)


class VesselGeneration(IntEnum):
    """Vessel hierarchy levels based on radius."""
    CENTRAL = 0      # Main PA, >8mm
    LOBAR = 1        # Lobar branches, 4-8mm
    SEGMENTAL = 2    # Segmental, 2-4mm
    SUBSEGMENTAL = 3 # Peripheral, <2mm


# Radius thresholds in mm
RADIUS_THRESHOLDS = {
    VesselGeneration.CENTRAL: 8.0,
    VesselGeneration.LOBAR: 4.0,
    VesselGeneration.SEGMENTAL: 2.0,
    VesselGeneration.SUBSEGMENTAL: 0.0,
}


def classify_generation(radius_mm: float) -> VesselGeneration:
    """
    Classify vessel generation based on radius.

    Args:
        radius_mm: Vessel radius in mm

    Returns:
        VesselGeneration enum
    """
    if radius_mm >= RADIUS_THRESHOLDS[VesselGeneration.CENTRAL]:
        return VesselGeneration.CENTRAL
    elif radius_mm >= RADIUS_THRESHOLDS[VesselGeneration.LOBAR]:
        return VesselGeneration.LOBAR
    elif radius_mm >= RADIUS_THRESHOLDS[VesselGeneration.SEGMENTAL]:
        return VesselGeneration.SEGMENTAL
    else:
        return VesselGeneration.SUBSEGMENTAL


def add_hierarchy_to_graph(G: nx.Graph) -> nx.Graph:
    """
    Add vessel hierarchy (generation) to all graph edges based on radius.

    Args:
        G: Vessel graph with radius_mean on edges

    Returns:
        Graph with added edge attributes:
        - generation: VesselGeneration enum (int)
        - generation_name: String name
    """
    print(f"Assigning hierarchy to {G.number_of_edges()} segments...")

    counts = {gen: 0 for gen in VesselGeneration}

    for u, v, data in G.edges(data=True):
        if 'radius_mean' not in data:
            continue

        generation = classify_generation(data['radius_mean'])
        counts[generation] += 1

        G.edges[u, v]['generation'] = int(generation)
        G.edges[u, v]['generation_name'] = generation.name.lower()

    print(f"  Central (>8mm): {counts[VesselGeneration.CENTRAL]}")
    print(f"  Lobar (4-8mm): {counts[VesselGeneration.LOBAR]}")
    print(f"  Segmental (2-4mm): {counts[VesselGeneration.SEGMENTAL]}")
    print(f"  Subsegmental (<2mm): {counts[VesselGeneration.SUBSEGMENTAL]}")

    return G


DEFAULT_OUTPUT_DIR = os.environ.get("CF_OUT", "/path/to/cf_mask_pipeline_output") + "/phase1_vessel_tree/output"


def add_hierarchy_for_patient(patient_id: str, save_output: bool = True,
                              output_dir: str = None) -> Tuple[nx.Graph, dict]:
    """
    Load graph and add hierarchy labels.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Load vessel graph (with artery/vein classification if available)
    graph_path = f"{output_dir}/{patient_id}_vessel_graph_scored.pkl"
    try:
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        print(f"Loaded scored graph: {graph_path}")
    except FileNotFoundError:
        # Fall back to basic graph
        graph_path = f"{output_dir}/{patient_id}_vessel_graph.pkl"
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        print(f"Loaded basic graph: {graph_path}")

    G = graph_data['graph']
    spacing = graph_data['spacing']
    affine = graph_data['affine']

    # Add hierarchy
    G = add_hierarchy_to_graph(G)

    # Compute stats
    stats = {
        'n_central': sum(1 for _, _, d in G.edges(data=True) if d.get('generation') == VesselGeneration.CENTRAL),
        'n_lobar': sum(1 for _, _, d in G.edges(data=True) if d.get('generation') == VesselGeneration.LOBAR),
        'n_segmental': sum(1 for _, _, d in G.edges(data=True) if d.get('generation') == VesselGeneration.SEGMENTAL),
        'n_subsegmental': sum(1 for _, _, d in G.edges(data=True) if d.get('generation') == VesselGeneration.SUBSEGMENTAL),
    }

    if save_output:
        output_path = f"{output_dir}/{patient_id}_vessel_graph_hierarchy.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({'graph': G, 'spacing': spacing, 'affine': affine}, f)
        print(f"Saved graph with hierarchy: {output_path}")
        stats['output_path'] = output_path

    return G, stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {os.path.basename(__file__)} <STUDY_ID>")
    patient_id = sys.argv[1]
    G, stats = add_hierarchy_for_patient(patient_id)
    print(f"\nHierarchy assignment complete. Stats: {stats}")
