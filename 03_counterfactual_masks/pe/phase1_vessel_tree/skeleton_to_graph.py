"""Extract graph structure from vessel skeleton.

Converts 3D skeleton into a networkx graph where:
- Nodes = branch points and endpoints
- Edges = vessel segments between nodes
- Edge attributes include: voxel coordinates, mean radius, length
"""

import os
import numpy as np
import networkx as nx
from scipy import ndimage
from typing import Tuple, List
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pe/ folder (for utils)
from utils.nifti_io import load_nifti


def find_branch_and_endpoints(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find branch points (3+ neighbors) and endpoints (1 neighbor) in skeleton.

    Args:
        skeleton: Binary skeleton mask

    Returns:
        branch_points: Boolean mask of branch points
        endpoints: Boolean mask of endpoints
    """
    # 26-connectivity kernel for neighbor counting
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    # Count neighbors at each skeleton voxel
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    neighbor_count = neighbor_count * skeleton  # Only count at skeleton voxels

    # Branch points: 3+ neighbors, Endpoints: exactly 1 neighbor
    branch_points = (neighbor_count >= 3) & (skeleton > 0)
    endpoints = (neighbor_count == 1) & (skeleton > 0)

    print(f"  Found {np.sum(branch_points)} branch points, {np.sum(endpoints)} endpoints")
    return branch_points, endpoints


def trace_segment(skeleton: np.ndarray, start: Tuple[int, int, int],
                  visited: np.ndarray, branch_points: np.ndarray,
                  endpoints: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Trace a single vessel segment from start point until reaching a branch/endpoint.

    Returns list of voxel coordinates along the segment.
    """
    segment = [start]
    visited[start] = True

    current = start
    while True:
        # Find unvisited skeleton neighbors
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = current[0] + dz, current[1] + dy, current[2] + dx
                    if (0 <= nz < skeleton.shape[0] and
                        0 <= ny < skeleton.shape[1] and
                        0 <= nx < skeleton.shape[2]):
                        if skeleton[nz, ny, nx] and not visited[nz, ny, nx]:
                            neighbors.append((nz, ny, nx))

        if len(neighbors) == 0:
            # Dead end or reached visited area
            break
        elif len(neighbors) == 1:
            # Continue along segment
            current = neighbors[0]
            segment.append(current)
            visited[current] = True

            # Stop if we hit a branch point or endpoint
            if branch_points[current] or endpoints[current]:
                break
        else:
            # Multiple unvisited neighbors - we're at an unmarked junction
            # Pick one and continue (others will be traced separately)
            current = neighbors[0]
            segment.append(current)
            visited[current] = True
            break

    return segment


def skeleton_to_graph(skeleton: np.ndarray, radius_map: np.ndarray,
                      spacing: Tuple[float, float, float]) -> nx.Graph:
    """
    Convert skeleton to networkx graph.

    Args:
        skeleton: Binary skeleton mask
        radius_map: Radius at each skeleton voxel (in mm)
        spacing: Voxel spacing (z, y, x) in mm

    Returns:
        G: networkx Graph with nodes and edges
    """
    print("Building graph from skeleton...")

    # Find special points
    branch_points, endpoints = find_branch_and_endpoints(skeleton)

    # Create graph
    G = nx.Graph()

    # Add all branch points and endpoints as nodes
    node_coords = np.argwhere(branch_points | endpoints)
    node_id_map = {}  # (z, y, x) -> node_id

    for i, coord in enumerate(node_coords):
        coord_tuple = tuple(coord)
        node_id_map[coord_tuple] = i
        G.add_node(i,
                   coord=coord_tuple,
                   coord_mm=tuple(coord * spacing),
                   radius=float(radius_map[coord_tuple]),
                   is_branch=bool(branch_points[coord_tuple]),
                   is_endpoint=bool(endpoints[coord_tuple]))

    print(f"  Added {G.number_of_nodes()} nodes")

    # Trace segments between nodes
    visited = np.zeros_like(skeleton, dtype=bool)
    # Mark nodes as visited but traceable
    visited[branch_points | endpoints] = False

    edge_count = 0
    for coord in node_coords:
        coord_tuple = tuple(coord)
        visited[coord_tuple] = True

        # Try to trace from this node in all directions
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue

                    n_z, n_y, n_x = coord[0] + dz, coord[1] + dy, coord[2] + dx
                    neighbor = (n_z, n_y, n_x)

                    if (0 <= n_z < skeleton.shape[0] and
                        0 <= n_y < skeleton.shape[1] and
                        0 <= n_x < skeleton.shape[2]):

                        if skeleton[neighbor] and not visited[neighbor]:
                            # Trace segment from this neighbor
                            segment = [coord_tuple] + trace_segment(
                                skeleton, neighbor, visited, branch_points, endpoints)

                            # Check if segment ends at another node
                            end_coord = segment[-1]
                            if end_coord in node_id_map and end_coord != coord_tuple:
                                start_node = node_id_map[coord_tuple]
                                end_node = node_id_map[end_coord]

                                # Compute segment properties
                                segment_radii = [radius_map[c] for c in segment]
                                segment_coords_mm = [tuple(np.array(c) * spacing) for c in segment]

                                # Compute length along segment
                                length_mm = 0.0
                                for j in range(1, len(segment)):
                                    diff = np.array(segment[j]) - np.array(segment[j-1])
                                    length_mm += np.sqrt(np.sum((diff * spacing) ** 2))

                                G.add_edge(start_node, end_node,
                                           segment_coords=segment,
                                           segment_coords_mm=segment_coords_mm,
                                           length_mm=length_mm,
                                           radius_mean=float(np.mean(segment_radii)),
                                           radius_min=float(np.min(segment_radii)),
                                           radius_max=float(np.max(segment_radii)),
                                           n_voxels=len(segment))
                                edge_count += 1

    print(f"  Added {G.number_of_edges()} edges")

    # Add isolated skeleton regions as single-node components if any
    # (skeleton voxels not connected to any node)
    labeled_skeleton, n_comp = ndimage.label(skeleton)
    node_labels = set(labeled_skeleton[branch_points | endpoints])

    return G


DEFAULT_OUTPUT_DIR = os.environ.get("CF_OUT", "/path/to/cf_mask_pipeline_output") + "/phase1_vessel_tree/output"


def build_graph_for_patient(patient_id: str, save_output: bool = True,
                            output_dir: str = None) -> Tuple[nx.Graph, dict]:
    """
    Build vessel graph for a patient from skeleton.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Load skeleton and radius
    skel_path = f"{output_dir}/{patient_id}_skeleton.nii.gz"
    radius_path = f"{output_dir}/{patient_id}_radius.nii.gz"

    print(f"Loading skeleton: {skel_path}")
    skeleton, affine, header = load_nifti(skel_path)
    skeleton = skeleton.astype(bool)

    print(f"Loading radius map: {radius_path}")
    radius_map, _, _ = load_nifti(radius_path)

    spacing = np.array(header.get_zooms()[:3])
    print(f"  Spacing: {spacing} mm")

    # Build graph
    G = skeleton_to_graph(skeleton, radius_map, spacing)

    # Compute stats
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_components': nx.number_connected_components(G),
    }

    if G.number_of_edges() > 0:
        lengths = [G.edges[e]['length_mm'] for e in G.edges]
        radii = [G.edges[e]['radius_mean'] for e in G.edges]
        stats['total_length_mm'] = sum(lengths)
        stats['mean_segment_length_mm'] = np.mean(lengths)
        stats['mean_radius_mm'] = np.mean(radii)

    print(f"\nGraph stats: {stats}")

    if save_output:
        graph_path = f"{output_dir}/{patient_id}_vessel_graph.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump({'graph': G, 'spacing': spacing, 'affine': affine}, f)
        print(f"Saved graph: {graph_path}")
        stats['graph_path'] = graph_path

    return G, spacing, affine, stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {os.path.basename(__file__)} <STUDY_ID>")
    patient_id = sys.argv[1]
    G, spacing, affine, stats = build_graph_for_patient(patient_id)
    print(f"\nGraph building complete.")
