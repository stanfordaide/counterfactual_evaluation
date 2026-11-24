"""Phase 1: Vessel Tree Extraction Pipeline

Runs all Phase 1 steps for a patient:
1. Morphological cleaning of vessel mask
2. 3D skeletonization
3. Graph extraction from skeleton
4. Artery/vein classification (from the v2 arteries/veins segmentation labels)
5. Hierarchy assignment (radius-based)
6. Lobe labeling

Usage:
    python run_phase1.py <STUDY_ID>
"""

import os
import argparse
import json
import time
from pathlib import Path

# Phase 1 modules
from vessel_preprocessing import preprocess_for_patient
from skeletonize import skeletonize_for_patient
from skeleton_to_graph import build_graph_for_patient
from vessel_type import add_v2_vessel_type_for_patient
from assign_hierarchy import add_hierarchy_for_patient
from assign_lobes import add_lobe_labels_for_patient


DEFAULT_OUTPUT_DIR = os.environ.get("CF_OUT", "/path/to/cf_mask_pipeline_output") + "/phase1_vessel_tree/output"


def run_phase1_pipeline(patient_id: str,
                        min_component_size: int = 100,
                        closing_iterations: int = 1,
                        dilation_iterations: int = 0,
                        output_dir: str = None) -> dict:
    """
    Run the complete Phase 1 pipeline for a patient.

    Args:
        patient_id: Patient ID (e.g., <STUDY_ID>)
        min_component_size: Minimum vessel component size in voxels
        closing_iterations: Morphological closing iterations
        dilation_iterations: Morphological dilation iterations (0 = off)
        output_dir: Output directory for all Phase 1 files (default: phase1_vessel_tree/output/)

    Returns:
        Dictionary with all pipeline statistics
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_stats = {
        'patient_id': patient_id,
        'parameters': {
            'min_component_size': min_component_size,
            'closing_iterations': closing_iterations,
            'dilation_iterations': dilation_iterations,
        }
    }

    start_time = time.time()

    # Step 1: Preprocessing
    print("\n" + "="*60)
    print("STEP 1: Vessel mask preprocessing")
    print("="*60)
    cleaned_mask, affine, header, preproc_stats = preprocess_for_patient(
        patient_id,
        min_component_size=min_component_size,
        closing_iterations=closing_iterations,
        dilation_iterations=dilation_iterations,
        output_dir=output_dir,
    )
    all_stats['preprocessing'] = preproc_stats

    # Step 2: Skeletonization
    print("\n" + "="*60)
    print("STEP 2: 3D Skeletonization")
    print("="*60)
    skeleton, radius, _, _, skel_stats = skeletonize_for_patient(
        patient_id, output_dir=output_dir)
    all_stats['skeletonization'] = skel_stats

    # Step 3: Graph extraction
    print("\n" + "="*60)
    print("STEP 3: Graph extraction from skeleton")
    print("="*60)
    G, spacing, affine, graph_stats = build_graph_for_patient(
        patient_id, output_dir=output_dir)
    all_stats['graph'] = graph_stats

    # Step 4: Artery/vein classification (from the v2 arteries/veins labels)
    print("\n" + "="*60)
    print("STEP 4: Artery/vein classification (v2 label-based)")
    print("="*60)
    G, vessel_type_stats = add_v2_vessel_type_for_patient(
        patient_id, output_dir=output_dir,
    )
    all_stats['vessel_classification'] = vessel_type_stats

    # Step 5: Hierarchy assignment
    print("\n" + "="*60)
    print("STEP 5: Vessel hierarchy assignment (radius-based)")
    print("="*60)
    G, hierarchy_stats = add_hierarchy_for_patient(
        patient_id, output_dir=output_dir)
    all_stats['hierarchy'] = hierarchy_stats

    # Step 6: Lobe labeling
    print("\n" + "="*60)
    print("STEP 6: Lobe label assignment")
    print("="*60)
    G, lobe_stats = add_lobe_labels_for_patient(
        patient_id, output_dir=output_dir)
    all_stats['lobes'] = lobe_stats

    # Summary
    elapsed = time.time() - start_time
    all_stats['elapsed_seconds'] = elapsed

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save summary stats
    stats_path = f"{output_dir}/{patient_id}_phase1_stats.json"
    with open(stats_path, 'w') as f:
        # Convert non-serializable items
        json.dump(all_stats, f, indent=2, default=str)
    print(f"Saved stats: {stats_path}")

    return all_stats


def print_graph_summary(patient_id: str, output_dir: str = None):
    """Print a summary of the final vessel graph."""
    import pickle

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    graph_path = f"{output_dir}/{patient_id}_vessel_graph_final.pkl"

    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    G = data['graph']

    print("\n" + "="*60)
    print("FINAL GRAPH SUMMARY")
    print("="*60)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges (segments): {G.number_of_edges()}")

    # Count by vessel type
    vessel_types = {}
    for _, _, d in G.edges(data=True):
        vt = d.get('vessel_type_str', 'unknown')
        vessel_types[vt] = vessel_types.get(vt, 0) + 1
    print(f"\nVessel type (artery/vein):")
    for vt, count in sorted(vessel_types.items()):
        print(f"  {vt}: {count}")

    # Count by hierarchy
    generations = {}
    for _, _, d in G.edges(data=True):
        gen = d.get('generation_name', 'unknown')
        generations[gen] = generations.get(gen, 0) + 1
    print(f"\nHierarchy (generation):")
    for gen, count in sorted(generations.items()):
        print(f"  {gen}: {count}")

    # Count by lobe
    lobes = {}
    for _, _, d in G.edges(data=True):
        lobe = d.get('lobe_name', 'unknown')
        lobes[lobe] = lobes.get(lobe, 0) + 1
    print(f"\nLobe distribution:")
    for lobe, count in sorted(lobes.items()):
        print(f"  {lobe}: {count}")

    # Segments suitable for PE injection (likely_artery or unknown, not vein)
    injectable = sum(1 for _, _, d in G.edges(data=True)
                     if d.get('vessel_type_str') in ['likely_artery', 'unknown'])
    print(f"\nSegments suitable for PE injection: {injectable}/{G.number_of_edges()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 vessel tree extraction")
    parser.add_argument("patient_id", help="Patient ID (e.g., <STUDY_ID>)")
    parser.add_argument("--min-component-size", type=int, default=100,
                        help="Minimum vessel component size in voxels")
    parser.add_argument("--closing-iterations", type=int, default=1,
                        help="Morphological closing iterations")
    parser.add_argument("--dilation-iterations", type=int, default=0,
                        help="Morphological dilation iterations (0 = off)")

    args = parser.parse_args()

    stats = run_phase1_pipeline(
        args.patient_id,
        min_component_size=args.min_component_size,
        closing_iterations=args.closing_iterations,
        dilation_iterations=args.dilation_iterations,
    )

    print_graph_summary(args.patient_id)
