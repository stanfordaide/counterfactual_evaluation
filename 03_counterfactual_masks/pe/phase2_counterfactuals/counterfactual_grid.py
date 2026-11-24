"""Phase 2 -- Counterfactual PE grid generation.

Given a patient's vessel graph (from Phase 1), generates the 16-configuration
counterfactual grid that sweeps location, occlusion degree, and burden.

Usage:
    python phase2_counterfactuals/counterfactual_grid.py <STUDY_ID>
    python phase2_counterfactuals/counterfactual_grid.py <STUDY_ID> --seed 42
"""

import os
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)
# Phase 1 modules must be importable for pickle deserialization (VesselType enum)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "phase1_vessel_tree"))
from phase2_counterfactuals.axis_definitions import (
    PELocation, OcclusionDegree, Burden, LOCATION_NAMES,
    OCCLUSION_FRACTION_RANGE, OCCLUSION_QANADLI_WEIGHT, SegmentSpec,
    CounterfactualConfig, severity_from_qanadli,
)

# Default paths
PHASE1_OUTPUT = Path(os.environ.get("CF_OUT", "/path/to/cf_mask_pipeline_output")) / "phase1_vessel_tree" / "output"
PHASE2_OUTPUT = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------------------
# EdgeCatalog: indexed view of graph edges
# ---------------------------------------------------------------------------

class EdgeCatalog:
    """Pre-indexed view of vessel graph edges for fast querying.

    Indexes edges by (generation, lobe_name) and tracks which edges
    are artery-like (suitable for PE injection).
    """

    def __init__(self, G: nx.Graph):
        self.G = G
        self._by_gen_lobe: Dict[Tuple[int, str], List[Tuple[int, int]]] = {}
        self._by_gen: Dict[int, List[Tuple[int, int]]] = {}
        self._artery_edges: set = set()
        self._edge_data: Dict[Tuple[int, int], dict] = {}
        self._build_index()

    def _build_index(self):
        for u, v, data in self.G.edges(data=True):
            gen = data.get('generation')
            if gen is None:
                continue

            lobe = data.get('lobe_name', 'unknown')
            vtype = data.get('vessel_type_str', 'unknown')
            key = (u, v)

            self._edge_data[key] = data
            self._by_gen.setdefault(gen, []).append(key)
            self._by_gen_lobe.setdefault((gen, lobe), []).append(key)

            if vtype in ('likely_artery', 'unknown'):
                self._artery_edges.add(key)

    def query(self,
              generation: Optional[int] = None,
              lobe_name: Optional[str] = None,
              artery_only: bool = True) -> List[Tuple[int, int]]:
        """Return edge keys matching criteria."""
        if generation is not None and lobe_name is not None:
            candidates = self._by_gen_lobe.get((generation, lobe_name), [])
        elif generation is not None:
            candidates = self._by_gen.get(generation, [])
        else:
            candidates = list(self._edge_data.keys())

        if artery_only:
            candidates = [e for e in candidates if e in self._artery_edges]
        return candidates

    def get_edge_info(self, edge_key: Tuple[int, int]) -> dict:
        """Return stored edge attributes."""
        return self._edge_data.get(edge_key, {})

    def get_connected_subsets(self,
                              edges: List[Tuple[int, int]],
                              max_subset_size: int = 15) -> List[List[Tuple[int, int]]]:
        """Find connected components among a set of edges.

        For realistic PE placement, we prefer edges that form a
        connected subtree (PE flows distally from an occlusion point).
        Returns components sorted by size descending.
        """
        if not edges:
            return []

        edge_set = set(edges)
        subG = nx.Graph()
        for u, v in edges:
            subG.add_edge(u, v)

        result = []
        for comp_nodes in nx.connected_components(subG):
            comp_edges = [
                (u, v) for u, v in edge_set
                if u in comp_nodes and v in comp_nodes
            ]
            if comp_edges:
                result.append(comp_edges[:max_subset_size])

        result.sort(key=len, reverse=True)
        return result

    def summary(self) -> dict:
        """Return summary counts by (generation, lobe)."""
        summary = {}
        for (gen, lobe), edges in sorted(self._by_gen_lobe.items()):
            artery_count = sum(1 for e in edges if e in self._artery_edges)
            summary[(gen, lobe)] = {
                'total': len(edges),
                'artery_like': artery_count,
            }
        return summary


# ---------------------------------------------------------------------------
# Qanadli score computation
# ---------------------------------------------------------------------------

def compute_qanadli_score(segments: List[SegmentSpec]) -> Tuple[int, int, float, str]:
    """Compute Qanadli-style score from affected segments.

    Standard Qanadli index: 20 pulmonary artery segments (10 per side),
    each scored 0/1/2 (none/partial/complete). Max = 40.

    Our approach: group graph edges by (lobe, generation_slot) where
    subsegmental edges roll up to segmental level. Each unique slot
    counts as one Qanadli segment, weighted by max occlusion in that slot.

    Returns:
        (raw_score, max_score, percent, severity_category)
    """
    if not segments:
        return (0, 40, 0.0, "none")

    # Group by (lobe, generation_slot), take max occlusion per group
    affected_slots: Dict[Tuple[str, int], OcclusionDegree] = {}
    for seg in segments:
        # Collapse subsegmental -> segmental for Qanadli counting
        gen_slot = min(seg.generation, 2)
        key = (seg.lobe_name, gen_slot)

        current = affected_slots.get(key)
        if current is None or \
           OCCLUSION_QANADLI_WEIGHT[seg.occlusion] > OCCLUSION_QANADLI_WEIGHT[current]:
            affected_slots[key] = seg.occlusion

    raw_score = sum(OCCLUSION_QANADLI_WEIGHT[occ] for occ in affected_slots.values())
    raw_score = min(raw_score, 40)
    percent = (raw_score / 40) * 100
    category = severity_from_qanadli(raw_score)

    return (raw_score, 40, percent, category)


# ---------------------------------------------------------------------------
# Edge selection helpers
# ---------------------------------------------------------------------------

def _select_edges(catalog: EdgeCatalog,
                  generation: int,
                  lobe_name: Optional[str],
                  n_edges: int,
                  rng: np.random.Generator,
                  prefer_connected: bool = True,
                  artery_only: bool = True) -> List[Tuple[int, int]]:
    """Select n_edges from the catalog, preferring connected subsets.

    Falls back to relaxing artery_only if no candidates found.
    """
    candidates = catalog.query(generation=generation, lobe_name=lobe_name,
                               artery_only=artery_only)

    # Fallback: relax artery filter
    if not candidates and artery_only:
        candidates = catalog.query(generation=generation, lobe_name=lobe_name,
                                   artery_only=False)
    if not candidates:
        return []

    if n_edges == 1:
        return [candidates[rng.integers(len(candidates))]]

    if prefer_connected and n_edges > 1:
        components = catalog.get_connected_subsets(candidates, max_subset_size=n_edges)
        for comp in components:
            if len(comp) >= n_edges:
                indices = rng.choice(len(comp), size=n_edges, replace=False)
                return [comp[i] for i in indices]
        # No single component large enough -- take from largest first
        all_edges = []
        for comp in components:
            all_edges.extend(comp)
            if len(all_edges) >= n_edges:
                break
        return all_edges[:n_edges]

    # Random selection
    n_select = min(n_edges, len(candidates))
    indices = rng.choice(len(candidates), size=n_select, replace=False)
    return [candidates[i] for i in indices]


def _make_segment_specs(catalog: EdgeCatalog,
                        edge_keys: List[Tuple[int, int]],
                        occlusion: OcclusionDegree,
                        acute: bool = False) -> List[SegmentSpec]:
    """Convert edge keys to SegmentSpec objects."""
    specs = []
    for ek in edge_keys:
        data = catalog.get_edge_info(ek)
        specs.append(SegmentSpec(
            edge_key=ek,
            generation=data.get('generation', 3),
            generation_name=data.get('generation_name', 'subsegmental'),
            lobe_name=data.get('lobe_name', 'unknown'),
            vessel_type_str=data.get('vessel_type_str', 'unknown'),
            radius_mean_mm=data.get('radius_mean', 0.0),
            length_mm=data.get('length_mm', 0.0),
            n_voxels=data.get('n_voxels', 0),
            occlusion=occlusion,
            acute=acute,
        ))
    return specs


# ---------------------------------------------------------------------------
# Per-location config generators
# ---------------------------------------------------------------------------

def _generate_central_configs(patient_id, catalog, rng, next_id):
    """Central/saddle PE: 0-2 configs. Not empirically observed."""
    configs = []
    central_edges = catalog.query(generation=0, artery_only=False)

    if not central_edges:
        print("  Central: no generation-0 edges found -- skipping")
        return configs

    for occlusion in [OcclusionDegree.PARTIAL, OcclusionDegree.COMPLETE]:
        edges = _select_edges(catalog, 0, None, 1, rng, artery_only=False)
        if edges:
            segments = _make_segment_specs(catalog, edges, occlusion)
            configs.append(CounterfactualConfig(
                config_id=next_id(),
                patient_id=patient_id,
                location=PELocation.CENTRAL,
                occlusion=occlusion,
                burden=Burden.ISOLATED,
                target_lobes=['central'],
                segments=segments,
                empirically_observed=False,
                clinical_note="Central/saddle PE: rare but life-threatening. "
                              "Not observed in real-PE empirical data.",
            ))

    return configs


def _generate_lobar_configs(patient_id, catalog, rng, next_id):
    """Lobar PE: 0-4 configs (1-2 lobes x 2 occlusions). Not empirically observed."""
    configs = []
    lobar_edges = catalog.query(generation=1, artery_only=True)

    if not lobar_edges:
        # Try without artery filter
        lobar_edges = catalog.query(generation=1, artery_only=False)
    if not lobar_edges:
        print("  Lobar: no generation-1 edges found -- skipping")
        return configs

    # Find which lobes have lobar edges
    lobes_available = set()
    for ek in lobar_edges:
        info = catalog.get_edge_info(ek)
        lobes_available.add(info.get('lobe_name', 'central'))

    # Pick up to 2 lobes (one right, one left if possible)
    right = [l for l in lobes_available if l.startswith('R')]
    left = [l for l in lobes_available if l.startswith('L')]

    target_lobes = []
    if right:
        target_lobes.append(str(rng.choice(right)))
    if left:
        target_lobes.append(str(rng.choice(left)))
    if not target_lobes:
        target_lobes.append(str(rng.choice(list(lobes_available))))

    for lobe in target_lobes:
        for occlusion in [OcclusionDegree.PARTIAL, OcclusionDegree.COMPLETE]:
            edges = _select_edges(catalog, 1, lobe, 1, rng, artery_only=False)
            if edges:
                segments = _make_segment_specs(catalog, edges, occlusion)
                configs.append(CounterfactualConfig(
                    config_id=next_id(),
                    patient_id=patient_id,
                    location=PELocation.LOBAR,
                    occlusion=occlusion,
                    burden=Burden.ISOLATED,
                    target_lobes=[lobe],
                    segments=segments,
                    empirically_observed=False,
                    clinical_note=f"Lobar PE in {lobe}: not observed in real-PE data.",
                ))

    return configs


def _generate_segmental_configs(patient_id, catalog, rng, next_id):
    """Segmental PE: 3-5 configs spanning isolated to multi-bilateral."""
    configs = []
    gen = 2
    all_lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']

    # --- Isolated, varying occlusion ---
    for occlusion in [OcclusionDegree.PARTIAL, OcclusionDegree.NEAR_COMPLETE]:
        lobe = str(rng.choice(all_lobes))
        edges = _select_edges(catalog, gen, lobe, 1, rng)
        if edges:
            segments = _make_segment_specs(catalog, edges, occlusion)
            configs.append(CounterfactualConfig(
                config_id=next_id(),
                patient_id=patient_id,
                location=PELocation.SEGMENTAL,
                occlusion=occlusion,
                burden=Burden.ISOLATED,
                target_lobes=[lobe],
                segments=segments,
                empirically_observed=True,
                clinical_note="Segmental PE: ~5.7% of empirical components.",
            ))

    # --- Multi-unilateral (right side, 2-3 segments, complete) ---
    right_lobe = str(rng.choice(['RUL', 'RML', 'RLL']))
    n_seg = int(rng.integers(2, 4))
    edges = _select_edges(catalog, gen, right_lobe, n_seg, rng)
    if edges:
        segments = _make_segment_specs(catalog, edges, OcclusionDegree.COMPLETE)
        configs.append(CounterfactualConfig(
            config_id=next_id(),
            patient_id=patient_id,
            location=PELocation.SEGMENTAL,
            occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.MULTI_UNILATERAL,
            target_lobes=[right_lobe],
            segments=segments,
            empirically_observed=True,
        ))

    # --- Multi-bilateral (1 per side from different lobes) ---
    left_lobe = str(rng.choice(['LUL', 'LLL']))
    right_lobe2 = str(rng.choice(['RUL', 'RML', 'RLL']))
    left_edges = _select_edges(catalog, gen, left_lobe, 1, rng)
    right_edges = _select_edges(catalog, gen, right_lobe2, 1, rng)
    if left_edges and right_edges:
        all_edges = left_edges + right_edges
        segments = _make_segment_specs(catalog, all_edges, OcclusionDegree.NEAR_COMPLETE)
        configs.append(CounterfactualConfig(
            config_id=next_id(),
            patient_id=patient_id,
            location=PELocation.SEGMENTAL,
            occlusion=OcclusionDegree.NEAR_COMPLETE,
            burden=Burden.MULTI_BILATERAL,
            target_lobes=[left_lobe, right_lobe2],
            segments=segments,
            empirically_observed=True,
        ))

    return configs


def _generate_subsegmental_configs(patient_id, catalog, rng, next_id):
    """Subsegmental PE: 8-12 configs with full burden sweep.

    Subsegmental accounts for ~94% of empirical PE components.
    """
    configs = []
    gen = 3
    all_lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']

    # --- Isolated, all 3 occlusions ---
    for occlusion in [OcclusionDegree.PARTIAL, OcclusionDegree.NEAR_COMPLETE,
                      OcclusionDegree.COMPLETE]:
        lobe = str(rng.choice(all_lobes))
        edges = _select_edges(catalog, gen, lobe, 1, rng)
        if edges:
            segments = _make_segment_specs(catalog, edges, occlusion)
            configs.append(CounterfactualConfig(
                config_id=next_id(),
                patient_id=patient_id,
                location=PELocation.SUBSEGMENTAL,
                occlusion=occlusion,
                burden=Burden.ISOLATED,
                target_lobes=[lobe],
                segments=segments,
                empirically_observed=True,
            ))

    # --- Multi-unilateral right (3 segments), partial and complete ---
    for occlusion in [OcclusionDegree.PARTIAL, OcclusionDegree.COMPLETE]:
        right_lobe = str(rng.choice(['RUL', 'RML', 'RLL']))
        edges = _select_edges(catalog, gen, right_lobe, 3, rng)
        if edges:
            segments = _make_segment_specs(catalog, edges, occlusion)
            configs.append(CounterfactualConfig(
                config_id=next_id(),
                patient_id=patient_id,
                location=PELocation.SUBSEGMENTAL,
                occlusion=occlusion,
                burden=Burden.MULTI_UNILATERAL,
                target_lobes=[right_lobe],
                segments=segments,
                empirically_observed=True,
            ))

    # --- Multi-bilateral (both sides), partial and near_complete ---
    for occlusion in [OcclusionDegree.PARTIAL, OcclusionDegree.NEAR_COMPLETE]:
        left_lobe = str(rng.choice(['LUL', 'LLL']))
        right_lobe = str(rng.choice(['RUL', 'RML', 'RLL']))
        n_left = int(rng.integers(1, 3))
        n_right = int(rng.integers(2, 4))
        left_edges = _select_edges(catalog, gen, left_lobe, n_left, rng)
        right_edges = _select_edges(catalog, gen, right_lobe, n_right, rng)
        all_edges = (left_edges or []) + (right_edges or [])
        if all_edges:
            segments = _make_segment_specs(catalog, all_edges, occlusion)
            configs.append(CounterfactualConfig(
                config_id=next_id(),
                patient_id=patient_id,
                location=PELocation.SUBSEGMENTAL,
                occlusion=occlusion,
                burden=Burden.MULTI_BILATERAL,
                target_lobes=[left_lobe, right_lobe],
                segments=segments,
                empirically_observed=True,
            ))

    # --- Extensive (8-12 segments across 3-5 lobes), partial and complete ---
    for occlusion in [OcclusionDegree.PARTIAL, OcclusionDegree.COMPLETE]:
        n_lobes = int(rng.integers(3, 6))
        selected_lobes = list(rng.choice(all_lobes, size=n_lobes, replace=False))
        all_edges = []
        for lobe in selected_lobes:
            n_seg = int(rng.integers(2, 4))
            edges = _select_edges(catalog, gen, lobe, n_seg, rng)
            all_edges.extend(edges or [])
        if len(all_edges) >= 7:
            segments = _make_segment_specs(catalog, all_edges, occlusion)
            configs.append(CounterfactualConfig(
                config_id=next_id(),
                patient_id=patient_id,
                location=PELocation.SUBSEGMENTAL,
                occlusion=occlusion,
                burden=Burden.EXTENSIVE,
                target_lobes=selected_lobes,
                segments=segments,
                empirically_observed=True,
            ))

    # --- Massive bilateral: all 5 lobes, complete occlusion ---
    # Pushes Qanadli into moderate range (5 Qanadli segments * weight 2 = 10)
    massive_edges = []
    for lobe in all_lobes:
        edges = _select_edges(catalog, gen, lobe, 3, rng)
        massive_edges.extend(edges or [])
    if len(massive_edges) >= 10:
        segments = _make_segment_specs(catalog, massive_edges, OcclusionDegree.COMPLETE)
        configs.append(CounterfactualConfig(
            config_id=next_id(),
            patient_id=patient_id,
            location=PELocation.SUBSEGMENTAL,
            occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.EXTENSIVE,
            target_lobes=all_lobes.copy(),
            segments=segments,
            empirically_observed=True,
            clinical_note="Massive bilateral PE: all lobes, complete occlusion.",
        ))

    return configs


def _generate_mixed_configs(patient_id, catalog, rng, next_id):
    """Mixed-generation extensive configs (segmental + subsegmental).

    Simulates severe PE spanning multiple vessel generations.
    Each lobe gets both segmental AND subsegmental PE, which means
    two Qanadli slots per lobe (segmental gen=2, subsegmental->collapses
    to same slot). To maximize Qanadli, we use complete occlusion on
    all 5 lobes at segmental level.
    """
    configs = []
    all_lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']

    # --- Mixed moderate: 3 lobes, near-complete ---
    selected_3 = list(rng.choice(all_lobes, size=3, replace=False))
    edges_3 = []
    for lobe in selected_3:
        seg_edges = _select_edges(catalog, 2, lobe, 1, rng)
        subseg_edges = _select_edges(catalog, 3, lobe, int(rng.integers(2, 4)), rng)
        edges_3.extend(seg_edges or [])
        edges_3.extend(subseg_edges or [])

    if len(edges_3) >= 5:
        segments = _make_segment_specs(catalog, edges_3, OcclusionDegree.NEAR_COMPLETE)
        configs.append(CounterfactualConfig(
            config_id=next_id(),
            patient_id=patient_id,
            location=PELocation.SEGMENTAL,
            occlusion=OcclusionDegree.NEAR_COMPLETE,
            burden=Burden.EXTENSIVE,
            target_lobes=selected_3,
            segments=segments,
            empirically_observed=True,
            clinical_note="Mixed generation PE (segmental + subsegmental), 3 lobes.",
        ))

    # --- Mixed severe: all 5 lobes, complete, both generations ---
    # Each lobe scored at segmental level → 5 slots * weight 2 = 10 (moderate)
    edges_5 = []
    for lobe in all_lobes:
        seg_edges = _select_edges(catalog, 2, lobe, 2, rng)
        subseg_edges = _select_edges(catalog, 3, lobe, 3, rng)
        edges_5.extend(seg_edges or [])
        edges_5.extend(subseg_edges or [])

    if len(edges_5) >= 15:
        segments = _make_segment_specs(catalog, edges_5, OcclusionDegree.COMPLETE)
        configs.append(CounterfactualConfig(
            config_id=next_id(),
            patient_id=patient_id,
            location=PELocation.SEGMENTAL,
            occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.EXTENSIVE,
            target_lobes=all_lobes.copy(),
            segments=segments,
            empirically_observed=True,
            clinical_note="Massive mixed generation PE: all lobes, complete occlusion.",
        ))

    # --- Worst-case: lobar + segmental + subsegmental, all 5 lobes, complete ---
    # Lobar edges add separate Qanadli slots (generation 1), pushing score higher.
    # 5 lobar slots * 2 + 5 segmental slots * 2 = 20 (severe)
    lobar_available = catalog.query(generation=1, artery_only=False)
    if lobar_available:
        worst_edges = []
        worst_lobes = []
        for lobe in all_lobes:
            lob_edges = _select_edges(catalog, 1, lobe, 1, rng, artery_only=False)
            seg_edges = _select_edges(catalog, 2, lobe, 2, rng)
            subseg_edges = _select_edges(catalog, 3, lobe, 2, rng)
            worst_edges.extend(lob_edges or [])
            worst_edges.extend(seg_edges or [])
            worst_edges.extend(subseg_edges or [])
            if lob_edges or seg_edges or subseg_edges:
                worst_lobes.append(lobe)
        if len(worst_edges) >= 10:
            segments = _make_segment_specs(catalog, worst_edges, OcclusionDegree.COMPLETE)
            configs.append(CounterfactualConfig(
                config_id=next_id(),
                patient_id=patient_id,
                location=PELocation.LOBAR,
                occlusion=OcclusionDegree.COMPLETE,
                burden=Burden.EXTENSIVE,
                target_lobes=worst_lobes,
                segments=segments,
                empirically_observed=False,
                clinical_note="Worst-case PE: lobar+segmental+subsegmental, "
                              "all lobes, complete occlusion.",
            ))

    return configs


# ---------------------------------------------------------------------------
# Base grid (severity-stratified, acute/chronic balanced)
# ---------------------------------------------------------------------------

def _build_base_grid(patient_id: str,
                         G: nx.Graph,
                         catalog: EdgeCatalog,
                         rng: np.random.Generator = None) -> List[CounterfactualConfig]:
    """Generate ~15 configs with balanced acute/chronic PE and realistic severity.

    Targets a distribution matching real-world PE prevalence:
      ~60% mild  (Qanadli 1-8)   - 9 configs
      ~33% moderate (Qanadli 9-16) - 5 configs
      ~7%  severe (Qanadli 17+)   - 1 config

    Approximately half the configs use acute (concentric) morphology.
    """
    if rng is None:
        rng = np.random.default_rng(seed=hash(patient_id) % 2**32)

    config_counter = [0]
    def next_id():
        config_counter[0] += 1
        return f"{patient_id}_cf{config_counter[0]:03d}"

    configs = []
    all_lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']

    def _pick_lobe_with_edges(gen: int, lobe_pool: list, n_needed: int = 1,
                               artery_only: bool = True) -> str:
        """Pick a random lobe from lobe_pool that has >= n_needed edges."""
        shuffled = list(rng.permutation(lobe_pool))
        for lobe in shuffled:
            if len(catalog.query(generation=gen, lobe_name=lobe,
                                 artery_only=artery_only)) >= n_needed:
                return lobe
            if artery_only and len(catalog.query(generation=gen, lobe_name=lobe,
                                                  artery_only=False)) >= n_needed:
                return lobe
        return str(rng.choice(lobe_pool))

    # =====================================================================
    # MILD (9 configs, ~60%): Qanadli 1-8
    # These use isolated / small multi-segment configs that stay in the
    # mild range because few Qanadli slots are affected.
    # =====================================================================

    # 1. Subsegmental, partial, isolated - CHRONIC
    lobe = _pick_lobe_with_edges(3, all_lobes)
    edges = _select_edges(catalog, 3, lobe, 1, rng)
    if edges:
        segments = _make_segment_specs(catalog, edges, OcclusionDegree.PARTIAL, acute=False)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SUBSEGMENTAL, occlusion=OcclusionDegree.PARTIAL,
            burden=Burden.ISOLATED, target_lobes=[lobe],
            segments=segments, empirically_observed=True,
            clinical_note="Chronic PE: eccentric partial, subsegmental.",
        ))

    # 2. Subsegmental, complete, isolated - ACUTE
    lobe = _pick_lobe_with_edges(3, all_lobes)
    edges = _select_edges(catalog, 3, lobe, 1, rng)
    if edges:
        segments = _make_segment_specs(catalog, edges, OcclusionDegree.COMPLETE, acute=True)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SUBSEGMENTAL, occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.ISOLATED, target_lobes=[lobe],
            segments=segments, empirically_observed=True,
            clinical_note="Acute PE: concentric complete, subsegmental.",
        ))

    # 3. Segmental, partial, isolated - ACUTE
    lobe = _pick_lobe_with_edges(2, all_lobes)
    edges = _select_edges(catalog, 2, lobe, 1, rng)
    if edges:
        segments = _make_segment_specs(catalog, edges, OcclusionDegree.PARTIAL, acute=True)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SEGMENTAL, occlusion=OcclusionDegree.PARTIAL,
            burden=Burden.ISOLATED, target_lobes=[lobe],
            segments=segments, empirically_observed=True,
            clinical_note="Acute PE: concentric partial, segmental.",
        ))

    # 4. Subsegmental, near-complete, multi-unilateral - CHRONIC
    right_lobe = _pick_lobe_with_edges(3, ['RUL', 'RML', 'RLL'], n_needed=3)
    edges = _select_edges(catalog, 3, right_lobe, 3, rng)
    if edges:
        segments = _make_segment_specs(catalog, edges, OcclusionDegree.NEAR_COMPLETE, acute=False)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SUBSEGMENTAL, occlusion=OcclusionDegree.NEAR_COMPLETE,
            burden=Burden.MULTI_UNILATERAL, target_lobes=[right_lobe],
            segments=segments, empirically_observed=True,
            clinical_note="Chronic PE: eccentric near-complete, multi subsegmental.",
        ))

    # 5. Segmental, complete, bilateral (2 lobes) - ACUTE  (Q~4)
    left_lobe = _pick_lobe_with_edges(2, ['LUL', 'LLL'])
    right_lobe = _pick_lobe_with_edges(2, ['RUL', 'RML', 'RLL'])
    left_e = _select_edges(catalog, 2, left_lobe, 1, rng)
    right_e = _select_edges(catalog, 2, right_lobe, 1, rng)
    seg_edges = (left_e or []) + (right_e or [])
    if seg_edges:
        segments = _make_segment_specs(catalog, seg_edges, OcclusionDegree.COMPLETE, acute=True)
        lobes_used = list(set(s.lobe_name for s in segments))
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SEGMENTAL, occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.MULTI_BILATERAL, target_lobes=lobes_used,
            segments=segments, empirically_observed=True,
            clinical_note="Acute PE: concentric complete, bilateral segmental, 2 lobes.",
        ))

    # 6. Lobar, complete, isolated - CHRONIC  (Q=2)
    lobar_edges = catalog.query(generation=1, artery_only=False)
    if lobar_edges:
        info = catalog.get_edge_info(lobar_edges[0])
        lobe_name = info.get('lobe_name', 'central')
        edges = _select_edges(catalog, 1, lobe_name, 1, rng, artery_only=False)
        if edges:
            segments = _make_segment_specs(catalog, edges, OcclusionDegree.COMPLETE, acute=False)
            configs.append(CounterfactualConfig(
                config_id=next_id(), patient_id=patient_id,
                location=PELocation.LOBAR, occlusion=OcclusionDegree.COMPLETE,
                burden=Burden.ISOLATED, target_lobes=[lobe_name],
                segments=segments, empirically_observed=False,
                clinical_note="Chronic PE: eccentric complete, single lobar.",
            ))

    # 7. Subsegmental, near-complete, bilateral (2 lobes) - ACUTE  (Q~2)
    left_lobe = _pick_lobe_with_edges(3, ['LUL', 'LLL'])
    right_lobe = _pick_lobe_with_edges(3, ['RUL', 'RML', 'RLL'])
    left_e = _select_edges(catalog, 3, left_lobe, 1, rng)
    right_e = _select_edges(catalog, 3, right_lobe, 2, rng)
    subseg_edges = (left_e or []) + (right_e or [])
    if subseg_edges:
        segments = _make_segment_specs(catalog, subseg_edges, OcclusionDegree.NEAR_COMPLETE, acute=True)
        lobes_used = list(set(s.lobe_name for s in segments))
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SUBSEGMENTAL, occlusion=OcclusionDegree.NEAR_COMPLETE,
            burden=Burden.MULTI_BILATERAL, target_lobes=lobes_used,
            segments=segments, empirically_observed=True,
            clinical_note="Acute PE: concentric near-complete, bilateral subseg.",
        ))

    # 8. Segmental, near-complete, isolated - CHRONIC  (Q=1)
    lobe = _pick_lobe_with_edges(2, all_lobes)
    edges = _select_edges(catalog, 2, lobe, 1, rng)
    if edges:
        segments = _make_segment_specs(catalog, edges, OcclusionDegree.NEAR_COMPLETE, acute=False)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SEGMENTAL, occlusion=OcclusionDegree.NEAR_COMPLETE,
            burden=Burden.ISOLATED, target_lobes=[lobe],
            segments=segments, empirically_observed=True,
            clinical_note="Chronic PE: eccentric near-complete, segmental.",
        ))

    # 9. Subsegmental, complete, multi-bilateral (3 lobes) - ACUTE  (Q~3)
    selected_3 = list(rng.choice(all_lobes, size=3, replace=False))
    subseg_edges = []
    for lobe in selected_3:
        e = _select_edges(catalog, 3, lobe, 1, rng)
        subseg_edges.extend(e or [])
    if subseg_edges:
        segments = _make_segment_specs(catalog, subseg_edges, OcclusionDegree.COMPLETE, acute=True)
        lobes_used = list(set(s.lobe_name for s in segments))
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SUBSEGMENTAL, occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.MULTI_BILATERAL, target_lobes=lobes_used,
            segments=segments, empirically_observed=True,
            clinical_note="Acute PE: concentric complete, bilateral subseg, 3 lobes.",
        ))

    # =====================================================================
    # MODERATE (5 configs, ~33%): Qanadli 9-16
    # To reach moderate, we need lobar+seg across 3+ lobes with complete
    # occlusion. Each (lobe, gen_slot) pair = 1 Qanadli slot.
    # =====================================================================

    # 10. Lobar+seg, complete, 3 lobes - CHRONIC  (Q~12)
    selected_3 = list(rng.choice(all_lobes, size=3, replace=False))
    mixed_edges = []
    for lobe in selected_3:
        lob_e = _select_edges(catalog, 1, lobe, 1, rng, artery_only=False)
        seg_e = _select_edges(catalog, 2, lobe, 1, rng)
        mixed_edges.extend(lob_e or [])
        mixed_edges.extend(seg_e or [])
    if mixed_edges:
        segments = _make_segment_specs(catalog, mixed_edges, OcclusionDegree.COMPLETE, acute=False)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.LOBAR, occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.EXTENSIVE, target_lobes=selected_3,
            segments=segments, empirically_observed=False,
            clinical_note="Chronic PE: eccentric complete, lobar+seg, 3 lobes.",
        ))

    # 11. Lobar+seg, complete, 4 lobes - ACUTE  (Q~16)
    selected_4 = list(rng.choice(all_lobes, size=4, replace=False))
    mixed_edges = []
    for lobe in selected_4:
        lob_e = _select_edges(catalog, 1, lobe, 1, rng, artery_only=False)
        seg_e = _select_edges(catalog, 2, lobe, 1, rng)
        mixed_edges.extend(lob_e or [])
        mixed_edges.extend(seg_e or [])
    if mixed_edges:
        segments = _make_segment_specs(catalog, mixed_edges, OcclusionDegree.COMPLETE, acute=True)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.LOBAR, occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.EXTENSIVE, target_lobes=selected_4,
            segments=segments, empirically_observed=False,
            clinical_note="Acute PE: concentric complete, lobar+seg, 4 lobes.",
        ))

    # 12. Seg complete, 5 lobes + subseg - CHRONIC  (Q~10)
    seg_edges = []
    for lobe in all_lobes:
        seg_e = _select_edges(catalog, 2, lobe, 1, rng)
        seg_edges.extend(seg_e or [])
    if seg_edges:
        segments = _make_segment_specs(catalog, seg_edges, OcclusionDegree.COMPLETE, acute=False)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.SEGMENTAL, occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.EXTENSIVE, target_lobes=all_lobes.copy(),
            segments=segments, empirically_observed=True,
            clinical_note="Chronic PE: eccentric complete, segmental all 5 lobes.",
        ))

    # 13. Lobar+seg, near-complete, 4 lobes - ACUTE  (Q~8-12)
    selected_4 = list(rng.choice(all_lobes, size=4, replace=False))
    mixed_edges = []
    for lobe in selected_4:
        lob_e = _select_edges(catalog, 1, lobe, 1, rng, artery_only=False)
        seg_e = _select_edges(catalog, 2, lobe, 1, rng)
        mixed_edges.extend(lob_e or [])
        mixed_edges.extend(seg_e or [])
    if mixed_edges:
        segments = _make_segment_specs(catalog, mixed_edges, OcclusionDegree.NEAR_COMPLETE, acute=True)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.LOBAR, occlusion=OcclusionDegree.NEAR_COMPLETE,
            burden=Burden.EXTENSIVE, target_lobes=selected_4,
            segments=segments, empirically_observed=False,
            clinical_note="Acute PE: concentric near-complete, lobar+seg, 4 lobes.",
        ))

    # 14. Lobar+seg+subseg, complete, 3 lobes - CHRONIC  (Q~12)
    selected_3 = list(rng.choice(all_lobes, size=3, replace=False))
    mixed_edges = []
    for lobe in selected_3:
        lob_e = _select_edges(catalog, 1, lobe, 1, rng, artery_only=False)
        seg_e = _select_edges(catalog, 2, lobe, 1, rng)
        subseg_e = _select_edges(catalog, 3, lobe, int(rng.integers(2, 4)), rng)
        mixed_edges.extend(lob_e or [])
        mixed_edges.extend(seg_e or [])
        mixed_edges.extend(subseg_e or [])
    if mixed_edges:
        segments = _make_segment_specs(catalog, mixed_edges, OcclusionDegree.COMPLETE, acute=False)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.LOBAR, occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.EXTENSIVE, target_lobes=selected_3,
            segments=segments, empirically_observed=False,
            clinical_note="Chronic PE: eccentric complete, all gens, 3 lobes.",
        ))

    # =====================================================================
    # SEVERE (1 config, ~7%): Qanadli 17+
    # =====================================================================

    # 15. All generations, complete, all 5 lobes + central - ACUTE  (Q~20+)
    all5_edges = []
    for lobe in all_lobes + ['central']:
        lob_e = _select_edges_by_size(catalog, 1, lobe, 1, rng, artery_only=False)
        seg_e = _select_edges_by_size(catalog, 2, lobe, 2, rng)
        subseg_e = _select_edges(catalog, 3, lobe, 3, rng)
        all5_edges.extend(lob_e or [])
        all5_edges.extend(seg_e or [])
        all5_edges.extend(subseg_e or [])
    if len(all5_edges) >= 10:
        segments = _make_segment_specs(catalog, all5_edges, OcclusionDegree.COMPLETE, acute=True)
        configs.append(CounterfactualConfig(
            config_id=next_id(), patient_id=patient_id,
            location=PELocation.LOBAR, occlusion=OcclusionDegree.COMPLETE,
            burden=Burden.EXTENSIVE, target_lobes=all_lobes.copy(),
            segments=segments, empirically_observed=False,
            clinical_note="Acute PE: concentric complete, all generations, all 5 lobes.",
        ))

    # ── Post-processing: compute derived metrics ─────────────────────────
    for cfg in configs:
        score, max_s, pct, cat = compute_qanadli_score(cfg.segments)
        cfg.qanadli_score = score
        cfg.qanadli_max = max_s
        cfg.qanadli_percent = pct
        cfg.severity_category = cat
        cfg.n_segments_affected = len(cfg.segments)

        lobes = set(s.lobe_name for s in cfg.segments)
        has_left = any(l.startswith('L') for l in lobes)
        has_right = any(l.startswith('R') for l in lobes)
        if has_left and has_right:
            cfg.laterality = "bilateral"
        elif has_right:
            cfg.laterality = "unilateral_right"
        elif has_left:
            cfg.laterality = "unilateral_left"
        else:
            cfg.laterality = "central"

        total_voxels = 0
        for seg in cfg.segments:
            frac_lo, frac_hi = OCCLUSION_FRACTION_RANGE[seg.occlusion]
            frac = (frac_lo + frac_hi) / 2
            total_voxels += int(seg.n_voxels * frac)
        cfg.estimated_total_pe_voxels = total_voxels

    n_acute = sum(1 for c in configs if any(s.acute for s in c.segments))
    print(f"\n--- Base grid: {len(configs)} configs "
          f"({n_acute} acute, {len(configs) - n_acute} chronic) ---")
    return configs


def build_counterfactual_grid(patient_id: str,
                         G: nx.Graph,
                         catalog: EdgeCatalog,
                         rng: np.random.Generator = None) -> List[CounterfactualConfig]:
    """Generate the 16-config counterfactual grid: a severity-stratified base grid
    (9 mild, 5 moderate, 1 severe; acute/chronic balanced) plus one extra severe
    central/saddle PE case spanning central (gen 0) + lobar + segmental +
    subsegmental edges across all 5 lobes.

    Vessel classification comes from the v2 labels (direct artery/vein segmentation).
    """
    configs = _build_base_grid(patient_id, G, catalog, rng)

    if rng is None:
        rng = np.random.default_rng(seed=hash(patient_id) % 2**32)

    all_lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']

    # Extra severe: anatomically realistic central/saddle PE.
    # - Central (gen 0): near-complete (saddle thrombus at PA bifurcation)
    # - 3 random lobes: complete lobar + segmental downstream occlusion
    # - 2 remaining lobes: partial segmental (residual flow)
    # This mimics a survivable saddle PE where some perfusion is maintained.
    next_num = len(configs) + 1
    config_id = f"{patient_id}_cf{next_num:03d}"

    segments = []

    # Central generation-0 edges - partial (saddle straddling bifurcation,
    # 30-50% occlusion with visible flow around the thrombus). Artery (or unknown)
    # edges are preferred; _select_edges falls back to all edges if there are none.
    central_edges = _select_edges(catalog, 0, None, 2, rng, artery_only=True)
    if central_edges:
        segments.extend(_make_segment_specs(
            catalog, central_edges, OcclusionDegree.PARTIAL, acute=True))

    # Split lobes: 3 severely affected, 2 with residual flow
    rng.shuffle(all_lobes)
    severe_lobes = all_lobes[:3]
    partial_lobes = all_lobes[3:]

    # Severe lobes: complete lobar + segmental
    for lobe in severe_lobes:
        lob_e = _select_edges_by_size(catalog, 1, lobe, 1, rng, artery_only=False)
        seg_e = _select_edges_by_size(catalog, 2, lobe, 2, rng)
        subseg_e = _select_edges(catalog, 3, lobe, int(rng.integers(1, 3)), rng)
        if lob_e:
            segments.extend(_make_segment_specs(
                catalog, lob_e, OcclusionDegree.COMPLETE, acute=True))
        if seg_e:
            segments.extend(_make_segment_specs(
                catalog, seg_e, OcclusionDegree.COMPLETE, acute=True))
        if subseg_e:
            segments.extend(_make_segment_specs(
                catalog, subseg_e, OcclusionDegree.NEAR_COMPLETE, acute=True))

    # Partial lobes: partial segmental only (residual flow)
    for lobe in partial_lobes:
        seg_e = _select_edges(catalog, 2, lobe, 1, rng)
        subseg_e = _select_edges(catalog, 3, lobe, int(rng.integers(1, 3)), rng)
        if seg_e:
            segments.extend(_make_segment_specs(
                catalog, seg_e, OcclusionDegree.PARTIAL, acute=True))
        if subseg_e:
            segments.extend(_make_segment_specs(
                catalog, subseg_e, OcclusionDegree.PARTIAL, acute=True))

    if len(segments) >= 3:
        extra = CounterfactualConfig(
            config_id=config_id, patient_id=patient_id,
            location=PELocation.CENTRAL, occlusion=OcclusionDegree.PARTIAL,
            burden=Burden.EXTENSIVE, target_lobes=all_lobes.copy(),
            segments=segments, empirically_observed=False,
            clinical_note=(
                "Acute saddle PE: partial central (straddling bifurcation), "
                "complete in 3 lobes, partial in 2 lobes (residual flow)."
            ),
        )
        # Compute derived metrics - override severity to "severe" since central
        # PE is clinically severe by definition regardless of Qanadli index
        score, max_s, pct, _cat = compute_qanadli_score(extra.segments)
        extra.qanadli_score = score
        extra.qanadli_max = max_s
        extra.qanadli_percent = pct
        extra.severity_category = "severe"
        extra.n_segments_affected = len(extra.segments)

        lobes = set(s.lobe_name for s in extra.segments)
        has_left = any(l.startswith('L') for l in lobes)
        has_right = any(l.startswith('R') for l in lobes)
        if has_left and has_right:
            extra.laterality = "bilateral"
        elif has_right:
            extra.laterality = "unilateral_right"
        elif has_left:
            extra.laterality = "unilateral_left"
        else:
            extra.laterality = "central"

        total_voxels = 0
        for seg in extra.segments:
            frac_lo, frac_hi = OCCLUSION_FRACTION_RANGE[seg.occlusion]
            frac = (frac_lo + frac_hi) / 2
            total_voxels += int(seg.n_voxels * frac)
        extra.estimated_total_pe_voxels = total_voxels

        configs.append(extra)

    n_severe = sum(1 for c in configs if c.severity_category in ('severe', 'massive'))
    print(f"--- Counterfactual grid: {len(configs)} configs ({n_severe} severe) ---")
    return configs


# ---------------------------------------------------------------------------
# Replacement config generation (for retry after validation failures)
# ---------------------------------------------------------------------------

def _select_edges_by_size(catalog: EdgeCatalog,
                          generation: int,
                          lobe_name: str,
                          n_edges: int,
                          rng: np.random.Generator,
                          artery_only: bool = True) -> List[Tuple[int, int]]:
    """Select edges preferring larger vessels (more voxels).

    Sorts candidates by n_voxels descending and samples from the top half,
    giving replacement configs a better chance of producing PE voxels.
    """
    candidates = catalog.query(generation=generation, lobe_name=lobe_name,
                               artery_only=artery_only)
    if not candidates and artery_only:
        candidates = catalog.query(generation=generation, lobe_name=lobe_name,
                                   artery_only=False)
    if not candidates:
        return []

    # Sort by n_voxels descending
    sized = [(e, catalog.get_edge_info(e).get('n_voxels', 0)) for e in candidates]
    sized.sort(key=lambda x: x[1], reverse=True)

    # Sample from top half (larger vessels)
    top = sized[:max(1, len(sized) // 2)]
    n_select = min(n_edges, len(top))
    if n_select == 0:
        return []
    indices = rng.choice(len(top), size=n_select, replace=False)
    return [top[int(i)][0] for i in indices]


def generate_replacement_configs(
    failed_config_dicts: List[dict],
    patient_id: str,
    G: nx.Graph,
    catalog: EdgeCatalog,
    rng: np.random.Generator,
    start_config_num: int,
    attempt: int = 0,
) -> List[CounterfactualConfig]:
    """Generate replacement configs for failed validations.

    Escalation strategy by attempt number:
      0: Same params, different lobe, prefer larger vessels
      1: Bump occlusion one level up
      2: Increase burden (isolated -> multi_unilateral, 3 segments)

    Args:
        failed_config_dicts: Failed config dicts from grid JSON.
        patient_id: Patient ID.
        G: Vessel graph from Phase 1.
        catalog: Pre-built EdgeCatalog.
        rng: Random generator.
        start_config_num: Starting number for new config IDs.
        attempt: Retry attempt (0, 1, 2).

    Returns:
        List of replacement CounterfactualConfig objects.
    """
    all_lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']

    OCCLUSION_UP = {
        'partial': OcclusionDegree.NEAR_COMPLETE,
        'near_complete': OcclusionDegree.COMPLETE,
        'complete': OcclusionDegree.COMPLETE,
    }
    LOCATION_TO_GEN = {
        'central': 0, 'lobar': 1, 'segmental': 2, 'subsegmental': 3,
    }
    LOCATION_TO_ENUM = {
        'central': PELocation.CENTRAL, 'lobar': PELocation.LOBAR,
        'segmental': PELocation.SEGMENTAL, 'subsegmental': PELocation.SUBSEGMENTAL,
    }
    BURDEN_TO_ENUM = {
        'isolated': Burden.ISOLATED, 'multi_unilateral': Burden.MULTI_UNILATERAL,
        'multi_bilateral': Burden.MULTI_BILATERAL, 'extensive': Burden.EXTENSIVE,
    }
    OCCLUSION_TO_ENUM = {
        'partial': OcclusionDegree.PARTIAL, 'near_complete': OcclusionDegree.NEAR_COMPLETE,
        'complete': OcclusionDegree.COMPLETE,
    }

    replacements = []
    config_num = start_config_num

    for fc in failed_config_dicts:
        config_num += 1
        new_id = f"{patient_id}_cf{config_num:03d}"

        loc_str = fc.get('location', 'subsegmental')
        bur_str = fc.get('burden', 'isolated')
        occ_str = fc.get('occlusion', 'partial')

        orig_lobes = fc.get('target_lobes', [])
        if isinstance(orig_lobes, str):
            orig_lobes = [l.strip() for l in orig_lobes.split(',') if l.strip()]

        location = LOCATION_TO_ENUM.get(loc_str, PELocation.SUBSEGMENTAL)
        burden = BURDEN_TO_ENUM.get(bur_str, Burden.ISOLATED)
        occlusion = OCCLUSION_TO_ENUM.get(occ_str, OcclusionDegree.PARTIAL)
        gen = LOCATION_TO_GEN.get(loc_str, 3)

        # Escalation
        if attempt >= 1:
            occlusion = OCCLUSION_UP.get(occ_str, OcclusionDegree.COMPLETE)
        if attempt >= 2 and burden == Burden.ISOLATED:
            burden = Burden.MULTI_UNILATERAL

        # Number of edges based on burden
        if burden == Burden.ISOLATED:
            n_edges = 1
        elif burden in (Burden.MULTI_UNILATERAL, Burden.MULTI_BILATERAL):
            n_edges = 3
        else:
            n_edges = max(7, fc.get('n_segments_affected', 7))

        # Pick a different lobe than the original
        other_lobes = [l for l in all_lobes if l not in orig_lobes]
        if not other_lobes:
            other_lobes = all_lobes

        target_lobe = str(rng.choice(other_lobes))
        edges = _select_edges_by_size(catalog, gen, target_lobe, n_edges, rng)

        # Fallback: try any lobe
        if not edges:
            for lobe in rng.permutation(all_lobes):
                edges = _select_edges_by_size(catalog, gen, str(lobe), n_edges, rng)
                if edges:
                    target_lobe = str(lobe)
                    break

        if not edges:
            print(f"    WARNING: no edges for replacement of "
                  f"{fc.get('config_id')} - skipping")
            continue

        target_lobes = [target_lobe]

        # Handle bilateral burden
        if burden == Burden.MULTI_BILATERAL:
            if target_lobe.startswith('R'):
                other_side = ([l for l in ['LUL', 'LLL']
                               if l not in orig_lobes] or ['LUL', 'LLL'])
            else:
                other_side = ([l for l in ['RUL', 'RML', 'RLL']
                               if l not in orig_lobes] or ['RUL', 'RML', 'RLL'])
            other_lobe = str(rng.choice(other_side))
            other_edges = _select_edges_by_size(
                catalog, gen, other_lobe, n_edges, rng)
            if other_edges:
                edges = edges + other_edges
                target_lobes.append(other_lobe)

        segments = _make_segment_specs(catalog, edges, occlusion)

        cfg = CounterfactualConfig(
            config_id=new_id,
            patient_id=patient_id,
            location=location,
            occlusion=occlusion,
            burden=burden,
            target_lobes=target_lobes,
            segments=segments,
            empirically_observed=fc.get('empirically_observed', True),
            clinical_note=f"Retry {attempt} replacing {fc.get('config_id', '?')}",
        )

        # Compute derived metrics
        score, max_s, pct, cat = compute_qanadli_score(cfg.segments)
        cfg.qanadli_score = score
        cfg.qanadli_max = max_s
        cfg.qanadli_percent = pct
        cfg.severity_category = cat
        cfg.n_segments_affected = len(cfg.segments)

        lobes = set(s.lobe_name for s in cfg.segments)
        has_left = any(l.startswith('L') for l in lobes)
        has_right = any(l.startswith('R') for l in lobes)
        if has_left and has_right:
            cfg.laterality = "bilateral"
        elif has_right:
            cfg.laterality = "unilateral_right"
        elif has_left:
            cfg.laterality = "unilateral_left"
        else:
            cfg.laterality = "central"

        total_voxels = 0
        for seg in cfg.segments:
            frac_lo, frac_hi = OCCLUSION_FRACTION_RANGE[seg.occlusion]
            frac = (frac_lo + frac_hi) / 2
            total_voxels += int(seg.n_voxels * frac)
        cfg.estimated_total_pe_voxels = total_voxels

        replacements.append(cfg)

    return replacements


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_grid(configs: List[CounterfactualConfig],
              patient_id: str,
              output_dir: str) -> Tuple[str, str]:
    """Save counterfactual grid as CSV and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV: one row per config (flat)
    rows = []
    for cfg in configs:
        rows.append({
            'config_id': cfg.config_id,
            'patient_id': cfg.patient_id,
            'location': LOCATION_NAMES[cfg.location],
            'occlusion': cfg.occlusion.value,
            'burden': cfg.burden.value,
            'target_lobes': ','.join(cfg.target_lobes),
            'n_segments_affected': cfg.n_segments_affected,
            'qanadli_score': cfg.qanadli_score,
            'qanadli_percent': round(cfg.qanadli_percent, 1),
            'severity_category': cfg.severity_category,
            'laterality': cfg.laterality,
            'estimated_pe_voxels': cfg.estimated_total_pe_voxels,
            'acute': any(s.acute for s in cfg.segments),
            'empirically_observed': cfg.empirically_observed,
            'clinical_note': cfg.clinical_note,
            'edge_keys': str([(s.edge_key[0], s.edge_key[1]) for s in cfg.segments]),
        })

    csv_path = output_dir / f"{patient_id}_counterfactual_grid.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # JSON: full detail
    json_data = {
        'patient_id': patient_id,
        'n_configs': len(configs),
        'grid_summary': {
            'by_location': {},
            'by_severity': {},
            'by_burden': {},
            'by_laterality': {},
        },
        'configs': [cfg.to_dict() for cfg in configs],
    }

    for cfg in configs:
        loc = LOCATION_NAMES[cfg.location]
        json_data['grid_summary']['by_location'][loc] = \
            json_data['grid_summary']['by_location'].get(loc, 0) + 1
        sev = cfg.severity_category
        json_data['grid_summary']['by_severity'][sev] = \
            json_data['grid_summary']['by_severity'].get(sev, 0) + 1
        bur = cfg.burden.value
        json_data['grid_summary']['by_burden'][bur] = \
            json_data['grid_summary']['by_burden'].get(bur, 0) + 1
        lat = cfg.laterality
        json_data['grid_summary']['by_laterality'][lat] = \
            json_data['grid_summary']['by_laterality'].get(lat, 0) + 1

    json_path = output_dir / f"{patient_id}_counterfactual_grid.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)

    return str(csv_path), str(json_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase2_for_patient(patient_id: str,
                           output_dir: str = None,
                           seed: int = None,
                           max_configs: int = None,
                           phase1_output_dir: str = None) -> List[CounterfactualConfig]:
    """Run Phase 2 for a single patient.

    Args:
        patient_id: Patient ID.
        output_dir: Where to save grid CSV/JSON.
        seed: Random seed.
        max_configs: Unused (kept for call-site compatibility).
        phase1_output_dir: Where to find Phase 1 outputs (vessel graph).
    """
    if output_dir is None:
        output_dir = str(PHASE2_OUTPUT)
    if phase1_output_dir is None:
        phase1_output_dir = str(PHASE1_OUTPUT)

    # Load graph from Phase 1
    graph_path = Path(phase1_output_dir) / f"{patient_id}_vessel_graph_final.pkl"
    print(f"Loading vessel graph: {graph_path}")

    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    G = graph_data['graph']
    spacing = graph_data['spacing']

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Spacing: {spacing}")

    # Build catalog
    catalog = EdgeCatalog(G)
    summary = catalog.summary()
    print(f"\nEdge catalog ({len(catalog._edge_data)} indexed edges):")
    for (gen, lobe), counts in sorted(summary.items()):
        print(f"  gen={gen} ({LOCATION_NAMES.get(gen, '?'):14s}) "
              f"lobe={lobe:8s}: {counts['total']:6d} total, "
              f"{counts['artery_like']:5d} artery-like")

    # Generate grid
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng(seed=hash(patient_id) % 2**32)

    configs = build_counterfactual_grid(patient_id, G, catalog, rng)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Counterfactual grid for {patient_id}: {len(configs)} configs")
    print(f"{'='*80}")
    print(f"{'ID':<22s} {'Location':<14s} {'Burden':<18s} {'Occlusion':<15s} "
          f"{'Segs':>4s} {'Qanadli':>7s} {'Severity':<10s} {'Laterality':<18s}")
    print(f"{'-'*80}")
    for cfg in configs:
        print(f"{cfg.config_id:<22s} {LOCATION_NAMES[cfg.location]:<14s} "
              f"{cfg.burden.value:<18s} {cfg.occlusion.value:<15s} "
              f"{cfg.n_segments_affected:>4d} {cfg.qanadli_score:>3d}/{cfg.qanadli_max:<3d} "
              f"{cfg.severity_category:<10s} {cfg.laterality:<18s}")

    # Save
    csv_path, json_path = save_grid(configs, patient_id, output_dir)
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")

    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Generate counterfactual PE grid for a patient")
    parser.add_argument("patient_id", help="Patient ID (e.g., <STUDY_ID>)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: phase2_counterfactuals/output/)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: hash(patient_id); Python string hashes change per process unless PYTHONHASHSEED is set)")
    args = parser.parse_args()

    run_phase2_for_patient(args.patient_id, args.output_dir, args.seed)
