"""Phase 2 -- Axis definitions for counterfactual PE generation.

Defines the structured parameter space:
  Axis 1: Location (vessel generation)
  Axis 2: Occlusion degree (fraction of lumen cross-section)
  Axis 3: Burden (number and distribution of affected segments)

Plus SegmentSpec and CounterfactualConfig dataclasses that fully
specify one synthetic PE case for downstream Phase 3 mask construction.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Axis 1: Location (vessel generation)
# ---------------------------------------------------------------------------

class PELocation(IntEnum):
    """Vessel generation where PE is placed.

    Values align with Phase 1's VesselGeneration enum (assign_hierarchy.py).
    """
    CENTRAL = 0       # Main PA / saddle, radius >= 8 mm
    LOBAR = 1         # Lobar branches, radius 4-8 mm
    SEGMENTAL = 2     # Segmental branches, radius 2-4 mm
    SUBSEGMENTAL = 3  # Peripheral, radius < 2 mm


LOCATION_NAMES = {
    PELocation.CENTRAL: "central",
    PELocation.LOBAR: "lobar",
    PELocation.SEGMENTAL: "segmental",
    PELocation.SUBSEGMENTAL: "subsegmental",
}


# ---------------------------------------------------------------------------
# Lobe specification
# ---------------------------------------------------------------------------

class Lobe(Enum):
    """Lung lobe for targeted PE placement.

    Integer labels match Phase 1's assign_lobes.py mapping.
    """
    LUL = "LUL"       # label 10
    LLL = "LLL"       # label 11
    RUL = "RUL"       # label 12
    RML = "RML"       # label 13
    RLL = "RLL"       # label 14
    CENTRAL = "central"  # label 0, hilar region
    ANY = "any"          # no lobe constraint


LEFT_LOBES = {Lobe.LUL, Lobe.LLL}
RIGHT_LOBES = {Lobe.RUL, Lobe.RML, Lobe.RLL}

LOBE_TO_LABEL = {
    Lobe.LUL: 10, Lobe.LLL: 11,
    Lobe.RUL: 12, Lobe.RML: 13, Lobe.RLL: 14,
    Lobe.CENTRAL: 0,
}


# ---------------------------------------------------------------------------
# Axis 2: Occlusion degree
# ---------------------------------------------------------------------------

class OcclusionDegree(Enum):
    """How much of the vessel lumen is occluded."""
    PARTIAL = "partial"              # 30-50% cross-section
    NEAR_COMPLETE = "near_complete"  # 70-90% cross-section
    COMPLETE = "complete"            # 100% cross-section


# Numeric fraction ranges consumed by Phase 3 mask construction
OCCLUSION_FRACTION_RANGE = {
    OcclusionDegree.PARTIAL: (0.30, 0.50),
    OcclusionDegree.NEAR_COMPLETE: (0.70, 0.90),
    OcclusionDegree.COMPLETE: (1.0, 1.0),
}

# Qanadli per-segment weight: partial = 1, complete = 2
OCCLUSION_QANADLI_WEIGHT = {
    OcclusionDegree.PARTIAL: 1,
    OcclusionDegree.NEAR_COMPLETE: 1,  # "partial" in Qanadli terms
    OcclusionDegree.COMPLETE: 2,
}


# ---------------------------------------------------------------------------
# Axis 3: Burden
# ---------------------------------------------------------------------------

class Burden(Enum):
    """Number and distribution of affected segments."""
    ISOLATED = "isolated"                    # Single segment
    MULTI_UNILATERAL = "multi_unilateral"    # 2-4 segments, one side
    MULTI_BILATERAL = "multi_bilateral"      # 2-6 segments, both sides
    EXTENSIVE = "extensive"                  # 7+ segments, bilateral


# Target segment count ranges for each burden level
BURDEN_SEGMENT_RANGE = {
    Burden.ISOLATED: (1, 1),
    Burden.MULTI_UNILATERAL: (2, 4),
    Burden.MULTI_BILATERAL: (2, 6),
    Burden.EXTENSIVE: (7, 20),
}


# ---------------------------------------------------------------------------
# Qanadli severity helper
# ---------------------------------------------------------------------------

def severity_from_qanadli(score: int) -> str:
    """Map raw Qanadli score to severity category.

    Ranges: mild (1-8), moderate (9-16), severe (17-28), massive (29+).
    """
    if score <= 0:
        return "none"
    elif score <= 8:
        return "mild"
    elif score <= 16:
        return "moderate"
    elif score <= 28:
        return "severe"
    else:
        return "massive"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SegmentSpec:
    """Specification for PE in a single vessel segment (graph edge).

    Captures everything Phase 3 needs to locate and fill a segment:
    the edge key for graph lookup, descriptive attributes, and the
    target occlusion degree.
    """
    edge_key: Tuple[int, int]       # (node_u, node_v) in the graph
    generation: int                  # 0-3
    generation_name: str             # "central", "lobar", etc.
    lobe_name: str                   # "RUL", "LLL", etc.
    vessel_type_str: str             # "likely_artery", "unknown", etc.
    radius_mean_mm: float
    length_mm: float
    n_voxels: int
    occlusion: OcclusionDegree
    acute: bool = False              # True = concentric (acute PE), False = eccentric (chronic PE)


@dataclass
class CounterfactualConfig:
    """Complete specification for one synthetic PE case.

    Fully describes a counterfactual: which segments to fill,
    at what occlusion, and computed severity metrics. This is the
    unit of work for Phase 3 (mask construction), Phase 4 (validation),
    and Phase 5 (cohort assembly).
    """
    config_id: str                    # "{patient_id}_cf{NNN}"
    patient_id: str
    location: PELocation              # Axis 1 (dominant generation)
    occlusion: OcclusionDegree        # Axis 2 (dominant occlusion)
    burden: Burden                    # Axis 3
    target_lobes: List[str]           # Which lobes are targeted
    segments: List[SegmentSpec] = field(default_factory=list)

    # Computed severity metrics (filled by grid generator)
    qanadli_score: int = 0
    qanadli_max: int = 40             # 20 segments * max weight 2
    qanadli_percent: float = 0.0
    severity_category: str = ""
    n_segments_affected: int = 0
    laterality: str = ""              # "unilateral_left" / "unilateral_right" / "bilateral" / "central"
    estimated_total_pe_voxels: int = 0

    # Clinical realism flags
    empirically_observed: bool = False
    clinical_note: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        d = {
            'config_id': self.config_id,
            'patient_id': self.patient_id,
            'location': LOCATION_NAMES[self.location],
            'occlusion': self.occlusion.value,
            'burden': self.burden.value,
            'target_lobes': self.target_lobes,
            'n_segments_affected': self.n_segments_affected,
            'qanadli_score': self.qanadli_score,
            'qanadli_max': self.qanadli_max,
            'qanadli_percent': round(self.qanadli_percent, 1),
            'severity_category': self.severity_category,
            'laterality': self.laterality,
            'estimated_total_pe_voxels': self.estimated_total_pe_voxels,
            'empirically_observed': self.empirically_observed,
            'clinical_note': self.clinical_note,
            'segments': [
                {
                    'edge_key': list(s.edge_key),
                    'generation': s.generation,
                    'generation_name': s.generation_name,
                    'lobe_name': s.lobe_name,
                    'vessel_type_str': s.vessel_type_str,
                    'radius_mean_mm': round(s.radius_mean_mm, 3),
                    'length_mm': round(s.length_mm, 3),
                    'n_voxels': s.n_voxels,
                    'occlusion': s.occlusion.value,
                    'acute': s.acute,
                }
                for s in self.segments
            ],
        }
        return d
