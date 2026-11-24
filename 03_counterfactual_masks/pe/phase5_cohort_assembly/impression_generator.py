"""Impression text generator for counterfactual PE cases.

Takes an original PE-negative radiology impression and modifies it to
describe the synthetic PE from a counterfactual configuration.  The "no PE"
finding is replaced; all other findings are preserved verbatim.
"""

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lobe name mapping (abbreviation → anatomical name used in reports)
# ---------------------------------------------------------------------------

LOBE_IMPRESSION_NAMES = {
    "RUL": "right upper lobe",
    "RML": "right middle lobe",
    "RLL": "right lower lobe",
    "LUL": "left upper lobe",
    "LLL": "left lower lobe",
}

# ---------------------------------------------------------------------------
# "No PE" detection patterns (covers >95 % of PE-negative impressions)
# ---------------------------------------------------------------------------

_NO_PE_PATTERNS = [
    # "No evidence of pulmonary embolism/embolus/emboli"
    # Handles: No / No evidence of / No CT evidence of / No CT sign(s) of
    # Adjectives: acute, central, large, major, significant, a (article)
    r"no\s+(?:evidence\s+of\s+)?(?:CT\s+evidence\s+of\s+)?(?:CT\s+signs?\s+of\s+)?"
    r"(?:an?\s+)?(?:acute\s+)?(?:central\s+)?(?:large\s+)?(?:major\s+)?(?:significant\s+)?"
    r"pulmonary\s+"
    r"(?:embolism|embolus|emboli|thromboembolic\s+disease)",
    # "No filling defect(s) to suggest / suggestive of pulmonary embolism"
    r"no\s+filling\s+defects?\s+(?:to\s+suggest|suggestive\s+of)\s+pulmonary\s+"
    r"(?:embolism|embolus|emboli)",
    # "Negative for pulmonary embolism (or right ventricular strain)"
    # Also handles an optional "to the level of ..." or "or ..." tail.
    r"negative\s+for\s+(?:acute\s+)?(?:central\s+)?pulmonary\s+"
    r"(?:embolism|embolus|emboli|thromboembolic\s+disease)"
    r"(?:\s+(?:to\s+the\s+level\s+of\s+[^.]*|or\s+[^.]*))?",
]

_NO_PE_RE = re.compile("|".join(f"(?:{p})" for p in _NO_PE_PATTERNS), re.IGNORECASE)

# Regex to split numbered findings: "1.  Finding text.  2.  Finding text."
_NUMBERED_FINDING_RE = re.compile(r"(\d+)\.\s+")


# ---------------------------------------------------------------------------
# PE finding text generation
# ---------------------------------------------------------------------------

def _format_lobe_list(lobes: List[str]) -> str:
    """Convert lobe abbreviations to a natural-language list."""
    names = [LOBE_IMPRESSION_NAMES.get(l, l) for l in lobes]
    if len(names) == 1:
        return names[0]
    elif len(names) == 2:
        return f"{names[0]} and {names[1]}"
    else:
        return ", ".join(names[:-1]) + f", and {names[-1]}"


def generate_pe_finding(config: Dict) -> str:
    """Generate a realistic PE-positive finding sentence from config.

    Args:
        config: Dict with keys: location, burden, occlusion,
                target_lobes (list of abbreviations), laterality,
                n_segments_affected, severity_category.

    Returns:
        A single finding sentence describing the PE.
    """
    location = config.get("location", "subsegmental")
    burden = config.get("burden", "isolated")
    occlusion = config.get("occlusion", "partial")
    target_lobes = config.get("target_lobes", [])
    laterality = config.get("laterality", "")
    n_seg = config.get("n_segments_affected", 1)

    # Normalize values (may come as enum values or plain strings)
    location = str(location).lower()
    burden = str(burden).lower()
    occlusion = str(occlusion).lower()

    if isinstance(target_lobes, str):
        target_lobes = [l.strip() for l in target_lobes.split(",") if l.strip()]

    lobe_text = _format_lobe_list(target_lobes) if target_lobes else "bilateral lungs"

    is_bilateral = laterality == "bilateral" or (
        any(l.startswith("L") for l in target_lobes) and
        any(l.startswith("R") for l in target_lobes)
    )
    bilateral_word = "bilateral " if is_bilateral else ""

    # Occlusion qualifier (check "near" first since "complete" is in "near_complete")
    if "near" in occlusion:
        occ_adj = "near-occlusive "
    elif "complete" in occlusion:
        occ_adj = "occlusive "
    else:
        occ_adj = ""

    # Build finding based on location + burden
    if location == "lobar":
        if burden == "extensive":
            finding = (
                f"Extensive {bilateral_word}pulmonary emboli involving lobar, "
                f"segmental, and subsegmental branches across the {lobe_text}. "
                f"No saddle embolus."
            )
        else:
            finding = (
                f"{occ_adj.capitalize().strip()} pulmonary embolus in the "
                f"{lobe_text} lobar pulmonary artery.".replace("  ", " ")
            )

    elif location == "segmental":
        if burden in ("extensive", "multi_bilateral", "multi-bilateral"):
            finding = (
                f"{occ_adj.capitalize().strip()} {bilateral_word}segmental "
                f"pulmonary emboli involving the {lobe_text}.".replace("  ", " ")
            )
        elif burden in ("multi_unilateral", "multi-unilateral"):
            finding = (
                f"Multiple {occ_adj}segmental pulmonary emboli in the "
                f"{lobe_text}.".replace("  ", " ")
            )
        else:
            finding = (
                f"Segmental pulmonary embolus in a branch supplying the "
                f"{lobe_text}."
            )

    elif location == "subsegmental":
        if burden == "extensive":
            finding = (
                f"Extensive {bilateral_word}subsegmental pulmonary emboli "
                f"involving the {lobe_text}."
            )
        elif burden in ("multi_bilateral", "multi-bilateral"):
            finding = (
                f"{bilateral_word.capitalize().strip()} subsegmental pulmonary "
                f"emboli involving the {lobe_text}.".replace("  ", " ")
            )
        elif burden in ("multi_unilateral", "multi-unilateral"):
            finding = (
                f"Multiple subsegmental pulmonary emboli in the {lobe_text}."
            )
        else:
            finding = (
                f"Subsegmental pulmonary embolus in the {lobe_text}."
            )

    else:
        # Central / fallback
        finding = (
            f"Pulmonary embolus identified in the main pulmonary artery."
        )

    # Clean up double spaces
    finding = re.sub(r"\s{2,}", " ", finding).strip()
    return finding


# ---------------------------------------------------------------------------
# Impression modification
# ---------------------------------------------------------------------------

def _split_impression(text: str) -> Tuple[str, str, str]:
    """Split text_prompt into (demographics, impression_body, closing).

    Returns:
        demographics: e.g. "<age> year old <race> <sex>."
        impression_body: everything after "IMPRESSION:" up to closing
        closing: boilerplate like "There are no substantial differences..."
    """
    # Find IMPRESSION section
    imp_match = re.search(r"IMPRESSION\s*:\s*", text, re.IGNORECASE)
    if imp_match:
        demographics = text[:imp_match.start()].rstrip()
        rest = text[imp_match.end():]
    else:
        # Try splitting on ". " after first sentence (demographics)
        first_dot = text.find(". ")
        if first_dot > 0:
            demographics = text[:first_dot + 1]
            rest = text[first_dot + 2:]
        else:
            demographics = ""
            rest = text

    # Identify closing boilerplate
    closing = ""
    closing_patterns = [
        r"There are no substantial differences.*$",
        r"Preliminary results were reviewed.*$",
        r"Findings (?:discussed|communicated|were discussed).*$",
    ]
    for pat in closing_patterns:
        m = re.search(pat, rest, re.IGNORECASE)
        if m:
            closing = rest[m.start():]
            rest = rest[:m.start()].rstrip()
            break

    return demographics, rest.strip(), closing.strip()


def _parse_numbered_findings(body: str) -> Optional[List[Tuple[int, str]]]:
    """Parse numbered findings from impression body.

    Returns list of (number, text) tuples, or None if not numbered.
    """
    # Check if body starts with a number
    if not re.match(r"\s*\d+\.\s", body):
        return None

    findings = []
    parts = _NUMBERED_FINDING_RE.split(body)
    # parts looks like: ['', '1', 'finding text', '2', 'finding text', ...]
    i = 1
    while i < len(parts) - 1:
        num = int(parts[i])
        text = parts[i + 1].strip().rstrip()
        findings.append((num, text))
        i += 2

    return findings if findings else None


def modify_impression(original_text_prompt: str, config: Dict) -> str:
    """Replace the 'no PE' finding with a PE-positive finding.

    Args:
        original_text_prompt: Full text_prompt from the datalist.
        config: Counterfactual config dict with location, burden, etc.

    Returns:
        Modified text_prompt with PE finding replacing the no-PE finding.
    """
    pe_finding = generate_pe_finding(config)

    demographics, body, closing = _split_impression(original_text_prompt)

    # Try numbered findings first (79% of cases)
    findings = _parse_numbered_findings(body)

    if findings:
        # Replace the FIRST "no PE" finding with the positive PE finding,
        # and DROP any additional "no PE" findings entirely.
        replaced = False
        new_findings = []
        for _num, text in findings:
            if _NO_PE_RE.search(text):
                if not replaced:
                    # Replace first match with PE-positive finding
                    new_findings.append(pe_finding)
                    replaced = True
                # else: drop this finding (additional "no PE" statement)
            else:
                new_findings.append(text)

        if not replaced:
            # No "no PE" finding found - prepend PE finding as #1
            new_findings.insert(0, pe_finding)

        # Renumber sequentially
        body_parts = []
        for i, text in enumerate(new_findings, start=1):
            text = text.rstrip()
            if not text.endswith("."):
                text += "."
            body_parts.append(f"{i}.  {text}")
        new_body = " ".join(body_parts)

    else:
        # Non-numbered body - do sentence-level replacement
        if _NO_PE_RE.search(body):
            # Replace first "no PE" sentence, drop any additional ones
            sentences = re.split(r"(?<=[.!?])\s+", body)
            new_sentences = []
            replaced = False
            for sent in sentences:
                if _NO_PE_RE.search(sent):
                    if not replaced:
                        new_sentences.append(pe_finding)
                        replaced = True
                    # else: drop additional "no PE" sentences
                else:
                    new_sentences.append(sent)
            new_body = " ".join(new_sentences)
        else:
            # No match at all - prepend
            new_body = pe_finding + " " + body

    # Reassemble full text
    parts = []
    if demographics:
        parts.append(demographics)
    parts.append("IMPRESSION: " + new_body)
    if closing:
        parts.append(closing)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def generate_impressions_for_patient(original_text_prompt: str,
                                      configs: List[Dict]) -> Dict[str, str]:
    """Generate modified impressions for all configs of a patient.

    Args:
        original_text_prompt: Original PE-negative impression.
        configs: List of config dicts (from Phase 2 grid).

    Returns:
        Dict mapping config_id -> modified text_prompt.
    """
    impressions = {}
    for cfg in configs:
        config_id = cfg.get("config_id", "unknown")
        impressions[config_id] = modify_impression(original_text_prompt, cfg)
    return impressions

