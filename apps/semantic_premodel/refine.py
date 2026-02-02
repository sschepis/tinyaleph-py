"""
Optional refinement stage for semantic landscapes.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .landscape import PrimeEntry, SemanticLandscape


def _clone_landscape(landscape: SemanticLandscape) -> SemanticLandscape:
    entries = {
        prime: PrimeEntry.from_dict(entry.to_dict())
        for prime, entry in landscape.entries.items()
    }
    return SemanticLandscape(
        metadata=dict(landscape.metadata),
        stats=dict(landscape.stats),
        entries=entries,
    )


def refine_with_mapping(
    landscape: SemanticLandscape,
    mapping: Dict[int, str],
    confidence_delta: float = 0.1,
) -> SemanticLandscape:
    """
    Refine meanings using a provided mapping {prime: meaning}.
    """
    refined = _clone_landscape(landscape)
    for prime, meaning in mapping.items():
        entry = refined.entries.get(prime)
        if entry is None:
            continue
        if meaning:
            entry.metadata["raw_meaning"] = entry.meaning
            entry.meaning = meaning
            entry.refined = True
            entry.confidence = min(0.95, entry.confidence + confidence_delta)
    return refined


def refine_with_callable(
    landscape: SemanticLandscape,
    refine_fn: Callable[[List[PrimeEntry]], List[Tuple[int, str]]],
) -> SemanticLandscape:
    """
    Refine meanings using a callable returning (prime, meaning) pairs.
    """
    refined = _clone_landscape(landscape)
    entries_list = list(refined.entries.values())
    results = refine_fn(entries_list)
    for prime, meaning in results:
        entry = refined.entries.get(prime)
        if entry is None:
            continue
        if meaning:
            entry.metadata["raw_meaning"] = entry.meaning
            entry.meaning = meaning
            entry.refined = True
    return refined
