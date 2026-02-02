"""
Configuration objects for semantic premodel generation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LandscapeConfig:
    """
    Configuration for deterministic semantic landscape building.

    Args:
        num_primes: Number of primes to include in the landscape.
        max_routes_per_prime: Max fusion routes stored per prime (ranked by score).
        canonical_only: If True, store only the canonical fusion route.
        include_adjectives: Include adjective metadata when present in PRQS.
        min_prime: Minimum prime to include (default 2).
        mirror_prime: Prime reserved for mirror/dual operator (default 2).
        mirror_mode: Mirror behavior description (default "phase_flip").
        source_lexicon: Name of lexicon used for seeding.
    """

    num_primes: int = 200
    max_routes_per_prime: int = 5
    canonical_only: bool = False
    include_adjectives: bool = True
    min_prime: int = 2
    mirror_prime: int = 2
    mirror_mode: str = "phase_flip"
    source_lexicon: str = "PRQS_LEXICON"
