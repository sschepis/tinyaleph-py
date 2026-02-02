"""
Deterministic semantic landscape builder.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple

from tinyaleph.core.primes import first_n_primes, is_prime
from tinyaleph.semantic.lambda_calc import PRQS_LEXICON, ConceptInterpreter, classify_prime
from tinyaleph.semantic.reduction import FusionCanonicalizer
from tinyaleph.semantic.types import FusionTerm, NounTerm

from .config import LandscapeConfig
from .landscape import FusionRoute, PrimeEntry, SemanticLandscape


def _route_entropy(route_count: int) -> float:
    """
    Simple entropy proxy based on route count.
    More routes imply higher ambiguity.
    """
    if route_count <= 0:
        return 0.0
    return min(1.0, route_count / (route_count + 3.0))


def _find_triads_for_target(
    target: int,
    primes: List[int],
    prime_set: set,
    mirror_prime: int,
) -> List[FusionTerm]:
    """
    Find valid triads p < q < r with p+q+r = target.
    Excludes mirror_prime (default 2).
    """
    if not is_prime(target):
        return []

    triads: List[FusionTerm] = []

    for i, p in enumerate(primes):
        if p == mirror_prime:
            continue
        if p >= target:
            break
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if q == mirror_prime:
                continue
            if p + q >= target:
                break
            r = target - p - q
            if r <= q:
                continue
            if r == mirror_prime:
                continue
            if r in prime_set:
                triads.append(FusionTerm(p, q, r))

    return triads


def build_landscape(config: LandscapeConfig) -> SemanticLandscape:
    primes = first_n_primes(config.num_primes)
    primes = [p for p in primes if p >= config.min_prime]
    prime_set = set(primes)

    lexicon_nouns = PRQS_LEXICON.get("nouns", {})
    lexicon_adjs = PRQS_LEXICON.get("adjectives", {})

    interpreter = ConceptInterpreter(PRQS_LEXICON)
    canonicalizer = FusionCanonicalizer()

    entries: Dict[int, PrimeEntry] = {}
    seeded_count = 0
    fused_count = 0
    fallback_count = 0
    entropy_sum = 0.0

    for prime in primes:
        classification = classify_prime(prime)
        adjective = None
        if config.include_adjectives and prime in lexicon_adjs:
            adjective = lexicon_adjs[prime].get("concept")

        if prime in lexicon_nouns:
            noun_meta = lexicon_nouns[prime]
            meaning = noun_meta.get("concept", f"concept_{prime}")
            category = noun_meta.get("category", classification.get("primary", ""))
            role = noun_meta.get("role")
            origin = "seeded"
            confidence = 1.0
            routes: List[FusionRoute] = []
            components: List[int] = []
            route_count = 0
            entropy = 0.0
            coherence = 1.0
            metadata = {"seed": noun_meta}
            seeded_count += 1
        else:
            triads = _find_triads_for_target(prime, primes, prime_set, config.mirror_prime)
            route_count = len(triads)
            routes_scored: List[Tuple[FusionTerm, float]] = []
            for triad in triads:
                score = canonicalizer.resonance_score(triad)
                routes_scored.append((triad, score))

            routes_scored.sort(key=lambda x: (-x[1], x[0].p, x[0].q, x[0].r))

            if routes_scored:
                canonical_triad = routes_scored[0][0]
                fusion_sem = interpreter.interpret_fusion_semantic(
                    canonical_triad.p, canonical_triad.q, canonical_triad.r
                )
                meaning = fusion_sem.get("emergent", f"fusion_{prime}")
                category = fusion_sem.get("dominant", classification.get("primary", ""))
                role = None
                origin = "fusion"
                confidence = 0.7
                components = [canonical_triad.p, canonical_triad.q, canonical_triad.r]

                routes = []
                for idx, (triad, score) in enumerate(routes_scored):
                    if config.canonical_only and idx > 0:
                        break
                    if idx >= config.max_routes_per_prime:
                        break
                    routes.append(
                        FusionRoute(
                            p=triad.p,
                            q=triad.q,
                            r=triad.r,
                            score=score,
                            canonical=(idx == 0),
                        )
                    )

                entropy = _route_entropy(route_count)
                coherence = 1.0 - entropy
                metadata = {"fusion": fusion_sem}
                fused_count += 1
            else:
                meaning = interpreter.interpret_noun(NounTerm(prime))
                category = classification.get("primary", "")
                role = None
                origin = "fallback"
                confidence = 0.4
                routes = []
                components = []
                entropy = 1.0
                coherence = 0.0
                metadata = {"note": "no_valid_fusion_routes"}
                fallback_count += 1

        entry = PrimeEntry(
            prime=prime,
            meaning=meaning,
            category=category,
            role=role,
            origin=origin,
            confidence=confidence,
            coherence=coherence,
            entropy=entropy,
            route_count=route_count,
            components=components,
            routes=routes,
            classification=classification,
            adjective=adjective,
            refined=False,
            metadata=metadata,
        )
        entries[prime] = entry
        entropy_sum += entropy

    stats = {
        "prime_count": len(entries),
        "seeded": seeded_count,
        "fusion": fused_count,
        "fallback": fallback_count,
        "mean_entropy": entropy_sum / max(1, len(entries)),
    }

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_lexicon": config.source_lexicon,
        "config": asdict(config),
        "mirror": {
            "prime": config.mirror_prime,
            "mode": config.mirror_mode,
            "description": "mirror/dual operator, modeled as phase inversion",
        },
    }

    return SemanticLandscape(metadata=metadata, entries=entries, stats=stats)
