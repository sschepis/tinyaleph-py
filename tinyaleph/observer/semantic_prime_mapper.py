from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
import math
import os
import random

from tinyaleph.observer.symbols import SymbolDatabase


@dataclass
class TriadicFusion:
    p: int
    q: int
    r: int
    result: str
    fusion_type: str
    strength: float


@dataclass
class PrimeMeaning:
    prime: int
    meaning: str
    raw_meaning: Optional[str] = None
    confidence: float = 0.7
    derived_from: List[TriadicFusion] = field(default_factory=list)
    entropy: float = 0.5
    is_seeded: bool = False
    is_refined: bool = False
    category: Optional[str] = None
    resonant_with: List[int] = field(default_factory=list)
    prime_signature: Optional[List[int]] = None


class SemanticPrimeMapper:
    """
    Python port of the TS SemanticPrimeMapper.

    Responsibilities:
      - Seed meanings from SymbolDatabase.
      - Maintain prime->meaning assignments and metrics.
      - Propose meanings for uncatalogued primes via triadic fusion.
      - Compute local/global entropy; minimize entropy via alternate fusions.
      - Optionally refine meanings via an external callable.
    """

    def __init__(self, num_primes: int = 128):
        self.symbol_db = SymbolDatabase()
        self.target_primes = self._first_n_primes(num_primes)
        self.field: Dict[int, PrimeMeaning] = {}
        self.global_entropy: float = 1.0
        self.coherence: float = 0.0
        self._seed_from_symbols()
        self._update_metrics()

    # --------- Seeding ---------
    def _seed_from_symbols(self) -> None:
        for sym in self.symbol_db.get_all_symbols():
            pm = PrimeMeaning(
                prime=sym.prime,
                meaning=sym.name,
                confidence=1.0,
                derived_from=[],
                entropy=0.0,
                is_seeded=True,
                is_refined=True,
                category=sym.category.name if hasattr(sym, "category") else None,
                resonant_with=[],
            )
            self.field[sym.prime] = pm

    # --------- Prime utilities ---------
    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        r = int(math.sqrt(n)) + 1
        for i in range(3, r, 2):
            if n % i == 0:
                return False
        return True

    def _first_n_primes(self, n: int) -> List[int]:
        out: List[int] = []
        candidate = 2
        while len(out) < n:
            if self._is_prime(candidate):
                out.append(candidate)
            candidate += 1
        return out

    # --------- Public API ---------
    def get_uncatalogued_primes(self) -> List[int]:
        return [p for p in self.target_primes if p not in self.field]

    def get_field(self) -> Dict[int, PrimeMeaning]:
        return self.field

    def add_learned_meaning(self, prime: int, meaning: str, confidence: float = 0.8, category: Optional[str] = None) -> None:
        existing = self.field.get(prime)
        if existing and existing.is_seeded:
            return
        pm = PrimeMeaning(
            prime=prime,
            meaning=meaning,
            raw_meaning=meaning,
            confidence=confidence,
            derived_from=[],
            entropy=0.1,
            is_seeded=False,
            is_refined=True,
            category=category,
            resonant_with=[],
        )
        pm.entropy = self._calculate_local_entropy(pm)
        self.field[prime] = pm
        self._update_metrics()

    def expand_by_fusion(self) -> Tuple[int, List[PrimeMeaning]]:
        uncatalogued = self.get_uncatalogued_primes()
        new_meanings: List[PrimeMeaning] = []
        for target in uncatalogued:
            cands = self._find_fusion_candidates(target)
            if not cands:
                continue
            best = max(cands, key=lambda c: c.strength)
            pm = PrimeMeaning(
                prime=target,
                meaning=best.result,
                raw_meaning=best.result,
                confidence=min(0.8, best.strength),
                derived_from=[best],
                entropy=0.5,
                is_seeded=False,
                is_refined=False,
                resonant_with=[best.p, best.q, best.r],
            )
            pm.entropy = self._calculate_local_entropy(pm)
            self.field[target] = pm
            new_meanings.append(pm)
        self._update_metrics()
        return len(new_meanings), new_meanings

    def minimize_entropy(self) -> Tuple[int, float]:
        initial = self.global_entropy
        improvements = 0
        for prime, meaning in list(self.field.items()):
            if meaning.is_seeded:
                continue
            alts = self._find_fusion_candidates(prime)
            for alt in alts:
                test = PrimeMeaning(
                    prime=prime,
                    meaning=meaning.meaning,
                    raw_meaning=meaning.raw_meaning,
                    confidence=min(0.95, meaning.confidence + 0.05 * alt.strength),
                    derived_from=meaning.derived_from + [alt],
                    entropy=meaning.entropy,
                    is_seeded=False,
                    is_refined=meaning.is_refined,
                    category=meaning.category,
                    resonant_with=list(set(meaning.resonant_with + [alt.p, alt.q, alt.r])),
                )
                new_entropy = self._calculate_local_entropy(test)
                if new_entropy < meaning.entropy:
                    test.entropy = new_entropy
                    self.field[prime] = test
                    improvements += 1
        self._update_metrics()
        return improvements, initial - self.global_entropy

    def refine_meanings(self, refine_fn) -> Tuple[int, List[str]]:
        """refine_fn expects List[PrimeMeaning] -> List[(prime, refined_str)]."""
        unrefined = [m for m in self.field.values() if not m.is_refined and not m.is_seeded]
        if not unrefined:
            return 0, []
        errors: List[str] = []
        refined_count = 0
        try:
            results = refine_fn(unrefined)
            for prime, refined in results:
                pm = self.field.get(prime)
                if pm and refined:
                    pm.raw_meaning = pm.meaning
                    pm.meaning = refined
                    pm.is_refined = True
                    pm.confidence = min(0.95, pm.confidence + 0.1)
                    refined_count += 1
        except Exception as e:
            errors.append(str(e))
        self._update_metrics()
        return refined_count, errors

    def get_unrefined_count(self) -> int:
        return len([m for m in self.field.values() if not m.is_refined and not m.is_seeded])

    def expand_cycle(self, refine_fn=None) -> Dict[str, Any]:
        expanded, _ = self.expand_by_fusion()
        improved, delta = self.minimize_entropy()
        refined = 0
        if refine_fn:
            refined, _ = self.refine_meanings(refine_fn)
        return {"expanded": expanded, "improved": improved, "entropy_delta": delta, "global_entropy": self.global_entropy, "refined": refined}

    def export_knowledge(self) -> Dict[str, Any]:
        return {
            "meanings": [self._pm_to_dict(pm) for pm in self.field.values() if not pm.is_seeded],
            "global_entropy": self.global_entropy,
            "coherence": self.coherence,
        }

    def import_knowledge(self, data: Dict[str, Any]) -> None:
        meanings = data.get("meanings", [])
        for m in meanings:
            pm = self._dict_to_pm(m)
            self.field[pm.prime] = pm
        self._update_metrics()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_knowledge(), f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.import_knowledge(data)

    # --------- Internal fusion/entropy ---------
    def _find_fusion_candidates(self, target: int) -> List[TriadicFusion]:
        cands: List[TriadicFusion] = []
        known = [p for p, m in self.field.items() if m.confidence > 0.5]
        for i in range(len(known)):
            for j in range(i + 1, len(known)):
                p = known[i]; q = known[j]
                fusion = self._triadic_fuse(target, p, q)
                if fusion and fusion.strength > 0.1:
                    cands.append(fusion)
        cands.sort(key=lambda c: c.strength, reverse=True)
        return cands[:5]

    def _triadic_fuse(self, p: int, q: int, r: int) -> Optional[TriadicFusion]:
        if p == q or q == r or p == r:
            return None
        if not self._is_prime(p + q + r):
            return None
        # Require at least two known meanings
        known = [self.field.get(x) for x in (p, q, r) if self.field.get(x)]
        if len(known) < 2:
            return None
        sum_ = p + q + r
        harmonic_mean = 3 / (1/p + 1/q + 1/r)
        prime_gap = abs(p - q) + abs(q - r) + abs(r - p)
        strength = math.exp(-prime_gap / sum_) * 1.5
        fused_meaning = self._combine_meanings([k.meaning for k in known], p, q, r)
        fusion_type = "harmonic" if math.gcd(math.gcd(p, q), r) > 1 else "additive"
        return TriadicFusion(p=p, q=q, r=r, result=fused_meaning, fusion_type=fusion_type, strength=strength)

    def _combine_meanings(self, meanings: List[str], p: int, q: int, r: int) -> str:
        words = [w for w in " ".join(meanings).lower().replace(",", " ").split() if len(w) > 3]
        uniq = list(dict.fromkeys(words))
        prefixes = ["meta-", "proto-", "trans-", "hyper-", "neo-"]
        suffixes = ["-essence", "-nature", "-force", "-aspect", "-resonance"]
        prefix = prefixes[p % len(prefixes)]
        suffix = suffixes[r % len(suffixes)]
        if len(uniq) >= 2:
            return f"{prefix}{uniq[0]}-{uniq[1]}{suffix}"
        if len(uniq) == 1:
            return f"{prefix}{uniq[0]}{suffix}"
        return f"resonance-{p}-{q}-{r}"

    def _calculate_local_entropy(self, pm: PrimeMeaning) -> float:
        if pm.is_seeded:
            return 0.0
        confidence_entropy = -math.log2(max(0.001, pm.confidence))
        resonance_entropy = 1.0
        if pm.resonant_with:
            resonance_entropy = 0.0
            for neighbor in pm.resonant_with:
                neighbor_pm = self.field.get(neighbor)
                if neighbor_pm:
                    similarity = 1 / (1 + abs(math.log(pm.prime / neighbor_pm.prime))) if neighbor_pm.prime else 0
                    resonance_entropy += similarity * neighbor_pm.confidence
            resonance_entropy = -math.log2(max(0.001, resonance_entropy / max(1, len(pm.resonant_with))))
        derivation_entropy = math.log2(1 + len(pm.derived_from)) * 0.1
        return (confidence_entropy + resonance_entropy + derivation_entropy) / 3

    def _update_metrics(self) -> None:
        if not self.field:
            self.global_entropy = 1.0
            self.coherence = 0.0
            return
        total_entropy = sum(pm.entropy for pm in self.field.values())
        self.global_entropy = total_entropy / len(self.field)
        # Coherence proxy: inverse of entropy normalized
        self.coherence = 1.0 / (1.0 + self.global_entropy)

    # --------- Serialization helpers ---------
    def _pm_to_dict(self, pm: PrimeMeaning) -> Dict[str, Any]:
        return {
            "prime": pm.prime,
            "meaning": pm.meaning,
            "raw_meaning": pm.raw_meaning,
            "confidence": pm.confidence,
            "derived_from": [fusion.__dict__ for fusion in pm.derived_from],
            "entropy": pm.entropy,
            "is_seeded": pm.is_seeded,
            "is_refined": pm.is_refined,
            "category": pm.category,
            "resonant_with": pm.resonant_with,
            "prime_signature": pm.prime_signature,
        }

    def _dict_to_pm(self, data: Dict[str, Any]) -> PrimeMeaning:
        derived = [TriadicFusion(**d) for d in data.get("derived_from", [])]
        return PrimeMeaning(
            prime=data["prime"],
            meaning=data["meaning"],
            raw_meaning=data.get("raw_meaning"),
            confidence=data.get("confidence", 0.7),
            derived_from=derived,
            entropy=data.get("entropy", 0.5),
            is_seeded=data.get("is_seeded", False),
            is_refined=data.get("is_refined", False),
            category=data.get("category"),
            resonant_with=data.get("resonant_with", []),
            prime_signature=data.get("prime_signature"),
        )

