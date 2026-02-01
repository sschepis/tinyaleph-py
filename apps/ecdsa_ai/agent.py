"""Exploration loop for evolving key-derivation encodings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import random

from tinyaleph.observer.smf import SedenionMemoryField
from tinyaleph.physics.entropy import EntropyTracker, shannon_entropy

from apps.ecdsa_ai.curve import CurvePoint, generate_keypair
from apps.ecdsa_ai.encodings import SedenionSpiralEncoding, PrimeResidueEncoding
from apps.ecdsa_ai.methods import SedenionSpiralMethod, DerivationResult
from apps.ecdsa_ai.hnp import hnp_consistency
from apps.ecdsa_ai.datasets import ReplayDataset
from apps.ecdsa_ai.logging import JsonlLogger


@dataclass
class PopulationMember:
    encoding: SedenionSpiralEncoding
    score: float = 0.0
    last_result: Optional[DerivationResult] = None
    hnp_metrics: Optional[Dict[str, float]] = None


class KeyDerivationExplorer:
    def __init__(
        self,
        population_size: int = 32,
        elite_ratio: float = 0.25,
        mutation_temp: float = 0.25,
        memory_size: int = 200,
        sample_size: int = 16,
        dataset: Optional[ReplayDataset] = None,
        ecdsa_weight: float = 1.0,
        hnp_weight: float = 1.0,
        residue_primes: int = 32,
        logger: Optional[JsonlLogger] = None,
    ) -> None:
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_temp = mutation_temp
        self.sample_size = sample_size
        self.dataset = dataset
        self.ecdsa_weight = ecdsa_weight
        self.hnp_weight = hnp_weight
        self.logger = logger
        self.residue_encoder = PrimeResidueEncoding(num_primes=residue_primes)

        self.population: List[PopulationMember] = []
        self.best_member: Optional[PopulationMember] = None

        self.memory = SedenionMemoryField(decay_rate=0.01, max_moments=memory_size)
        self.entropy_tracker = EntropyTracker(window_size=25)

        self._seed_population()

    def _seed_population(self) -> None:
        self.population = []
        for _ in range(self.population_size):
            encoding = SedenionSpiralEncoding().mutate(random.uniform(0.05, 0.4))
            self.population.append(PopulationMember(encoding=encoding))

    def _evaluate_member(self, member: PopulationMember, samples: List[Tuple[int, CurvePoint]]) -> float:
        method = SedenionSpiralMethod(member.encoding, residue_encoding=self.residue_encoder)
        errors: List[float] = []
        last_result: Optional[DerivationResult] = None
        for private_key, public_key in samples:
            result = method.predict(public_key, private_key)
            last_result = result
            if result.error is not None:
                errors.append(result.error)
        if not errors:
            return 0.0
        avg_error = sum(errors) / len(errors)
        member.last_result = last_result
        ecdsa_score = 1.0 / (avg_error + 1e-9)

        hnp_match = 0.0
        hnp_error = 1.0
        if self.dataset is not None:
            hnp_result = method.predict(self.dataset.hnp_public_key, None)
            best_score = -1.0
            best_match = 0.0
            best_error = 1.0
            best_candidate = None
            for candidate in hnp_result.candidates:
                match_ratio, low_error = hnp_consistency(
                    candidate,
                    self.dataset.hnp_signatures,
                    bits=self.dataset.hnp_bits,
                    mode=self.dataset.hnp_mode,
                )
                score = match_ratio - low_error
                if score > best_score:
                    best_score = score
                    best_match = match_ratio
                    best_error = low_error
                    best_candidate = candidate
            hnp_match = best_match
            hnp_error = best_error
        member.hnp_metrics = {
            "match_ratio": hnp_match,
            "low_error": hnp_error,
            "best_candidate": best_candidate,
        }
        hnp_score = max(0.0, hnp_match - hnp_error)

        member.score = self.ecdsa_weight * ecdsa_score + self.hnp_weight * hnp_score
        return avg_error

    def _sample_dataset(self) -> List[Tuple[int, CurvePoint]]:
        if self.dataset is not None:
            return self.dataset.ecdsa_pairs
        return [generate_keypair() for _ in range(self.sample_size)]

    def _record_memory(self, member: PopulationMember, avg_error: float) -> None:
        signature = member.encoding.signature()
        self.memory.encode(f"{signature}::error={avg_error:.6f}", importance=1.0)

    def _update_entropy(self) -> None:
        scores = [m.score for m in self.population if m.score > 0]
        if not scores:
            self.entropy_tracker.record(0.0)
            return
        total = sum(scores)
        probs = {i: s / total for i, s in enumerate(scores)}
        self.entropy_tracker.record(shannon_entropy(probs))

    def _select_elite(self) -> List[PopulationMember]:
        sorted_pop = sorted(self.population, key=lambda m: m.score, reverse=True)
        elite_count = max(1, int(len(sorted_pop) * self.elite_ratio))
        return sorted_pop[:elite_count]

    def _mutate_population(self, elite: List[PopulationMember]) -> None:
        next_population = [PopulationMember(encoding=e.encoding) for e in elite]
        while len(next_population) < self.population_size:
            parent = random.choice(elite)
            mutated = parent.encoding.mutate(self.mutation_temp)
            next_population.append(PopulationMember(encoding=mutated))
        self.population = next_population

    def step(self, generation: int = 0) -> float:
        samples = self._sample_dataset()
        avg_errors = []

        for member in self.population:
            avg_error = self._evaluate_member(member, samples)
            avg_errors.append(avg_error)

        self._update_entropy()
        elite = self._select_elite()

        best = elite[0]
        if self.best_member is None or best.score > self.best_member.score:
            self.best_member = PopulationMember(
                encoding=best.encoding,
                score=best.score,
                last_result=best.last_result,
                hnp_metrics=best.hnp_metrics,
            )

        best_error = min(avg_errors) if avg_errors else float("inf")
        self._record_memory(best, best_error)

        entropy = self.entropy_tracker.current_entropy()
        if entropy < 0.5:
            self.mutation_temp = min(self.mutation_temp * 1.1, 0.6)
        else:
            self.mutation_temp = max(self.mutation_temp * 0.95, 0.05)

        self._mutate_population(elite)
        if self.logger is not None and best.hnp_metrics is not None:
            self.logger.log({
                "generation": generation,
                "best_error": best_error,
                "best_score": best.score,
                "hnp_match_ratio": best.hnp_metrics["match_ratio"],
                "hnp_low_error": best.hnp_metrics["low_error"],
                "hnp_best_candidate": best.hnp_metrics.get("best_candidate"),
                "hnp_bits": self.dataset.hnp_bits if self.dataset else None,
                "hnp_mode": self.dataset.hnp_mode if self.dataset else None,
                "mutation_temp": self.mutation_temp,
                "entropy": self.entropy_tracker.current_entropy(),
                "signature": best.encoding.signature(),
            })
        return best_error

    def run(self, generations: int = 10) -> PopulationMember:
        for generation in range(generations):
            self.step(generation=generation)
        if self.best_member is None:
            raise RuntimeError("No best member recorded")
        return self.best_member
