"""Encodings for mapping curve points into hypercomplex state space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math
import random
import numpy as np

from tinyaleph.core.hypercomplex import Hypercomplex
from tinyaleph.core.primes import first_n_primes
from apps.ecdsa_ai.curve import P, CurvePoint

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class SpiralEncodingParams:
    spiral_scale: float
    curvature: float
    frequency: float
    phase_offsets: np.ndarray
    weights: np.ndarray

    @staticmethod
    def default() -> "SpiralEncodingParams":
        phase = np.zeros(16, dtype=np.float64)
        weights = np.ones(16, dtype=np.float64)
        return SpiralEncodingParams(
            spiral_scale=1.0,
            curvature=0.0,
            frequency=1.0,
            phase_offsets=phase,
            weights=weights,
        )

    def mutate(self, temperature: float = 0.1) -> "SpiralEncodingParams":
        def jitter(value: float, scale: float) -> float:
            return value * (1 + random.gauss(0.0, scale))

        phase = self.phase_offsets + np.random.normal(0.0, temperature * 0.15, 16)
        weights = self.weights + np.random.normal(0.0, temperature * 0.2, 16)
        weights = np.clip(weights, 0.05, 5.0)

        return SpiralEncodingParams(
            spiral_scale=jitter(self.spiral_scale, temperature * 0.4),
            curvature=self.curvature + random.gauss(0.0, temperature * 0.3),
            frequency=jitter(self.frequency, temperature * 0.3),
            phase_offsets=phase,
            weights=weights,
        )


class SedenionSpiralEncoding:
    def __init__(self, params: SpiralEncodingParams | None = None) -> None:
        self.params = params or SpiralEncodingParams.default()

    def encode(self, point: CurvePoint) -> Hypercomplex:
        log_x = math.log(point.x + 1) / math.log(P)
        log_y = math.log(point.y + 1) / math.log(P)
        r = math.sqrt(log_x * log_x + log_y * log_y) * self.params.spiral_scale
        theta = math.atan2(log_y, log_x) * self.params.frequency

        components: List[float] = []
        for i in range(16):
            spiral = theta + r * math.pi * (i / 8) * (1 + self.params.curvature * r)
            spiral += float(self.params.phase_offsets[i])
            if i < 8:
                base = math.cos(spiral) * (1 - i / 16)
            else:
                base = math.sin(spiral) * ((i - 8) / 16)
            components.append(base * float(self.params.weights[i]))

        arr = np.array(components, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-12:
            arr /= norm

        return Hypercomplex(16, arr)

    def mutate(self, temperature: float = 0.1) -> "SedenionSpiralEncoding":
        return SedenionSpiralEncoding(self.params.mutate(temperature))

    def signature(self) -> str:
        phase_sig = "|".join(f"{v:.3f}" for v in self.params.phase_offsets[:4])
        return f"spiral:{self.params.spiral_scale:.3f}:{self.params.frequency:.3f}:{phase_sig}"


class CRTReconstructor:
    def reconstruct(self, residues: List[int], moduli: List[int]) -> int:
        if len(residues) != len(moduli):
            raise ValueError("Residues and moduli must align")
        result = 0
        modulus = 1
        for residue, mod in zip(residues, moduli):
            inv = pow(modulus, -1, mod)
            t = ((residue - result) * inv) % mod
            result += t * modulus
            modulus *= mod
        return result


class HomologyLoss:
    def score(self, residues_x: List[int], residues_y: List[int], moduli: List[int]) -> float:
        if not residues_x or not residues_y:
            return 0.0
        total = 0.0
        for rx, ry, mod in zip(residues_x, residues_y, moduli):
            diff = abs(rx - ry)
            diff = min(diff, mod - diff)
            total += diff / mod
        return total / len(moduli)


class PrimeResidueEncoding:
    def __init__(self, num_primes: int = 32) -> None:
        self.primes = first_n_primes(num_primes)
        self.reconstructor = CRTReconstructor()
        self.homology = HomologyLoss()

    def encode(self, point: CurvePoint) -> dict:
        residues_x = [point.x % p for p in self.primes]
        residues_y = [point.y % p for p in self.primes]
        crt_x = self.reconstructor.reconstruct(residues_x, self.primes)
        crt_y = self.reconstructor.reconstruct(residues_y, self.primes)
        homology_loss = self.homology.score(residues_x, residues_y, self.primes)
        return {
            "residues_x": residues_x,
            "residues_y": residues_y,
            "crt_x": crt_x,
            "crt_y": crt_y,
            "homology_loss": homology_loss,
        }
