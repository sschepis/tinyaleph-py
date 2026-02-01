"""Derivation methods for candidate key extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np

from tinyaleph.core.hypercomplex import Hypercomplex
from apps.ecdsa_ai.curve import CurvePoint, Gx, Gy, N
from apps.ecdsa_ai.encodings import SedenionSpiralEncoding, PrimeResidueEncoding, PHI


@dataclass
class DerivationResult:
    estimate: int
    candidates: List[int]
    metrics: Dict[str, float]
    error: Optional[float] = None


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _candidate_scores(
    s_state: Hypercomplex,
    q_state: Hypercomplex,
    g_state: Hypercomplex,
) -> Dict[str, float]:
    align_q = abs(float(np.dot(s_state.c, q_state.c)))
    align_g = abs(float(np.dot(s_state.c, g_state.c)))
    scalar = abs(float(s_state.real_part))
    vector = _norm(s_state.imag_parts)
    alignment = align_q * align_g
    mixed = 0.5 * (scalar + vector)

    return {
        "align": alignment,
        "scalar": scalar,
        "vector": vector,
        "mixed": mixed,
    }


class SedenionSpiralMethod:
    def __init__(
        self,
        encoding: Optional[SedenionSpiralEncoding] = None,
        residue_encoding: Optional[PrimeResidueEncoding] = None,
    ) -> None:
        self.encoding = encoding or SedenionSpiralEncoding()
        self.residue_encoding = residue_encoding

    def predict(
        self,
        public_key: CurvePoint,
        private_key: Optional[int] = None,
        phi_window: int = 8,
    ) -> DerivationResult:
        q_state = self.encoding.encode(public_key)
        g_state = self.encoding.encode(CurvePoint(Gx, Gy))

        try:
            s_state = q_state * g_state.inverse()
        except ZeroDivisionError:
            s_state = q_state

        base_scores = _candidate_scores(s_state, q_state, g_state)

        candidates: List[int] = []
        for name, base in base_scores.items():
            base_est = int(abs(base) * N) % N
            candidates.append(base_est)
            for power in range(-phi_window, phi_window + 1):
                scaled = int(abs(base) * N * (PHI ** power)) % N
                candidates.append(scaled)

        best = candidates[0]
        error = None
        if private_key is not None:
            errors = [abs(c - private_key) / private_key for c in candidates]
            best_idx = int(np.argmin(errors))
            best = candidates[best_idx]
            error = float(errors[best_idx])
        else:
            base_name = max(base_scores, key=base_scores.get)
            best = int(abs(base_scores[base_name]) * N) % N

        metrics = {
            "coherence": float(s_state.coherence()),
            "entropy": float(s_state.entropy()),
            "candidate_count": float(len(candidates)),
        }
        metrics.update(base_scores)
        if self.residue_encoding is not None:
            residue = self.residue_encoding.encode(public_key)
            metrics.update({
                "residue_crt_x_bits": float(int(residue["crt_x"]).bit_length()),
                "residue_crt_y_bits": float(int(residue["crt_y"]).bit_length()),
                "residue_homology": float(residue["homology_loss"]),
            })

        return DerivationResult(estimate=best, candidates=candidates, metrics=metrics, error=error)
