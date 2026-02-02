"""
Mirror/dual operator for prime states.
"""
from __future__ import annotations

from tinyaleph.hilbert.state import PrimeState
from tinyaleph.core.complex import Complex


def mirror_state(state: PrimeState) -> PrimeState:
    """
    Mirror a PrimeState by phase inversion (multiply by -1).
    """
    mirrored = state.copy()
    for p in mirrored.primes:
        amp = mirrored.amplitudes[p]
        mirrored.amplitudes[p] = Complex(-amp.re, -amp.im)
    return mirrored
