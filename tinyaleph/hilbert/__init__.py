"""
Prime Hilbert Space H_P

Provides quantum state representations and operators for the
prime Hilbert space where primes form an orthonormal basis.

Components:
- PrimeState: Quantum state |ψ⟩ = Σ α_p |p⟩
- Operators: P̂ (shift), F̂ (Fourier), R̂ (resonance), Ĉ (collapse)
"""

from tinyaleph.hilbert.state import PrimeState

# Import operators if available
try:
    from tinyaleph.hilbert.operators import (
        Operator,
        PrimeShiftOperator,
        PrimeFourierOperator,
        ResonanceOperator,
        CollapseOperator,
        PhaseOperator,
        ProjectionOperator,
        HadamardLikeOperator,
        TimeEvolutionOperator,
        GoldenPhaseOperator,
        # Convenience functions
        shift,
        fourier,
        resonance,
        collapse,
        phase,
        project,
        hadamard,
        evolve,
        golden_phase,
        identity,
    )
except ImportError:
    pass

__all__ = [
    "PrimeState",
    "Operator",
    "PrimeShiftOperator",
    "PrimeFourierOperator",
    "ResonanceOperator",
    "CollapseOperator",
    "PhaseOperator",
    "ProjectionOperator",
    "HadamardLikeOperator",
    "TimeEvolutionOperator",
    "GoldenPhaseOperator",
    "shift",
    "fourier",
    "resonance",
    "collapse",
    "phase",
    "project",
    "hadamard",
    "evolve",
    "golden_phase",
    "identity",
]