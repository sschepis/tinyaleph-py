"""
TinyAleph: Prime-Resonant Quantum Computing Framework

A Python library unifying concepts from TinyAleph and ResoLang for
prime-based quantum-inspired computing.

Core Modules:
- core: Complex numbers, quaternions, hypercomplex algebra, primes
- hilbert: Prime Hilbert space and quantum operators (requires numpy)
- physics: Kuramoto oscillators and entropy analysis
- resonance: Holographic memory fragments
- network: Distributed computing with entanglement
- observer: SMF and PRSC semantic layers
- ml: Sparse prime states and resonant attention
- runtime: Engine and execution context
- distributed: Transport and networking

Note: Some modules (hilbert, physics, resonance, observer, ml) require numpy.
The core module works with pure Python.
"""

__version__ = "0.1.0"
__author__ = "Sebastian Schepis"

# Core mathematical primitives (always available - pure Python)
from tinyaleph.core import (
    Complex,
    Quaternion,
    is_prime,
    nth_prime,
    prime_sieve,
    factorize,
    prime_index,
    PHI,
    DELTA_S,
    LAMBDA_STABILITY_THRESHOLD,
)

# Optional imports that require numpy
_HAS_NUMPY = False
try:
    import numpy
    _HAS_NUMPY = True
except ImportError:
    pass

# Conditional imports
if _HAS_NUMPY:
    try:
        from tinyaleph.hilbert import PrimeState
    except ImportError:
        PrimeState = None
    
    try:
        from tinyaleph.physics import (
            KuramotoOscillator,
            CoupledOscillatorNetwork,
            EntropyTracker,
            StabilityClass,
        )
    except ImportError:
        KuramotoOscillator = None
        CoupledOscillatorNetwork = None
        EntropyTracker = None
        StabilityClass = None
    
    try:
        from tinyaleph.resonance import ResonantFragment
    except ImportError:
        ResonantFragment = None
    
    try:
        from tinyaleph.network import PrimeResonanceIdentity, EntangledNode
    except ImportError:
        PrimeResonanceIdentity = None
        EntangledNode = None
    
    try:
        from tinyaleph.runtime import AlephEngine
    except ImportError:
        AlephEngine = None
    
    try:
        from tinyaleph.core import Octonion, Sedenion, CayleyDicksonAlgebra
    except ImportError:
        Octonion = None
        Sedenion = None
        CayleyDicksonAlgebra = None
else:
    # Numpy not available - set optional types to None
    PrimeState = None
    KuramotoOscillator = None
    CoupledOscillatorNetwork = None
    EntropyTracker = None
    StabilityClass = None
    ResonantFragment = None
    PrimeResonanceIdentity = None
    EntangledNode = None
    AlephEngine = None
    Octonion = None
    Sedenion = None
    CayleyDicksonAlgebra = None


__all__ = [
    # Version
    "__version__",
    # Core (always available)
    "Complex",
    "Quaternion",
    "is_prime",
    "nth_prime",
    "prime_sieve",
    "factorize",
    "prime_index",
    "PHI",
    "DELTA_S",
    "LAMBDA_STABILITY_THRESHOLD",
    # Optional (may be None if numpy not available)
    "Octonion",
    "Sedenion",
    "CayleyDicksonAlgebra",
    "PrimeState",
    "KuramotoOscillator",
    "CoupledOscillatorNetwork",
    "EntropyTracker",
    "StabilityClass",
    "ResonantFragment",
    "PrimeResonanceIdentity",
    "EntangledNode",
    "AlephEngine",
]