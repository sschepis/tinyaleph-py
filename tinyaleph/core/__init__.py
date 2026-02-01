"""
Core mathematical primitives for TinyAleph.

Provides:
- Complex: Complex numbers with full arithmetic
- Quaternion: Hamilton quaternions for 3D rotations
- Hypercomplex: Cayley-Dickson algebras (requires numpy)
- Prime utilities: Factorization, sieves, prime tests

Note: Hypercomplex (Octonion, Sedenion) requires numpy installation.
The core Complex and Quaternion classes work with pure Python.
"""

from tinyaleph.core.complex import Complex
from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.primes import (
    is_prime,
    nth_prime,
    prime_sieve,
    factorize,
    prime_index,
    next_prime,
    prev_prime,
)
from tinyaleph.core.constants import (
    PHI,
    DELTA_S,
    LAMBDA_STABILITY_THRESHOLD,
    ENTROPY_COLLAPSE_THRESHOLD,
    COHERENCE_THRESHOLD,
)

# Try to import hypercomplex (requires numpy)
try:
    from tinyaleph.core.hypercomplex import Hypercomplex
    
    # Create convenience aliases for specific dimensions
    def Octonion(components=None):
        """Create 8-dimensional octonion."""
        import numpy as np
        if components is None:
            components = np.zeros(8)
        return Hypercomplex(8, components)
    
    def Sedenion(components=None):
        """Create 16-dimensional sedenion."""
        import numpy as np
        if components is None:
            components = np.zeros(16)
        return Hypercomplex(16, components)
    
    # Also create Cayley-Dickson alias
    CayleyDicksonAlgebra = Hypercomplex
    
    _HAS_NUMPY = True
except ImportError:
    # Numpy not available - hypercomplex types not available
    Hypercomplex = None
    Octonion = None
    Sedenion = None
    CayleyDicksonAlgebra = None
    _HAS_NUMPY = False


__all__ = [
    # Core types (always available)
    "Complex",
    "Quaternion",
    # Prime utilities
    "is_prime",
    "nth_prime", 
    "prime_sieve",
    "factorize",
    "prime_index",
    "next_prime",
    "prev_prime",
    # Constants
    "PHI",
    "DELTA_S",
    "LAMBDA_STABILITY_THRESHOLD",
    "ENTROPY_COLLAPSE_THRESHOLD",
    "COHERENCE_THRESHOLD",
    # Hypercomplex (may be None if numpy not available)
    "Hypercomplex",
    "Octonion",
    "Sedenion",
    "CayleyDicksonAlgebra",
]