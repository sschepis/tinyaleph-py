"""
Mathematical and physical constants used throughout TinyAleph.

Uses pure Python math - no numpy required.
"""
import math


# Golden ratio φ = (1 + √5) / 2
PHI: float = (1 + math.sqrt(5)) / 2

# Conjugate golden ratio ψ = φ - 1 = 1/φ
PHI_CONJUGATE: float = PHI - 1

# Euler-Mascheroni constant γ
EULER_MASCHERONI: float = 0.5772156649015328606065120900824024310421593359

# Base entropy step (from TinyAleph theory)
DELTA_S: float = 0.01

# Prime coherence threshold
COHERENCE_THRESHOLD: float = 0.7

# Default entropy threshold for halting
ENTROPY_THRESHOLD: float = 2.0

# Entropy collapse threshold
ENTROPY_COLLAPSE_THRESHOLD: float = 0.5

# Lambda stability threshold for Lyapunov analysis
LAMBDA_STABILITY_THRESHOLD: float = 0.1

# Maximum number of primes in default Hilbert space
DEFAULT_PRIME_COUNT: int = 25

# Sedenion dimension
SEDENION_DIM: int = 16

# Octonion dimension
OCTONION_DIM: int = 8

# Quaternion dimension
QUATERNION_DIM: int = 4

# Complex dimension
COMPLEX_DIM: int = 2

# Planck-like constant for discrete evolution
H_PRIME: float = 0.1

# Critical coupling for Kuramoto transition
KURAMOTO_CRITICAL_K: float = 2.0

# Default memory decay rate
MEMORY_DECAY_RATE: float = 0.01

# Maximum Lyapunov exponent threshold for stability
LYAPUNOV_STABILITY_THRESHOLD: float = 0.0

# Prime Resonance Identity primes (commonly used)
PRI_GAUSSIAN_PRIMES: list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
PRI_EISENSTEIN_PRIMES: list = [2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
PRI_QUATERNIONIC_PRIMES: list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

# Resonance weights from TinyAleph
RESONANCE_WEIGHTS: dict = {
    "harmonic": 1.0,
    "golden": PHI,
    "prime": 1.0 / math.log(2),
    "fibonacci": PHI ** 2,
}

# Fibonacci sequence (first 20 terms)
FIBONACCI: list = [
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
    89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765
]

# Lucas numbers (first 20 terms)
LUCAS: list = [
    2, 1, 3, 4, 7, 11, 18, 29, 47, 76,
    123, 199, 322, 521, 843, 1364, 2207, 3571, 5778, 9349
]

# Golden angle in radians: 2π/φ²
GOLDEN_ANGLE: float = 2 * math.pi / (PHI ** 2)

# Natural logarithm of 2
LN2: float = math.log(2)

# Natural logarithm of golden ratio
LN_PHI: float = math.log(PHI)