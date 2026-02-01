# TinyAleph Reference Guide

## Table of Contents

1. [Module Overview](#module-overview)
2. [Core Module (`tinyaleph.core`)](#core-module)
3. [Hilbert Module (`tinyaleph.hilbert`)](#hilbert-module)
4. [Physics Module (`tinyaleph.physics`)](#physics-module)
5. [Observer Module (`tinyaleph.observer`)](#observer-module)
6. [Semantic Module (`tinyaleph.semantic`)](#semantic-module)
7. [ML Module (`tinyaleph.ml`)](#ml-module)
8. [Network Module (`tinyaleph.network`)](#network-module)
9. [Engine Module (`tinyaleph.engine`)](#engine-module)
10. [Runtime Module (`tinyaleph.runtime`)](#runtime-module)
11. [Backends Module (`tinyaleph.backends`)](#backends-module)
12. [Distributed Module (`tinyaleph.distributed`)](#distributed-module)
13. [Resonance Module (`tinyaleph.resonance`)](#resonance-module)
14. [Constants Reference](#constants-reference)

---

## Module Overview

| Module | Description | Primary Classes |
|--------|-------------|-----------------|
| `core` | Mathematical primitives | `Complex`, `Quaternion`, `Octonion`, `Sedenion` |
| `hilbert` | Prime Hilbert space | `PrimeState`, `PrimeOperator` |
| `physics` | Oscillator dynamics | `KuramotoOscillator`, `KuramotoField` |
| `observer` | Cognitive architecture | `SMF`, `PRSC`, `TemporalLayer`, `AgencyLayer` |
| `semantic` | Type system & inference | `NounTerm`, `ChainTerm`, `SemanticInference` |
| `ml` | Machine learning | `SparsePrimeState`, `ResoFormer` |
| `network` | Distributed computing | `EntanglementNetwork`, `PrimeResonanceIdentity` |
| `engine` | Field-based computation | `AlephEngine` |
| `runtime` | Execution context | `AlephEngine`, `AlephConfig` |
| `backends` | Cryptographic operations | `PrimeStateKeyGenerator` |
| `distributed` | Transport layer | `LocalTransport` |
| `resonance` | Holographic memory | `ResonantFragment` |

---

## Core Module

### `tinyaleph.core.complex`

#### `Complex`

A pure Python implementation of complex numbers.

```python
class Complex:
    """Complex number with real and imaginary parts."""
    
    def __init__(self, re: float = 0.0, im: float = 0.0):
        """
        Create a complex number.
        
        Args:
            re: Real part (default: 0.0)
            im: Imaginary part (default: 0.0)
        """
```

**Properties:**
- `re: float` - Real part
- `im: float` - Imaginary part

**Class Methods:**
- `zero() -> Complex` - Return complex zero (0 + 0i)
- `one() -> Complex` - Return complex one (1 + 0i)
- `i() -> Complex` - Return imaginary unit (0 + 1i)
- `from_polar(r: float, theta: float) -> Complex` - Create from polar form

**Instance Methods:**
- `norm() -> float` - Return |z| = √(re² + im²)
- `conjugate() -> Complex` - Return z̄ = re - im·i
- `inverse() -> Complex` - Return 1/z
- `phase() -> float` - Return arg(z) = atan2(im, re)
- `exp() -> Complex` - Return e^z
- `log() -> Complex` - Return ln(z)

**Operators:**
- `+`, `-`, `*`, `/` - Standard arithmetic
- `==`, `!=` - Equality comparison
- `abs()` - Absolute value (norm)

---

### `tinyaleph.core.quaternion`

#### `Quaternion`

Hamilton quaternions for 3D rotations.

```python
class Quaternion:
    """Quaternion: q = w + xi + yj + zk where i² = j² = k² = ijk = -1"""
    
    def __init__(self, w: float = 0.0, i: float = 0.0, 
                 j: float = 0.0, k: float = 0.0):
        """
        Create a quaternion.
        
        Args:
            w: Scalar (real) part
            i: First imaginary component
            j: Second imaginary component
            k: Third imaginary component
        """
```

**Properties:**
- `w: float` - Scalar part
- `i: float` - i component
- `j: float` - j component
- `k: float` - k component

**Class Methods:**
- `zero() -> Quaternion` - Return (0, 0, 0, 0)
- `one() -> Quaternion` - Return (1, 0, 0, 0)
- `i() -> Quaternion` - Return (0, 1, 0, 0)
- `j() -> Quaternion` - Return (0, 0, 1, 0)
- `k() -> Quaternion` - Return (0, 0, 0, 1)
- `from_axis_angle(x, y, z, angle) -> Quaternion` - Create rotation quaternion

**Instance Methods:**
- `norm() -> float` - Return |q| = √(w² + i² + j² + k²)
- `conjugate() -> Quaternion` - Return q̄ = w - xi - yj - zk
- `inverse() -> Quaternion` - Return q⁻¹ = q̄/|q|²
- `normalize() -> Quaternion` - Return unit quaternion q/|q|
- `rotate_point(point: Tuple[float, float, float]) -> Tuple[float, float, float]` - Rotate 3D point
- `slerp(other: Quaternion, t: float) -> Quaternion` - Spherical linear interpolation
- `to_matrix() -> List[List[float]]` - Convert to 3×3 rotation matrix
- `to_euler() -> Tuple[float, float, float]` - Convert to Euler angles

**Operators:**
- `+`, `-`, `*`, `/` - Quaternion arithmetic (multiplication is non-commutative)
- `==`, `!=` - Equality comparison

---

### `tinyaleph.core.hypercomplex`

#### `CayleyDicksonAlgebra`

Generic hypercomplex algebra construction.

```python
class CayleyDicksonAlgebra:
    """Cayley-Dickson construction for hypercomplex algebras."""
    
    def __init__(self, dim: int):
        """
        Create a Cayley-Dickson algebra of given dimension.
        
        Args:
            dim: Dimension (must be power of 2: 2, 4, 8, 16, 32, ...)
        """
```

**Methods:**
- `create(components: List[float]) -> CayleyDicksonElement` - Create element
- `zero() -> CayleyDicksonElement` - Return zero element
- `one() -> CayleyDicksonElement` - Return identity element
- `basis(i: int) -> CayleyDicksonElement` - Return i-th basis element

#### `CayleyDicksonElement`

Element of a Cayley-Dickson algebra.

**Properties:**
- `components: List[float]` - Component values
- `dim: int` - Dimension of algebra

**Methods:**
- `norm() -> float` - Euclidean norm
- `conjugate() -> CayleyDicksonElement` - Cayley-Dickson conjugate
- `inverse() -> CayleyDicksonElement` - Multiplicative inverse

#### `Octonion`

8-dimensional octonions (non-associative).

```python
class Octonion(CayleyDicksonElement):
    """8-dimensional octonions."""
    
    def __init__(self, components: List[float]):
        """
        Create an octonion.
        
        Args:
            components: 8 real components [e0, e1, ..., e7]
        """
```

#### `Sedenion`

16-dimensional sedenions (contains zero divisors).

```python
class Sedenion(CayleyDicksonElement):
    """16-dimensional sedenions."""
    
    def __init__(self, components: List[float]):
        """
        Create a sedenion.
        
        Args:
            components: 16 real components [e0, e1, ..., e15]
        """
```

**Pre-built Algebras:**
- `COMPLEX_ALGEBRA` - 2D complex numbers
- `QUATERNION_ALGEBRA` - 4D quaternions
- `OCTONION_ALGEBRA` - 8D octonions
- `SEDENION_ALGEBRA` - 16D sedenions

---

### `tinyaleph.core.primes`

Prime number utilities.

#### Functions

```python
def is_prime(n: int) -> bool:
    """
    Check if n is prime.
    
    Args:
        n: Integer to check
        
    Returns:
        True if n is prime, False otherwise
    """

def nth_prime(n: int) -> int:
    """
    Return the n-th prime number (1-indexed).
    
    Args:
        n: Index (1 for first prime, 2 for second, etc.)
        
    Returns:
        The n-th prime number
        
    Examples:
        nth_prime(1) → 2
        nth_prime(10) → 29
    """

def prime_sieve(limit: int) -> List[int]:
    """
    Return all primes up to limit using Sieve of Eratosthenes.
    
    Args:
        limit: Upper bound (inclusive)
        
    Returns:
        List of primes ≤ limit
    """

def first_n_primes(n: int) -> List[int]:
    """
    Return the first n prime numbers.
    
    Args:
        n: Number of primes to return
        
    Returns:
        List of first n primes
    """

def factorize(n: int) -> Dict[int, int]:
    """
    Return prime factorization.
    
    Args:
        n: Integer to factorize
        
    Returns:
        Dictionary mapping prime → exponent
        
    Example:
        factorize(12) → {2: 2, 3: 1}  (12 = 2² × 3)
    """

def prime_index(p: int) -> int:
    """
    Return 1-based index of prime p.
    
    Args:
        p: A prime number
        
    Returns:
        Index (1 for 2, 2 for 3, 3 for 5, etc.)
        
    Raises:
        ValueError: If p is not prime
    """

def next_prime(n: int) -> int:
    """
    Return the smallest prime greater than n.
    
    Args:
        n: Starting integer
        
    Returns:
        Next prime after n
    """
```

---

### `tinyaleph.core.fano`

Fano plane for octonion multiplication.

#### `FanoPlane`

```python
class FanoPlane:
    """The Fano plane - 7 points, 7 lines, projective geometry."""
    
    POINTS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    
    LINES: Tuple[Tuple[int, int, int], ...] = (
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 6),
        (4, 5, 7),
        (5, 6, 1),
        (6, 7, 2),
        (7, 1, 3),
    )
```

**Methods:**
- `points_on_line(line_idx: int) -> Tuple[int, int, int]` - Points on given line
- `lines_through_point(point: int) -> List[int]` - Lines containing point
- `multiplication_sign(i: int, j: int) -> int` - Get sign for e_i × e_j

---

### `tinyaleph.core.constants`

Mathematical and physical constants.

```python
# Golden ratio
PHI: float = (1 + math.sqrt(5)) / 2  # ≈ 1.618034

# Coherence threshold
COHERENCE_THRESHOLD: float = 1.0 / PHI  # ≈ 0.618034

# Entropy threshold
ENTROPY_THRESHOLD: float = math.log(PHI)  # ≈ 0.481212

# Entropy change threshold for stability
DELTA_S: float = 0.01

# Lyapunov stability threshold
LAMBDA_STABILITY_THRESHOLD: float = 0.0

# Symbolic vacuum coupling
EPSILON_0: float = 0.0023

# Critical coupling for Kuramoto transition
CRITICAL_COUPLING: float = math.pi / 2  # ≈ 1.5708

# Fine structure constant
ALPHA: float = 1.0 / 137.035999
```

---

## Hilbert Module

### `tinyaleph.hilbert.state`

#### `PrimeState`

Quantum state in the Prime Hilbert Space H_P.

```python
class PrimeState:
    """
    A quantum state in the Prime Hilbert Space.
    
    |ψ⟩ = Σ_p α_p|p⟩ where p runs over primes
    """
    
    def __init__(self, primes: List[int], amplitudes: Optional[Dict[int, Complex]] = None):
        """
        Create a prime state.
        
        Args:
            primes: List of prime numbers in the basis
            amplitudes: Optional dictionary mapping prime → amplitude
        """
```

**Properties:**
- `primes: List[int]` - Primes in the basis
- `amplitudes: Dict[int, Complex]` - Amplitude for each prime

**Class Methods:**
- `basis(p: int) -> PrimeState` - Create |p⟩ eigenstate
- `single_prime(p: int) -> PrimeState` - Alias for basis()
- `uniform_superposition(primes: List[int]) -> PrimeState` - Equal superposition
- `first_n_superposition(n: int) -> PrimeState` - First n primes superposition
- `composite(n: int) -> PrimeState` - State weighted by prime factors

**Instance Methods:**
- `get(p: int) -> Complex` - Get amplitude for prime p
- `set(p: int, amplitude: Complex) -> None` - Set amplitude for prime p
- `norm() -> float` - Calculate √(Σ|α_p|²)
- `normalize() -> PrimeState` - Return normalized state
- `probabilities() -> Dict[int, float]` - Get |α_p|² for each prime
- `entropy() -> float` - Shannon entropy H = -Σ p log p
- `coherence() -> float` - Coherence measure (inverse of entropy)
- `inner_product(other: PrimeState) -> Complex` - Calculate ⟨self|other⟩
- `measure() -> Tuple[int, float]` - Collapse to eigenstate, return (prime, probability)

**Operators:**
- `+` - Add states (creates superposition)
- `*` (scalar) - Scalar multiplication
- `@` (operator) - Apply operator

---

### `tinyaleph.hilbert.operators`

#### `PrimeOperator`

Base class for operators on Prime Hilbert Space.

```python
class PrimeOperator:
    """Abstract base class for operators on H_P."""
    
    def apply(self, state: PrimeState) -> PrimeState:
        """Apply operator to state."""
        raise NotImplementedError
    
    def is_unitary(self) -> bool:
        """Check if operator is unitary."""
        raise NotImplementedError
```

**Operators:**
- `@` - Operator composition

#### `PhaseShiftOperator`

```python
class PhaseShiftOperator(PrimeOperator):
    """
    Apply phase e^(iθ) to specified primes.
    
    U_θ|p⟩ = e^(iθ)|p⟩ for p in target primes
    """
    
    def __init__(self, primes: List[int], phase: float):
        """
        Create phase shift operator.
        
        Args:
            primes: Primes to shift
            phase: Phase angle in radians
        """
```

#### `HadamardOperator`

```python
class HadamardOperator(PrimeOperator):
    """
    Hadamard-like operator creating superposition.
    
    H|p⟩ = (1/√2)(|p⟩ + |q⟩) for pairs (p, q)
    """
    
    def __init__(self, primes: List[int]):
        """
        Create Hadamard operator.
        
        Args:
            primes: List of primes (pairs will be created)
        """
```

#### `ProjectionOperator`

```python
class ProjectionOperator(PrimeOperator):
    """
    Project onto subspace spanned by target primes.
    
    P|ψ⟩ = Σ_{p ∈ targets} |p⟩⟨p|ψ⟩
    """
    
    def __init__(self, target_primes: Union[int, List[int]]):
        """
        Create projection operator.
        
        Args:
            target_primes: Prime(s) to project onto
        """
```

#### `PrimeTranslationOperator`

```python
class PrimeTranslationOperator(PrimeOperator):
    """
    Shift each prime to next/previous prime.
    
    T|p_n⟩ = |p_{n+1}⟩ (next prime)
    """
    
    def __init__(self, direction: int = 1):
        """
        Create translation operator.
        
        Args:
            direction: +1 for next prime, -1 for previous
        """
```

#### `Observable`

```python
class Observable(PrimeOperator):
    """
    Observable with prime eigenvalues.
    
    Â = Σ_p a_p |p⟩⟨p|
    """
    
    def __init__(self, eigenvalues: Dict[int, float]):
        """
        Create observable.
        
        Args:
            eigenvalues: Mapping prime → eigenvalue
        """
    
    def expectation(self, state: PrimeState) -> float:
        """Calculate ⟨ψ|Â|ψ⟩."""
        
    def variance(self, state: PrimeState) -> float:
        """Calculate ⟨ψ|Â²|ψ⟩ - ⟨ψ|Â|ψ⟩²."""
```

---

## Physics Module

### `tinyaleph.physics.kuramoto`

#### `KuramotoOscillator`

```python
class KuramotoOscillator:
    """
    Single Kuramoto oscillator with phase dynamics.
    
    dθ/dt = ω + F(t)
    """
    
    def __init__(self, omega: float = 1.0, phase: float = 0.0):
        """
        Create oscillator.
        
        Args:
            omega: Natural frequency
            phase: Initial phase in radians
        """
```

**Properties:**
- `omega: float` - Natural frequency
- `phase: float` - Current phase

**Methods:**
- `step(force: float, dt: float) -> None` - Evolve by dt
- `frequency() -> float` - Instantaneous frequency

#### `KuramotoField`

```python
class KuramotoField:
    """
    Coupled oscillator field with mean-field coupling.
    
    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)
    """
    
    def __init__(self, num_oscillators: int, coupling: float = 1.0,
                 omega_spread: float = 0.5, random_seed: Optional[int] = None):
        """
        Create oscillator field.
        
        Args:
            num_oscillators: Number of oscillators N
            coupling: Coupling strength K
            omega_spread: Standard deviation of natural frequencies
            random_seed: For reproducibility
        """
```

**Properties:**
- `N: int` - Number of oscillators
- `K: float` - Coupling strength
- `phases: List[float]` - Current phases
- `omegas: List[float]` - Natural frequencies

**Methods:**
- `step(dt: float) -> None` - Evolve system by dt
- `order_parameter() -> float` - Calculate r = |Σ e^(iθ)|/N
- `mean_phase() -> float` - Mean phase ψ
- `reset() -> None` - Reset to initial random phases

---

### `tinyaleph.physics.stochastic`

#### `StochasticKuramotoOscillator`

```python
class StochasticKuramotoOscillator(KuramotoOscillator):
    """
    Kuramoto oscillator with additive noise.
    
    dθ/dt = ω + F(t) + σξ(t)
    """
    
    def __init__(self, omega: float = 1.0, phase: float = 0.0,
                 noise_strength: float = 0.1):
        """
        Create stochastic oscillator.
        
        Args:
            omega: Natural frequency
            phase: Initial phase
            noise_strength: σ in noise term
        """
```

#### `ThermalKuramotoField`

```python
class ThermalKuramotoField(KuramotoField):
    """
    Kuramoto field with thermal fluctuations.
    
    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i) + √(2T)ξ_i(t)
    """
    
    def __init__(self, num_oscillators: int, coupling: float = 1.0,
                 temperature: float = 0.1, omega_spread: float = 0.5):
        """
        Create thermal field.
        
        Args:
            num_oscillators: Number of oscillators
            coupling: Coupling strength K
            temperature: Temperature T
            omega_spread: Natural frequency spread
        """
```

#### `ColoredNoiseOscillator`

```python
class ColoredNoiseOscillator(KuramotoOscillator):
    """
    Oscillator with colored (correlated) noise.
    
    Uses Ornstein-Uhlenbeck process for noise.
    """
    
    def __init__(self, omega: float = 1.0, phase: float = 0.0,
                 noise_strength: float = 0.1, correlation_time: float = 1.0):
        """
        Create colored noise oscillator.
        
        Args:
            omega: Natural frequency
            phase: Initial phase
            noise_strength: Noise amplitude
            correlation_time: τ for OU process
        """
```

---

### `tinyaleph.physics.lyapunov`

#### `LyapunovExponent`

```python
class LyapunovExponent:
    """
    Track Lyapunov exponent for stability analysis.
    
    λ = lim_{t→∞} (1/t) ln|δx(t)/δx(0)|
    """
    
    def __init__(self, window_size: int = 100):
        """
        Create Lyapunov tracker.
        
        Args:
            window_size: Number of observations to use
        """
```

**Methods:**
- `observe(value: float) -> None` - Add observation
- `current_exponent() -> float` - Current λ estimate
- `classify() -> StabilityClass` - Get stability classification
- `reset() -> None` - Clear observations

#### `StabilityClass`

```python
class StabilityClass(Enum):
    """Classification of dynamical stability."""
    
    COLLAPSED = "collapsed"      # λ < -0.1
    STABLE = "stable"            # -0.1 ≤ λ ≤ 0.1
    UNSTABLE = "unstable"        # λ > 0.1
```

---

### `tinyaleph.physics.entropy`

#### `EntropyTracker`

```python
class EntropyTracker:
    """
    Track Shannon entropy over time.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Create entropy tracker.
        
        Args:
            window_size: Observations for rate calculation
        """
```

**Methods:**
- `observe(distribution: Dict[Any, float]) -> float` - Record distribution, return entropy
- `current_entropy() -> float` - Latest entropy value
- `entropy_rate() -> float` - Rate of entropy change dH/dt
- `is_converging() -> bool` - True if entropy decreasing
- `classify() -> StabilityClass` - Based on entropy dynamics

---

### `tinyaleph.physics.collapse`

#### `CollapseEngine`

```python
class CollapseEngine:
    """
    Wave function collapse with configurable dynamics.
    """
    
    def __init__(self, collapse_rate: float = 0.1,
                 entropy_threshold: float = ENTROPY_THRESHOLD):
        """
        Create collapse engine.
        
        Args:
            collapse_rate: Base collapse probability
            entropy_threshold: Trigger collapse when H < threshold
        """
```

**Methods:**
- `should_collapse(state: PrimeState) -> bool` - Check collapse conditions
- `collapse(state: PrimeState) -> Tuple[int, float]` - Perform collapse
- `conditional_collapse(state: PrimeState) -> PrimeState` - Collapse if conditions met

---

## Observer Module

### `tinyaleph.observer.smf`

#### `SedenionMemoryField`

```python
class SedenionMemoryField:
    """
    16-dimensional sedenion-based memory field.
    
    Stores patterns as sedenions with temporal decay.
    """
    
    def __init__(self, decay_rate: float = 0.1, dimension: int = 16):
        """
        Create SMF.
        
        Args:
            decay_rate: Memory decay rate per timestep
            dimension: Component dimension (16 for sedenions)
        """
```

**Properties:**
- `memories: Dict[str, Sedenion]` - Stored patterns
- `timestamps: Dict[str, float]` - Storage times

**Methods:**
- `store(key: str, pattern: List[float]) -> None` - Store pattern
- `query(key: str, pattern: List[float]) -> float` - Query similarity
- `recall(key: str) -> Optional[Sedenion]` - Retrieve stored pattern
- `step(dt: float) -> None` - Apply temporal decay
- `prune(threshold: float = 0.01) -> int` - Remove weak memories

---

### `tinyaleph.observer.symbolic_smf`

#### `SymbolicSMF`

```python
class SymbolicSMF(SedenionMemoryField):
    """
    SMF with symbolic interpretation layer.
    
    Maps patterns to 128 symbolic indices.
    """
    
    NUM_SYMBOLS: int = 128
```

**Methods:**
- `store_symbolic(symbol_id: int, pattern: List[float]) -> None` - Store by symbol
- `query_symbolic(symbol_id: int) -> float` - Query symbol strength
- `active_symbols(threshold: float = 0.5) -> List[int]` - Currently active symbols
- `symbol_resonance(a: int, b: int) -> float` - Resonance between symbols

---

### `tinyaleph.observer.prsc`

#### `PRSCLayer`

```python
class PRSCLayer:
    """
    Prime Resonance Semantic Coherence layer.
    
    Binds semantic concepts to prime state representations.
    """
    
    def __init__(self, num_primes: int = 100):
        """
        Create PRSC layer.
        
        Args:
            num_primes: Size of prime vocabulary
        """
```

**Methods:**
- `bind_concept(name: str, primes: List[int]) -> SemanticConcept` - Bind concept
- `get_state() -> PrimeState` - Current coherent state
- `semantic_similarity(a: str, b: str) -> float` - Similarity between concepts
- `coherence(concept: str, state: PrimeState) -> float` - Concept-state coherence
- `activate(concept: str, strength: float = 1.0) -> None` - Activate concept
- `deactivate(concept: str) -> None` - Deactivate concept

#### `SemanticConcept`

```python
@dataclass
class SemanticConcept:
    """A semantic concept bound to primes."""
    
    name: str
    primes: List[int]
    state: PrimeState
    activation: float = 0.0
```

---

### `tinyaleph.observer.temporal`

#### `TemporalLayer`

```python
class TemporalLayer:
    """
    Emergent time from coherence events.
    
    Subjective duration weighted by coherence.
    """
    
    def __init__(self):
        """Create temporal layer."""
```

**Methods:**
- `register_event(coherence: float, timestamp: float) -> None` - Record event
- `subjective_duration(start: float, end: float) -> float` - Weighted duration
- `event_density(window: float) -> float` - Events per unit time
- `peak_coherence(start: float, end: float) -> float` - Maximum coherence in range

---

### `tinyaleph.observer.symbolic_temporal`

#### `SymbolicTemporalLayer`

```python
class SymbolicTemporalLayer(TemporalLayer):
    """
    Temporal layer with 64 hexagram classification.
    
    Maps moments to I Ching archetypes.
    """
```

**Methods:**
- `classify_moment(state: PrimeState) -> Dict[str, Any]` - Get hexagram for state
- `hexagram_sequence(n: int) -> List[int]` - Recent hexagram sequence
- `hexagram_transition(from_id: int, to_id: int) -> Dict[str, Any]` - Transition meaning

#### `HEXAGRAM_ARCHETYPES`

```python
HEXAGRAM_ARCHETYPES: Dict[int, Dict[str, Any]] = {
    0: {'name': 'creative', 'symbol': 'creation', 'tags': ['beginning', 'potential']},
    1: {'name': 'receptive', 'symbol': 'reception', 'tags': ['nurturing', 'acceptance']},
    # ... 64 total hexagrams
    63: {'name': 'completion', 'symbol': 'fulfillment', 'tags': ['ending', 'wholeness']},
}
```

---

### `tinyaleph.observer.agency`

#### `AgencyLayer`

```python
class AgencyLayer:
    """
    Agent decision-making layer.
    
    Handles goals, attention, action selection.
    """
    
    def __init__(self):
        """Create agency layer."""
```

**Methods:**
- `set_goal(name: str, priority: float) -> None` - Set goal with priority
- `remove_goal(name: str) -> None` - Remove goal
- `attend_to(stimulus: str, salience: float) -> None` - Direct attention
- `select_action() -> Optional[str]` - Choose action based on state
- `get_attention_focus() -> Optional[str]` - Current attention target
- `goal_satisfaction(name: str) -> float` - Satisfaction level for goal

---

### `tinyaleph.observer.boundary`

#### `BoundaryLayer`

```python
class BoundaryLayer:
    """
    Self/other boundary distinction.
    
    Implements objectivity gate R(ω) ≥ τ_R.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Create boundary layer.
        
        Args:
            threshold: τ_R for objectivity gate
        """
```

**Methods:**
- `is_self(pattern: List[float]) -> bool` - Check if pattern belongs to self
- `objectivity_gate(observation: Any) -> float` - R(ω) objectivity score
- `register_self_pattern(pattern: List[float]) -> None` - Add self-pattern
- `boundary_strength() -> float` - Current boundary coherence

---

### `tinyaleph.observer.safety`

#### `SafetyLayer`

```python
class SafetyLayer:
    """
    Safety constraints and violation detection.
    """
    
    def __init__(self):
        """Create safety layer."""
```

**Methods:**
- `add_constraint(name: str, check: Callable) -> None` - Add safety constraint
- `check_all() -> List[str]` - Return violated constraint names
- `is_safe() -> bool` - True if no violations
- `emergency_shutdown() -> None` - Trigger emergency stop

---

### `tinyaleph.observer.hqe`

#### `HolographicQuantumEncoder`

```python
class HolographicQuantumEncoder:
    """
    Holographic Quantum Encoding via DFT projection.
    
    Projects state onto holographic basis.
    """
    
    def __init__(self, num_modes: int = 64):
        """
        Create HQE.
        
        Args:
            num_modes: Number of holographic modes
        """
```

**Methods:**
- `encode(state: PrimeState) -> List[Complex]` - Encode to holographic representation
- `decode(hologram: List[Complex]) -> PrimeState` - Decode from hologram
- `interference(a: List[Complex], b: List[Complex]) -> List[Complex]` - Combine holograms

---

## Semantic Module

### `tinyaleph.semantic.types`

#### `Term`

Base class for semantic terms.

```python
class Term(ABC):
    """Abstract base for semantic terms."""
    
    @abstractmethod
    def is_well_formed(self) -> bool:
        """Check well-formedness."""
```

#### `NounTerm`

```python
class NounTerm(Term):
    """
    Noun term indexed by prime.
    
    N(p) represents an entity.
    """
    
    def __init__(self, p: int):
        """
        Create noun term.
        
        Args:
            p: Prime index (must be prime)
        """
```

**Properties:**
- `p: int` - Prime index

#### `AdjTerm`

```python
class AdjTerm(Term):
    """
    Adjective term indexed by prime.
    
    A(p) represents a property/operator.
    """
    
    def __init__(self, p: int):
        """
        Create adjective term.
        
        Args:
            p: Prime index (must be prime)
        """
```

#### `ChainTerm`

```python
class ChainTerm(Term):
    """
    Chain of adjectives applied to noun.
    
    A(p₁)...A(pₖ)N(q) is well-formed iff p_i < q for all i.
    """
    
    def __init__(self, adjectives: List[AdjTerm], noun: NounTerm):
        """
        Create chain term.
        
        Args:
            adjectives: List of adjective terms
            noun: The noun term
        """
```

**Methods:**
- `is_well_formed() -> bool` - Check ordering constraint

#### `FusionTerm`

```python
class FusionTerm(Term):
    """
    Triadic fusion of primes.
    
    FUSE(p, q, r) is well-formed iff p+q+r is prime.
    """
    
    def __init__(self, p: int, q: int, r: int):
        """
        Create fusion term.
        
        Args:
            p, q, r: Prime components
        """
```

**Methods:**
- `is_well_formed() -> bool` - Check sum is prime
- `fused_value() -> int` - Return p + q + r

#### `TypeChecker`

```python
class TypeChecker:
    """Check well-formedness of terms."""
    
    def check(self, term: Term) -> bool:
        """
        Check if term is well-formed.
        
        Args:
            term: Term to check
            
        Returns:
            True if well-formed
        """
    
    def infer_type(self, term: Term) -> Optional[str]:
        """Infer type of term."""
```

---

### `tinyaleph.semantic.reduction`

#### `PrimeOperator`

Base for prime-preserving operators.

```python
class PrimeOperator(ABC):
    """Abstract prime-preserving operator ⊕."""
    
    @abstractmethod
    def apply(self, p: int) -> int:
        """Apply operator to prime, return prime."""
```

#### `NextPrimeOperator`

```python
class NextPrimeOperator(PrimeOperator):
    """⊕: p → next_prime(p)"""
    
    def apply(self, p: int) -> int:
        """Return next prime after p."""
```

#### `ReductionSystem`

```python
class ReductionSystem:
    """
    Term reduction with strong normalization.
    
    Guarantees termination via decreasing size measure.
    """
    
    def __init__(self, max_steps: int = 1000):
        """
        Create reduction system.
        
        Args:
            max_steps: Maximum reduction steps
        """
```

**Methods:**
- `reduce(term: Term) -> Term` - Reduce to normal form
- `is_normal(term: Term) -> bool` - Check if in normal form
- `reduction_sequence(term: Term) -> List[Term]` - Full reduction trace

#### `ProofGenerator`

```python
class ProofGenerator:
    """Generate formal proofs of reductions."""
    
    def generate(self, term: Term, normal: Term) -> ProofTrace:
        """
        Generate proof from term to normal form.
        
        Returns:
            ProofTrace object with steps
        """
```

---

### `tinyaleph.semantic.lambda_calc`

#### `LambdaTranslator`

```python
class LambdaTranslator:
    """Translate semantic terms to lambda calculus."""
    
    def translate(self, term: Term) -> LambdaExpr:
        """
        τ translation: Term → LambdaExpr
        
        N(p) → ConstExpr(p)
        A(p) → λx.⊕(p, x)
        FUSE(p,q,r) → ConstExpr(p+q+r)
        """
```

#### `LambdaEvaluator`

```python
class LambdaEvaluator:
    """Evaluate lambda expressions via β-reduction."""
    
    def evaluate(self, expr: LambdaExpr, max_steps: int = 100) -> LambdaExpr:
        """Evaluate to normal form."""
```

#### `PRQS_LEXICON`

30 core semantic primes.

```python
PRQS_LEXICON: Dict[int, Dict[str, Any]] = {
    2: {'name': 'duality', 'category': 'structure'},
    3: {'name': 'structure', 'category': 'form'},
    5: {'name': 'change', 'category': 'process'},
    7: {'name': 'identity', 'category': 'being'},
    11: {'name': 'complexity', 'category': 'measure'},
    13: {'name': 'emergence', 'category': 'process'},
    # ... additional primes
}
```

---

### `tinyaleph.semantic.inference`

#### `SemanticInference`

```python
class SemanticInference:
    """Inference engine for semantic reasoning."""
```

**Methods:**
- `forward_chain(premises: List[int], rules: List) -> Set[int]` - Derive consequences
- `backward_chain(goal: int, rules: List) -> List[int]` - Find premises for goal
- `abduct(observation: int, kb: List) -> List[int]` - Explain observation
- `analogy(source: Tuple[int, int], target: int) -> Optional[int]` - Analogical mapping

---

### `tinyaleph.semantic.crt_homology`

#### `ResidueEncoder`

```python
class ResidueEncoder:
    """Chinese Remainder Theorem encoder."""
    
    def __init__(self, moduli: List[int]):
        """Create encoder with coprime moduli."""
```

**Methods:**
- `encode(value: int) -> List[int]` - Encode as residues
- `capacity() -> int` - Maximum encodable value

#### `CRTReconstructor`

```python
class CRTReconstructor:
    """Reconstruct value from residues."""
    
    def reconstruct(self, residues: List[int]) -> int:
        """Recover original value."""
```

#### `BirkhoffProjector`

```python
class BirkhoffProjector:
    """Project matrix onto Birkhoff polytope."""
    
    def project(self, matrix: List[List[float]]) -> List[List[float]]:
        """Project to doubly stochastic using Sinkhorn-Knopp."""
```

---

### `tinyaleph.semantic.topology`

#### `Knot`

```python
class Knot:
    """Mathematical knot with invariants."""
```

**Methods:**
- `crossing_number() -> int` - Number of crossings
- `writhe() -> int` - Sum of crossing signs
- `jones_polynomial() -> Dict[int, int]` - Jones polynomial coefficients

#### `FreeEnergyDynamics`

```python
class FreeEnergyDynamics:
    """Friston Free Energy Principle implementation."""
```

**Methods:**
- `free_energy(q: List[float], observations: List[float]) -> float` - Compute F
- `update_beliefs(observations: List[float]) -> List[float]` - Update q(s)

---

## ML Module

### `tinyaleph.ml.sparse_state`

#### `SparsePrimeState`

```python
class SparsePrimeState:
    """Sparse prime state with quaternionic amplitudes."""
```

**Class Methods:**
- `from_primes(primes: List[int]) -> SparsePrimeState` - Uniform amplitudes
- `single_prime(p: int) -> SparsePrimeState` - Single eigenstate
- `first_n_superposition(n: int) -> SparsePrimeState` - First n primes

**Instance Methods:**
- `entropy() -> float` - Shannon entropy
- `is_coherent() -> bool` - entropy < ln(φ)
- `prime_spectrum() -> Dict[int, float]` - Probability distribution
- `top_k_primes(k: int) -> List[int]` - Highest probability primes

---

### `tinyaleph.ml.resoformer`

#### `Tensor`

```python
class Tensor:
    """N-dimensional tensor with automatic broadcasting."""
```

**Methods:**
- `reshape(new_shape: Tuple) -> Tensor`
- `sum(axis: Optional[int] = None) -> Tensor`
- `mean(axis: Optional[int] = None) -> Tensor`
- `relu() -> Tensor`
- `sigmoid() -> Tensor`
- `softmax(axis: int = -1) -> Tensor`

#### `Dense`

```python
class Dense(Layer):
    """Fully connected layer."""
    
    def __init__(self, units: int, activation: Optional[str] = None):
        """Create dense layer."""
```

#### `QuaternionDense`

```python
class QuaternionDense(Layer):
    """Dense layer with quaternion weight structure."""
    
    def __init__(self, units: int):
        """Output dimension is units * 4."""
```

#### `ResonantAttentionLayer`

```python
class ResonantAttentionLayer(Layer):
    """Multi-head attention with resonance operators."""
    
    def __init__(self, num_heads: int, key_dim: int, dropout: float = 0.0):
        """Create attention layer."""
```

#### `CoherenceGatingLayer`

```python
class CoherenceGatingLayer(Layer):
    """Gate computation by coherence."""
    
    def __init__(self, threshold: float = 0.618):
        """Create gating layer."""
```

#### `EntropyCollapseLayer`

```python
class EntropyCollapseLayer(Layer):
    """VQ-style collapse to discrete attractors."""
    
    def __init__(self, num_attractors: int = 64):
        """Create collapse layer."""
```

#### `ResoFormerBlock`

```python
class ResoFormerBlock(Layer):
    """
    Complete transformer block with resonance.
    
    Architecture:
    1. LayerNorm → Attention → Residual
    2. ResonanceOperator
    3. LayerNorm → FFN → Residual
    4. CoherenceGating
    5. EntropyCollapse (optional)
    """
    
    def __init__(self, dim: int, num_heads: int, ffn_dim: int,
                 dropout_rate: float = 0.1, use_collapse: bool = True):
        """Create ResoFormer block."""
```

#### `ResoFormerConfig`

```python
@dataclass
class ResoFormerConfig:
    """Configuration for ResoFormer model."""
    
    vocab_size: int = 1000
    seq_len: int = 32
    dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
```

#### Model Builders

```python
def create_resoformer_model(vocab_size, seq_len, dim, num_layers, num_heads,
                            ffn_dim, dropout=0.1) -> ResoFormerModel:
    """Create language model."""

def create_resoformer_classifier(vocab_size, seq_len, dim, num_layers, num_heads,
                                  ffn_dim, num_classes, dropout=0.1) -> ResoFormerModel:
    """Create classifier."""

def create_resoformer_embedder(vocab_size, seq_len, dim, num_layers, num_heads,
                                ffn_dim, embedding_dim, dropout=0.1) -> ResoFormerModel:
    """Create embedder."""
```

---

## Network Module

### `tinyaleph.network.identity`

#### `PrimeResonanceIdentity`

```python
class PrimeResonanceIdentity:
    """
    Cryptographic identity based on prime resonance.
    
    Triple of Gaussian, Eisenstein, and Quaternionic primes.
    """
```

**Class Methods:**
- `generate() -> PrimeResonanceIdentity` - Generate new identity

**Instance Methods:**
- `prove(challenge: bytes) -> IdentityProof` - Create proof
- `verify(proof: IdentityProof, challenge: bytes) -> bool` - Verify proof

---

### `tinyaleph.network.entanglement`

#### `BellState`

```python
class BellState(Enum):
    """Four Bell states for entanglement."""
    
    PHI_PLUS = "Φ+"    # (|00⟩ + |11⟩)/√2
    PHI_MINUS = "Φ-"   # (|00⟩ - |11⟩)/√2
    PSI_PLUS = "Ψ+"    # (|01⟩ + |10⟩)/√2
    PSI_MINUS = "Ψ-"   # (|01⟩ - |10⟩)/√2
```

#### `EntangledPair`

```python
class EntangledPair:
    """Entangled pair of primes in Bell state."""
    
    def __init__(self, prime_a: int, prime_b: int, state: BellState):
        """Create entangled pair."""
```

#### `EntanglementNetwork`

```python
class EntanglementNetwork:
    """Network of entangled nodes."""
```

**Methods:**
- `register_node(name: str, identity: PrimeResonanceIdentity) -> None` - Add node
- `establish(a: str, b: str) -> EntangledPair` - Create entanglement
- `establish_long_distance(source: str, target: str) -> Optional[EntangledPair]` - Via swapping
- `teleport(source: str, target: str, state: PrimeState) -> bool` - Quantum teleportation

---

## Engine Module

### `tinyaleph.engine.aleph`

#### `AlephEngine` (Field-based)

```python
class AlephEngine:
    """
    Field-based computation engine.
    
    Pipeline: encode → excite → evolve → sample → decode
    """
    
    def __init__(self, num_oscillators: int = 50, coupling: float = 2.0,
                 coherence_threshold: float = 0.8):
        """Create field engine."""
```

**Methods:**
- `run(input_data: Any) -> RunResult` - Execute full pipeline
- `encode(data: Any) -> List[int]` - Encode to primes
- `excite(primes: List[int]) -> None` - Excite oscillators
- `evolve(max_steps: int = 1000) -> List[Frame]` - Evolve until coherent
- `decode(frames: List[Frame]) -> Any` - Decode result

#### `RunResult`

```python
@dataclass
class RunResult:
    """Result of AlephEngine computation."""
    
    frames: List[Frame]
    coherence: float
    decoded: Any
    steps: int
```

---

## Runtime Module

### `tinyaleph.runtime.engine`

#### `AlephConfig`

```python
@dataclass
class AlephConfig:
    """Configuration for AlephEngine runtime."""
    
    name: str = "aleph"
    max_history: int = 100
    coherence_threshold: float = 0.618
    default_primes: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
```

#### `ExecutionPhase`

```python
class ExecutionPhase(Enum):
    """Execution state machine phases."""
    
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    COLLAPSING = "collapsing"
    COMPLETE = "complete"
```

#### `EngineHooks`

```python
@dataclass
class EngineHooks:
    """Callback hooks for engine events."""
    
    on_state_change: Optional[Callable] = None
    on_phase_change: Optional[Callable] = None
    on_collapse: Optional[Callable] = None
    on_error: Optional[Callable] = None
```

#### `AlephEngine` (Runtime)

```python
class AlephEngine:
    """Unified runtime engine."""
    
    def __init__(self, config: Optional[AlephConfig] = None,
                 initial_state: Optional[PrimeState] = None,
                 hooks: Optional[EngineHooks] = None):
        """Create runtime engine."""
```

**Properties:**
- `state: PrimeState` - Current quantum state
- `phase: ExecutionPhase` - Current execution phase
- `coherence: float` - Current coherence level
- `history: List[PrimeState]` - State history

**Methods:**
- `set_state(state: PrimeState) -> None` - Set current state
- `apply_phase_shift(prime: int, phase: float) -> None` - Apply phase shift
- `evolve(dt: float) -> None` - Evolve state
- `collapse() -> int` - Collapse and return measured prime
- `transition_phase(phase: ExecutionPhase) -> None` - Change phase
- `checkpoint() -> Dict` - Create checkpoint
- `restore(checkpoint: Dict) -> None` - Restore from checkpoint
- `metrics() -> Dict[str, Any]` - Get engine metrics
- `store_fragment(key: str, pattern: List[float]) -> None` - Store resonant fragment
- `query_fragment(key: str, pattern: List[float]) -> float` - Query fragment
- `entangle_primes(a: int, b: int) -> None` - Create entanglement
- `process_batch(states: List[PrimeState]) -> List[Dict]` - Batch processing

**Async Methods:**
- `async_evolve(steps: int, dt: float) -> None` - Async evolution
- `async_process_batch(states: List[PrimeState]) -> List[Dict]` - Parallel batch

---

## Backends Module

### `tinyaleph.backends.cryptographic`

#### `PrimeStateKeyGenerator`

```python
class PrimeStateKeyGenerator:
    """
    Generate cryptographic keys from prime resonance.
    
    K = Σ_i θ_{p_i} mod 2π where θ_{p_i} = 2π log_{p_i}(n)
    """
    
    def __init__(self, primes: List[int]):
        """Create key generator with prime basis."""
```

**Methods:**
- `generate(n: int) -> Key` - Generate key from seed n
- `resonance_phase(p: int, n: int) -> float` - Calculate phase for prime

#### `EntropySensitiveEncryptor`

```python
class EntropySensitiveEncryptor:
    """Phase modulation encryption."""
    
    def __init__(self, key: Key):
        """Create encryptor with key."""
```

**Methods:**
- `encrypt(plaintext: bytes) -> bytes` - Encrypt data
- `decrypt(ciphertext: bytes) -> bytes` - Decrypt data

#### `HolographicKeyDistributor`

```python
class HolographicKeyDistributor:
    """Secret sharing via holographic interference."""
    
    def __init__(self, num_shares: int, threshold: int):
        """Create distributor with k-of-n scheme."""
```

**Methods:**
- `split(key: Key) -> List[Share]` - Split key into shares
- `reconstruct(shares: List[Share]) -> Key` - Reconstruct from threshold shares

---

## Distributed Module

### `tinyaleph.distributed.transport`

#### `LocalTransport`

```python
class LocalTransport:
    """In-process message transport."""
```

**Methods:**
- `register_handler(msg_type: MessageType, handler: Callable) -> None` - Register handler
- `send(destination: str, message: Message) -> Any` - Send message

#### `MessageType`

```python
class MessageType(Enum):
    """Message types for transport."""
    
    QUERY = "query"
    RESPONSE = "response"
    SYNC = "sync"
    ENTANGLE = "entangle"
```

---

## Resonance Module

### `tinyaleph.resonance.fragment`

#### `ResonantFragment`

```python
class ResonantFragment:
    """
    Holographic memory fragment.
    
    Encodes patterns as: A_p * e^(-S) * e^(ipθ)
    """
    
    def __init__(self, pattern: List[float], prime: int):
        """Create fragment for prime."""
```

**Properties:**
- `pattern: List[float]` - Encoded pattern
- `prime: int` - Associated prime
- `strength: float` - Current activation strength

**Methods:**
- `resonate(query: List[float]) -> float` - Compute resonance with query
- `decay(rate: float) -> None` - Apply temporal decay
- `interfere(other: ResonantFragment) -> ResonantFragment` - Combine fragments

---

## Constants Reference

### Core Constants

| Name | Value | Description |
|------|-------|-------------|
| `PHI` | 1.618034... | Golden ratio (1 + √5)/2 |
| `COHERENCE_THRESHOLD` | 0.618034... | 1/φ, minimum coherence |
| `ENTROPY_THRESHOLD` | 0.481212... | ln(φ), maximum stable entropy |
| `DELTA_S` | 0.01 | Entropy change threshold |
| `LAMBDA_STABILITY_THRESHOLD` | 0.0 | Lyapunov boundary |
| `EPSILON_0` | 0.0023 | Symbolic vacuum coupling |
| `CRITICAL_COUPLING` | 1.5708 | π/2, Kuramoto transition |
| `ALPHA` | 0.007297... | 1/137, fine structure |

### Stability Classes

| Class | λ Range | Description |
|-------|---------|-------------|
| `COLLAPSED` | λ < -0.1 | Converging to attractor |
| `STABLE` | -0.1 ≤ λ ≤ 0.1 | Bounded dynamics |
| `UNSTABLE` | λ > 0.1 | Diverging trajectory |

### Bell States

| State | Form | Description |
|-------|------|-------------|
| `PHI_PLUS` | (|00⟩ + |11⟩)/√2 | Maximally entangled, same |
| `PHI_MINUS` | (|00⟩ - |11⟩)/√2 | Maximally entangled, anti-correlated |
| `PSI_PLUS` | (|01⟩ + |10⟩)/√2 | Maximally entangled, flipped |
| `PSI_MINUS` | (|01⟩ - |10⟩)/√2 | Singlet state |

### Execution Phases

| Phase | Description |
|-------|-------------|
| `IDLE` | Ready, no computation |
| `INITIALIZING` | Setting up state |
| `PROCESSING` | Active computation |
| `COLLAPSING` | Measurement in progress |
| `COMPLETE` | Finished |

---

## Index

### Classes by Module

**core:**
Complex, Quaternion, Octonion, Sedenion, CayleyDicksonAlgebra, FanoPlane

**hilbert:**
PrimeState, PrimeOperator, PhaseShiftOperator, HadamardOperator, ProjectionOperator, Observable

**physics:**
KuramotoOscillator, KuramotoField, StochasticKuramotoOscillator, ThermalKuramotoField, LyapunovExponent, EntropyTracker, CollapseEngine

**observer:**
SedenionMemoryField, SymbolicSMF, PRSCLayer, TemporalLayer, SymbolicTemporalLayer, AgencyLayer, BoundaryLayer, SafetyLayer, HolographicQuantumEncoder

**semantic:**
NounTerm, AdjTerm, ChainTerm, FusionTerm, TypeChecker, ReductionSystem, LambdaTranslator, SemanticInference, ResidueEncoder, CRTReconstructor, Knot, FreeEnergyDynamics

**ml:**
SparsePrimeState, Tensor, Dense, QuaternionDense, ResonantAttentionLayer, CoherenceGatingLayer, EntropyCollapseLayer, ResoFormerBlock, ResoFormerConfig

**network:**
PrimeResonanceIdentity, EntangledPair, EntanglementNetwork, BellState

**engine:**
AlephEngine (field-based), RunResult

**runtime:**
AlephEngine, AlephConfig, ExecutionPhase, EngineHooks

**backends:**
PrimeStateKeyGenerator, EntropySensitiveEncryptor, HolographicKeyDistributor

**distributed:**
LocalTransport, MessageType

**resonance:**
ResonantFragment

---

*TinyAleph v0.1.0 - Reference Guide*