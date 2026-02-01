# TinyAleph

**Prime-Resonant Quantum Computing Framework**

TinyAleph is a Python library unifying concepts from TinyAleph and ResoLang for prime-based quantum-inspired computing. It provides mathematical primitives and algorithms based on the Prime Hilbert Space formalism.

## Installation

```bash
# Basic installation (pure Python core)
pip install tinyaleph

# With numpy support (recommended for full functionality)
pip install tinyaleph[numpy]

# Full installation with all optional dependencies
pip install tinyaleph[full]
```

## Quick Start

```python
from tinyaleph import Complex, Quaternion, is_prime, nth_prime, PHI

# Complex numbers
z = Complex(3, 4)
print(f"Complex: {z}, magnitude: {z.magnitude()}")

# Quaternions for 3D rotations
q = Quaternion(1, 2, 3, 4)
print(f"Quaternion: {q}, norm: {q.norm()}")

# Prime utilities
print(f"Is 7 prime? {is_prime(7)}")
print(f"10th prime: {nth_prime(10)}")

# Golden ratio constant
print(f"PHI = {PHI}")
```

## Core Concepts

### Prime Hilbert Space

TinyAleph operates in a Hilbert space where prime numbers form an orthonormal basis:

```
H_P = {|ψ⟩ = Σ α_p |p⟩ : Σ|α_p|² = 1, p ∈ P}
```

States exist as superpositions over primes, with quantum-like operations defined on this space.

### Quaternionic Amplitudes

For richer geometric structure, amplitudes can be quaternionic:

```
H_Q = H_P ⊗ ℍ
```

This enables 3D rotations and geometric transformations on prime states.

### Entropy and Stability

The Lyapunov exponent λ characterizes state stability:
- **λ < -0.1**: Collapsed state (high certainty)
- **-0.1 ≤ λ ≤ 0.1**: Metastable (edge of chaos)
- **λ > 0.1**: Divergent (chaotic)

## Modules

### Core (Pure Python)

Works without numpy installation:

```python
from tinyaleph.core import Complex, Quaternion
from tinyaleph.core import is_prime, nth_prime, factorize, prime_sieve
from tinyaleph.core import PHI, DELTA_S, LAMBDA_STABILITY_THRESHOLD
```

### Hilbert Space (requires numpy)

```python
from tinyaleph.hilbert import PrimeState
from tinyaleph.hilbert import shift, fourier, collapse, hadamard

# Create superposition
state = PrimeState.superposition([2, 3, 5, 7])

# Apply operators
shifted = shift(1)(state)
measured, outcome = collapse()(state)
```

### Physics

```python
from tinyaleph.physics import KuramotoOscillator, CoupledOscillatorNetwork
from tinyaleph.physics import EntropyTracker, StabilityClass

# Kuramoto oscillator network
network = CoupledOscillatorNetwork(n_oscillators=10, coupling=0.5)
network.step(dt=0.1)
print(f"Order parameter: {network.order_parameter()}")

# Entropy tracking
tracker = EntropyTracker()
tracker.record(entropy_value)
print(f"Lyapunov: {tracker.lyapunov_exponent()}")
print(f"Stability: {tracker.stability()}")
```

### Resonance

```python
from tinyaleph.resonance import ResonantFragment

# Create holographic memory fragment
fragment = ResonantFragment(
    coefficients={2: Complex(0.5, 0.1), 3: Complex(0.3, -0.2)}
)

# Compute overlap (memory retrieval)
overlap = fragment.overlap(query_fragment)
```

### Network

```python
from tinyaleph.network import PrimeResonanceIdentity, EntangledNode
from tinyaleph.network import EntanglementNetwork, BellState

# Create network identity
identity = PrimeResonanceIdentity.generate()

# Entanglement network
network = EntanglementNetwork()
network.add_node("alice")
network.add_node("bob")
pair = network.establish_link("alice", "bob")
```

### Observer (SMF/PRSC)

```python
from tinyaleph.observer import SedenionMemoryField, PRSCLayer

# 16-dimensional holographic memory
smf = SedenionMemoryField()
smf.store("concept_key", fragment)
retrieved = smf.query(query_fragment)

# Semantic binding to primes
prsc = PRSCLayer()
prsc.bind("cat", [2, 3, 5])
prsc.bind("dog", [7, 11, 13])
composed = prsc.compose(["cat", "dog"])
```

### ML (Machine Learning)

```python
from tinyaleph.ml import SparsePrimeState, resonant_attention

# Sparse quaternionic prime states
state = SparsePrimeState.from_primes([2, 3, 5, 7])

# Resonant attention mechanism
attended = resonant_attention(query, keys, values)
```

### Runtime

```python
from tinyaleph.runtime import AlephEngine

# Create engine with hooks
engine = AlephEngine()
engine.register_hook("pre_step", my_callback)

# Run computation
result = engine.run(initial_state, steps=100)
```

## Mathematical Foundations

### Cayley-Dickson Construction

The library supports hypercomplex algebras via Cayley-Dickson construction:

| Dimension | Algebra | Properties |
|-----------|---------|------------|
| 2 | Complex ℂ | Commutative, Associative |
| 4 | Quaternion ℍ | Non-commutative, Associative |
| 8 | Octonion 𝕆 | Non-associative, Alternative |
| 16 | Sedenion 𝕊 | Has zero divisors |

### Kuramoto Model

Phase synchronization via coupled oscillators:

```
dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
```

Order parameter r ∈ [0, 1] measures synchronization:
- r = 0: Incoherent (random phases)
- r = 1: Fully synchronized

### Prime Resonance Identity

Network nodes are identified by triples of algebraic primes:
- Gaussian primes: Z[i]
- Eisenstein primes: Z[ω] where ω = e^(2πi/3)
- Quaternionic primes: Lipschitz integers

## Examples

See the `examples/` directory for:
- Basic usage and arithmetic
- Prime state manipulation
- Kuramoto synchronization
- Memory fragment storage and retrieval
- Distributed network simulation

## Theory

Based on the mathematical frameworks described in:
- TinyAleph: Prime Hilbert space and entropy-driven reasoning
- ResoLang: Resonant fragment protocols and network identity

Key innovations:
1. **Prime Basis**: Natural numbers factor uniquely into primes, providing orthogonal basis
2. **Golden Ratio Scaling**: φ-based attention weights for optimal information spread
3. **Coherence Gating**: Adaptive computation time based on entropy thresholds
4. **Quaternionic Geometry**: Rich geometric transformations on quantum states

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.