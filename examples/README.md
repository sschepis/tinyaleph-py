# TinyAleph Examples

This directory contains 20 comprehensive examples demonstrating all major features of the TinyAleph library.

## Examples Overview

| # | File | Topic | Concepts Covered |
|---|------|-------|-----------------|
| 01 | `01_hello_world.py` | Getting Started | PrimeState, superposition, entropy, collapse |
| 02 | `02_quantum_operators.py` | Operators | Phase shifts, Hadamard, measurement, unitarity |
| 03 | `03_quaternion_rotations.py` | Quaternions | Hamilton quaternions, 3D rotations, gimbal lock |
| 04 | `04_kuramoto_synchronization.py` | Physics | Coupled oscillators, phase locking, order parameter |
| 05 | `05_holographic_memory.py` | Resonance | ResonantFragment, interference, associative recall |
| 06 | `06_entanglement_networks.py` | Networks | PRI, Bell states, entanglement graphs |
| 07 | `07_sedenion_memory_field.py` | Observer | SMF, 16D holography, temporal decay |
| 08 | `08_prsc_semantic_binding.py` | Semantics | PRSC, concept binding, coherence metrics |
| 09 | `09_ml_attention_mechanisms.py` | ML Basics | SparsePrimeState, attention, ACT |
| 10 | `10_aleph_engine_runtime.py` | Runtime | AlephEngine, hooks, async, checkpoints |
| 11 | `11_symbolic_hexagrams.py` | Symbolic | SymbolicSMF, 64 hexagrams, I Ching classification |
| 12 | `12_advanced_physics.py` | Physics | Stochastic Kuramoto, Lyapunov, colored noise |
| 13 | `13_cryptographic_backend.py` | Cryptography | Prime key generation, encryption, distribution |
| 14 | `14_semantic_types.py` | Type System | NounTerm, AdjTerm, ChainTerm, FusionTerm, Sentences |
| 15 | `15_reduction_semantics.py` | Reduction | Prime operators, reduction system, proofs |
| 16 | `16_lambda_calculus.py` | Lambda | Translation, evaluation, PRQS lexicon |
| 17 | `17_crt_homology.py` | CRT/Homology | Residue encoding, Birkhoff, homological loss |
| 18 | `18_topology_physics.py` | Topology | Knot invariants, gauge symmetry, free energy |
| 19 | `19_semantic_inference.py` | Inference | CompoundBuilder, reasoning, entity extraction |
| 20 | `20_resoformer_ml.py` | ML Transformer | ResoFormer, attention, training pipeline |

## Running Examples

Each example is self-contained and can be run directly:

```bash
# Navigate to examples directory
cd /Users/sschepis/Development/TinyAleph/examples

# Run any example
python3 01_hello_world.py
python3 11_symbolic_hexagrams.py
python3 20_resoformer_ml.py
# ... etc
```

Or from the project root:

```bash
cd /Users/sschepis/Development/TinyAleph
python3 -m examples.01_hello_world
```

## Example Categories

### Core (01-05): Foundations

**01: Hello World - Basic Prime State Operations**
- Creating single prime eigenstates
- Uniform superpositions
- Born rule and probability distributions
- Entropy as coherence measure
- State collapse (measurement)

**02: Quantum Operators and Measurements**
- Phase shift operators
- Hadamard (superposition) operators
- Prime translation operators
- Unitarity verification
- Observable construction

**03: Quaternion Rotations**
- Quaternion arithmetic (i² = j² = k² = ijk = -1)
- Unit quaternions and normalization
- Rotation representation (axis-angle)
- Composition of rotations
- Avoiding gimbal lock

**04: Kuramoto Synchronization**
- Kuramoto model: dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
- Natural frequencies and coupling strength
- Order parameter r = |Σ e^(iθ)|/N
- Phase transitions
- Collective behavior emergence

**05: Holographic Memory (ResonantFragment)**
- Holographic encoding principles
- ResonantFragment storage
- Query by partial pattern
- Interference patterns
- Associative memory networks

### Networks & Observer (06-10): Higher-Level Structures

**06: Entanglement Networks**
- Bell state construction
- Entanglement entropy
- Network topology
- Correlation propagation
- Graph-theoretic analysis

**07: Sedenion Memory Field (SMF)**
- Sedenion algebra (16D hypercomplex)
- Multi-dimensional memory encoding
- Temporal decay processes
- Field dynamics
- 16-basis visualization

**08: PRSC Semantic Binding**
- Binding concepts to prime states
- Coherence metrics
- Semantic similarity
- Concept networks
- Reasoning through resonance

**09: ML Attention Mechanisms**
- SparsePrimeState (H_Q = H_P ⊗ ℍ)
- Resonant attention mechanism
- Multi-head attention
- Coherence-gated computation (ACT)
- Transformer blocks

**10: AlephEngine Runtime**
- Configuration management
- Hook-based callbacks
- Checkpoint and restore
- Async execution
- Full pipeline integration

### Symbolic & Advanced (11-15): Extended Capabilities

**11: Symbolic Hexagrams**
- SymbolicSMF with 128 symbols
- 64 I Ching hexagram classification
- Yin/Yang line encoding
- Hexagram archetype mapping
- Symbolic resonance patterns

**12: Advanced Physics**
- Stochastic Kuramoto with noise
- Lyapunov exponent analysis
- Colored noise oscillators
- Thermal Kuramoto dynamics
- Phase space trajectories

**13: Cryptographic Backend**
- Prime-based key generation
- Resonant encryption/decryption
- Key distribution protocols
- Signature generation
- Security analysis

**14: Semantic Types**
- NounTerm (prime indices)
- AdjTerm (feature application)
- ChainTerm (compositional chains)
- FusionTerm (semantic fusion)
- Complete sentences

**15: Reduction Semantics**
- Prime operator definitions
- Reduction system rules
- Proof generation
- Normalization (strong/weak)
- Term rewriting

### Semantic Theory (16-20): Deep Semantics & ML

**16: Lambda Calculus**
- Prime lambda terms
- β-reduction evaluation
- PRQS semantic lexicon
- Concept interpretation
- Type inference

**17: CRT-Homology**
- Chinese Remainder Theorem encoding
- Multi-modulus residue channels
- Birkhoff polytope projection
- Sinkhorn-Knopp algorithm
- Homological constraints (Betti numbers)

**18: Topology & Physics**
- Knot invariants (Jones polynomial, writhe)
- Physical constants (α, Λ_QCD, G_F)
- Gauge symmetry groups (U(1), SU(2), SU(3))
- Friston free energy dynamics
- Variational inference

**19: Semantic Inference**
- Wierzbicka semantic primes (~65 NSM primes)
- CompoundBuilder for composition
- Forward/backward chaining
- Subsumption and generalization
- Entity and relation extraction

**20: ResoFormer ML**
- Complete transformer architecture
- SparsePrimeState representations
- Multi-head prime attention
- Training pipeline utilities
- Sequence generation

## Mathematical Background

### Prime Hilbert Space
```
H_P = {|ψ⟩ = Σ αp|p⟩ : Σ|αp|² = 1, αp ∈ ℂ, p prime}
```

### Hypercomplex Algebras
```
Quaternions: ℍ = {a + bi + cj + dk : i² = j² = k² = ijk = -1}
Sedenions: 𝕊 = 16-dimensional, non-associative
```

### Key Constants
- **φ (PHI)**: Golden ratio ≈ 1.618034
- **ε₀ (EPSILON_0)**: Symbolic vacuum coupling ≈ 0.0023
- **r_c**: Critical coupling ≈ 1.5708
- **α**: Fine structure constant ≈ 1/137

### Core Equations

**Kuramoto Dynamics:**
```
dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)
```

**Order Parameter:**
```
r·e^(iψ) = (1/N) Σ_j e^(iθ_j)
```

**PRSC Coherence:**
```
C = |⟨ψ|φ⟩|² · S_rel
```

**Entropy:**
```
H = -Σ p_i log(p_i)
```

**Lyapunov Exponent:**
```
λ = lim_{t→∞} (1/t) ln|δx(t)/δx(0)|
```

**Chinese Remainder Theorem:**
```
x ≡ a_i (mod m_i) for coprime m_i ⟹ unique x (mod Π m_i)
```

**Free Energy (Friston):**
```
F = E_q[ln q(s) - ln p(o,s)] = D_KL(q||p) - ln p(o)
```

## Module Coverage

| Module | Examples |
|--------|----------|
| `core` | 01, 02, 03 |
| `hilbert` | 01, 02, 06 |
| `physics` | 04, 12 |
| `resonance` | 05 |
| `network` | 06 |
| `observer` | 07, 08, 11 |
| `ml` | 09, 20 |
| `engine` | 10 |
| `backends` | 13 |
| `semantic.types` | 14 |
| `semantic.reduction` | 15 |
| `semantic.lambda_calc` | 16 |
| `semantic.crt_homology` | 17 |
| `semantic.topology` | 18 |
| `semantic.inference` | 19 |

## Dependencies

Core examples work with pure Python. Advanced features require:

- `numpy` - Numerical operations (optional for core)
- `networkx` - Graph operations (optional)

Install all optional dependencies:

```bash
pip install tinyaleph[full]
```

## Further Reading

- [Design Document](../docs/design/PYTHON_PORT_DESIGN.md)
- [API Reference](../docs/api/)
- [Theory Papers](../docs/theory/)

## License

MIT License - see [LICENSE](../LICENSE)