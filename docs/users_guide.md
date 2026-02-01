# TinyAleph User's Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start](#quick-start)
5. [Prime States and the Hilbert Space](#prime-states-and-the-hilbert-space)
6. [Quaternions and Hypercomplex Algebras](#quaternions-and-hypercomplex-algebras)
7. [Physics: Kuramoto Oscillators](#physics-kuramoto-oscillators)
8. [Observer Architecture](#observer-architecture)
9. [Semantic Types and Inference](#semantic-types-and-inference)
10. [Machine Learning: ResoFormer](#machine-learning-resoformer)
11. [The AlephEngine Runtime](#the-alephengine-runtime)
12. [Network and Distributed Computing](#network-and-distributed-computing)
13. [Cryptographic Backend](#cryptographic-backend)
14. [Examples Gallery](#examples-gallery)
15. [Best Practices](#best-practices)
16. [Appendix: Mathematical Foundations](#appendix-mathematical-foundations)

---

## Introduction

**TinyAleph** is a Python library that unifies concepts from prime number theory, quantum computing formalism, hypercomplex algebras, and semantic reasoning into a coherent computational framework. It provides:

- **Prime Hilbert Space (H_P)**: Quantum-like states where basis vectors are prime numbers
- **Quaternion/Sedenion Algebras**: 4D and 16D hypercomplex number systems for rotations and memory
- **Kuramoto Oscillators**: Coupled phase dynamics for synchronization and collective behavior
- **Observer Architecture**: Multi-layer cognitive model (agency, boundaries, safety)
- **Semantic Type System**: Formal types based on prime ordering
- **ResoFormer**: Transformer-style ML architecture with prime-based attention
- **Distributed Computing**: Entanglement networks and cryptographic backends

### Philosophy

TinyAleph is built on several key insights:

1. **Prime Numbers as Semantic Atoms**: Just as atoms form matter, primes form the basis of meaning. Each prime number p corresponds to a semantic primitive.

2. **Golden Ratio (φ ≈ 1.618)**: The golden ratio appears throughout the library as a fundamental threshold for coherence, entropy, and phase spacing.

3. **Resonance Over Statistics**: Computation proceeds through phase synchronization (resonance) rather than purely statistical pattern matching.

4. **Observer-Centric**: The framework explicitly models an observer with agency, boundaries, and temporal experience.

---

## Installation

### Basic Installation

```bash
pip install tinyaleph
```

### From Source

```bash
git clone https://github.com/sschepis/TinyAleph.git
cd TinyAleph
pip install -e .
```

### Optional Dependencies

```bash
# Full installation with all optional dependencies
pip install tinyaleph[full]

# Or install numpy separately for advanced features
pip install numpy
```

### Verify Installation

```python
import tinyaleph
print(tinyaleph.__version__)

# Core features (always available)
from tinyaleph import Complex, Quaternion, is_prime, PHI
print(f"Golden ratio: {PHI}")
print(f"Is 17 prime? {is_prime(17)}")

# Advanced features (require numpy)
from tinyaleph import PrimeState
state = PrimeState.basis(7)
print(f"Prime state |7⟩: {state}")
```

---

## Core Concepts

### Prime Hilbert Space (H_P)

The fundamental mathematical structure is the **Prime Hilbert Space**:

```
H_P = {|ψ⟩ = Σ_p α_p|p⟩ : Σ|α_p|² = 1, α_p ∈ ℂ, p prime}
```

- Each prime number p defines a basis state |p⟩
- States are complex superpositions of prime basis states
- Normalization ensures probabilities sum to 1
- Measurement collapses to a single prime with probability |α_p|²

### Extended State Space (H_Q)

For richer representations, states can have quaternionic amplitudes:

```
H_Q = H_P ⊗ ℍ = {|ψ⟩ = Σ_p q_p|p⟩ : q_p ∈ ℍ}
```

This gives each prime a 4-dimensional amplitude, enabling rotation-like operations.

### Key Constants

| Constant | Symbol | Value | Usage |
|----------|--------|-------|-------|
| Golden Ratio | φ (PHI) | 1.618034... | Coherence thresholds, phase spacing |
| Coherence Threshold | τ_c | 1/φ ≈ 0.618 | Minimum coherence for stability |
| Entropy Threshold | τ_H | ln(φ) ≈ 0.481 | Maximum entropy for stability |
| Stability Threshold | λ | 0.0 | Lyapunov exponent boundary |

### Entropy and Coherence

**Shannon Entropy** measures uncertainty:
```python
H = -Σ p_i log(p_i)
```

- Zero entropy: Perfect certainty (single prime eigenstate)
- Maximum entropy: Complete uncertainty (uniform superposition)

**Coherence** is the inverse relationship:
```python
coherence ≈ 1 / (1 + entropy)
```

High coherence (> 1/φ) indicates stable, focused states.

---

## Quick Start

### Hello World: Creating Prime States

```python
from tinyaleph.hilbert.state import PrimeState
from tinyaleph.core.complex import Complex

# Create a single prime eigenstate |7⟩
state = PrimeState.basis(7)
print(f"State: {state}")

# Create uniform superposition of first 5 primes
superposition = PrimeState.first_n_superposition(5)
print(f"Superposition: {superposition}")
print(f"Entropy: {superposition.entropy():.4f}")

# Measure the state (collapses to single prime)
prime, probability = superposition.measure()
print(f"Measured: |{prime}⟩ with probability {probability:.4f}")
```

### Quaternion Basics

```python
from tinyaleph.core.quaternion import Quaternion
import math

# Create quaternion: q = 1 + 2i + 3j + 4k
q = Quaternion(1, 2, 3, 4)
print(f"Quaternion: {q}")
print(f"Norm: {q.norm():.4f}")
print(f"Conjugate: {q.conjugate()}")

# Create rotation quaternion (90° around z-axis)
rotation = Quaternion.from_axis_angle(0, 0, 1, math.pi/2)
print(f"Rotation: {rotation}")

# Rotate a point
point = (1, 0, 0)
rotated = rotation.rotate_point(point)
print(f"Rotated (1,0,0): {rotated}")  # → (0, 1, 0)
```

### Kuramoto Oscillators

```python
from tinyaleph.physics.kuramoto import KuramotoOscillator, KuramotoField
import math

# Create oscillator field
field = KuramotoField(num_oscillators=10, coupling=2.0)
print(f"Initial order parameter: {field.order_parameter():.4f}")

# Evolve the system
for step in range(100):
    field.step(dt=0.1)

print(f"Final order parameter: {field.order_parameter():.4f}")
# High order parameter indicates synchronization
```

---

## Prime States and the Hilbert Space

### Creating States

```python
from tinyaleph.hilbert.state import PrimeState
from tinyaleph.core.complex import Complex

# Method 1: Single prime eigenstate
state_2 = PrimeState.basis(2)
state_7 = PrimeState.basis(7)
state_11 = PrimeState.single_prime(11)  # Alias for basis()

# Method 2: Uniform superposition
uniform = PrimeState.uniform_superposition([2, 3, 5, 7])
first_n = PrimeState.first_n_superposition(10)  # First 10 primes

# Method 3: Custom amplitudes
custom = PrimeState([2, 3, 5])
custom.set(2, Complex(0.6, 0))
custom.set(3, Complex(0.6, 0))
custom.set(5, Complex(0.5, 0))
custom = custom.normalize()  # Ensure Σ|α|² = 1

# Method 4: From composite number (weighted by prime factors)
composite = PrimeState.composite(12)  # 12 = 2² × 3
```

### State Properties

```python
# Probabilities
probs = state.probabilities()
for p, prob in probs.items():
    print(f"P(|{p}⟩) = {prob:.4f}")

# Entropy (uncertainty measure)
entropy = state.entropy()

# Coherence (focus measure)
coherence = state.coherence()

# Norm (should be 1 for normalized states)
norm = state.norm()

# Inner product ⟨ψ|φ⟩
overlap = state1.inner_product(state2)
```

### State Operations

```python
# Addition (creates superposition)
combined = state1 + state2
combined = combined.normalize()

# Scalar multiplication
scaled = state * 0.5

# Inner product
overlap = state1.inner_product(state2)

# Measurement (collapses state)
prime, probability = state.measure()
```

### Operators

```python
from tinyaleph.hilbert.operators import (
    PrimeOperator,
    PhaseShiftOperator,
    HadamardOperator,
    ProjectionOperator,
)

# Create operators
phase = PhaseShiftOperator(primes=[2, 3, 5], phase=math.pi/4)
hadamard = HadamardOperator(primes=[2, 3])
projection = ProjectionOperator(target_prime=7)

# Apply operator to state
new_state = phase.apply(state)

# Check unitarity
is_unitary = phase.is_unitary()

# Compose operators
composed = phase @ hadamard  # Apply hadamard first, then phase
```

---

## Quaternions and Hypercomplex Algebras

### Quaternion Arithmetic

```python
from tinyaleph.core.quaternion import Quaternion

# Creation
q1 = Quaternion(1, 2, 3, 4)  # w + xi + yj + zk
q2 = Quaternion.from_axis_angle(0, 0, 1, math.pi/2)  # From rotation

# Static constructors
zero = Quaternion.zero()
one = Quaternion.one()
unit_i = Quaternion.i()  # Pure quaternion i
unit_j = Quaternion.j()  # Pure quaternion j
unit_k = Quaternion.k()  # Pure quaternion k

# Arithmetic
q3 = q1 + q2
q4 = q1 - q2
q5 = q1 * q2   # Quaternion multiplication (non-commutative!)
q6 = q1 / q2   # q1 * q2.inverse()

# Properties
norm = q1.norm()
conjugate = q1.conjugate()  # w - xi - yj - zk
inverse = q1.inverse()      # conjugate / |q|²

# Normalize to unit quaternion
q1_unit = q1.normalize()
```

### 3D Rotations with Quaternions

```python
# Create rotation (axis-angle representation)
# 90° rotation around z-axis
rotation = Quaternion.from_axis_angle(0, 0, 1, math.pi/2)

# Rotate a 3D point
point = (1, 0, 0)
rotated = rotation.rotate_point(point)  # Returns (0, 1, 0)

# Compose rotations (apply rotation2, then rotation1)
rotation1 = Quaternion.from_axis_angle(1, 0, 0, math.pi/2)  # 90° around x
rotation2 = Quaternion.from_axis_angle(0, 1, 0, math.pi/2)  # 90° around y
combined = rotation1 * rotation2

# Interpolate between rotations (slerp)
t = 0.5  # Interpolation parameter
interpolated = rotation1.slerp(rotation2, t)
```

### Octonions (8D)

```python
from tinyaleph.core.hypercomplex import Octonion

# Create octonion from 8 components
o = Octonion([1, 2, 3, 4, 5, 6, 7, 8])

# Arithmetic (WARNING: non-associative!)
o1 = Octonion([1, 0, 0, 0, 0, 0, 0, 0])
o2 = Octonion([0, 1, 0, 0, 0, 0, 0, 0])
o3 = o1 * o2

# Properties
norm = o.norm()
conjugate = o.conjugate()
```

### Sedenions (16D)

```python
from tinyaleph.core.hypercomplex import Sedenion

# Create sedenion from 16 components
s = Sedenion([1] + [0]*15)  # e_0 = 1

# Sedenion multiplication follows Cayley-Dickson construction
# WARNING: Contains zero divisors!

# Access components
components = s.components
```

### Cayley-Dickson Construction

```python
from tinyaleph.core.hypercomplex import CayleyDicksonAlgebra

# Create custom hypercomplex algebra
algebra_32 = CayleyDicksonAlgebra(dim=32)

# Create element
elem = algebra_32.create([1] + [0]*31)

# Standard algebras are pre-built
from tinyaleph.core.hypercomplex import (
    COMPLEX_ALGEBRA,     # 2D
    QUATERNION_ALGEBRA,  # 4D
    OCTONION_ALGEBRA,    # 8D
    SEDENION_ALGEBRA,    # 16D
)
```

---

## Physics: Kuramoto Oscillators

### Single Oscillator

```python
from tinyaleph.physics.kuramoto import KuramotoOscillator

# Create oscillator with natural frequency ω
osc = KuramotoOscillator(omega=1.0, phase=0.0)

# Evolve with external force
dt = 0.01
external_force = 0.5
osc.step(force=external_force, dt=dt)

print(f"Phase: {osc.phase}")
print(f"Frequency: {osc.omega}")
```

### Coupled Oscillator Field

```python
from tinyaleph.physics.kuramoto import KuramotoField

# Create field of coupled oscillators
field = KuramotoField(
    num_oscillators=100,
    coupling=2.0,  # K in Kuramoto model
    omega_spread=0.5,  # Natural frequency spread
)

# Initial state
print(f"Initial r: {field.order_parameter():.4f}")

# Evolve the system
for step in range(1000):
    field.step(dt=0.1)
    
    if step % 100 == 0:
        r = field.order_parameter()
        print(f"Step {step}: r = {r:.4f}")

# Order parameter interpretation:
# r ≈ 0: Incoherent (random phases)
# r ≈ 1: Fully synchronized
# 0 < r < 1: Partial synchronization
```

### Stochastic Kuramoto

```python
from tinyaleph.physics.stochastic import (
    StochasticKuramotoOscillator,
    ThermalKuramotoField,
)

# Single oscillator with noise
stoch_osc = StochasticKuramotoOscillator(
    omega=1.0,
    noise_strength=0.1,  # Noise intensity
)

# Thermal field (temperature-dependent noise)
thermal_field = ThermalKuramotoField(
    num_oscillators=50,
    coupling=2.0,
    temperature=0.3,
)

# Evolution with thermal fluctuations
for step in range(500):
    thermal_field.step(dt=0.1)
```

### Lyapunov Stability Analysis

```python
from tinyaleph.physics.lyapunov import (
    LyapunovExponent,
    StabilityClass,
)

# Track Lyapunov exponent over time
lyap = LyapunovExponent()

for step in range(1000):
    # Get some observable trajectory
    value = field.order_parameter()
    lyap.observe(value)

# Get stability classification
lambda_exp = lyap.current_exponent()
stability = lyap.classify()

# Stability classes:
# StabilityClass.COLLAPSED: λ < -0.1 (converging)
# StabilityClass.STABLE: -0.1 ≤ λ ≤ 0.1 (bounded)
# StabilityClass.UNSTABLE: λ > 0.1 (diverging)
```

---

## Observer Architecture

The observer architecture models a conscious agent with multiple layers:

### Sedenion Memory Field (SMF)

```python
from tinyaleph.observer.smf import SedenionMemoryField

# Create 16D memory field
smf = SedenionMemoryField(
    decay_rate=0.1,  # Temporal decay
    dimension=16,
)

# Store memory patterns
smf.store("concept_tree", [0.5, 0.3, 0.2, ...])  # 16 components
smf.store("concept_leaf", [0.2, 0.6, 0.1, ...])

# Query memory (returns coherence score)
similarity = smf.query("concept_tree", [0.5, 0.3, 0.2, ...])

# Memory decays over time
smf.step(dt=0.1)
```

### PRSC: Prime Resonance Semantic Coherence

```python
from tinyaleph.observer.prsc import (
    PRSCLayer,
    SemanticConcept,
)

# Create PRSC layer
prsc = PRSCLayer(num_primes=100)

# Bind concept to prime state
concept = prsc.bind_concept("democracy", primes=[2, 11, 17, 23])

# Query semantic similarity
similarity = prsc.semantic_similarity("democracy", "freedom")

# Get coherence between concepts
coherence = prsc.coherence("democracy", prsc.get_state())
```

### Temporal Layer

```python
from tinyaleph.observer.temporal import TemporalLayer

# Temporal layer tracks emergent time from coherence events
temporal = TemporalLayer()

# Register coherent moment (high coherence event)
temporal.register_event(coherence=0.85, timestamp=1.0)
temporal.register_event(coherence=0.92, timestamp=2.3)

# Get subjective duration
duration = temporal.subjective_duration(start=0, end=3)

# Subjective time is weighted by coherence:
# High coherence moments feel longer
```

### Agency Layer

```python
from tinyaleph.observer.agency import AgencyLayer

# Agency layer handles goals, attention, action selection
agency = AgencyLayer()

# Set goals
agency.set_goal("explore", priority=0.8)
agency.set_goal("conserve_energy", priority=0.3)

# Attention focus
agency.attend_to("stimulus_1", salience=0.9)

# Select action based on goals and state
action = agency.select_action()
```

### Boundary Layer

```python
from tinyaleph.observer.boundary import BoundaryLayer

# Boundary layer distinguishes self from other
boundary = BoundaryLayer()

# Test if pattern belongs to self
is_self = boundary.is_self(pattern)

# Objectivity gate: R(ω) ≥ τ_R for external validity
objectivity = boundary.objectivity_gate(observation)
```

### Symbolic Temporal (64 Hexagrams)

```python
from tinyaleph.observer.symbolic_temporal import (
    SymbolicTemporalLayer,
    HEXAGRAM_ARCHETYPES,
)

# 64 hexagram archetypes (I Ching inspired)
temporal = SymbolicTemporalLayer()

# Get hexagram for current state
hexagram = temporal.classify_moment(state)
print(f"Hexagram: {hexagram['name']}")
print(f"Tags: {hexagram['tags']}")

# Example hexagrams:
# 0: 'creative' - beginning, potential
# 1: 'receptive' - nurturing, acceptance
# 63: 'completion' - ending, fulfillment
```

---

## Semantic Types and Inference

### Type System

```python
from tinyaleph.semantic.types import (
    NounTerm,
    AdjTerm,
    ChainTerm,
    FusionTerm,
    SentenceTerm,
    TypeChecker,
)

# NounTerm: Subject indexed by prime
noun = NounTerm(7)  # Entity indexed by prime 7

# AdjTerm: Operator with prime ordering constraint
adj = AdjTerm(3)  # Adjective indexed by prime 3

# ChainTerm: Composition A(p₁)...A(pₖ)N(q)
# Requires p_i < q for all adjectives
chain = ChainTerm(adjectives=[AdjTerm(2), AdjTerm(3)], noun=NounTerm(7))

# FusionTerm: FUSE(p, q, r) where p+q+r is prime
fusion = FusionTerm(2, 3, 6)  # 2+3+6 = 11 (prime) ✓

# SentenceTerm: Complete sentence
sentence = SentenceTerm(subject=noun, predicate=adj)

# Type checking
checker = TypeChecker()
is_valid = checker.check(chain)
```

### Reduction Semantics

```python
from tinyaleph.semantic.reduction import (
    ReductionSystem,
    NextPrimeOperator,
    ProofGenerator,
)

# Prime-preserving operators
# ⊕: Takes prime to next prime
op = NextPrimeOperator()
print(op.apply(7))  # → 11

# Reduction system with strong normalization
system = ReductionSystem()

# Reduce term to normal form
normal_form = system.reduce(term)

# Generate proof trace
proof = ProofGenerator()
trace = proof.generate(term, normal_form)
print(trace.to_latex())  # LaTeX output
```

### Lambda Calculus Translation

```python
from tinyaleph.semantic.lambda_calc import (
    LambdaTranslator,
    LambdaEvaluator,
    ConceptInterpreter,
    PRQS_LEXICON,
)

# Translate semantic terms to lambda calculus
translator = LambdaTranslator()
lambda_expr = translator.translate(chain_term)

# Evaluate lambda expression
evaluator = LambdaEvaluator()
result = evaluator.evaluate(lambda_expr)

# PRQS Lexicon: 30 core semantic primes
for prime, concept in PRQS_LEXICON.items():
    print(f"Prime {prime}: {concept['name']} ({concept['category']})")

# Example mappings:
# 2: duality, 3: structure, 5: change, 7: identity
# 11: complexity, 13: emergence, 17: boundary, ...
```

### Semantic Inference

```python
from tinyaleph.semantic.inference import (
    SemanticInference,
    CompoundBuilder,
    EntityExtractor,
)

# Build compound concepts from primes
builder = CompoundBuilder()
compound = builder.build_and(prime_a=2, prime_b=3)  # A ∧ B = 2*3 = 6

# Semantic inference engine
inference = SemanticInference()

# Forward chaining
consequences = inference.forward_chain(premises=[2, 3], rules=rules)

# Abductive reasoning (explain observation)
explanation = inference.abduct(observation=7, knowledge_base=kb)

# Analogical reasoning
analogy = inference.analogy(source=(2, 3), target=5)

# Entity extraction from text
extractor = EntityExtractor()
entities = extractor.extract("The cat sat on the mat")
```

### CRT-Homology

```python
from tinyaleph.semantic.crt_homology import (
    ResidueEncoder,
    CRTReconstructor,
    BirkhoffProjector,
    HomologyLoss,
)

# Chinese Remainder Theorem encoding
encoder = ResidueEncoder(moduli=[3, 5, 7])
residues = encoder.encode(value=23)  # [23 mod 3, 23 mod 5, 23 mod 7]

# Reconstruct original value
reconstructor = CRTReconstructor(moduli=[3, 5, 7])
original = reconstructor.reconstruct(residues)

# Birkhoff projection to doubly stochastic matrices
projector = BirkhoffProjector()
doubly_stochastic = projector.project(matrix)

# Homological loss for topological regularization
loss = HomologyLoss()
betti_loss = loss.compute(representation)
```

---

## Machine Learning: ResoFormer

### SparsePrimeState

```python
from tinyaleph.ml.sparse_state import (
    SparsePrimeState,
    coherent_superposition,
    golden_superposition,
)

# Create sparse state with quaternionic amplitudes
state = SparsePrimeState.from_primes([2, 3, 5, 7])

# Single prime eigenstate
single = SparsePrimeState.single_prime(7)

# First n primes
first_5 = SparsePrimeState.first_n_superposition(5)

# Coherent superposition with custom phases
phases = [0, math.pi/4, math.pi/2, 3*math.pi/4]
coherent = coherent_superposition([2, 3, 5, 7], phases)

# Golden ratio phase spacing
golden = golden_superposition(5)  # 2π/φ² ≈ 137.5° spacing

# Properties
entropy = state.entropy()
is_coherent = state.is_coherent()  # entropy < ln(φ)
spectrum = state.prime_spectrum()  # Probability distribution
top_k = state.top_k_primes(3)  # Highest probability primes
```

### Tensor Operations

```python
from tinyaleph.ml.resoformer import (
    Tensor,
    zeros,
    ones,
    randn,
    glorot_uniform,
    quaternion_normalize,
)

# Create tensors
z = zeros((3, 4))
o = ones((2, 3))
r = randn((4, 4))
g = glorot_uniform((32, 64))  # Glorot initialization

# Tensor operations
t = Tensor([[1, 2, 3], [4, 5, 6]])
reshaped = t.reshape((3, 2))

# Mathematical operations
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([0.5, 0.5, 0.5, 0.5])
c = a + b
d = a * b
s = a.sum()
m = a.mean()

# Activation functions
relu_out = a.relu()
sigmoid_out = a.sigmoid()
tanh_out = a.tanh()

# Quaternion normalization
q = Tensor([1.0, 2.0, 3.0, 4.0])
q_norm = quaternion_normalize(q)  # Unit quaternion
```

### Layers

```python
from tinyaleph.ml.resoformer import (
    Dense,
    QuaternionDense,
    LayerNorm,
    Dropout,
    SparsePrimeEmbedding,
    ResonantAttentionLayer,
    CoherenceGatingLayer,
    EntropyCollapseLayer,
    ResonanceOperator,
    ResoFormerBlock,
)

# Standard dense layer
dense = Dense(units=16, activation="relu")
y = dense(x)

# Quaternion dense layer
q_dense = QuaternionDense(units=8)  # Output: units * 4
y_q = q_dense(x)

# Sparse prime embedding
embed = SparsePrimeEmbedding(
    num_primes=1000,  # Vocabulary
    k=8,              # Top-k active primes
    embedding_dim=32,
)
result = embed(x)  # Returns dict with indices, amplitudes, logits

# Resonant attention (multi-head)
attn = ResonantAttentionLayer(num_heads=4, key_dim=16)
attended = attn(x)

# Coherence gating
gating = CoherenceGatingLayer(threshold=0.7)
result = gating(x)  # Returns output, coherence, gate

# Entropy collapse (VQ-style quantization)
collapse = EntropyCollapseLayer(num_attractors=64)
result = collapse(x)  # Returns output, probs, entropy, indices

# Resonance operator
res_op = ResonanceOperator()
resonant = res_op(x)  # R̂(n)|p⟩ = e^(2πi log(n))|p⟩
```

### ResoFormer Model

```python
from tinyaleph.ml.resoformer import (
    ResoFormerConfig,
    create_resoformer_model,
    create_resoformer_classifier,
    create_resoformer_embedder,
)

# Configuration
config = ResoFormerConfig(
    vocab_size=1000,
    seq_len=32,
    dim=64,
    num_layers=4,
    num_heads=4,
    ffn_dim=256,
    dropout=0.1,
)

# Language model
lm = create_resoformer_model(
    vocab_size=1000,
    seq_len=32,
    dim=64,
    num_layers=2,
    num_heads=4,
    ffn_dim=128,
)
logits = lm(tokens, training=False)

# Classifier
clf = create_resoformer_classifier(
    vocab_size=1000,
    seq_len=32,
    dim=64,
    num_layers=2,
    num_heads=4,
    ffn_dim=128,
    num_classes=10,
)
class_probs = clf(tokens)

# Embedder
emb = create_resoformer_embedder(
    vocab_size=1000,
    seq_len=32,
    dim=64,
    num_layers=2,
    num_heads=4,
    ffn_dim=128,
    embedding_dim=32,
)
embeddings = emb(tokens)
```

---

## The AlephEngine Runtime

### Basic Usage

```python
from tinyaleph.runtime.engine import (
    AlephEngine,
    AlephConfig,
    EngineHooks,
    ExecutionPhase,
)
from tinyaleph.hilbert.state import PrimeState

# Create engine with default configuration
engine = AlephEngine()

# Create engine with custom configuration
config = AlephConfig(
    name="quantum_processor",
    max_history=100,
    coherence_threshold=0.7,
    default_primes=[2, 3, 5, 7, 11],
)
engine = AlephEngine(config=config)

# Initialize state
engine.set_state(PrimeState.uniform_superposition([2, 3, 5]))

# Get engine properties
print(f"Coherence: {engine.coherence}")
print(f"Phase: {engine.phase}")
print(f"State: {engine.state}")
```

### State Evolution

```python
# Apply phase shift
engine.apply_phase_shift(prime=2, phase=math.pi/4)

# Evolve state
for step in range(100):
    engine.evolve(dt=0.1)
    
    # Check coherence
    if engine.coherence < config.coherence_threshold:
        result = engine.collapse()
        print(f"Collapsed to |{result}⟩")
        break
```

### Execution Phases

```python
# Phase transitions
engine.transition_phase(ExecutionPhase.INITIALIZING)
engine.transition_phase(ExecutionPhase.PROCESSING)
engine.transition_phase(ExecutionPhase.COLLAPSING)
engine.transition_phase(ExecutionPhase.IDLE)

# Phases:
# IDLE         - Ready, no active computation
# INITIALIZING - Setting up initial state
# PROCESSING   - Active computation
# COLLAPSING   - Measurement in progress
# COMPLETE     - Computation finished
```

### Hooks and Callbacks

```python
def on_state_change(old_state, new_state):
    print(f"State changed: {len(old_state.amplitudes)} → {len(new_state.amplitudes)}")

def on_phase_change(old_phase, new_phase):
    print(f"Phase: {old_phase.name} → {new_phase.name}")

def on_collapse(state, result):
    print(f"Collapsed to |{result}⟩")

hooks = EngineHooks(
    on_state_change=on_state_change,
    on_phase_change=on_phase_change,
    on_collapse=on_collapse,
)

engine = AlephEngine(hooks=hooks)
```

### Checkpoint and Restore

```python
# Create checkpoint
checkpoint = engine.checkpoint()
print(f"Checkpoint at step {checkpoint['step']}")

# Modify state
engine.set_state(PrimeState.single_prime(7))
engine.evolve(dt=0.5)

# Restore from checkpoint
engine.restore(checkpoint)
print(f"Restored to step {checkpoint['step']}")
```

### Resonant Memory

```python
# Store resonant fragments
engine.store_fragment("query", [0.6, 0.3, 0.1])
engine.store_fragment("key", [0.5, 0.4, 0.1])
engine.store_fragment("value", [0.2, 0.3, 0.5])

# Query similarity
query = [0.5, 0.3, 0.2]
similarity = engine.query_fragment("query", query)

# Find best match
best = engine.find_best_fragment(query)
```

### Entanglement Network

```python
# Create entangled pairs
engine.entangle_primes(2, 3)
engine.entangle_primes(5, 7)
engine.entangle_primes(2, 5)

# Get network
network = engine.get_entanglement_network()
print(f"Edges: {list(network.edges())}")

# Check entanglement
engine.are_entangled(2, 3)  # True
engine.are_entangled(2, 7)  # False

# Find path
path = engine.entanglement_path(2, 7)  # [2, 5, 7]
```

### Async Execution

```python
import asyncio

async def async_demo():
    engine = AlephEngine()
    
    # Async evolution
    await engine.async_evolve(steps=100, dt=0.1)
    
    # Parallel batch processing
    states = [
        PrimeState.uniform_superposition([2, 3, 5]),
        PrimeState.uniform_superposition([7, 11, 13]),
    ]
    results = await engine.async_process_batch(states)

asyncio.run(async_demo())
```

### Field-Based Engine

```python
from tinyaleph.engine.aleph import AlephEngine as FieldEngine

# Create field-based engine
engine = FieldEngine(
    num_oscillators=50,
    coupling=2.0,
    coherence_threshold=0.8,
)

# Run computation
# Pipeline: encode → excite → evolve → sample → decode
result = engine.run(input_data)

print(f"Frames collected: {result.frames}")
print(f"Final coherence: {result.coherence}")
print(f"Answer: {result.decoded}")
```

---

## Network and Distributed Computing

### Prime Resonance Identity (PRI)

```python
from tinyaleph.network.identity import (
    PrimeResonanceIdentity,
    IdentityProof,
)

# Generate identity
pri = PrimeResonanceIdentity.generate()

# Identity components:
# - Gaussian prime (ℤ[i])
# - Eisenstein prime (ℤ[ω])
# - Quaternionic prime (ℤ[i,j,k])

# Create proof of identity
proof = pri.prove(challenge=b"random_challenge")

# Verify proof
is_valid = PrimeResonanceIdentity.verify(proof, challenge=b"random_challenge")
```

### Entanglement Networks

```python
from tinyaleph.network.entanglement import (
    EntangledPair,
    EntanglementNetwork,
    BellState,
)

# Create Bell states
phi_plus = BellState.PHI_PLUS    # (|00⟩ + |11⟩)/√2
phi_minus = BellState.PHI_MINUS  # (|00⟩ - |11⟩)/√2
psi_plus = BellState.PSI_PLUS    # (|01⟩ + |10⟩)/√2
psi_minus = BellState.PSI_MINUS  # (|01⟩ - |10⟩)/√2

# Create entangled pair
pair = EntangledPair(prime_a=2, prime_b=3, state=BellState.PHI_PLUS)

# Entanglement network
network = EntanglementNetwork()

# Register nodes
network.register_node("alice", pri_alice)
network.register_node("bob", pri_bob)
network.register_node("charlie", pri_charlie)

# Establish entanglement
network.establish("alice", "bob")
network.establish("bob", "charlie")

# Long-distance entanglement via swapping
network.establish_long_distance("alice", "charlie")

# Teleportation (requires pre-shared entanglement)
state = PrimeState.basis(7)
network.teleport("alice", "charlie", state)
```

### Distributed Transport

```python
from tinyaleph.distributed.transport import (
    LocalTransport,
    Message,
    MessageType,
)

# Create local transport
transport = LocalTransport()

# Register handlers
def handle_query(message):
    return {"result": "processed"}

transport.register_handler(MessageType.QUERY, handle_query)

# Send message
response = transport.send(
    destination="processor_1",
    message=Message(
        type=MessageType.QUERY,
        payload={"data": [1, 2, 3]},
    ),
)
```

---

## Cryptographic Backend

### Key Generation

```python
from tinyaleph.backends.cryptographic import (
    PrimeStateKeyGenerator,
    EntropySensitiveEncryptor,
    HolographicKeyDistributor,
)

# Generate keys from prime resonance phases
keygen = PrimeStateKeyGenerator(primes=[2, 3, 5, 7, 11])

# Key = Σ_i θ_{p_i} mod 2π where θ_{p_i} = 2π log_{p_i}(n)
key = keygen.generate(n=42)

print(f"Key phase: {key.phase}")
print(f"Key entropy: {key.entropy}")
```

### Encryption/Decryption

```python
# Phase modulation encryption
encryptor = EntropySensitiveEncryptor(key)

# Encrypt data
plaintext = b"secret message"
ciphertext = encryptor.encrypt(plaintext)

# Decrypt data
decrypted = encryptor.decrypt(ciphertext)
assert decrypted == plaintext
```

### Key Distribution

```python
# Holographic key distribution (secret sharing)
distributor = HolographicKeyDistributor(
    num_shares=5,
    threshold=3,  # 3-of-5 reconstruction
)

# Split key into shares
shares = distributor.split(key)

# Reconstruct from any 3 shares
reconstructed = distributor.reconstruct(shares[:3])
assert reconstructed == key
```

---

## Examples Gallery

The library includes 20 comprehensive examples in the `examples/` directory:

| # | File | Topic |
|---|------|-------|
| 01 | `01_hello_world.py` | Basic prime state operations |
| 02 | `02_quantum_operators.py` | Operators and measurements |
| 03 | `03_quaternion_rotations.py` | 3D rotations with quaternions |
| 04 | `04_kuramoto_synchronization.py` | Coupled oscillators |
| 05 | `05_holographic_memory.py` | ResonantFragment storage |
| 06 | `06_entanglement_networks.py` | Bell states and networks |
| 07 | `07_sedenion_memory_field.py` | 16D SMF |
| 08 | `08_prsc_semantic_binding.py` | Concept-prime binding |
| 09 | `09_ml_attention_mechanisms.py` | Attention layers |
| 10 | `10_aleph_engine_runtime.py` | Full engine demo |
| 11 | `11_symbolic_hexagrams.py` | 64 I Ching hexagrams |
| 12 | `12_advanced_physics.py` | Stochastic Kuramoto |
| 13 | `13_cryptographic_backend.py` | Prime-based crypto |
| 14 | `14_semantic_types.py` | Type system |
| 15 | `15_reduction_semantics.py` | Proof generation |
| 16 | `16_lambda_calculus.py` | Lambda translation |
| 17 | `17_crt_homology.py` | CRT encoding |
| 18 | `18_topology_physics.py` | Knot invariants |
| 19 | `19_semantic_inference.py` | Reasoning engine |
| 20 | `20_resoformer_ml.py` | Complete ML architecture |

Run any example:
```bash
cd examples/
python 01_hello_world.py
```

---

## Best Practices

### State Normalization

Always normalize states after arithmetic operations:

```python
combined = state1 + state2
combined = combined.normalize()  # Essential!
```

### Coherence Monitoring

Monitor coherence during evolution:

```python
for step in range(max_steps):
    engine.evolve(dt=0.1)
    
    if engine.coherence < threshold:
        # Collapse before coherence loss
        result = engine.collapse()
        break
```

### Memory Management

Use sparse representations for large prime spaces:

```python
# Prefer SparsePrimeState for large vocabularies
from tinyaleph.ml.sparse_state import SparsePrimeState

state = SparsePrimeState.first_n_superposition(1000)  # Efficient
```

### Golden Ratio Thresholds

Use φ-based thresholds for consistency:

```python
from tinyaleph.core.constants import PHI

coherence_threshold = 1 / PHI  # ≈ 0.618
entropy_threshold = math.log(PHI)  # ≈ 0.481
```

### Error Handling

Validate prime inputs:

```python
from tinyaleph.core.primes import is_prime

def process_prime(p):
    if not is_prime(p):
        raise ValueError(f"{p} is not prime")
    # ... process ...
```

---

## Appendix: Mathematical Foundations

### Prime Hilbert Space (H_P)

The Prime Hilbert Space is the fundamental mathematical structure:

```
H_P = {|ψ⟩ = Σ_p α_p|p⟩ : Σ|α_p|² = 1, α_p ∈ ℂ, p prime}
```

**Properties:**
- Countably infinite dimensional (one dimension per prime)
- Separable Hilbert space
- Basis states |p⟩ are orthonormal: ⟨p|q⟩ = δ_{pq}

### Extended State Space (H_Q)

For quaternionic amplitudes:

```
H_Q = H_P ⊗ ℍ = {|ψ⟩ = Σ_p q_p|p⟩ : q_p ∈ ℍ}
```

Inner product: ⟨ψ|φ⟩ = Σ_p q̄_p · r_p (quaternion conjugate product)

### Cayley-Dickson Construction

Hypercomplex algebras are constructed via doubling:

```
ℝ → ℂ → ℍ → 𝕆 → 𝕊 → ...
1D   2D   4D   8D   16D
```

**Properties lost at each step:**
- ℂ: Ordering (no "greater than")
- ℍ: Commutativity (ab ≠ ba)
- 𝕆: Associativity ((ab)c ≠ a(bc))
- 𝕊: Alternativity (zero divisors exist)

### Kuramoto Model

Coupled oscillator dynamics:

```
dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)
```

**Order parameter:**
```
r·e^(iψ) = (1/N) Σ_j e^(iθ_j)
```

- r = 0: Incoherent (random phases)
- r = 1: Fully synchronized
- Critical coupling: K_c ≈ 2σ_ω/π (for Lorentzian frequency distribution)

### Semantic Type Rules

**Well-formedness for ChainTerm:**
```
A(p₁)...A(pₖ)N(q) is well-formed iff p_i < q for all i
```

**FusionTerm:**
```
FUSE(p, q, r) is well-formed iff p + q + r is prime
```

### Chinese Remainder Theorem

For coprime moduli m₁, m₂, ..., mₖ:

```
ℤ/(m₁·m₂·...·mₖ)ℤ ≅ ℤ/m₁ℤ × ℤ/m₂ℤ × ... × ℤ/mₖℤ
```

This enables encoding large values as tuples of small residues.

### Free Energy Principle (Friston)

```
F = E_q[ln q(s) - ln p(o,s)]
  = D_KL(q||p) - ln p(o)
```

Where:
- q(s): Approximate posterior (beliefs)
- p(o,s): Generative model (world model)
- D_KL: Kullback-Leibler divergence
- p(o): Model evidence

### Lyapunov Exponent

```
λ = lim_{t→∞} (1/t) ln|δx(t)/δx(0)|
```

**Classification:**
- λ < -0.1: Collapsed (converging)
- -0.1 ≤ λ ≤ 0.1: Stable (bounded)
- λ > 0.1: Unstable (diverging)

---

## Getting Help

- **Documentation**: This guide and the Reference Guide
- **Examples**: 20 examples in `examples/` directory
- **Issues**: Report bugs on GitHub
- **Source Code**: Full implementation in `tinyaleph/` directory

---

*TinyAleph v0.1.0 - Prime-Resonant Quantum Computing Framework*