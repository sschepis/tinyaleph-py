#!/usr/bin/env python3
"""
Example 09: ML Attention Mechanisms

This example demonstrates machine learning components:
- SparsePrimeState (H_Q = H_P ⊗ ℍ)
- Resonant attention
- Multi-head attention
- Coherence-gated computation (ACT)
- Transformer blocks

The key insight is extending PrimeState with quaternionic amplitudes
for richer geometric transformations.
"""

from tinyaleph.ml.sparse_state import (
    SparsePrimeState,
    coherent_superposition,
    golden_superposition,
)
from tinyaleph.ml.attention import (
    resonant_attention,
    softmax,
    prime_resonance_weight,
    resonance_kernel,
    golden_ratio_attention_weights,
    AttentionHead,
    MultiHeadResonantAttention,
    CoherenceGatedComputation,
    ResonantTransformerBlock,
    create_resonant_transformer,
)
from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.constants import PHI
import math

def main():
    print("=" * 60)
    print("TinyAleph: ML Attention Mechanisms")
    print("=" * 60)
    print()
    
    # ===== PART 1: SparsePrimeState =====
    print("PART 1: SparsePrimeState (H_Q = H_P ⊗ ℍ)")
    print("-" * 40)
    
    # SparsePrimeState uses quaternionic amplitudes
    # This enables richer rotations than complex amplitudes
    
    # Vacuum state
    vacuum = SparsePrimeState.vacuum()
    print(f"Vacuum state: {vacuum}")
    print(f"  Length: {len(vacuum)}")
    print()
    
    # Single prime eigenstate
    state_7 = SparsePrimeState.single_prime(7)
    print(f"Single prime |7⟩: {state_7}")
    print()
    
    # Superposition of first 5 primes
    superpos = SparsePrimeState.first_n_superposition(5)
    print(f"First 5 primes superposition: {superpos}")
    print(f"  Norm: {superpos.norm():.4f}")
    print(f"  Entropy: {superpos.entropy():.4f}")
    print()
    
    # Custom quaternionic amplitudes
    q1 = Quaternion(0.5, 0.5, 0, 0)
    q2 = Quaternion(0.5, 0, 0.5, 0)
    q3 = Quaternion(0.5, 0, 0, 0.5)
    
    custom = SparsePrimeState.from_primes([2, 3, 5], [q1, q2, q3])
    print(f"Custom quaternionic state: {custom}")
    print()
    
    # ===== PART 2: State Operations =====
    print("PART 2: State Operations")
    print("-" * 40)
    
    s1 = SparsePrimeState.single_prime(2)
    s2 = SparsePrimeState.single_prime(3)
    
    # Addition
    combined = s1 + s2
    print(f"|2⟩ + |3⟩ = {combined}")
    
    # Scalar multiplication
    scaled = s1 * 2.0
    print(f"2.0 * |2⟩: norm = {scaled.norm():.4f}")
    
    # Quaternion multiplication
    q = Quaternion(0, 1, 0, 0)  # Pure i
    rotated = s1.quaternion_mul(q)
    print(f"|2⟩ * i = {rotated}")
    print()
    
    # ===== PART 3: Inner Product =====
    print("PART 3: Quaternionic Inner Product")
    print("-" * 40)
    
    # Inner product ⟨ψ|φ⟩ uses quaternion conjugate
    s1 = SparsePrimeState.first_n_superposition(3)
    s2 = SparsePrimeState.first_n_superposition(3)
    
    ip = s1.inner_product(s2)
    print(f"⟨s1|s2⟩ = {ip}")
    print(f"  Norm of inner product: {ip.norm():.4f}")
    
    # Overlap probability
    overlap = s1.overlap(s2)
    print(f"Overlap |⟨s1|s2⟩|²: {overlap:.4f}")
    print()
    
    # ===== PART 4: Prime Spectrum =====
    print("PART 4: Prime Spectrum and Entropy")
    print("-" * 40)
    
    state = SparsePrimeState.first_n_superposition(5)
    spectrum = state.prime_spectrum()
    
    print("Prime probability distribution:")
    for p, prob in spectrum.items():
        bar = '█' * int(prob * 20)
        print(f"  |{p}⟩: {prob:.4f} {bar}")
    
    print(f"\nEntropy: {state.entropy():.4f} bits")
    print(f"Is coherent (entropy < threshold): {state.is_coherent()}")
    print()
    
    # ===== PART 5: Rotations =====
    print("PART 5: Quaternionic Rotations")
    print("-" * 40)
    
    state = SparsePrimeState.single_prime(2)
    print(f"Original: {state}")
    
    # Apply rotation around j axis
    axis = Quaternion(0, 0, 1, 0)  # j axis
    angle = math.pi / 4  # 45 degrees
    
    rotated = state.apply_rotation(axis, angle)
    print(f"After 45° rotation around j: {rotated}")
    
    # Apply phase to specific prime
    phased = state.apply_phase(2, math.pi / 2)
    print(f"After π/2 phase on |2⟩: {phased}")
    print()
    
    # ===== PART 6: Resonant Attention =====
    print("PART 6: Resonant Attention")
    print("-" * 40)
    
    # Create query, keys, values
    query = SparsePrimeState.single_prime(2)
    keys = [
        SparsePrimeState.single_prime(2),
        SparsePrimeState.single_prime(3),
        SparsePrimeState.single_prime(5),
    ]
    values = [
        SparsePrimeState.single_prime(7),
        SparsePrimeState.single_prime(11),
        SparsePrimeState.single_prime(13),
    ]
    
    print(f"Query: |2⟩")
    print(f"Keys: |2⟩, |3⟩, |5⟩")
    print(f"Values: |7⟩, |11⟩, |13⟩")
    
    result = resonant_attention(query, keys, values, temperature=1.0)
    print(f"\nAttention result: {result}")
    print("  (Weighted combination based on query-key similarity)")
    print()
    
    # Prime resonance weighting
    print("Prime resonance weights:")
    for p in [2, 3, 5, 7, 11]:
        w = prime_resonance_weight(p)
        bar = '█' * int(w * 20)
        print(f"  w({p}) = {w:.4f} {bar}")
    print()
    
    # ===== PART 7: Multi-Head Attention =====
    print("PART 7: Multi-Head Attention")
    print("-" * 40)
    
    mha = MultiHeadResonantAttention(num_heads=4, dim=16)
    print(f"Created multi-head attention:")
    print(f"  Number of heads: {mha.num_heads}")
    print(f"  Dimension: {mha.dim}")
    
    # Process through multi-head attention
    query = SparsePrimeState.first_n_superposition(3)
    keys = [SparsePrimeState.first_n_superposition(3)]
    values = [SparsePrimeState.first_n_superposition(5)]
    
    output = mha.forward(query, keys, values)
    print(f"\nOutput: {output}")
    print()
    
    # ===== PART 8: Coherence-Gated Computation =====
    print("PART 8: Coherence-Gated Computation (ACT)")
    print("-" * 40)
    
    # ACT: Halt computation when coherence threshold reached
    gate = CoherenceGatedComputation(max_steps=5, coherence_threshold=0.7)
    
    def step_fn(state, step):
        """Simple step that adds entropy."""
        # Add more primes to increase entropy
        new_primes = [2, 3, 5, 7, 11, 13][:step+2]
        return SparsePrimeState.from_primes(new_primes)
    
    initial = SparsePrimeState.single_prime(2)
    result, steps, halt_prob = gate.compute(initial, step_fn)
    
    print(f"Coherence-gated computation:")
    print(f"  Initial state: {initial}")
    print(f"  Steps taken: {steps}")
    print(f"  Halt probability: {halt_prob:.4f}")
    print(f"  Final state: {result}")
    print()
    
    # ===== PART 9: Golden Ratio Patterns =====
    print("PART 9: Golden Ratio Patterns")
    print("-" * 40)
    
    # Golden superposition uses Φ-spaced phases
    golden = golden_superposition(5)
    print(f"Golden superposition (5 primes): {golden}")
    print(f"  Golden ratio Φ = {PHI:.6f}")
    
    # Golden attention weights
    weights = golden_ratio_attention_weights(5)
    print(f"\nGolden attention weights:")
    for i, w in enumerate(weights):
        bar = '█' * int(w * 30)
        print(f"  w[{i}] = {w:.4f} {bar}")
    print(f"  Sum: {sum(weights):.4f}")
    print()
    
    # ===== PART 10: Transformer Blocks =====
    print("PART 10: Resonant Transformer")
    print("-" * 40)
    
    # Create transformer stack
    blocks = create_resonant_transformer(
        num_layers=2,
        num_heads=4,
        dim=16,
        max_steps=3
    )
    
    print(f"Created transformer with {len(blocks)} layers")
    
    # Process through transformer
    state = SparsePrimeState.first_n_superposition(3)
    print(f"\nInput: {state}")
    
    for i, block in enumerate(blocks):
        state = block.forward(state)
        print(f"After layer {i+1}: {state}")
    
    print()
    
    # ===== PART 11: Collapse and Measurement =====
    print("PART 11: Collapse and Measurement")
    print("-" * 40)
    
    state = SparsePrimeState.first_n_superposition(5)
    print(f"Before collapse: {state}")
    print(f"  Entropy: {state.entropy():.4f}")
    
    prime, phase = state.collapse()
    print(f"\nAfter collapse:")
    print(f"  Measured prime: {prime}")
    print(f"  Phase: {phase}")
    print(f"  State entropy: {state.entropy():.4f}")
    print()
    
    # ===== PART 12: Vector Conversion =====
    print("PART 12: Vector Conversion")
    print("-" * 40)
    
    state = SparsePrimeState.first_n_superposition(3)
    
    # To real vector (4 components per prime)
    vec = state.to_real_vector(max_prime_idx=10)
    print(f"State as vector (first 12 components):")
    print(f"  {vec[:12]}")
    print(f"  Total length: {len(vec)} (10 primes × 4 quaternion components)")
    
    # Back to state
    recovered = SparsePrimeState.from_real_vector(vec)
    print(f"\nRecovered: {recovered}")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: ML Attention Mechanisms")
    print("=" * 60)
    print("""
SparsePrimeState (H_Q = H_P ⊗ ℍ):
- Prime states with quaternionic amplitudes
- Enables 3D rotations in amplitude space
- Sparse representation for efficiency

Resonant Attention:
    Attention(Q, K, V) = softmax(R(Q, K)) V
    
- Prime resonance modulates attention scores
- Golden ratio weighting for heads
- Coherence-gated halting (ACT)

Multi-Head Attention:
- Multiple attention heads with Φ-spaced axes
- Golden ratio weighting for head combination
- Projects through Q, K, V transformations

Coherence-Gated Computation:
- ACT-style adaptive computation time
- Halts when coherence exceeds threshold
- Prevents over-computation

Golden Ratio Patterns:
- Φ-based phase spacing
- Optimal coverage in phase space
- Fibonacci-weighted combinations

Applications:
- Quantum-inspired neural networks
- Symbolic reasoning
- Semantic search
- Generative models
    """)

if __name__ == "__main__":
    main()