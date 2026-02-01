#!/usr/bin/env python3
"""
Example 01: Hello World - Basic Prime State Operations

This example introduces the fundamental concepts of TinyAleph:
- Creating prime states in the Hilbert space H_P
- Superposition and probability amplitudes
- Measurement and state collapse

The Prime Hilbert Space is defined as:
    H_P = {|ψ⟩ = Σ αp|p⟩ : Σ|αp|² = 1, αp ∈ ℂ, p prime}

Each basis state |p⟩ corresponds to a prime number p.
"""

from tinyaleph.hilbert.state import PrimeState
from tinyaleph.core.complex import Complex
from tinyaleph.core.primes import first_n_primes
import math

def main():
    print("=" * 60)
    print("TinyAleph: Hello World - Prime State Basics")
    print("=" * 60)
    print()
    
    # ===== PART 1: Creating Basic States =====
    print("PART 1: Creating Basic States")
    print("-" * 40)
    
    # Create a basis state |2⟩ - the first prime
    state_2 = PrimeState.basis(2)
    print(f"Basis state |2⟩: {state_2}")
    print(f"  Number of primes in basis: {len(state_2.primes)}")
    print()
    
    # Create a basis state |7⟩
    state_7 = PrimeState.basis(7)
    print(f"Basis state |7⟩: {state_7}")
    print()
    
    # Single prime (alias for basis)
    state_11 = PrimeState.single_prime(11)
    print(f"Single prime |11⟩: {state_11}")
    print()
    
    # ===== PART 2: Superposition States =====
    print("PART 2: Superposition States")
    print("-" * 40)
    
    # Create equal superposition of first 5 primes: (|2⟩ + |3⟩ + |5⟩ + |7⟩ + |11⟩)/√5
    uniform = PrimeState.first_n_superposition(5)
    print(f"Uniform superposition of first 5 primes: {uniform}")
    print(f"  Primes in basis: {uniform.primes}")
    
    # Get probabilities for each prime
    probs = uniform.probabilities()
    print(f"  Probabilities:")
    for p, prob in probs.items():
        print(f"    P(|{p}⟩) = {prob:.4f} ({prob*100:.1f}%)")
    print(f"  Sum of probabilities: {sum(probs.values()):.4f}")
    print()
    
    # Create superposition with custom amplitudes
    custom_primes = [2, 3, 5]
    custom = PrimeState(custom_primes)
    custom.set(2, Complex(0.6, 0))   # |2⟩ with amplitude 0.6
    custom.set(3, Complex(0.6, 0))   # |3⟩ with amplitude 0.6
    custom.set(5, Complex(0.5, 0))   # |5⟩ with amplitude 0.5
    
    print(f"Custom superposition (before normalization): {custom}")
    print(f"  Norm: {custom.norm():.4f}")
    print(f"  Is normalized: {abs(custom.norm() - 1.0) < 0.01}")
    
    # Normalize the state
    normalized = custom.normalize()
    print(f"After normalization: {normalized}")
    print(f"  Norm: {normalized.norm():.6f}")
    print()
    
    # ===== PART 3: State Properties =====
    print("PART 3: State Properties")
    print("-" * 40)
    
    # Shannon entropy measures the "uncertainty" in the state
    # Basis state has zero entropy (perfectly determined)
    # Uniform superposition has maximum entropy
    
    basis_entropy = state_2.entropy()
    uniform_entropy = uniform.entropy()
    
    print(f"Entropy of |2⟩ (basis state): {basis_entropy:.4f}")
    print(f"Entropy of uniform superposition (5 primes): {uniform_entropy:.4f}")
    print(f"Maximum entropy for 5 primes: {math.log2(5):.4f}")
    print()
    
    # Coherence is a measure of how "focused" the state is
    # Low entropy = high coherence
    print(f"Coherence of |2⟩: {state_2.coherence():.4f}")
    print(f"Coherence of uniform: {uniform.coherence():.4f}")
    print()
    
    # ===== PART 4: State Arithmetic =====
    print("PART 4: State Arithmetic")
    print("-" * 40)
    
    # Need same basis for arithmetic
    primes_small = [2, 3, 5, 7]
    s2 = PrimeState(primes_small)
    s2.set(2, Complex.one())
    s7 = PrimeState(primes_small)
    s7.set(7, Complex.one())
    
    # Add two states (creates superposition)
    combined = s2 + s7
    print(f"|2⟩ + |7⟩ = {combined}")
    print(f"  Norm before normalization: {combined.norm():.4f}")
    
    combined_norm = combined.normalize()
    print(f"  After normalization: {combined_norm}")
    print()
    
    # Scalar multiplication
    scaled = s2 * 0.5
    print(f"0.5 * |2⟩ = {scaled}")
    print()
    
    # ===== PART 5: Inner Products =====
    print("PART 5: Inner Products")
    print("-" * 40)
    
    # Inner product ⟨ψ|φ⟩ measures overlap between states
    # Orthogonal states have zero inner product
    
    overlap_same = s2.inner_product(s2)
    print(f"⟨2|2⟩ = {overlap_same.re:.4f} (same state)")
    
    overlap_diff = s2.inner_product(s7)
    print(f"⟨2|7⟩ = {overlap_diff.re:.4f} (orthogonal)")
    
    # Overlap with superposition
    uniform_small = PrimeState.uniform_superposition(primes_small)
    overlap_super = s2.inner_product(uniform_small)
    print(f"⟨2|uniform⟩ = {overlap_super.norm():.4f}")
    print(f"  (Expected: 1/√4 = {1/math.sqrt(4):.4f})")
    print()
    
    # ===== PART 6: Measurement =====
    print("PART 6: Measurement (Collapse)")
    print("-" * 40)
    
    # Measurement collapses the state to a single basis state
    # The probability of each outcome is |amplitude|²
    
    print("Measuring uniform superposition 10 times:")
    outcomes = {}
    for i in range(10):
        # Create fresh uniform state (measurement modifies state)
        test_state = PrimeState.first_n_superposition(5)
        prime, prob = test_state.measure()
        outcomes[prime] = outcomes.get(prime, 0) + 1
        print(f"  Trial {i+1}: Collapsed to |{prime}⟩ with probability {prob:.4f}")
    
    print(f"\nOutcome distribution: {outcomes}")
    print()
    
    # ===== PART 7: Composite States =====
    print("PART 7: Composite States from Numbers")
    print("-" * 40)
    
    # Create state from a composite number (spreads over factors)
    composite = PrimeState.composite(12)  # 12 = 2² × 3
    print(f"State from 12 = 2² × 3:")
    print(f"  {composite}")
    probs = composite.probabilities()
    print(f"  Non-zero probabilities:")
    for p, prob in probs.items():
        if prob > 0.001:
            print(f"    P(|{p}⟩) = {prob:.4f}")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Key Concepts")
    print("=" * 60)
    print("""
1. PrimeState represents quantum states in the Prime Hilbert Space
2. Basis states |p⟩ are created with PrimeState.basis(p) or single_prime(p)
3. Uniform superpositions are created with first_n_superposition(n)
4. States must be normalized: Σ|αp|² = 1
5. Entropy measures uncertainty (0 for basis, log₂(n) for uniform)
6. Inner products measure overlap: ⟨ψ|φ⟩
7. Measurement collapses to basis state with probability |αp|²
    """)

if __name__ == "__main__":
    main()