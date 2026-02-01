#!/usr/bin/env python3
"""
Example 02: Quantum Operators and Measurements

This example demonstrates quantum operators on prime states:
- Identity operator
- Prime shift operator (raising/lowering)
- Phase operators
- Resonance operators
- Fourier transform in prime space
- Projection and measurement operators

These operators form the algebra acting on H_P.
"""

from tinyaleph.hilbert.state import PrimeState
from tinyaleph.hilbert.operators import (
    IdentityOperator,
    PrimeShiftOperator,
    ResonanceOperator,
    PhaseOperator,
    ProjectionOperator,
    CollapseOperator,
    FourierOperator,
)
from tinyaleph.core.complex import Complex
import math

def main():
    print("=" * 60)
    print("TinyAleph: Quantum Operators in Prime Hilbert Space")
    print("=" * 60)
    print()
    
    # ===== PART 1: Identity Operator =====
    print("PART 1: Identity Operator")
    print("-" * 40)
    
    # The identity operator I leaves all states unchanged: I|ψ⟩ = |ψ⟩
    identity = IdentityOperator()
    state = PrimeState.uniform(n=5)
    
    result = identity.apply(state)
    print(f"Original state: {state}")
    print(f"After identity: {result}")
    print(f"States are equal: {state.primes == result.primes}")
    print()
    
    # ===== PART 2: Prime Shift Operator =====
    print("PART 2: Prime Shift Operator")
    print("-" * 40)
    
    # The shift operator S_k maps |p_n⟩ → |p_{n+k}⟩
    # S_1|2⟩ = |3⟩, S_1|3⟩ = |5⟩, etc.
    
    shift_up = PrimeShiftOperator(shift=1)
    shift_down = PrimeShiftOperator(shift=-1)
    shift_2 = PrimeShiftOperator(shift=2)
    
    basis_2 = PrimeState.basis(2)
    print(f"Original: {basis_2}")
    
    shifted_up = shift_up.apply(basis_2)
    print(f"S₁|2⟩ = {shifted_up} (shifted to 3rd prime)")
    
    shifted_2 = shift_2.apply(basis_2)
    print(f"S₂|2⟩ = {shifted_2} (shifted to 4th prime)")
    
    # Shift down from |5⟩
    basis_5 = PrimeState.basis(5)
    shifted_down = shift_down.apply(basis_5)
    print(f"S₋₁|5⟩ = {shifted_down} (shifted to 2nd prime)")
    print()
    
    # Apply to superposition
    superpos = PrimeState.uniform(n=3)  # |2⟩ + |3⟩ + |5⟩
    print(f"Superposition: {superpos}")
    shifted_superpos = shift_up.apply(superpos)
    print(f"After S₁: {shifted_superpos}")
    print()
    
    # ===== PART 3: Phase Operator =====
    print("PART 3: Phase Operator")
    print("-" * 40)
    
    # Phase operator applies e^(iφ) to specific prime component
    # P_p(φ)|p⟩ = e^(iφ)|p⟩, P_p(φ)|q⟩ = |q⟩ for q ≠ p
    
    phase_2 = PhaseOperator(prime=2, phase=math.pi / 2)  # 90° phase on |2⟩
    
    superpos = PrimeState.superposition(
        primes=[2, 3],
        amplitudes=[Complex(1/math.sqrt(2), 0), Complex(1/math.sqrt(2), 0)]
    )
    print(f"Original: {superpos}")
    print(f"  Amplitude of |2⟩: {superpos.amplitudes[0]}")
    
    phased = phase_2.apply(superpos)
    print(f"After π/2 phase on |2⟩: {phased}")
    print(f"  Amplitude of |2⟩: {phased.amplitudes[0]}")
    print(f"  (Expected: i/√2 = {Complex(0, 1/math.sqrt(2))})")
    print()
    
    # ===== PART 4: Resonance Operator =====
    print("PART 4: Resonance Operator")
    print("-" * 40)
    
    # Resonance operator applies prime-dependent phase
    # R_κ|p⟩ = e^(iκ log p)|p⟩
    # Creates "harmonic" phase relationships between primes
    
    resonance = ResonanceOperator(coupling=1.0)
    
    print(f"Resonance operator with κ=1.0:")
    for p in [2, 3, 5, 7, 11]:
        basis_p = PrimeState.basis(p)
        res_p = resonance.apply(basis_p)
        phase = math.log(p)  # The applied phase
        print(f"  R|{p}⟩: phase = κ·log({p}) = {phase:.4f} rad = {math.degrees(phase):.1f}°")
    print()
    
    # Apply to superposition - each component gets different phase
    uniform = PrimeState.uniform(n=5)
    print(f"Original uniform: {uniform}")
    resonated = resonance.apply(uniform)
    print(f"After resonance: {resonated}")
    print("  (Each component now has prime-dependent phase)")
    print()
    
    # ===== PART 5: Projection Operator =====
    print("PART 5: Projection Operator")
    print("-" * 40)
    
    # Projection operator projects onto subspace spanned by specific primes
    # Π_S|ψ⟩ = Σ_{p∈S} αp|p⟩ / ||Σ_{p∈S} αp|p⟩||
    
    project_23 = ProjectionOperator(primes=[2, 3])
    
    uniform_5 = PrimeState.uniform(n=5)  # |2⟩ + |3⟩ + |5⟩ + |7⟩ + |11⟩
    print(f"Original (5 primes): {uniform_5}")
    
    projected = project_23.apply(uniform_5)
    print(f"Projected onto {{2, 3}}: {projected}")
    print(f"  Remaining primes: {projected.primes}")
    print(f"  Is normalized: {abs(projected.norm() - 1.0) < 0.01}")
    print()
    
    # ===== PART 6: Collapse (Measurement) Operator =====
    print("PART 6: Collapse (Measurement) Operator")
    print("-" * 40)
    
    # Collapse operator performs a measurement on specific primes
    # Used for selective measurement without full collapse
    
    collapse_2 = CollapseOperator(primes=[2])
    
    superpos = PrimeState.uniform(n=3)
    print(f"Original: {superpos}")
    print(f"  P(|2⟩) = {superpos.probabilities()[2]:.4f}")
    
    # Collapse to |2⟩ if 2 is in the state
    collapsed = collapse_2.apply(superpos)
    print(f"After collapse onto {{2}}: {collapsed}")
    print(f"  (Deterministic collapse to |2⟩)")
    print()
    
    # ===== PART 7: Fourier Transform in Prime Space =====
    print("PART 7: Fourier Transform in Prime Space")
    print("-" * 40)
    
    # The prime Fourier transform creates superposition with
    # phase relationships based on prime indices
    # F|p_n⟩ = (1/√N) Σ_m e^(2πi·n·m/N) |p_m⟩
    
    fourier = FourierOperator()
    
    # Fourier transform of basis state
    basis_2 = PrimeState.basis(2)
    print(f"Original basis |2⟩: {basis_2}")
    
    transformed = fourier.apply(basis_2)
    print(f"After Fourier: {transformed}")
    print("  (Now a superposition with phase structure)")
    
    # The inverse returns to the original (approximately)
    inverse = FourierOperator(inverse=True)
    recovered = inverse.apply(transformed)
    print(f"After inverse Fourier: {recovered}")
    print()
    
    # ===== PART 8: Operator Composition =====
    print("PART 8: Operator Composition")
    print("-" * 40)
    
    # Operators can be composed: (A∘B)|ψ⟩ = A(B|ψ⟩)
    
    state = PrimeState.basis(2)
    print(f"Original: {state}")
    
    # First shift, then apply resonance
    shifted = shift_up.apply(state)
    print(f"After shift S₁: {shifted}")
    
    final = resonance.apply(shifted)
    print(f"After resonance R: {final}")
    print()
    
    # Create a circuit-like sequence of operations
    print("Circuit: S₁ → R → S₁ on |2⟩")
    circuit_state = PrimeState.basis(2)
    circuit_state = shift_up.apply(circuit_state)   # |2⟩ → |3⟩
    circuit_state = resonance.apply(circuit_state)  # Apply phase
    circuit_state = shift_up.apply(circuit_state)   # |3⟩ → |5⟩
    print(f"Final state: {circuit_state}")
    print()
    
    # ===== PART 9: Measuring Operator Effects =====
    print("PART 9: Measuring Operator Effects")
    print("-" * 40)
    
    # Operators can change the entropy/coherence of states
    
    uniform = PrimeState.uniform(n=5)
    print(f"Uniform state entropy: {uniform.entropy():.4f}")
    
    # Projection reduces entropy (increases coherence)
    project_2 = ProjectionOperator(primes=[2])
    projected = project_2.apply(uniform)
    print(f"After projection to |2⟩: entropy = {projected.entropy():.4f}")
    
    # Resonance preserves entropy (unitary operation)
    resonated = resonance.apply(uniform)
    print(f"After resonance: entropy = {resonated.entropy():.4f}")
    
    # Fourier transform preserves entropy (unitary)
    fourier_state = fourier.apply(uniform)
    print(f"After Fourier: entropy = {fourier_state.entropy():.4f}")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Quantum Operators")
    print("=" * 60)
    print("""
Key Operators in Prime Hilbert Space:

1. Identity (I): I|ψ⟩ = |ψ⟩
   - Preserves all states

2. Prime Shift (S_k): S_k|p_n⟩ = |p_{n+k}⟩
   - Moves between prime indices

3. Phase (P): P_p(φ)|p⟩ = e^(iφ)|p⟩
   - Applies phase to specific prime

4. Resonance (R): R_κ|p⟩ = e^(iκ log p)|p⟩
   - Prime-dependent phase modulation

5. Projection (Π): Π_S projects onto subspace
   - Reduces state to subset of primes

6. Collapse (C): C_S collapses to specific primes
   - Measurement-like operation

7. Fourier (F): Creates phase-structured superpositions
   - Analogous to quantum Fourier transform

Properties:
- Unitary operators preserve norm and entropy
- Projection/collapse reduce entropy (increase coherence)
- Operators can be composed to form circuits
    """)

if __name__ == "__main__":
    main()