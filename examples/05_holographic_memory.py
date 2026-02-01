#!/usr/bin/env python3
"""
Example 05: Holographic Memory with ResonantFragment

This example demonstrates the ResonantFragment system:
- Encoding patterns into holographic memory
- Prime-indexed coefficients
- Interference patterns
- Collapse and measurement
- Content-addressable retrieval

ResonantFragment provides holographic memory storage:
    A_p * e^(-S) * e^(ipθ)

where:
- A_p is the base amplitude for prime p
- S is the spatial entropy
- θ is the phase modulation
"""

import numpy as np
from tinyaleph.resonance.fragment import ResonantFragment

def main():
    print("=" * 60)
    print("TinyAleph: Holographic Memory with ResonantFragment")
    print("=" * 60)
    print()
    
    # ===== PART 1: Basic Fragment Creation =====
    print("PART 1: Basic Fragment Creation")
    print("-" * 40)
    
    # Create empty fragment
    empty = ResonantFragment()
    print(f"Empty fragment: {empty}")
    print(f"  Coefficients: {empty.coeffs}")
    print()
    
    # Encode a string pattern
    fragment = ResonantFragment.encode("Hello world")
    print(f"Encoded 'Hello world': {fragment}")
    print(f"  Number of primes: {len(fragment.coeffs)}")
    print(f"  Primes used: {fragment.primes()[:5]}... (first 5)")
    print(f"  Entropy: {fragment.entropy:.4f}")
    print(f"  Center: {fragment.center}")
    print()
    
    # Show coefficient structure
    print("Coefficient structure (first 5 primes):")
    for p in fragment.primes()[:5]:
        amp = fragment.coeffs[p]
        print(f"  |{p}⟩: amplitude = {amp:.4f}")
    print()
    
    # ===== PART 2: Different Encoding Methods =====
    print("PART 2: Different Encoding Methods")
    print("-" * 40)
    
    # Encode with different spatial entropy
    low_entropy = ResonantFragment.encode("Test", spatial_entropy=0.1)
    high_entropy = ResonantFragment.encode("Test", spatial_entropy=0.9)
    
    print(f"Same string with different spatial entropy:")
    print(f"  Low entropy (0.1): Shannon entropy = {low_entropy.entropy:.4f}")
    print(f"  High entropy (0.9): Shannon entropy = {high_entropy.entropy:.4f}")
    print()
    
    # Create from explicit prime-amplitude pairs
    from_primes = ResonantFragment.from_primes([
        (2, 1.0),
        (3, 0.8),
        (5, 0.6),
        (7, 0.4)
    ])
    print(f"From explicit primes: {from_primes}")
    print()
    
    # Random fragment
    random_frag = ResonantFragment.random(n_primes=10)
    print(f"Random fragment: {random_frag}")
    print()
    
    # ===== PART 3: Fragment Properties =====
    print("PART 3: Fragment Properties")
    print("-" * 40)
    
    fragment = ResonantFragment.encode("TinyAleph")
    
    # Norm (should be ~1 after normalization)
    print(f"Fragment: {fragment}")
    print(f"  Norm: {fragment.norm():.6f}")
    
    # Dominant prime (highest amplitude)
    dominant = fragment.dominant_prime()
    print(f"  Dominant prime: {dominant}")
    print(f"  Amplitude of |{dominant}⟩: {fragment.coeffs[dominant]:.4f}")
    
    # All primes
    primes = fragment.primes()
    print(f"  All primes: {primes}")
    print()
    
    # ===== PART 4: Fragment Operations =====
    print("PART 4: Fragment Operations")
    print("-" * 40)
    
    # Addition (unnormalized)
    f1 = ResonantFragment.encode("Alpha")
    f2 = ResonantFragment.encode("Beta")
    f_sum = f1 + f2
    
    print(f"f1 = encode('Alpha')")
    print(f"f2 = encode('Beta')")
    print(f"f1 + f2: {f_sum}")
    print(f"  Combined primes: {len(f_sum.coeffs)}")
    print()
    
    # Scalar multiplication
    scaled = f1 * 0.5
    print(f"0.5 * f1: norm = {scaled.norm():.4f} (half of original)")
    print()
    
    # Normalization
    normalized = f_sum.normalize()
    print(f"Normalized sum: norm = {normalized.norm():.6f}")
    print()
    
    # ===== PART 5: Tensor Product (Interference) =====
    print("PART 5: Tensor Product (Interference)")
    print("-" * 40)
    
    # Tensor product creates interference patterns
    pattern1 = ResonantFragment.encode("Signal")
    pattern2 = ResonantFragment.encode("Noise")
    
    interference = pattern1.tensor(pattern2)
    
    print(f"Pattern 1: {pattern1}")
    print(f"Pattern 2: {pattern2}")
    print(f"Interference: {interference}")
    print(f"  Combined entropy: {interference.entropy:.4f}")
    print(f"  Combined center: ({interference.center[0]:.2f}, {interference.center[1]:.2f})")
    print()
    
    # ===== PART 6: Collapse and Measurement =====
    print("PART 6: Collapse and Measurement")
    print("-" * 40)
    
    # Create superposition
    superpos = ResonantFragment.from_primes([
        (2, 1.0),
        (3, 1.0),
        (5, 1.0),
        (7, 1.0)
    ])
    
    print(f"Initial superposition: {superpos}")
    print(f"  Entropy: {superpos.entropy:.4f}")
    
    # Collapse collapses to single prime (probabilistic)
    print("\nCollapsing 5 times:")
    for i in range(5):
        test = ResonantFragment.from_primes([(2, 1.0), (3, 1.0), (5, 1.0), (7, 1.0)])
        collapsed = test.collapse()
        dominant = collapsed.dominant_prime()
        print(f"  Trial {i+1}: collapsed to |{dominant}⟩")
    
    print("\nAfter collapse:")
    collapsed = superpos.collapse()
    print(f"  {collapsed}")
    print(f"  Entropy: {collapsed.entropy:.4f} (should be 0)")
    print()
    
    # ===== PART 7: Phase Rotation =====
    print("PART 7: Phase Rotation")
    print("-" * 40)
    
    fragment = ResonantFragment.from_primes([(2, 1.0), (3, 0.5)])
    print(f"Original: {fragment}")
    print(f"  Coefficient of |2⟩: {fragment.coeffs[2]:.4f}")
    
    rotated = fragment.rotate_phase(np.pi / 4)
    print(f"After π/4 rotation: {rotated}")
    print(f"  Coefficient of |2⟩: {rotated.coeffs[2]:.4f}")
    print("  (Phase rotation applies cos(p * θ) to each component)")
    print()
    
    # ===== PART 8: Overlap and Distance =====
    print("PART 8: Overlap and Distance")
    print("-" * 40)
    
    # Similar patterns should have high overlap
    same1 = ResonantFragment.encode("Similarity test")
    same2 = ResonantFragment.encode("Similarity test")
    diff = ResonantFragment.encode("Different pattern")
    
    overlap_same = same1.overlap(same2)
    overlap_diff = same1.overlap(diff)
    
    print(f"Overlap between identical patterns: {overlap_same:.4f}")
    print(f"Overlap between different patterns: {overlap_diff:.4f}")
    print()
    
    dist_same = same1.distance(same2)
    dist_diff = same1.distance(diff)
    
    print(f"Distance between identical patterns: {dist_same:.4f}")
    print(f"Distance between different patterns: {dist_diff:.4f}")
    print()
    
    # ===== PART 9: Vector Conversion =====
    print("PART 9: Vector Conversion")
    print("-" * 40)
    
    fragment = ResonantFragment.encode("Convert")
    primes = fragment.primes()
    
    # Convert to numpy vector
    vec = fragment.to_vector(primes)
    print(f"As vector: {vec[:5]}... (first 5 elements)")
    print(f"  Vector length: {len(vec)}")
    
    # Convert back
    recovered = ResonantFragment.from_vector(vec, primes)
    print(f"Recovered: {recovered}")
    print(f"  Distance from original: {fragment.distance(recovered):.6f}")
    print()
    
    # ===== PART 10: Content-Addressable Memory =====
    print("PART 10: Content-Addressable Memory Example")
    print("-" * 40)
    
    # Store multiple patterns
    memories = {
        "cat": ResonantFragment.encode("The quick cat"),
        "dog": ResonantFragment.encode("The lazy dog"),
        "fox": ResonantFragment.encode("The brown fox"),
        "bird": ResonantFragment.encode("The singing bird")
    }
    
    # Query pattern
    query = ResonantFragment.encode("The cat")
    
    print("Stored memories:")
    for name, frag in memories.items():
        print(f"  '{name}': {len(frag.coeffs)} primes")
    
    print(f"\nQuery: 'The cat'")
    print("Finding most similar memory:")
    
    overlaps = {}
    for name, frag in memories.items():
        overlap = query.overlap(frag)
        overlaps[name] = overlap
        print(f"  Overlap with '{name}': {overlap:.4f}")
    
    best_match = max(overlaps, key=overlaps.get)
    print(f"\nBest match: '{best_match}' (overlap = {overlaps[best_match]:.4f})")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Holographic Memory")
    print("=" * 60)
    print("""
ResonantFragment provides holographic memory:

Encoding Formula:
    A_p * e^(-S) * e^(ipθ)
    
Key Properties:
1. Prime-Indexed: Information stored in prime coefficients
2. Normalized: Σ|A_p|² = 1 (like quantum state)
3. Holographic: Each part contains information about whole
4. Interference: Fragments can be superposed

Operations:
- encode(): String → Fragment
- tensor(): Interference pattern
- collapse(): Measure to single prime
- overlap(): Similarity measure
- distance(): L2 distance

Applications:
- Content-addressable memory
- Pattern recognition
- Semantic similarity
- Information compression

Connection to Quantum States:
- ResonantFragment ≈ PrimeState with spatial structure
- Entropy measures information content
- Collapse is probabilistic measurement
    """)

if __name__ == "__main__":
    main()