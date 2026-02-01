#!/usr/bin/env python3
"""
Example 08: Prime Resonance Semantic Coherence (PRSC)

This example demonstrates semantic concept binding:
- Binding concepts to prime states
- Computing semantic coherence
- Concept composition
- Semantic distance metrics
- State-based semantic reasoning

PRSC binds semantic concepts to prime states:
    concept → PrimeState → Semantic operations
"""

from tinyaleph.observer.prsc import PrimeResonanceSemanticCoherence
from tinyaleph.hilbert.state import PrimeState
from tinyaleph.core.complex import Complex
import math

def main():
    print("=" * 60)
    print("TinyAleph: Prime Resonance Semantic Coherence (PRSC)")
    print("=" * 60)
    print()
    
    # ===== PART 1: Creating PRSC System =====
    print("PART 1: Creating PRSC System")
    print("-" * 40)
    
    prsc = PrimeResonanceSemanticCoherence()
    print(f"Created PRSC system")
    print(f"  Initial concepts: {prsc.list_concepts()}")
    print()
    
    # ===== PART 2: Binding Concepts to Primes =====
    print("PART 2: Binding Concepts to Primes")
    print("-" * 40)
    
    # Bind concepts to prime sets
    # Each concept maps to a set of primes that represent it
    
    prsc.bind_concept("mathematics", [2, 3, 5, 7, 11])
    prsc.bind_concept("physics", [2, 5, 7, 13, 17])
    prsc.bind_concept("chemistry", [3, 7, 11, 19, 23])
    prsc.bind_concept("biology", [5, 11, 13, 29, 31])
    prsc.bind_concept("art", [37, 41, 43, 47, 53])
    
    print("Bound concepts:")
    for concept in prsc.list_concepts():
        binding = prsc.get_binding(concept)
        print(f"  {concept}: primes = {binding}")
    print()
    
    # ===== PART 3: Semantic Coherence =====
    print("PART 3: Semantic Coherence")
    print("-" * 40)
    
    # Coherence measures how related two concepts are
    # Based on overlap in their prime representations
    
    pairs = [
        ("mathematics", "physics"),      # High overlap (2, 5, 7)
        ("mathematics", "chemistry"),    # Medium overlap (3, 7, 11)
        ("physics", "chemistry"),        # Low overlap (7)
        ("mathematics", "art"),          # No overlap
    ]
    
    print("Semantic coherence between concepts:")
    for c1, c2 in pairs:
        coherence = prsc.compute_coherence(c1, c2)
        overlap_desc = "high" if coherence > 0.5 else "medium" if coherence > 0.2 else "low"
        print(f"  {c1} <-> {c2}: {coherence:.4f} ({overlap_desc})")
    print()
    
    # ===== PART 4: Semantic Distance =====
    print("PART 4: Semantic Distance")
    print("-" * 40)
    
    # Distance is inverse of coherence
    print("Semantic distances:")
    for c1, c2 in pairs:
        distance = prsc.semantic_distance(c1, c2)
        print(f"  d({c1}, {c2}) = {distance:.4f}")
    print()
    
    # ===== PART 5: Concept Composition =====
    print("PART 5: Concept Composition")
    print("-" * 40)
    
    # Compose multiple concepts into a unified representation
    
    composed = prsc.compose_concepts(["mathematics", "physics"])
    print(f"Composition of 'mathematics' + 'physics':")
    print(f"  {composed}")
    print()
    
    # Multi-concept composition
    interdisciplinary = prsc.compose_concepts([
        "mathematics", "physics", "chemistry"
    ])
    print(f"Interdisciplinary (math + physics + chemistry):")
    print(f"  {interdisciplinary}")
    print()
    
    # ===== PART 6: Resonance Strength =====
    print("PART 6: Resonance Strength")
    print("-" * 40)
    
    # Resonance strength indicates how "focused" a concept is
    # Fewer primes = higher resonance (more specific)
    
    print("Resonance strength of concepts:")
    for concept in prsc.list_concepts():
        strength = prsc.resonance_strength(concept)
        print(f"  {concept}: {strength:.4f}")
    print()
    
    # ===== PART 7: Binding PrimeStates =====
    print("PART 7: Binding PrimeStates (Advanced)")
    print("-" * 40)
    
    # Instead of just prime lists, bind full PrimeState objects
    # This allows for amplitude and phase information
    
    # Create states with phases
    state_logic = PrimeState.superposition(
        primes=[2, 3, 5],
        amplitudes=[
            Complex(0.6, 0),
            Complex(0.5, 0.3),
            Complex(0.4, 0.4)
        ]
    ).normalize()
    
    state_intuition = PrimeState.superposition(
        primes=[7, 11, 13],
        amplitudes=[
            Complex(0.5, 0.5),
            Complex(0.6, -0.2),
            Complex(0.4, 0.3)
        ]
    ).normalize()
    
    prsc.bind_state("logic", state_logic)
    prsc.bind_state("intuition", state_intuition)
    
    print("Bound states:")
    print(f"  logic: {prsc.get_state_binding('logic')}")
    print(f"  intuition: {prsc.get_state_binding('intuition')}")
    print()
    
    # Coherence between states
    coherence = prsc.state_coherence("logic", "intuition")
    print(f"Coherence(logic, intuition): {coherence:.4f}")
    print("  (Low because they use different primes)")
    print()
    
    # ===== PART 8: Semantic Reasoning =====
    print("PART 8: Semantic Reasoning")
    print("-" * 40)
    
    # Use PRSC for simple semantic reasoning
    # "If A is related to B, and B is related to C, is A related to C?"
    
    print("Semantic reasoning chain:")
    ab = prsc.compute_coherence("mathematics", "physics")
    bc = prsc.compute_coherence("physics", "chemistry")
    ac = prsc.compute_coherence("mathematics", "chemistry")
    
    print(f"  math <-> physics: {ab:.4f}")
    print(f"  physics <-> chemistry: {bc:.4f}")
    print(f"  math <-> chemistry: {ac:.4f}")
    
    # Transitive estimate (simplified)
    transitive_estimate = ab * bc
    print(f"  Transitive estimate (ab * bc): {transitive_estimate:.4f}")
    print(f"  Actual: {ac:.4f}")
    print()
    
    # ===== PART 9: Concept Unbinding =====
    print("PART 9: Concept Management")
    print("-" * 40)
    
    print(f"Concepts before unbind: {prsc.list_concepts()}")
    
    prsc.unbind_concept("art")
    print(f"After unbinding 'art': {prsc.list_concepts()}")
    
    # Check that unbinding worked
    binding = prsc.get_binding("art")
    print(f"  'art' binding: {binding}")
    print()
    
    # ===== PART 10: Clear All =====
    print("PART 10: Clear All Bindings")
    print("-" * 40)
    
    print(f"Concepts before clear: {len(prsc.list_concepts())}")
    prsc.clear_all()
    print(f"Concepts after clear: {len(prsc.list_concepts())}")
    print()
    
    # ===== PART 11: Practical Example - Knowledge Graph =====
    print("PART 11: Practical Example - Simple Knowledge Graph")
    print("-" * 40)
    
    # Create a fresh PRSC for knowledge graph demo
    kg = PrimeResonanceSemanticCoherence()
    
    # Define entities
    kg.bind_concept("dog", [2, 3, 5])
    kg.bind_concept("cat", [2, 3, 7])
    kg.bind_concept("animal", [2, 3])
    kg.bind_concept("mammal", [2, 3, 11])
    kg.bind_concept("pet", [2, 3, 13])
    kg.bind_concept("car", [17, 19, 23])
    kg.bind_concept("vehicle", [17, 19])
    
    print("Knowledge graph entities defined.")
    print("\nFinding related concepts for 'dog':")
    
    dog_coherences = {}
    for concept in kg.list_concepts():
        if concept != "dog":
            coh = kg.compute_coherence("dog", concept)
            dog_coherences[concept] = coh
    
    # Sort by coherence
    sorted_concepts = sorted(dog_coherences.items(), key=lambda x: -x[1])
    for concept, coh in sorted_concepts:
        bar = '█' * int(coh * 20)
        print(f"  {concept:10}: {coh:.4f} {bar}")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Prime Resonance Semantic Coherence")
    print("=" * 60)
    print("""
PRSC System Overview:

Concept Binding:
- Each concept maps to a set of primes
- Primes act as "semantic dimensions"
- Related concepts share primes

Coherence Computation:
- Coherence = overlap / union (Jaccard-like)
- High coherence = semantically related
- Zero coherence = unrelated concepts

State Binding:
- Bind full PrimeStates with amplitudes
- Enables phase-based semantics
- Supports quantum-inspired reasoning

Key Operations:
1. bind_concept(): Concept → Prime list
2. compute_coherence(): Similarity measure
3. compose_concepts(): Merge representations
4. semantic_distance(): Inverse of coherence
5. resonance_strength(): Focus measure

Applications:
- Knowledge representation
- Semantic similarity search
- Concept hierarchies
- Ontology alignment
- Question answering

Mathematical Foundation:
- Concepts live in prime Hilbert space
- Coherence = |⟨ψ|φ⟩|² for states
- Composition = superposition with normalization
    """)

if __name__ == "__main__":
    main()