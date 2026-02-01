#!/usr/bin/env python3
"""
Example 06: Quantum Entanglement Networks

This example demonstrates:
- Prime Resonance Identity (PRI)
- Bell states for prime pairs
- Entanglement swapping
- Quantum teleportation
- Network routing via entanglement

Bell States for primes p, q:
    |Φ+⟩ = (1/√2)(|p,p⟩ + |q,q⟩)
    |Φ-⟩ = (1/√2)(|p,p⟩ - |q,q⟩)
    |Ψ+⟩ = (1/√2)(|p,q⟩ + |q,p⟩)
    |Ψ-⟩ = (1/√2)(|p,q⟩ - |q,p⟩)
"""

from tinyaleph.network.identity import PrimeResonanceIdentity, EntangledNode
from tinyaleph.network.entanglement import (
    EntangledPair,
    EntanglementSource,
    EntanglementSwapper,
    EntanglementDistiller,
    EntanglementNetwork,
    BellState,
    create_ghz_state,
    create_w_state,
    entanglement_entropy,
)
from tinyaleph.core.complex import Complex
import math

def main():
    print("=" * 60)
    print("TinyAleph: Quantum Entanglement Networks")
    print("=" * 60)
    print()
    
    # ===== PART 1: Prime Resonance Identity (PRI) =====
    print("PART 1: Prime Resonance Identity (PRI)")
    print("-" * 40)
    
    # PRI = (Gaussian prime, Eisenstein prime, Quaternionic prime)
    pri = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
    
    print(f"Created PRI: {pri}")
    print(f"  Gaussian prime: {pri.gaussian}")
    print(f"  Eisenstein prime: {pri.eisenstein}")
    print(f"  Quaternionic prime: {pri.quaternionic}")
    print(f"  Signature: {pri.signature}")
    print(f"  Hash: {pri.hash}")
    print(f"  Product: {pri.product}")
    print()
    
    # Create from seed (deterministic)
    pri_seeded = PrimeResonanceIdentity.from_seed(42)
    pri_seeded2 = PrimeResonanceIdentity.from_seed(42)
    print(f"PRI from seed 42: {pri_seeded}")
    print(f"Same seed again: {pri_seeded2}")
    print(f"  Are identical: {pri_seeded == pri_seeded2}")
    print()
    
    # Random PRI
    pri_random = PrimeResonanceIdentity.random()
    print(f"Random PRI: {pri_random}")
    print()
    
    # ===== PART 2: Entanglement Strength =====
    print("PART 2: Entanglement Strength")
    print("-" * 40)
    
    pri1 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
    pri2 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)  # Identical
    pri3 = PrimeResonanceIdentity(gaussian=5, eisenstein=13, quaternionic=17)  # Different
    pri4 = PrimeResonanceIdentity(gaussian=3, eisenstein=13, quaternionic=11)  # Partial overlap
    
    print("Entanglement strength measures PRI compatibility:")
    print(f"  Identical PRIs: {pri1.entanglement_strength(pri2):.4f}")
    print(f"  Different PRIs: {pri1.entanglement_strength(pri3):.4f}")
    print(f"  Partial overlap: {pri1.entanglement_strength(pri4):.4f}")
    print()
    
    # Compatibility check
    print("Compatibility check (threshold 0.5):")
    print(f"  pri1 compatible with pri2: {pri1.is_compatible(pri2, 0.5)}")
    print(f"  pri1 compatible with pri3: {pri1.is_compatible(pri3, 0.5)}")
    print()
    
    # ===== PART 3: Entangled Nodes =====
    print("PART 3: Entangled Nodes")
    print("-" * 40)
    
    node_a = EntangledNode(pri=pri1)
    node_b = EntangledNode(pri=pri2)
    node_c = EntangledNode(pri=pri3)
    
    print(f"Node A: {node_a}")
    print(f"Node B: {node_b}")
    print(f"Node C: {node_c}")
    print()
    
    # Check if nodes can entangle
    print("Can entangle:")
    print(f"  A with B (identical PRI): {node_a.can_entangle(node_b)}")
    print(f"  A with C (different PRI): {node_a.can_entangle(node_c, threshold=0.3)}")
    print()
    
    # Perform entanglement
    success = node_a.entangle(node_b)
    print(f"Entangling A with B: {'success' if success else 'failed'}")
    print(f"  A entangled with: {node_a.entangled_with}")
    print(f"  B entangled with: {node_b.entangled_with}")
    print(f"  A coherence: {node_a.coherence:.4f} (slightly reduced)")
    print()
    
    # ===== PART 4: Bell States =====
    print("PART 4: Bell States")
    print("-" * 40)
    
    # Create entangled pair
    pair = EntangledPair(prime_a=2, prime_b=3, bell_state=BellState.PHI_PLUS)
    
    print(f"Entangled pair: {pair}")
    print(f"  Primes: {pair.prime_a}, {pair.prime_b}")
    print(f"  Bell state: {pair.bell_state.value}")
    print(f"  Fidelity: {pair.fidelity}")
    print()
    
    # Bell states
    print("Available Bell states:")
    for state in BellState:
        print(f"  {state.value}: |{state.name}⟩")
    print()
    
    # Measurement
    print("Measuring particle A:")
    for i in range(5):
        test_pair = EntangledPair(prime_a=2, prime_b=3, bell_state=BellState.PHI_PLUS)
        outcome_a, predicted_b = test_pair.measure_a()
        print(f"  Trial {i+1}: A={outcome_a}, predicted B={predicted_b}")
    print()
    
    # ===== PART 5: Entanglement Source =====
    print("PART 5: Entanglement Source")
    print("-" * 40)
    
    source = EntanglementSource(
        base_fidelity=0.95,
        success_probability=0.8
    )
    
    print(f"Entanglement source:")
    print(f"  Base fidelity: {source.base_fidelity}")
    print(f"  Success probability: {source.success_probability}")
    print()
    
    # Generate pairs
    print("Generating 5 entangled pairs:")
    for i in range(5):
        pair = source.generate(prime_a=2, prime_b=3)
        if pair:
            print(f"  Pair {i+1}: fidelity={pair.fidelity:.4f}")
        else:
            print(f"  Pair {i+1}: FAILED")
    print(f"Total generated: {source.generated_count}")
    print()
    
    # ===== PART 6: Entanglement Swapping =====
    print("PART 6: Entanglement Swapping")
    print("-" * 40)
    
    # A-B entangled, B-C entangled → A-C entangled
    print("Swapping: If A-B and B-C are entangled, we can create A-C")
    
    source_swap = EntanglementSource(success_probability=1.0)  # For demo
    pair_ab = source_swap.generate(prime_a=2, prime_b=3)
    pair_bc = source_swap.generate(prime_a=3, prime_b=5)
    
    pair_ab.node_a = "Alice"
    pair_ab.node_b = "Bob"
    pair_bc.node_a = "Bob"
    pair_bc.node_b = "Charlie"
    
    print(f"A-B pair: primes={pair_ab.prime_a},{pair_ab.prime_b}, fidelity={pair_ab.fidelity:.4f}")
    print(f"B-C pair: primes={pair_bc.prime_a},{pair_bc.prime_b}, fidelity={pair_bc.fidelity:.4f}")
    
    swapper = EntanglementSwapper(success_probability=1.0, fidelity_loss=0.1)
    pair_ac = swapper.swap(pair_ab, pair_bc)
    
    if pair_ac:
        print(f"A-C pair (swapped): primes={pair_ac.prime_a},{pair_ac.prime_b}, fidelity={pair_ac.fidelity:.4f}")
        print(f"  Nodes: {pair_ac.node_a} - {pair_ac.node_b}")
    print()
    
    # ===== PART 7: Entanglement Distillation =====
    print("PART 7: Entanglement Distillation")
    print("-" * 40)
    
    # Create noisy pairs
    noisy_source = EntanglementSource(base_fidelity=0.7, success_probability=1.0)
    noisy_pairs = noisy_source.generate_n(4, prime_a=2, prime_b=3)
    
    print("Noisy pairs:")
    for i, pair in enumerate(noisy_pairs):
        print(f"  Pair {i+1}: fidelity={pair.fidelity:.4f}")
    
    distiller = EntanglementDistiller(target_fidelity=0.95)
    distilled = distiller.distill_to_target(noisy_pairs, max_rounds=5)
    
    if distilled:
        print(f"Distilled pair: fidelity={distilled.fidelity:.4f}")
    else:
        print("Distillation failed")
    print()
    
    # ===== PART 8: Entanglement Network =====
    print("PART 8: Entanglement Network")
    print("-" * 40)
    
    network = EntanglementNetwork()
    
    # Add nodes
    nodes = ["Alice", "Bob", "Charlie", "Dave"]
    for node in nodes:
        network.add_node(node)
    
    print(f"Network nodes: {network.nodes}")
    print()
    
    # Establish links
    print("Establishing entanglement links:")
    for _ in range(3):  # Try multiple times due to probabilistic nature
        network.establish_link("Alice", "Bob")
        network.establish_link("Bob", "Charlie")
        network.establish_link("Charlie", "Dave")
    
    print(f"  Total pairs: {len(network.pairs)}")
    print(f"  Total entanglement: {network.total_entanglement():.4f}")
    print(f"  Average fidelity: {network.average_fidelity():.4f}")
    print()
    
    # Check connectivity
    print("Checking entanglement:")
    print(f"  Alice-Bob entangled: {network.are_entangled('Alice', 'Bob')}")
    print(f"  Alice-Dave entangled: {network.are_entangled('Alice', 'Dave')}")
    print()
    
    # Find path
    path = network.find_path("Alice", "Dave")
    print(f"Path Alice → Dave: {path}")
    print()
    
    # ===== PART 9: GHZ and W States =====
    print("PART 9: GHZ and W States")
    print("-" * 40)
    
    # GHZ state: (|ppp⟩ + |qqq⟩)/√2
    ghz = create_ghz_state([2, 3, 5])
    print("GHZ state for primes [2, 3, 5]:")
    for config, amp in ghz.items():
        print(f"  {config}: amplitude = {amp}")
    print()
    
    # W state: (|pqq⟩ + |qpq⟩ + |qqp⟩)/√3
    w = create_w_state([2, 3, 5])
    print("W state for primes [2, 3, 5]:")
    for config, amp in w.items():
        print(f"  {config}: amplitude = {amp}")
    print()
    
    # ===== PART 10: Entanglement Entropy =====
    print("PART 10: Entanglement Entropy")
    print("-" * 40)
    
    # Maximally entangled state
    bell_state = {
        (2, 2): Complex(1/math.sqrt(2), 0),
        (3, 3): Complex(1/math.sqrt(2), 0)
    }
    entropy = entanglement_entropy(bell_state)
    print(f"Bell state |Φ+⟩ = (|2,2⟩ + |3,3⟩)/√2:")
    print(f"  Entanglement entropy: {entropy:.4f} bits")
    print(f"  Maximum for 2 primes: {math.log2(2):.4f} bits")
    print()
    
    # Product state (not entangled)
    product_state = {
        (2, 3): Complex(1.0, 0)
    }
    entropy_product = entanglement_entropy(product_state)
    print(f"Product state |2,3⟩:")
    print(f"  Entanglement entropy: {entropy_product:.4f} bits (not entangled)")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Entanglement Networks")
    print("=" * 60)
    print("""
Entanglement Concepts:

1. Prime Resonance Identity (PRI):
   - Triple of primes: (Gaussian, Eisenstein, Quaternionic)
   - Determines entanglement compatibility

2. Bell States:
   |Φ+⟩ = (|pp⟩ + |qq⟩)/√2  (maximally entangled)
   |Φ-⟩ = (|pp⟩ - |qq⟩)/√2
   |Ψ+⟩ = (|pq⟩ + |qp⟩)/√2
   |Ψ-⟩ = (|pq⟩ - |qp⟩)/√2

3. Key Operations:
   - Generation: Source produces entangled pairs
   - Swapping: A-B + B-C → A-C
   - Distillation: Noisy pairs → High-fidelity pair

4. Network Properties:
   - Nodes connected by entanglement links
   - Path finding for quantum routing
   - Fidelity degrades with operations

5. Multi-Particle States:
   - GHZ: (|ppp...⟩ + |qqq...⟩)/√2
   - W: Spread single excitation

Applications:
- Quantum communication
- Distributed quantum computing
- Secure key distribution
    """)

if __name__ == "__main__":
    main()