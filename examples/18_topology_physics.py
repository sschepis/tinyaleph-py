#!/usr/bin/env python3
"""
Example 18: Topology and Physics Constants

Demonstrates the topology module:
- Knot invariants (Jones polynomial, writhe)
- Physical constants from prime structure
- Gauge symmetry (U(1), SU(2), SU(3))
- Free energy dynamics (Friston's FEP)
- Topological data analysis
"""

import sys
import math
sys.path.insert(0, '..')

from tinyaleph.semantic import (
    # Knot theory
    Knot, KnotDiagram, Crossing, CrossingSign,
    # Physical constants
    PhysicalConstants,
    # Gauge theory
    GaugeGroup, GaugeField, GaugeSymmetry,
    # Free energy
    BeliefState, Observation, FreeEnergyDynamics,
    # TDA
    TopologicalFeatures,
    # Utilities
    create_semantic_knot, analyze_semantic_topology,
    derive_physical_constant, free_energy_update
)


def demonstrate_knot_theory():
    """Demonstrate knot theory calculations."""
    print("=" * 60)
    print("KNOT THEORY")
    print("=" * 60)
    
    print("\nKnot invariants capture 'tangled' relationships")
    print("that cannot be untangled without cutting.")
    
    # Unknot (trivial)
    print("\n1. UNKNOT (trivial knot)")
    unknot = Knot.unknot()
    print(f"   Crossing number: {unknot.crossing_number()}")
    print(f"   Writhe: {unknot.writhe()}")
    print(f"   Is unknot: {unknot.unknot_detection()}")
    
    # Trefoil (simplest non-trivial)
    print("\n2. TREFOIL (simplest non-trivial)")
    trefoil = Knot.trefoil()
    print(f"   Braid word: σ₁³ = [1, 1, 1]")
    print(f"   Crossing number: {trefoil.crossing_number()}")
    print(f"   Writhe: {trefoil.writhe()}")
    print(f"   Is unknot: {trefoil.unknot_detection()}")
    
    # Jones polynomial
    jones = trefoil.jones_polynomial()
    print(f"   Jones polynomial (coefficients): {jones}")
    
    # Figure-8 (amphicheiral)
    print("\n3. FIGURE-8 (amphicheiral)")
    fig8 = Knot.figure_eight()
    print(f"   Braid word: σ₁σ₂⁻¹σ₁σ₂⁻¹ = [1, -2, 1, -2]")
    print(f"   Crossing number: {fig8.crossing_number()}")
    print(f"   Writhe: {fig8.writhe()}")  # Should be 0 for amphicheiral
    print(f"   Note: Writhe=0 indicates amphicheiral symmetry")
    
    # Semantic tangle complexity
    print("\n4. SEMANTIC TANGLE COMPLEXITY")
    knots = [unknot, trefoil, fig8]
    names = ["Unknot", "Trefoil", "Figure-8"]
    
    for name, knot in zip(names, knots):
        complexity = knot.semantic_tangle_complexity()
        print(f"   {name}: complexity = {complexity:.2f}")
    
    return trefoil


def demonstrate_physical_constants():
    """Demonstrate derivation of physical constants from primes."""
    print("\n" + "=" * 60)
    print("PHYSICAL CONSTANTS FROM PRIMES")
    print("=" * 60)
    
    pc = PhysicalConstants(precision=50)
    
    print("\nHypothesis: Fundamental constants emerge from prime structure")
    
    # Fine structure constant
    print("\n1. FINE STRUCTURE CONSTANT α")
    alpha = pc.fine_structure_alpha()
    print(f"   α = 1/137 ≈ {alpha:.10f}")
    print(f"   Note: 137 is the 33rd prime!")
    
    analysis = pc.analyze_137()
    print(f"   137 is prime: {analysis['is_prime']}")
    print(f"   Prime index: p_{analysis['prime_index']}")
    print(f"   Actual α: {analysis['actual_alpha']:.10f}")
    print(f"   Discrepancy: {analysis['discrepancy']*100:.4f}%")
    
    # Proton/electron mass ratio
    print("\n2. PROTON/ELECTRON MASS RATIO")
    ratio = pc.proton_electron_ratio()
    actual = 1836.15267
    print(f"   Derived: {ratio:.2f}")
    print(f"   Actual: {actual}")
    print(f"   Error: {abs(ratio - actual) / actual * 100:.2f}%")
    
    # π from primes
    print("\n3. π FROM EULER PRODUCT")
    pi_approx = pc.pi_from_primes(num_terms=50)
    print(f"   Approximation: {pi_approx:.10f}")
    print(f"   Actual π: {math.pi:.10f}")
    print(f"   Error: {abs(pi_approx - math.pi) / math.pi * 100:.4f}%")
    
    # Golden ratio from Fibonacci primes
    print("\n4. GOLDEN RATIO FROM FIBONACCI PRIMES")
    phi_approx = pc.golden_ratio_approximation()
    print(f"   Consecutive Fibonacci primes ratios:")
    for f1, f2, ratio in phi_approx[:4]:
        print(f"      {f2}/{f1} = {ratio:.6f}")
    print(f"   Actual φ = {(1 + math.sqrt(5))/2:.6f}")
    
    # Coupling constants
    print("\n5. COUPLING CONSTANTS FROM PRIME PRODUCTS")
    signatures = [
        ([1, 2, 3], "1/(p₁ × p₂ × p₃)"),
        ([5, 10, 15], "sparse primes"),
        ([33], "1/p₃₃ = 1/137"),
    ]
    
    for sig, desc in signatures:
        coupling = pc.coupling_constant_from_primes(sig)
        print(f"   {desc}: {coupling:.10f}")
    
    # Mass hierarchy
    print("\n6. MASS HIERARCHY FROM PRIME GAPS")
    hierarchy = pc.mass_hierarchy_from_gaps()
    print(f"   Hierarchy (first 5 levels): {[f'{h:.2f}' for h in hierarchy[:5]]}")
    
    return pc


def demonstrate_gauge_symmetry():
    """Demonstrate gauge symmetry structures."""
    print("\n" + "=" * 60)
    print("GAUGE SYMMETRY")
    print("=" * 60)
    
    print("\nGauge symmetry represents local redundancy in descriptions.")
    
    # U(1) symmetry
    print("\n1. U(1) SYMMETRY (Phase)")
    u1 = GaugeSymmetry(GaugeGroup.U1, dimension=1)
    
    state = [complex(1, 0)]
    theta = math.pi / 4
    
    transformed = u1.gauge_transform(state, [theta])
    print(f"   Original: {state[0]}")
    print(f"   After phase rotation by π/4:")
    print(f"   Transformed: {transformed[0]:.4f}")
    
    # SU(2) symmetry
    print("\n2. SU(2) SYMMETRY (Weak Isospin)")
    su2 = GaugeSymmetry(GaugeGroup.SU2, dimension=2)
    
    print(f"   Generators (Pauli/2):")
    for i, gen in enumerate(su2.generators):
        print(f"      T_{i+1} = {gen}")
    
    # Doublet transformation
    doublet = [complex(1, 0), complex(0, 0)]  # |up⟩
    params = [0.1, 0.0, 0.0]  # Small rotation around x
    
    transformed_doublet = su2.gauge_transform(doublet, params)
    print(f"\n   Doublet |↑⟩: {doublet}")
    print(f"   After small SU(2) rotation:")
    print(f"   Result: {[f'{c:.4f}' for c in transformed_doublet]}")
    
    # SU(3) symmetry
    print("\n3. SU(3) SYMMETRY (Color)")
    su3 = GaugeSymmetry(GaugeGroup.SU3, dimension=3)
    
    print(f"   Number of generators (Gell-Mann): 8")
    print(f"   (Showing 2 diagonal generators)")
    
    # Anomaly coefficients
    print("\n4. ANOMALY COEFFICIENTS")
    groups = [(GaugeGroup.U1, 1), (GaugeGroup.SU2, 2), (GaugeGroup.SU3, 3)]
    
    for group, dim in groups:
        gs = GaugeSymmetry(group, dim)
        anomaly = gs.anomaly_coefficient()
        print(f"   {group.value}: anomaly = {anomaly}")
    
    return su2


def demonstrate_free_energy():
    """Demonstrate Free Energy Principle dynamics."""
    print("\n" + "=" * 60)
    print("FREE ENERGY PRINCIPLE")
    print("=" * 60)
    
    print("\nFriston's Free Energy Principle:")
    print("  Biological systems minimize variational free energy F")
    print("  F = Complexity + Inaccuracy")
    print("  F = D_KL(q||p) - ⟨ln p(o|x)⟩")
    
    # Create agent
    dynamics = FreeEnergyDynamics(
        state_dim=4,
        learning_rate=0.1,
        precision_learning=0.01
    )
    
    print(f"\n1. INITIAL BELIEF STATE")
    print(f"   Mean: {dynamics.belief.mean}")
    print(f"   Precision: {dynamics.belief.precision}")
    print(f"   Entropy: {dynamics.belief.entropy():.4f}")
    
    # Present observations
    print("\n2. BELIEF UPDATING WITH OBSERVATIONS")
    
    observations = [
        Observation([1.0, 0.5, -0.5, 0.0], [2.0, 2.0, 2.0, 2.0]),
        Observation([0.8, 0.4, -0.3, 0.1], [2.0, 2.0, 2.0, 2.0]),
        Observation([0.9, 0.6, -0.4, 0.0], [2.0, 2.0, 2.0, 2.0]),
    ]
    
    for i, obs in enumerate(observations):
        F_before = dynamics.variational_free_energy(obs)
        delta_F = dynamics.gradient_descent_step(obs)
        F_after = dynamics.variational_free_energy(obs)
        
        print(f"\n   Observation {i+1}: {obs.value}")
        print(f"   F before: {F_before:.4f}, F after: {F_after:.4f}")
        print(f"   ΔF = {delta_F:.4f}")
        print(f"   Belief mean: {[f'{m:.3f}' for m in dynamics.belief.mean]}")
    
    # Surprise
    print("\n3. SURPRISE (UNEXPECTEDNESS)")
    
    expected_obs = Observation([0.9, 0.5, -0.4, 0.0], [2.0, 2.0, 2.0, 2.0])
    surprise_low = dynamics.surprise(expected_obs)
    
    unexpected_obs = Observation([5.0, -3.0, 2.0, 1.0], [2.0, 2.0, 2.0, 2.0])
    surprise_high = dynamics.surprise(unexpected_obs)
    
    print(f"   Expected observation surprise: {surprise_low:.4f}")
    print(f"   Unexpected observation surprise: {surprise_high:.4f}")
    
    # Active inference
    print("\n4. ACTIVE INFERENCE (Action Selection)")
    
    action = dynamics.active_inference_action(expected_obs)
    print(f"   Computed action: {[f'{a:.3f}' for a in action]}")
    print(f"   (Action minimizes expected prediction error)")
    
    # Analysis
    print("\n5. FREE ENERGY ANALYSIS")
    analysis = dynamics.get_analysis()
    print(f"   Knowledge count: {len(dynamics.free_energy_history)}")
    print(f"   Mean F: {analysis['mean_free_energy']:.4f}")
    print(f"   Belief entropy: {analysis['belief_entropy']:.4f}")
    
    return dynamics


def demonstrate_topological_features():
    """Demonstrate topological data analysis."""
    print("\n" + "=" * 60)
    print("TOPOLOGICAL DATA ANALYSIS")
    print("=" * 60)
    
    tda = TopologicalFeatures(max_dimension=2)
    
    # Create point cloud
    print("\n1. POINT CLOUD ANALYSIS")
    
    # Circle-like points
    import math
    n_points = 8
    points = [
        [math.cos(2 * math.pi * i / n_points),
         math.sin(2 * math.pi * i / n_points)]
        for i in range(n_points)
    ]
    
    print(f"   Points: {n_points} points on a circle")
    
    # Distance matrix
    dist = tda.distance_matrix(points)
    print(f"   Distance matrix computed (max dist: {max(max(row) for row in dist):.2f})")
    
    # Vietoris-Rips complex at different scales
    print("\n2. VIETORIS-RIPS COMPLEXES")
    
    scales = [0.5, 1.0, 1.5, 2.0]
    
    for epsilon in scales:
        v, e, t = tda.vietoris_rips_complex(dist, epsilon)
        betti = tda.betti_numbers(v, e, t)
        print(f"   ε = {epsilon:.1f}: vertices={len(v)}, edges={len(e)}, triangles={len(t)}")
        print(f"          β₀={betti[0]}, β₁={betti[1]}")
    
    # Persistence diagram
    print("\n3. PERSISTENCE DIAGRAM")
    
    diagram = tda.persistence_diagram(points, num_scales=10)
    print(f"   Features detected: {len(diagram)}")
    
    for birth, death, dim in diagram[:5]:
        lifetime = death - birth if death < float('inf') else "∞"
        print(f"   H_{dim}: born at {birth:.2f}, dies at {lifetime}")
    
    total_pers = tda.total_persistence(diagram)
    print(f"\n   Total persistence: {total_pers:.4f}")
    
    return tda


def demonstrate_semantic_knot():
    """Demonstrate semantic knot construction."""
    print("\n" + "=" * 60)
    print("SEMANTIC KNOT CONSTRUCTION")
    print("=" * 60)
    
    print("\nCreating knots from semantic relations:")
    print("  Each (subject, verb, object) triple becomes a braid generator")
    
    # Example relations
    relations = [
        (2, 1, 3),   # "Entity2 relation1 Entity3"
        (3, 2, 5),   # "Entity3 relation2 Entity5"
        (5, 1, 2),   # "Entity5 relation1 Entity2" (cycle!)
    ]
    
    print(f"\nRelations: {relations}")
    
    knot = create_semantic_knot(relations)
    
    print(f"\nResulting knot:")
    print(f"   Braid word: {knot.braid_word}")
    print(f"   Crossing number: {knot.crossing_number()}")
    print(f"   Writhe: {knot.writhe()}")
    print(f"   Tangle complexity: {knot.semantic_tangle_complexity():.2f}")
    
    # Interpretation
    print("\nInterpretation:")
    if knot.writhe() != 0:
        print("   Non-zero writhe → asymmetric relationships")
    if knot.crossing_number() > 0:
        print("   Crossings present → tangled semantic structure")
        print("   (Cannot simplify without 'cutting' relations)")
    
    return knot


def main():
    """Run all topology demonstrations."""
    print("ALEPH PRIME - TOPOLOGY AND PHYSICS EXAMPLES")
    print("=" * 60)
    
    demonstrate_knot_theory()
    demonstrate_physical_constants()
    demonstrate_gauge_symmetry()
    demonstrate_free_energy()
    demonstrate_topological_features()
    demonstrate_semantic_knot()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Topology module provides:
- Knot invariants (crossing number, writhe, Jones polynomial)
- Physical constants from prime structure
- Gauge symmetry (U(1), SU(2), SU(3))
- Free Energy Principle dynamics
- Topological data analysis

Key insights:
- 137 (fine structure) is the 33rd prime
- Knots model tangled semantic relationships
- Gauge symmetry = local semantic redundancy
- Free energy minimization = belief updating

Applications:
- Semantic complexity analysis
- Physical constant derivation
- Belief dynamics modeling
- Shape analysis of data
""")


if __name__ == "__main__":
    main()