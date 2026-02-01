#!/usr/bin/env python3
"""
Example 12: Advanced Physics - Stochastic Kuramoto & Lyapunov Analysis

Demonstrates advanced physics capabilities:
- Stochastic Kuramoto oscillators with noise
- Network-coupled oscillators
- Lyapunov exponent analysis for chaos detection
- Thermal and colored noise variants
"""

import sys
import math
import random
sys.path.insert(0, '..')

from tinyaleph.physics import (
    # Kuramoto
    KuramotoModel,
    # Stochastic variants
    StochasticKuramoto,
    ColoredNoiseKuramoto,
    ThermalKuramoto,
    # Network
    NetworkKuramoto,
    AdaptiveKuramoto,
    # Lyapunov
    estimate_lyapunov,
    LyapunovTracker,
    StabilityRegime,
    # Collapse
    collapse_probability,
    should_collapse,
    measure_state,
    CollapseMonitor,
)


def demonstrate_stochastic_kuramoto():
    """Demonstrate stochastic Kuramoto with noise."""
    print("=" * 60)
    print("STOCHASTIC KURAMOTO - Noisy Oscillator Dynamics")
    print("=" * 60)
    
    # Create stochastic Kuramoto system
    # Frequencies for oscillators (natural frequencies)
    import random
    n_oscillators = 10
    frequencies = [1.0 + 0.1 * random.gauss(0, 1) for _ in range(n_oscillators)]
    coupling = 2.0
    noise_intensity = 0.1
    
    kuramoto = StochasticKuramoto(
        frequencies=frequencies,
        coupling=coupling,
        noise_intensity=noise_intensity
    )
    
    print(f"\nSystem configuration:")
    print(f"  Oscillators: {n_oscillators}")
    print(f"  Coupling strength: {coupling}")
    print(f"  Noise intensity: {noise_intensity}")
    
    # Evolve and measure synchronization
    print("\nEvolution with noise:")
    for step in range(5):
        kuramoto.step(dt=0.1)
        order = abs(kuramoto.order_parameter())
        print(f"  Step {step}: Order parameter = {order:.4f}")
    
    # Final state
    phases = kuramoto.phases
    print(f"\nFinal phases: {[f'{p:.2f}' for p in phases[:5]]}...")
    
    return kuramoto


def demonstrate_colored_noise():
    """Demonstrate colored noise Kuramoto."""
    print("\n" + "=" * 60)
    print("COLORED NOISE KURAMOTO - Correlated Noise Dynamics")
    print("=" * 60)
    
    import random
    # Create colored noise variant
    frequencies = [1.0 + 0.1 * random.gauss(0, 1) for _ in range(8)]
    
    kuramoto = ColoredNoiseKuramoto(
        frequencies=frequencies,
        coupling=1.5,
        noise_intensity=0.2,
        correlation_time=0.5  # Slow noise
    )
    
    print(f"\nColored noise parameters:")
    print(f"  Correlation time: 0.5 (slow fluctuations)")
    print(f"  Noise intensity: 0.2")
    
    # Evolve colored noise system
    print("\nEvolution:")
    for step in range(5):
        kuramoto.step(dt=0.1)
        order = abs(kuramoto.order_parameter())
        print(f"  Step {step}: Order = {order:.4f}")
    
    return kuramoto


def demonstrate_thermal_kuramoto():
    """Demonstrate thermally-driven Kuramoto."""
    print("\n" + "=" * 60)
    print("THERMAL KURAMOTO - Temperature-Dependent Dynamics")
    print("=" * 60)
    
    import random
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("\nSynchronization vs Temperature:")
    print("  (Higher temperature → more disorder)")
    
    for temp in temperatures:
        frequencies = [1.0 + 0.1 * random.gauss(0, 1) for _ in range(20)]
        kuramoto = ThermalKuramoto(
            frequencies=frequencies,
            coupling=2.0,
            temperature=temp
        )
        
        # Equilibrate
        for _ in range(50):
            kuramoto.step(dt=0.05)
        
        order = abs(kuramoto.order_parameter())
        print(f"  T = {temp:.1f}: Order parameter = {order:.4f}")
    
    return kuramoto


def demonstrate_network_kuramoto():
    """Demonstrate network-coupled oscillators."""
    print("\n" + "=" * 60)
    print("NETWORK KURAMOTO - Graph-Structured Coupling")
    print("=" * 60)
    
    n = 10
    
    # Create different network topologies
    topologies = {
        'ring': create_ring_adjacency(n),
        'all-to-all': create_complete_adjacency(n),
        'random': create_random_adjacency(n, p=0.3),
    }
    
    print("\nComparing network topologies:")
    
    for name, adjacency in topologies.items():
        kuramoto = NetworkKuramoto(
            n_oscillators=n,
            adjacency=adjacency,
            base_coupling=1.0
        )
        
        # Evolve
        for _ in range(100):
            kuramoto.step(dt=0.1)
        
        order = kuramoto.order_parameter()
        edges = sum(sum(row) for row in adjacency) // 2
        print(f"  {name:12s}: edges={edges:3d}, order={order:.4f}")


def demonstrate_adaptive_kuramoto():
    """Demonstrate adaptive coupling dynamics."""
    print("\n" + "=" * 60)
    print("ADAPTIVE KURAMOTO - Self-Organizing Coupling")
    print("=" * 60)
    
    kuramoto = AdaptiveKuramoto(
        n_oscillators=8,
        initial_coupling=0.5,
        adaptation_rate=0.1
    )
    
    print("\nAdaptive coupling evolution:")
    print("  (Coupling adapts based on synchronization)")
    
    for step in range(10):
        kuramoto.step(dt=0.2)
        order = kuramoto.order_parameter()
        coupling = kuramoto.coupling
        print(f"  Step {step:2d}: coupling = {coupling:.3f}, order = {order:.4f}")
    
    return kuramoto


def demonstrate_lyapunov_analysis():
    """Demonstrate Lyapunov exponent computation."""
    print("\n" + "=" * 60)
    print("LYAPUNOV ANALYSIS - Chaos Detection")
    print("=" * 60)
    
    print("\nLyapunov exponent interpretations:")
    print("  λ > 0: Chaotic (exponential divergence)")
    print("  λ = 0: Quasiperiodic (neutral)")
    print("  λ < 0: Stable (exponential convergence)")
    
    # Analyze different Kuramoto regimes
    print("\nAnalyzing Kuramoto regimes:")
    
    regimes = [
        ("Weak coupling", 0.5),
        ("Critical coupling", 1.0),
        ("Strong coupling", 3.0),
    ]
    
    for name, coupling in regimes:
        kuramoto = StochasticKuramoto(
            n_oscillators=10,
            coupling=coupling,
            noise_strength=0.01
        )
        
        # Collect time series
        time_series = []
        for _ in range(100):
            kuramoto.step(dt=0.1)
            time_series.append(kuramoto.order_parameter())
        
        # Estimate Lyapunov exponent
        lyap = estimate_lyapunov(time_series)
        
        stability = "chaotic" if lyap > 0.01 else "stable" if lyap < -0.01 else "neutral"
        print(f"  {name:20s}: λ = {lyap:+.4f} ({stability})")
    
    # Using LyapunovTracker
    print("\nUsing LyapunovTracker:")
    tracker = LyapunovTracker()
    
    kuramoto = StochasticKuramoto(n_oscillators=5, coupling=2.0)
    for _ in range(50):
        kuramoto.step(0.1)
        tracker.update(kuramoto.order_parameter())
    
    print(f"  Tracked exponent: {tracker.current_estimate:.4f}")
    print(f"  Regime: {tracker.current_regime}")


def demonstrate_collapse():
    """Demonstrate quantum collapse mechanics."""
    print("\n" + "=" * 60)
    print("COLLAPSE MECHANICS - Quantum Measurement")
    print("=" * 60)
    
    from tinyaleph.hilbert import PrimeState
    from tinyaleph.core import Complex
    
    # Create superposition state
    amplitudes = [Complex(0.5), Complex(0.3), Complex(0.15), Complex(0.05)]
    state = PrimeState(amplitudes=amplitudes, primes=[2, 3, 5, 7])
    
    print(f"\nInitial superposition:")
    for i, (prime, amp) in enumerate(zip([2, 3, 5, 7], amplitudes)):
        prob = amp.abs() ** 2
        print(f"  |p={prime}⟩: amplitude={amp.real:.2f}, probability={prob:.2f}")
    
    # Check collapse probability at different entropies
    print("\nCollapse probability vs entropy:")
    for entropy in [0.2, 0.5, 0.8, 1.0]:
        prob = collapse_probability(entropy, threshold=0.7)
        will_collapse = should_collapse(entropy, threshold=0.7)
        print(f"  H={entropy:.1f}: P(collapse)={prob:.3f}, will_collapse={will_collapse}")
    
    # Use collapse monitor
    print("\nCollapse monitoring:")
    monitor = CollapseMonitor(threshold=0.5)
    
    for step in range(5):
        entropy = 0.3 + step * 0.2
        monitor.update(entropy)
        status = "COLLAPSED" if monitor.has_collapsed else "superposed"
        print(f"  Step {step}: entropy={entropy:.2f}, status={status}")


# Helper functions for network topologies
def create_ring_adjacency(n):
    """Create ring topology adjacency matrix."""
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        adj[i][(i + 1) % n] = 1
        adj[i][(i - 1) % n] = 1
    return adj


def create_complete_adjacency(n):
    """Create complete graph adjacency matrix."""
    return [[1 if i != j else 0 for j in range(n)] for i in range(n)]


def create_random_adjacency(n, p=0.5):
    """Create Erdős-Rényi random graph."""
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i][j] = 1
                adj[j][i] = 1
    return adj


def main():
    """Run all advanced physics demonstrations."""
    print("ALEPH PRIME - ADVANCED PHYSICS EXAMPLES")
    print("=" * 60)
    
    demonstrate_stochastic_kuramoto()
    demonstrate_colored_noise()
    demonstrate_thermal_kuramoto()
    demonstrate_network_kuramoto()
    demonstrate_adaptive_kuramoto()
    demonstrate_lyapunov_analysis()
    demonstrate_collapse()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Advanced physics capabilities:
- Stochastic Kuramoto with white/colored/thermal noise
- Network-coupled oscillators on arbitrary graphs
- Adaptive coupling that self-organizes
- Lyapunov exponent analysis for chaos detection
- Quantum collapse mechanics

Key applications:
- Neural synchronization modeling
- Network dynamics analysis  
- Stability analysis of dynamical systems
- Quantum measurement simulation
""")


if __name__ == "__main__":
    main()