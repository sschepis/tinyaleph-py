#!/usr/bin/env python3
"""
Example 04: Kuramoto Coupled Oscillator Synchronization

This example demonstrates the Kuramoto model for coupled oscillators:
- Phase synchronization dynamics
- Order parameter and critical coupling
- Phase transitions
- Entropy analysis

The Kuramoto model describes how coupled oscillators synchronize:
    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)

where:
- θ_i is the phase of oscillator i
- ω_i is the natural frequency of oscillator i
- K is the coupling strength
- N is the number of oscillators
"""

import numpy as np
from tinyaleph.physics.kuramoto import KuramotoModel

def main():
    print("=" * 60)
    print("TinyAleph: Kuramoto Coupled Oscillator Model")
    print("=" * 60)
    print()
    
    # ===== PART 1: Basic Kuramoto Model =====
    print("PART 1: Basic Kuramoto Model")
    print("-" * 40)
    
    # Create a population of 50 coupled oscillators
    n_oscillators = 50
    coupling = 2.0
    
    model = KuramotoModel(n_oscillators=n_oscillators, coupling=coupling)
    
    print(f"Created Kuramoto model:")
    print(f"  Number of oscillators: {model.n_oscillators}")
    print(f"  Coupling strength K: {model.coupling}")
    print(f"  Frequency distribution: mean={np.mean(model.frequencies):.3f}, std={np.std(model.frequencies):.3f}")
    print()
    
    # Initial state
    print("Initial state:")
    print(f"  Synchronization r: {model.synchronization():.4f}")
    print(f"  Mean phase ψ: {np.degrees(model.mean_phase()):.1f}°")
    print(f"  Phase entropy: {model.entropy():.4f}")
    print()
    
    # ===== PART 2: Simulating Time Evolution =====
    print("PART 2: Simulating Time Evolution")
    print("-" * 40)
    
    print("Evolving the system...")
    
    # Simulate for 10 time units
    duration = 10.0
    dt = 0.01
    history = model.simulate(duration=duration, dt=dt)
    
    print(f"Simulated {duration} time units with dt={dt}")
    print(f"  Steps taken: {len(history)}")
    print(f"  Initial r: {history[0]:.4f}")
    print(f"  Final r: {history[-1]:.4f}")
    print()
    
    # Show synchronization progress
    print("Synchronization over time:")
    checkpoints = [0, len(history)//4, len(history)//2, 3*len(history)//4, len(history)-1]
    for i in checkpoints:
        t = i * dt
        r = history[i]
        print(f"  t={t:.2f}: r={r:.4f} {'■' * int(r * 20)}")
    print()
    
    # ===== PART 3: Order Parameter =====
    print("PART 3: Order Parameter")
    print("-" * 40)
    
    # The order parameter z = r * e^(iψ) measures collective synchronization
    z = model.order_parameter()
    r = abs(z)
    psi = np.angle(z)
    
    print(f"Order parameter z = r * exp(iψ):")
    print(f"  z = {z.real:.4f} + {z.imag:.4f}i")
    print(f"  Magnitude r = {r:.4f}")
    print(f"  Phase ψ = {np.degrees(psi):.1f}°")
    print()
    
    print("Interpretation of r:")
    print(f"  r ≈ 0: Incoherent (random phases)")
    print(f"  r ≈ 1: Coherent (synchronized)")
    print(f"  Current: r = {r:.4f}")
    if r > 0.7:
        print("  → Strong synchronization!")
    elif r > 0.3:
        print("  → Partial synchronization")
    else:
        print("  → Weak or no synchronization")
    print()
    
    # ===== PART 4: Phase Transition =====
    print("PART 4: Phase Transition vs Coupling Strength")
    print("-" * 40)
    
    # The Kuramoto model exhibits a phase transition at critical coupling K_c
    # Below K_c: no synchronization
    # Above K_c: partial synchronization
    
    print("Scanning coupling strengths...")
    coupling_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    for K in coupling_values:
        # Create model with this coupling
        test_model = KuramotoModel(n_oscillators=50, coupling=K)
        
        # Simulate to steady state
        test_model.simulate(duration=20.0, dt=0.05, method="euler")
        
        r_final = test_model.synchronization()
        bar = '█' * int(r_final * 20)
        print(f"  K={K:.1f}: r={r_final:.4f} {bar}")
    
    # Estimate critical coupling
    k_c = model.critical_coupling()
    print(f"\nEstimated critical coupling K_c ≈ {k_c:.3f}")
    print("  (Gaussian frequencies: K_c ≈ 2/(π·g(0)) where g(0) is peak of distribution)")
    print()
    
    # ===== PART 5: Different Frequency Distributions =====
    print("PART 5: Different Frequency Distributions")
    print("-" * 40)
    
    # Uniform frequencies
    uniform_model = KuramotoModel.with_uniform_frequencies(
        n_oscillators=50, coupling=2.0, freq_range=1.0
    )
    uniform_model.simulate(duration=20.0, dt=0.05)
    print(f"Uniform frequencies [-1, 1]: final r = {uniform_model.synchronization():.4f}")
    
    # Lorentzian frequencies (fatter tails)
    lorentz_model = KuramotoModel.with_lorentzian_frequencies(
        n_oscillators=50, coupling=2.0, gamma=0.5
    )
    lorentz_model.simulate(duration=20.0, dt=0.05)
    print(f"Lorentzian frequencies (γ=0.5): final r = {lorentz_model.synchronization():.4f}")
    print()
    
    # ===== PART 6: Phase Coherence vs Entropy =====
    print("PART 6: Phase Coherence vs Entropy")
    print("-" * 40)
    
    # Reset and observe entropy during synchronization
    model.reset()
    
    print("Tracking entropy during synchronization:")
    print(f"{'Time':>8} {'r':>8} {'Entropy':>10}")
    print("-" * 28)
    
    for step in range(6):
        r = model.synchronization()
        entropy = model.entropy()
        t = step * 2.0
        print(f"{t:8.1f} {r:8.4f} {entropy:10.4f}")
        model.simulate(duration=2.0, dt=0.02)
    
    print()
    print("Note: As synchronization increases (r↑), entropy decreases")
    print("  - Random phases → high entropy")
    print("  - Aligned phases → low entropy")
    print()
    
    # ===== PART 7: RK4 vs Euler Integration =====
    print("PART 7: Numerical Integration Methods")
    print("-" * 40)
    
    # Compare Euler and RK4 integration
    model_euler = KuramotoModel(n_oscillators=30, coupling=2.0)
    model_rk4 = KuramotoModel(n_oscillators=30, coupling=2.0)
    
    # Set same initial conditions
    model_rk4.phases = model_euler.phases.copy()
    model_rk4.frequencies = model_euler.frequencies.copy()
    
    dt_large = 0.1  # Large time step
    
    history_euler = model_euler.simulate(duration=10.0, dt=dt_large, method="euler")
    history_rk4 = model_rk4.simulate(duration=10.0, dt=dt_large, method="rk4")
    
    print(f"Comparison with dt={dt_large} (large time step):")
    print(f"  Euler final r: {history_euler[-1]:.4f}")
    print(f"  RK4 final r: {history_rk4[-1]:.4f}")
    print("  (RK4 is more accurate for larger time steps)")
    print()
    
    # ===== PART 8: Frequency Histogram =====
    print("PART 8: Frequency Distribution")
    print("-" * 40)
    
    centers, hist = model.frequency_histogram(n_bins=10)
    
    max_h = max(hist)
    print("Frequency distribution:")
    for c, h in zip(centers, hist):
        bar_len = int(h / max_h * 30)
        print(f"  {c:6.2f}: {'█' * bar_len}")
    print()
    
    # ===== PART 9: Long-Time Behavior =====
    print("PART 9: Long-Time Steady State")
    print("-" * 40)
    
    # Run for a long time to reach steady state
    long_model = KuramotoModel(n_oscillators=100, coupling=3.0)
    long_history = long_model.simulate(duration=50.0, dt=0.05)
    
    # Analyze fluctuations in steady state
    steady_portion = long_history[len(long_history)//2:]  # Last half
    r_mean = np.mean(steady_portion)
    r_std = np.std(steady_portion)
    
    print(f"Steady state analysis (last half of simulation):")
    print(f"  Mean r: {r_mean:.4f}")
    print(f"  Std r: {r_std:.4f}")
    print(f"  Fluctuations: {100*r_std/r_mean:.2f}%")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Kuramoto Model")
    print("=" * 60)
    print("""
Kuramoto Model Dynamics:
    dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)

Key Concepts:
1. Order Parameter: r = |⟨exp(iθ)⟩|
   - r = 0: Incoherent (random phases)
   - r = 1: Fully synchronized

2. Critical Coupling: K_c ≈ 2/(π·g(0))
   - Below K_c: No synchronization
   - Above K_c: Partial synchronization

3. Entropy: Measures phase distribution
   - High entropy = random phases
   - Low entropy = aligned phases

4. Applications in TinyAleph:
   - Phase synchronization in prime networks
   - Coherence dynamics in quantum states
   - Entropy-based halting conditions

Physical Interpretation:
- Oscillators are like pendulums
- Natural frequencies ω_i: how fast each wants to swing
- Coupling K: how strongly they influence each other
- Synchronization: they eventually swing together
    """)

if __name__ == "__main__":
    main()