"""
Atomic Resonance Simulation
---------------------------
Simulates the emergence of atomic structure from prime-based consciousness fields
as described in "A Symbolic-Resonance Atomic Model".

Dynamics:
    d|Ψ⟩/dt = iĤ|Ψ⟩ - λ(R̂ - r_stable)|Ψ⟩

Hamiltonian:
    Ĥ = T̂ + V̂_res
    T̂ = -iℏ Σ log(p) |p⟩⟨p|  (Entropy Gradient)
    V̂_res = Σ_{p≠q} -γ log(pq) |p⟩⟨q|  (Resonance Potential)

This script evolves a superposition of primes [2, 3, 5, 7] and visualizes
the collapse into stable "orbital" states.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Dict, Tuple

from tinyaleph.core.complex import Complex
from tinyaleph.hilbert.state import PrimeState

# Constants from the paper
HBAR = 1.0  # Normalized Planck constant
GAMMA = 0.8 # Coupling constant for resonance (Increased for stronger interaction)
LAMBDA = 0.5 # Dissipation rate (Increased for faster collapse)
R_STABLE = 2.0 # Target stable resonance (e.g., favors prime 2)
DT = 0.01   # Time step
STEPS = 5000 # Increased steps to allow full collapse

class AtomicHamiltonian:
    def __init__(self, primes: List[int]):
        self.primes = primes
        self.dim = len(primes)
        self.prime_to_idx = {p: i for i, p in enumerate(primes)}
        
    def apply(self, state: PrimeState) -> PrimeState:
        """
        Apply Ĥ|Ψ⟩ = (T̂ + V̂_res)|Ψ⟩
        
        Returns a NEW un-normalized PrimeState representing Ĥ|Ψ⟩.
        """
        result_amplitudes = {p: Complex.zero() for p in self.primes}
        
        # Current amplitudes vector
        current_amps = {p: state.get(p) for p in self.primes}
        
        for p in self.primes:
            # 1. Apply Kinetic Term T̂ (Diagonal)
            # T̂|p⟩ = -iℏ log(p)|p⟩
            # Contribution to <p|H|Ψ>: -iℏ log(p) * ψ(p)
            
            # -i * HBAR * log(p)
            # Complex(0, -1) * HBAR * log(p) = Complex(0, -HBAR * log(p))
            t_factor = Complex(0.0, -HBAR * math.log(p))
            term_t = t_factor * current_amps[p]
            result_amplitudes[p] = result_amplitudes[p] + term_t
            
            # 2. Apply Potential Term V̂_res (Off-diagonal)
            # V̂_res = Σ_{q≠p} -γ log(pq) |p⟩⟨q|
            # Contribution to <p|H|Ψ>: Σ_{q≠p} -γ log(pq) * ψ(q)
            
            for q in self.primes:
                if p == q:
                    continue
                
                # -γ log(pq) is real
                v_val = -GAMMA * math.log(p * q)
                v_factor = Complex(v_val, 0.0)
                
                term_v = v_factor * current_amps[q]
                result_amplitudes[p] = result_amplitudes[p] + term_v
                
        return PrimeState(primes=self.primes).from_dict(result_amplitudes)

# Helper to extend PrimeState with direct dict init if not present
def state_from_dict(amplitudes: Dict[int, Complex], primes: List[int]) -> PrimeState:
    s = PrimeState(primes)
    s.amplitudes = amplitudes
    return s

PrimeState.from_dict = lambda self, d: state_from_dict(d, self.primes)


def resonance_operator(state: PrimeState) -> float:
    """
    Expectation value ⟨Ψ|R̂|Ψ⟩
    R̂ = Σ p |p⟩⟨p|
    """
    val = 0.0
    norm = state.norm2()
    if norm < 1e-10: return 0.0
    
    for p in state.primes:
        prob = state.get(p).norm2() / norm
        val += p * prob
    return val

def run_simulation():
    print("Initializing Atomic Resonance Simulation...")
    
    # 1. Initialize State
    primes = [2, 3, 5, 7]
    psi = PrimeState.uniform_superposition(primes)
    
    hamiltonian = AtomicHamiltonian(primes)
    
    # History for plotting
    history = {
        'time': [],
        'entropy': [],
        'r_expect': [],
        'probs': {p: [] for p in primes}
    }
    
    print(f"Initial State: {psi}")
    print(f"Initial Entropy: {psi.entropy():.4f}")
    
    # 2. Time Evolution Loop
    # d|Ψ⟩/dt = iĤ|Ψ⟩ - λ(R̂ - r_stable)|Ψ⟩
    
    for step in range(STEPS):
        t = step * DT
        
        # Calculate terms for derivative
        
        # Term 1: iĤ|Ψ⟩
        h_psi = hamiltonian.apply(psi)
        # Multiply by i: i * (a+bi) = -b + ai
        term1 = h_psi * Complex(0, 1)
        
        # Term 2: -λ(R̂ - r_stable)|Ψ⟩
        # We need (R̂ - r_stable)|Ψ⟩. 
        # Since R̂ is diagonal (p), this is just (p - r_stable) * ψ(p) for each component.
        
        term2_amps = {}
        for p in primes:
            amp = psi.get(p)
            # scalar factor = -λ * (p - r_stable)
            factor = -LAMBDA * (p - R_STABLE)
            term2_amps[p] = amp * factor
        
        term2 = state_from_dict(term2_amps, primes)
        
        # Total derivative: dΨ/dt
        d_psi = term1 + term2
        
        # Euler Integration: Ψ(t+dt) = Ψ(t) + dΨ/dt * dt
        delta = d_psi * DT
        psi = psi + delta
        
        # Normalize (though non-Hermitian evolution changes norm, we view it as projective)
        # The paper implies physical states are normalized rays.
        psi.normalize()
        
        # Record stats
        if step % 10 == 0:
            history['time'].append(t)
            history['entropy'].append(psi.entropy())
            history['r_expect'].append(resonance_operator(psi))
            probs = psi.probabilities()
            for p in primes:
                history['probs'][p].append(probs.get(p, 0.0))
                
    print("Simulation Complete.")
    print(f"Final State: {psi}")
    print(f"Final Entropy: {psi.entropy():.4f}")
    
    # 3. Visualization
    plot_results(history)

def plot_results(history):
    time = history['time']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Entropy & Resonance
    ax1.plot(time, history['entropy'], 'k-', linewidth=2, label='Symbolic Entropy')
    ax1.set_ylabel('Entropy (bits)')
    ax1.set_title('Resonance Collapse Dynamics')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add Resonance Expectation on twin axis
    ax1b = ax1.twinx()
    ax1b.plot(time, history['r_expect'], 'r--', alpha=0.6, label='<R> (Resonance)')
    ax1b.axhline(y=R_STABLE, color='g', linestyle=':', label='Attractor (p=2)')
    ax1b.set_ylabel('Resonance Expectation')
    ax1b.legend(loc='center right')
    
    # Probabilities
    primes = list(history['probs'].keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    ax2.stackplot(time, 
                 [history['probs'][p] for p in primes],
                 labels=[f'|{p}⟩' for p in primes],
                 colors=colors, alpha=0.8)
    
    ax2.set_xlabel('Time (symbolic)')
    ax2.set_ylabel('Probability Amplitude |ψ|²')
    ax2.set_title('Prime Mode Evolution')
    ax2.legend(loc='center right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('atomic_evolution.png', dpi=300)
    print("Plot saved to atomic_evolution.png")

if __name__ == "__main__":
    run_simulation()
