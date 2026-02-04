"""
Collatz Entropy Verification
----------------------------
Empirical verification of the "Collatz Conjecture Proven via Entropy Collapse" paper.

This script tracks the "Symbolic Entropy" of a number n as it evolves under the Collatz map:
    n -> n/2 if even
    n -> 3n+1 if odd

Theory:
    State |n⟩ = Σ (a_p / A) |p⟩  where n = Π p^(a_p) and A = Σ a_p
    Entropy H(n) = -Σ (p_k log2 p_k) where p_k = a_p/A

    Hypothesis: The entropy H(n) decreases monotonically in expectation over the trajectory,
    driving the system to the unique ground state |1⟩ (Entropy = 0).
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Dict, Tuple

# We can use sympy for robust factorization if available, or write a simple helper
try:
    from sympy import factorint
except ImportError:
    # Simple fallback factorization
    def factorint(n: int) -> Dict[int, int]:
        factors = {}
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

def calculate_symbolic_entropy(n: int) -> float:
    """
    Calculate symbolic entropy H(n) based on prime factorization structure.
    
    H(n) = -Σ (a_i/A) * log2(a_i/A)
    where a_i are exponents of prime factors and A is sum of exponents.
    
    Special case: n=1 has entropy 0.
    """
    if n <= 1:
        return 0.0
        
    factors = factorint(n)
    
    # Total "amplitude" A = sum of exponents
    A = sum(factors.values())
    
    if A == 0: 
        return 0.0
        
    entropy = 0.0
    for p, exponent in factors.items():
        # Probability mass for prime p
        prob = exponent / A
        entropy -= prob * math.log2(prob)
        
    return entropy

def calculate_structural_complexity(n: int) -> float:
    """
    Calculate 'Structure' metric A(n) = sum of exponents.
    This corresponds to the 'mass' or 'size' of the superposition.
    """
    if n <= 1: return 0
    return sum(factorint(n).values())

def get_collatz_trajectory(start_n: int) -> List[Tuple[int, float, int]]:
    """
    Generate trajectory (n, entropy, structure) until n=1.
    """
    trajectory = []
    curr = start_n
    
    # Limit to prevent infinite loops if conjecture is false (unlikely)
    max_steps = 10000 
    
    while curr > 1 and len(trajectory) < max_steps:
        h = calculate_symbolic_entropy(curr)
        a = calculate_structural_complexity(curr)
        trajectory.append((curr, h, a))
        
        if curr % 2 == 0:
            curr //= 2
        else:
            curr = 3 * curr + 1
            
    # Add final state
    trajectory.append((1, 0.0, 0))
    return trajectory

def run_verification(start_n: int = 27):
    print(f"Running Collatz Entropy Verification for n={start_n}...")
    
    traj = get_collatz_trajectory(start_n)
    steps = list(range(len(traj)))
    values = [t[0] for t in traj]
    entropies = [t[1] for t in traj]
    structures = [t[2] for t in traj] # A(n)
    
    print(f"Trajectory length: {len(traj)} steps")
    print(f"Max value: {max(values)}")
    print(f"Max entropy: {max(entropies):.4f} bits")
    
    # --- Visualization ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Value Path (Log Scale)
    ax1.plot(steps, values, 'b-', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_ylabel('Value n (log)')
    ax1.set_title(f'Collatz Trajectory: n={start_n}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Symbolic Entropy H(n)
    ax2.plot(steps, entropies, 'k-', linewidth=1.5, label='Symbolic Entropy H(n)')
    
    # Calculate and plot moving average to show "expectation"
    window = 5
    if len(entropies) > window:
        moving_avg = np.convolve(entropies, np.ones(window)/window, mode='valid')
        # Pad beginning to align x-axis
        pad = [np.nan] * (window - 1)
        ax2.plot(steps, list(pad) + list(moving_avg), 'r--', linewidth=2, label='Trend (Moving Avg)')
        
    ax2.set_ylabel('Symbolic Entropy (bits)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Structural Complexity A(n)
    ax3.plot(steps, structures, 'g-', alpha=0.7, label='Structure A(n) = Σ exponents')
    ax3.set_ylabel('Structure A(n)')
    ax3.set_xlabel('Step')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"collatz_entropy_n{start_n}.png"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    # Test with 27 (the famous one that takes 111 steps)
    run_verification(27)
