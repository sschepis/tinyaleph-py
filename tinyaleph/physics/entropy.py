"""
Entropy and Stability Analysis

Implements entropy measures and Lyapunov stability analysis for
prime-based quantum systems. Central to understanding when states
collapse (become certain) vs. diverge (become chaotic).

Core Concepts:
1. Symbolic Entropy: S = -Σ P(p) log₂ P(p) over prime distribution
2. Lyapunov Exponent: λ measures exponential divergence/convergence
3. Stability Classification: λ < 0 (stable), λ ≈ 0 (critical), λ > 0 (chaotic)

The stability indicator λ (lambda) is the key metric:
- λ < -0.1: Collapsed state, high certainty
- -0.1 ≤ λ ≤ 0.1: Metastable, can go either way  
- λ > 0.1: Divergent, increasing uncertainty

Mathematical Foundation:
    λ = lim_{t→∞} (1/t) ln|δ(t)/δ(0)|
    
    where δ(t) is the deviation of a trajectory from a reference.
    
    For discrete symbolic systems:
    λ_symbolic = (S(t) - S(t-1)) / S(t-1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, TypeVar
from enum import Enum
import math
from collections import deque

from tinyaleph.core.constants import (
    LAMBDA_STABILITY_THRESHOLD, 
    ENTROPY_COLLAPSE_THRESHOLD,
    COHERENCE_THRESHOLD,
    PHI
)
from tinyaleph.core.primes import is_prime, prime_index


class StabilityClass(Enum):
    """Classification of system stability based on Lyapunov exponent."""
    COLLAPSED = "collapsed"      # λ < -0.1: Converging to fixed point
    STABLE = "stable"            # -0.1 ≤ λ < 0: Slow convergence
    CRITICAL = "critical"        # λ ≈ 0: Edge of chaos
    METASTABLE = "metastable"    # 0 < λ ≤ 0.1: Temporary stability
    DIVERGENT = "divergent"      # λ > 0.1: Chaotic divergence


def classify_stability(lyapunov: float) -> StabilityClass:
    """Classify stability based on Lyapunov exponent."""
    if lyapunov < -LAMBDA_STABILITY_THRESHOLD:
        return StabilityClass.COLLAPSED
    elif lyapunov < 0:
        return StabilityClass.STABLE
    elif abs(lyapunov) <= LAMBDA_STABILITY_THRESHOLD:
        return StabilityClass.CRITICAL
    elif lyapunov <= LAMBDA_STABILITY_THRESHOLD:
        return StabilityClass.METASTABLE
    else:
        return StabilityClass.DIVERGENT


def shannon_entropy(probabilities: Dict[int, float]) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    S = -Σ P(x) log₂ P(x)
    
    Returns 0 for empty or single-element distributions.
    """
    if not probabilities:
        return 0.0
    
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 1e-10:
            entropy -= prob * math.log2(prob)
    
    return entropy


def relative_entropy(p: Dict[int, float], q: Dict[int, float]) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(P || Q).
    
    D_KL = Σ P(x) log(P(x) / Q(x))
    
    Returns infinity if Q has zero probability where P is non-zero.
    """
    if not p:
        return 0.0
    
    divergence = 0.0
    for x, p_x in p.items():
        if p_x > 1e-10:
            q_x = q.get(x, 1e-10)  # Avoid division by zero
            if q_x < 1e-10:
                return float('inf')
            divergence += p_x * math.log(p_x / q_x)
    
    return divergence


def joint_entropy(joint_prob: Dict[Tuple[int, int], float]) -> float:
    """
    Compute joint entropy H(X, Y).
    
    H(X,Y) = -Σ P(x,y) log₂ P(x,y)
    """
    if not joint_prob:
        return 0.0
    
    entropy = 0.0
    for prob in joint_prob.values():
        if prob > 1e-10:
            entropy -= prob * math.log2(prob)
    
    return entropy


def conditional_entropy(joint_prob: Dict[Tuple[int, int], float]) -> float:
    """
    Compute conditional entropy H(Y|X).
    
    H(Y|X) = H(X,Y) - H(X)
    
    where X is the first element of each tuple.
    """
    # Compute marginal P(X)
    p_x: Dict[int, float] = {}
    for (x, y), prob in joint_prob.items():
        p_x[x] = p_x.get(x, 0.0) + prob
    
    h_xy = joint_entropy(joint_prob)
    h_x = shannon_entropy(p_x)
    
    return h_xy - h_x


def mutual_information(joint_prob: Dict[Tuple[int, int], float]) -> float:
    """
    Compute mutual information I(X; Y).
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Measures how much knowing X tells us about Y.
    """
    # Compute marginals
    p_x: Dict[int, float] = {}
    p_y: Dict[int, float] = {}
    
    for (x, y), prob in joint_prob.items():
        p_x[x] = p_x.get(x, 0.0) + prob
        p_y[y] = p_y.get(y, 0.0) + prob
    
    h_x = shannon_entropy(p_x)
    h_y = shannon_entropy(p_y)
    h_xy = joint_entropy(joint_prob)
    
    return h_x + h_y - h_xy


@dataclass
class EntropyTracker:
    """
    Tracks entropy over time to compute Lyapunov exponent.
    
    Maintains a sliding window of entropy values and computes
    the rate of change (symbolic Lyapunov exponent).
    """
    
    window_size: int = 50
    history: deque = field(default_factory=lambda: deque(maxlen=50))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    _time: int = 0
    
    def __post_init__(self):
        self.history = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
    
    def record(self, entropy: float) -> None:
        """Record an entropy measurement."""
        self.history.append(entropy)
        self.timestamps.append(self._time)
        self._time += 1
    
    def current_entropy(self) -> float:
        """Get most recent entropy value."""
        return self.history[-1] if self.history else 0.0
    
    def mean_entropy(self) -> float:
        """Get mean entropy over window."""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)
    
    def entropy_variance(self) -> float:
        """Get variance of entropy over window."""
        if len(self.history) < 2:
            return 0.0
        mean = self.mean_entropy()
        return sum((e - mean) ** 2 for e in self.history) / len(self.history)
    
    def lyapunov_exponent(self) -> float:
        """
        Estimate Lyapunov exponent from entropy trajectory.
        
        Uses linear regression on log of entropy deviations
        from mean.
        
        λ = d/dt ln|S(t) - S̄|
        """
        if len(self.history) < 5:
            return 0.0
        
        mean = self.mean_entropy()
        if mean < 1e-10:
            return -1.0  # Collapsed state
        
        # Compute log deviations
        log_devs = []
        times = []
        
        for i, e in enumerate(self.history):
            dev = abs(e - mean)
            if dev > 1e-10:
                log_devs.append(math.log(dev))
                times.append(i)
        
        if len(log_devs) < 3:
            return 0.0
        
        # Linear regression: λ = slope of log_dev vs time
        n = len(times)
        mean_t = sum(times) / n
        mean_ld = sum(log_devs) / n
        
        numerator = sum((t - mean_t) * (ld - mean_ld) 
                       for t, ld in zip(times, log_devs))
        denominator = sum((t - mean_t) ** 2 for t in times)
        
        if abs(denominator) < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    def stability(self) -> StabilityClass:
        """Get current stability classification."""
        return classify_stability(self.lyapunov_exponent())
    
    def is_stable(self) -> bool:
        """Check if system is in stable/collapsed state."""
        lyap = self.lyapunov_exponent()
        return lyap < LAMBDA_STABILITY_THRESHOLD
    
    def is_collapsing(self) -> bool:
        """Check if entropy is decreasing (approaching certainty)."""
        if len(self.history) < 3:
            return False
        
        recent = list(self.history)[-3:]
        return all(recent[i] >= recent[i+1] for i in range(len(recent)-1))
    
    def is_diverging(self) -> bool:
        """Check if entropy is increasing (approaching chaos)."""
        if len(self.history) < 3:
            return False
        
        recent = list(self.history)[-3:]
        return all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
    
    def trend(self) -> float:
        """
        Compute entropy trend: positive = increasing, negative = decreasing.
        
        Returns smoothed derivative of entropy.
        """
        if len(self.history) < 3:
            return 0.0
        
        # Weighted average of recent changes
        changes = []
        weights = []
        
        h_list = list(self.history)
        for i in range(1, len(h_list)):
            changes.append(h_list[i] - h_list[i-1])
            weights.append(i)  # More weight to recent
        
        if not weights:
            return 0.0
        
        total_weight = sum(weights)
        weighted_sum = sum(c * w for c, w in zip(changes, weights))
        
        return weighted_sum / total_weight
    
    def time_to_collapse(self) -> Optional[float]:
        """
        Estimate time steps until entropy reaches collapse threshold.
        
        Returns None if not trending toward collapse.
        """
        current = self.current_entropy()
        trend = self.trend()
        
        if trend >= 0:
            return None  # Not collapsing
        
        if current <= ENTROPY_COLLAPSE_THRESHOLD:
            return 0.0  # Already collapsed
        
        # Estimate: (current - threshold) / |trend|
        return (current - ENTROPY_COLLAPSE_THRESHOLD) / abs(trend)
    
    def reset(self) -> None:
        """Clear history."""
        self.history.clear()
        self.timestamps.clear()
        self._time = 0


@dataclass
class LyapunovAnalyzer:
    """
    Detailed Lyapunov analysis for prime-state trajectories.
    
    Computes:
    - Maximum Lyapunov exponent (MLE)
    - Lyapunov spectrum (if multiple dimensions available)
    - Kaplan-Yorke dimension estimate
    """
    
    perturbation_size: float = 1e-6
    trajectory_length: int = 100
    transient_length: int = 20
    
    def compute_mle(
        self,
        evolution_fn: Callable[[Dict[int, float]], Dict[int, float]],
        initial_state: Dict[int, float]
    ) -> float:
        """
        Compute Maximum Lyapunov Exponent by trajectory divergence.
        
        Args:
            evolution_fn: Maps probability distribution to next state
            initial_state: Initial probability distribution over primes
            
        Returns:
            Maximum Lyapunov exponent
        """
        # Normalize initial state
        total = sum(initial_state.values())
        state = {p: v/total for p, v in initial_state.items()}
        
        # Create perturbed copy
        primes = list(state.keys())
        if not primes:
            return 0.0
        
        perturbed = dict(state)
        perturbed[primes[0]] += self.perturbation_size
        
        # Renormalize
        total_p = sum(perturbed.values())
        perturbed = {p: v/total_p for p, v in perturbed.items()}
        
        # Let transients die out
        for _ in range(self.transient_length):
            state = evolution_fn(state)
            perturbed = evolution_fn(perturbed)
        
        # Measure divergence
        lyapunov_sum = 0.0
        count = 0
        
        for _ in range(self.trajectory_length):
            # Compute separation
            all_primes = set(state.keys()) | set(perturbed.keys())
            separation_sq = sum(
                (state.get(p, 0) - perturbed.get(p, 0)) ** 2
                for p in all_primes
            )
            separation = math.sqrt(separation_sq)
            
            if separation > 1e-10:
                lyapunov_sum += math.log(separation / self.perturbation_size)
                count += 1
                
                # Renormalize perturbation
                for p in all_primes:
                    diff = perturbed.get(p, 0) - state.get(p, 0)
                    perturbed[p] = state.get(p, 0) + diff * self.perturbation_size / separation
            
            # Evolve both trajectories
            state = evolution_fn(state)
            perturbed = evolution_fn(perturbed)
        
        if count == 0:
            return 0.0
        
        return lyapunov_sum / count
    
    def stability_map(
        self,
        evolution_fn: Callable[[Dict[int, float]], Dict[int, float]],
        prime_range: List[int],
        resolution: int = 20
    ) -> Dict[Tuple[int, int], float]:
        """
        Create 2D map of Lyapunov exponents for different initial conditions.
        
        Uses first two primes as axes, sweeping through probability space.
        
        Returns dict mapping (p1_weight, p2_weight) indices to λ values.
        """
        if len(prime_range) < 2:
            raise ValueError("Need at least 2 primes for stability map")
        
        p1, p2 = prime_range[:2]
        others = prime_range[2:] if len(prime_range) > 2 else []
        
        result = {}
        
        for i in range(resolution):
            for j in range(resolution - i):
                # Create initial state
                w1 = i / resolution
                w2 = j / resolution
                w_other = (resolution - i - j) / resolution
                
                initial = {p1: w1, p2: w2}
                
                if others and w_other > 0:
                    per_other = w_other / len(others)
                    for p in others:
                        initial[p] = per_other
                
                # Compute MLE
                mle = self.compute_mle(evolution_fn, initial)
                result[(i, j)] = mle
        
        return result


def prime_entropy(amplitudes: Dict[int, complex]) -> float:
    """
    Compute entropy of a quantum state over primes.
    
    Args:
        amplitudes: Dict mapping primes to complex amplitudes
        
    Returns:
        Shannon entropy of the probability distribution
    """
    total = sum(abs(a) ** 2 for a in amplitudes.values())
    if total < 1e-10:
        return 0.0
    
    probs = {p: abs(a) ** 2 / total for p, a in amplitudes.items()}
    return shannon_entropy(probs)


def golden_entropy_threshold(dimension: int) -> float:
    """
    Compute golden-ratio based entropy threshold for given dimension.
    
    Uses: threshold = log₂(dim) / Φ
    
    This provides a dimension-dependent collapse criterion.
    """
    if dimension <= 1:
        return 0.0
    return math.log2(dimension) / PHI


@dataclass
class EntropyGradient:
    """
    Computes gradient of entropy with respect to amplitude changes.
    
    Useful for optimization and understanding which components
    most affect system stability.
    """
    
    epsilon: float = 1e-6
    
    def gradient(self, amplitudes: Dict[int, complex]) -> Dict[int, float]:
        """
        Compute ∂S/∂|α_p| for each prime p.
        
        Positive gradient: increasing amplitude increases entropy.
        Negative gradient: increasing amplitude decreases entropy.
        """
        base_entropy = prime_entropy(amplitudes)
        gradients = {}
        
        for p in amplitudes:
            # Perturb amplitude
            perturbed = dict(amplitudes)
            current_amp = perturbed[p]
            perturbed[p] = current_amp * (1 + self.epsilon)
            
            new_entropy = prime_entropy(perturbed)
            gradients[p] = (new_entropy - base_entropy) / self.epsilon
        
        return gradients
    
    def descent_direction(self, amplitudes: Dict[int, complex]) -> Dict[int, float]:
        """
        Find direction of steepest entropy decrease (toward collapse).
        
        Returns normalized direction vector.
        """
        grad = self.gradient(amplitudes)
        
        # Negate for descent
        descent = {p: -g for p, g in grad.items()}
        
        # Normalize
        norm = math.sqrt(sum(d ** 2 for d in descent.values()))
        if norm > 1e-10:
            descent = {p: d / norm for p, d in descent.items()}
        
        return descent
    
    def ascent_direction(self, amplitudes: Dict[int, complex]) -> Dict[int, float]:
        """
        Find direction of steepest entropy increase (toward chaos).
        
        Returns normalized direction vector.
        """
        grad = self.gradient(amplitudes)
        
        # Normalize
        norm = math.sqrt(sum(g ** 2 for g in grad.values()))
        if norm > 1e-10:
            grad = {p: g / norm for p, g in grad.items()}
        
        return grad


def coherence_from_entropy(entropy: float, max_entropy: float) -> float:
    """
    Convert entropy to coherence measure.
    
    coherence = 1 - S/S_max
    
    High entropy = low coherence (uncertain state)
    Low entropy = high coherence (definite state)
    """
    if max_entropy <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - entropy / max_entropy))


def entropy_production_rate(
    current: Dict[int, float],
    previous: Dict[int, float],
    dt: float = 1.0
) -> float:
    """
    Compute rate of entropy production.
    
    dS/dt = (S(t) - S(t-dt)) / dt
    
    Positive: entropy increasing (toward chaos)
    Negative: entropy decreasing (toward order)
    """
    s_curr = shannon_entropy(current)
    s_prev = shannon_entropy(previous)
    return (s_curr - s_prev) / dt


def thermodynamic_entropy(
    probabilities: Dict[int, float],
    energies: Dict[int, float],
    temperature: float = 1.0
) -> float:
    """
    Compute thermodynamic entropy S = -k_B Σ P_i ln P_i.
    
    For prime states, uses log(p) as energy: E_p = ln(p).
    
    At thermal equilibrium, this equals Boltzmann entropy.
    """
    if not probabilities:
        return 0.0
    
    # Use natural log for thermodynamic convention
    entropy = 0.0
    for p, prob in probabilities.items():
        if prob > 1e-10:
            entropy -= prob * math.log(prob)
    
    return entropy  # In units where k_B = 1


def state_entropy(state) -> float:
    """
    Compute entropy of a state vector (Hypercomplex or similar).
    
    Works with any object that has a 'components' attribute or
    can be iterated to get amplitude values.
    
    Args:
        state: State vector with components
        
    Returns:
        Shannon entropy of the probability distribution |component|^2
    """
    if state is None:
        return 0.0
    
    # Get components
    if hasattr(state, 'components'):
        components = state.components
    elif hasattr(state, '__iter__'):
        components = list(state)
    else:
        return 0.0
    
    if not components:
        return 0.0
    
    # Compute probabilities |c|^2
    probs = []
    for c in components:
        if hasattr(c, '__abs__'):
            probs.append(abs(c) ** 2)
        elif isinstance(c, (int, float)):
            probs.append(c ** 2)
        else:
            probs.append(0.0)
    
    total = sum(probs)
    if total < 1e-10:
        return 0.0
    
    # Normalize and compute entropy
    entropy = 0.0
    for p in probs:
        prob = p / total
        if prob > 1e-10:
            entropy -= prob * math.log2(prob)
    
    return entropy


def coherence(state) -> float:
    """
    Compute coherence of a state vector.
    
    Uses the norm of off-diagonal density matrix elements as a
    coherence measure. Higher coherence means more quantum superposition.
    
    For a pure state |ψ⟩ = Σ cᵢ|i⟩:
    coherence = Σᵢ≠ⱼ |cᵢ||cⱼ| / (Σᵢ |cᵢ|²)
    
    This is the l1-norm of coherence.
    
    Args:
        state: State vector
        
    Returns:
        Coherence measure in [0, 1]
    """
    if state is None:
        return 0.0
    
    # Get components
    if hasattr(state, 'components'):
        components = state.components
    elif hasattr(state, '__iter__'):
        components = list(state)
    else:
        return 0.0
    
    if not components:
        return 0.0
    
    # Get amplitudes
    amps = []
    for c in components:
        if hasattr(c, '__abs__'):
            amps.append(abs(c))
        elif isinstance(c, (int, float)):
            amps.append(abs(c))
        else:
            amps.append(0.0)
    
    n = len(amps)
    if n < 2:
        return 0.0
    
    # Normalization
    norm_sq = sum(a ** 2 for a in amps)
    if norm_sq < 1e-10:
        return 0.0
    
    # Sum of off-diagonal products
    off_diag_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag_sum += amps[i] * amps[j]
    
    # Normalize by maximum possible coherence (n-1)*norm²
    max_coherence = (n - 1) * norm_sq
    if max_coherence < 1e-10:
        return 0.0
    
    return min(1.0, off_diag_sum / max_coherence)