"""
Quantum-Inspired State Collapse Mechanics

Provides functions for simulating quantum-like measurement and collapse:
- collapse_probability: Probability of collapse based on entropy
- should_collapse: Decision function for collapse timing
- measure_state: Measure a hypercomplex state
- born_measurement: Probabilistic measurement using Born rule
- partial_collapse: Gradual collapse to target state
- apply_decoherence: Simulate environmental decoherence

These mechanisms are used for discrete decision-making in
continuous dynamical systems.
"""

import math
import random
from typing import List, Optional, Dict, Any, Tuple


def collapse_probability(
    entropy_integral: float,
    lyapunov_factor: float = 1.0
) -> float:
    """
    Calculate probability of state collapse.
    
    Higher entropy integral increases collapse probability.
    Negative Lyapunov factor (stable dynamics) increases probability.
    
    Args:
        entropy_integral: Accumulated entropy over time
        lyapunov_factor: Current Lyapunov exponent estimate
        
    Returns:
        Collapse probability in [0, 1]
    """
    factor = 1.5 if lyapunov_factor < 0 else 0.5
    return (1 - math.exp(-entropy_integral)) * factor


def should_collapse(
    coherence: float,
    entropy: float,
    probability: float,
    min_coherence: float = 0.7,
    min_entropy: float = 1.8
) -> bool:
    """
    Determine whether system should collapse.
    
    Collapse occurs when:
    1. Coherence exceeds threshold (system is synchronized)
    2. Entropy exceeds threshold (enough information accumulated)
    3. Random value falls below probability
    
    Args:
        coherence: Current coherence level [0, 1]
        entropy: Current entropy
        probability: Collapse probability
        min_coherence: Minimum coherence for collapse
        min_entropy: Minimum entropy for collapse
        
    Returns:
        True if system should collapse
    """
    return (
        coherence > min_coherence and 
        entropy > min_entropy and 
        random.random() < probability
    )


def measure_state(
    components: List[float],
    basis: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Measure a hypercomplex state.
    
    Without basis: returns the dominant component
    With basis: projects onto basis vector
    
    Args:
        components: State vector components
        basis: Optional basis vector for projection
        
    Returns:
        Dict with index and value of measurement
    """
    if not components:
        return {'index': 0, 'value': 0.0}
    
    if basis is None:
        # Find dominant component
        max_idx = 0
        max_val = 0.0
        for i, v in enumerate(components):
            if abs(v) > abs(max_val):
                max_val = v
                max_idx = i
        return {'index': max_idx, 'value': max_val}
    
    # Project onto basis
    if len(basis) != len(components):
        raise ValueError("Basis dimension must match state dimension")
    
    dot_product = sum(c * b for c, b in zip(components, basis))
    return {'projection': dot_product, 'basis_dim': len(basis)}


def collapse_to_index(
    components: List[float],
    index: int
) -> List[float]:
    """
    Collapse state to a specific basis vector.
    
    All components become zero except the target index,
    which takes value +1 or -1 based on original sign.
    
    Args:
        components: State vector components
        index: Target index to collapse to
        
    Returns:
        New collapsed state vector
    """
    dim = len(components)
    collapsed = [0.0] * dim
    
    if 0 <= index < dim:
        sign = 1.0 if components[index] >= 0 else -1.0
        collapsed[index] = sign
    
    return collapsed


def born_measurement(components: List[float]) -> Dict[str, Any]:
    """
    Probabilistic measurement using Born rule.
    
    Probability of measuring index i is proportional to |c_i|^2.
    
    Args:
        components: State vector components
        
    Returns:
        Dict with measured index and its probability
    """
    if not components:
        return {'index': 0, 'probability': 1.0}
    
    # Calculate norm
    norm_sq = sum(c * c for c in components)
    if norm_sq < 1e-20:
        return {'index': 0, 'probability': 1.0}
    
    norm = math.sqrt(norm_sq)
    
    # Calculate probabilities (Born rule)
    probabilities = [(c / norm) ** 2 for c in components]
    
    # Sample from distribution
    r = random.random()
    cumulative = 0.0
    
    for i, p in enumerate(probabilities):
        cumulative += p
        if r < cumulative:
            return {'index': i, 'probability': p}
    
    # Fallback to last index
    return {'index': len(probabilities) - 1, 'probability': probabilities[-1]}


def partial_collapse(
    components: List[float],
    target_index: int,
    strength: float = 0.5
) -> List[float]:
    """
    Partial collapse: mix between current state and collapsed state.
    
    Linear interpolation allows gradual collapse over time.
    
    Args:
        components: Current state vector
        target_index: Index to collapse toward
        strength: Collapse strength in [0, 1]
            0 = no change, 1 = full collapse
        
    Returns:
        Partially collapsed and normalized state
    """
    dim = len(components)
    
    if dim == 0 or not (0 <= target_index < dim):
        return components.copy()
    
    # Create target collapsed state
    sign = 1.0 if components[target_index] >= 0 else -1.0
    collapsed = [0.0] * dim
    collapsed[target_index] = sign
    
    # Linear interpolation
    result = [
        (1 - strength) * c + strength * t
        for c, t in zip(components, collapsed)
    ]
    
    # Normalize
    norm = math.sqrt(sum(r * r for r in result))
    if norm > 1e-10:
        result = [r / norm for r in result]
    
    return result


def apply_decoherence(
    components: List[float],
    rate: float = 0.1
) -> List[float]:
    """
    Apply decoherence: gradual loss of quantum coherence.
    
    Simulates environmental interaction that degrades
    superposition states toward classical mixtures.
    
    Args:
        components: State vector components
        rate: Decoherence rate in [0, 1]
        
    Returns:
        Decohered and normalized state
    """
    result = []
    for c in components:
        # Decay toward zero with random noise
        noise = (random.random() - 0.5) * rate
        decohered = c * (1 - rate) + noise
        result.append(decohered)
    
    # Normalize
    norm = math.sqrt(sum(r * r for r in result))
    if norm > 1e-10:
        result = [r / norm for r in result]
    
    return result


def continuous_collapse(
    components: List[float],
    measurement_rate: float,
    dt: float
) -> Tuple[List[float], Optional[int]]:
    """
    Continuous measurement with stochastic collapse.
    
    Simulates a continuous measurement process where
    stronger components are more likely to survive.
    
    Args:
        components: State vector components
        measurement_rate: Rate of measurement (higher = faster collapse)
        dt: Time step
        
    Returns:
        Tuple of (new_components, collapsed_index or None)
    """
    if not components:
        return components, None
    
    # Probability of a collapse event occurring
    collapse_prob = 1 - math.exp(-measurement_rate * dt)
    
    if random.random() < collapse_prob:
        # Collapse event - use Born measurement
        result = born_measurement(components)
        collapsed_idx = result['index']
        
        # Apply partial collapse toward measured state
        new_components = partial_collapse(components, collapsed_idx, 0.5)
        return new_components, collapsed_idx
    
    # No collapse - apply small decoherence
    new_components = apply_decoherence(components, measurement_rate * dt * 0.1)
    return new_components, None


def zeno_effect(
    components: List[float],
    measurement_frequency: float,
    target_index: int,
    dt: float
) -> List[float]:
    """
    Simulate quantum Zeno effect.
    
    Frequent measurement inhibits transitions away from
    the currently measured state.
    
    Args:
        components: State vector components
        measurement_frequency: How often measurements occur
        target_index: Index being "watched"
        dt: Time step
        
    Returns:
        State stabilized toward target
    """
    if not components or not (0 <= target_index < len(components)):
        return components.copy()
    
    # Zeno effect strength increases with measurement frequency
    zeno_strength = 1 - math.exp(-measurement_frequency * dt)
    
    # Project onto target
    target_amp = abs(components[target_index])
    
    result = []
    for i, c in enumerate(components):
        if i == target_index:
            # Strengthen target component
            result.append(c * (1 + zeno_strength))
        else:
            # Suppress other components
            suppression = zeno_strength * target_amp
            result.append(c * (1 - suppression))
    
    # Normalize
    norm = math.sqrt(sum(r * r for r in result))
    if norm > 1e-10:
        result = [r / norm for r in result]
    
    return result


def entropy_threshold_collapse(
    components: List[float],
    entropy: float,
    threshold: float = 2.0
) -> Tuple[List[float], bool]:
    """
    Collapse when entropy exceeds threshold.
    
    Used to force discrete decisions when system has
    accumulated enough information.
    
    Args:
        components: State vector components
        entropy: Current system entropy
        threshold: Entropy threshold for collapse
        
    Returns:
        Tuple of (new_components, did_collapse)
    """
    if entropy >= threshold:
        result = born_measurement(components)
        collapsed = collapse_to_index(components, result['index'])
        return collapsed, True
    
    return components.copy(), False


class CollapseMonitor:
    """
    Monitor and track collapse events.
    
    Provides statistics on collapse frequency, timing,
    and outcome distribution.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize collapse monitor.
        
        Args:
            dimension: Dimension of state space
        """
        self.dimension = dimension
        self.collapse_history: List[Dict[str, Any]] = []
        self.outcome_counts: Dict[int, int] = {i: 0 for i in range(dimension)}
        self.total_collapses = 0
        self.time_since_last = 0.0
        self.inter_collapse_times: List[float] = []
    
    def record_collapse(
        self,
        index: int,
        probability: float,
        timestamp: float
    ) -> None:
        """Record a collapse event."""
        if self.total_collapses > 0:
            self.inter_collapse_times.append(self.time_since_last)
        
        self.collapse_history.append({
            'index': index,
            'probability': probability,
            'timestamp': timestamp
        })
        
        self.outcome_counts[index] = self.outcome_counts.get(index, 0) + 1
        self.total_collapses += 1
        self.time_since_last = 0.0
        
        # Prune history
        if len(self.collapse_history) > 1000:
            self.collapse_history = self.collapse_history[-500:]
    
    def tick(self, dt: float) -> None:
        """Advance time tracker."""
        self.time_since_last += dt
    
    def get_outcome_distribution(self) -> Dict[int, float]:
        """Get normalized distribution of collapse outcomes."""
        if self.total_collapses == 0:
            return {}
        return {
            idx: count / self.total_collapses
            for idx, count in self.outcome_counts.items()
            if count > 0
        }
    
    def mean_inter_collapse_time(self) -> float:
        """Calculate mean time between collapses."""
        if not self.inter_collapse_times:
            return float('inf')
        return sum(self.inter_collapse_times) / len(self.inter_collapse_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collapse statistics."""
        return {
            'total_collapses': self.total_collapses,
            'outcome_distribution': self.get_outcome_distribution(),
            'mean_inter_collapse_time': self.mean_inter_collapse_time(),
            'time_since_last': self.time_since_last
        }