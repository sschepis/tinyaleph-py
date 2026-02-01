"""
Lyapunov Exponent Estimation and Stability Analysis

Provides tools for analyzing dynamical stability:
- estimate_lyapunov: Estimate from oscillator phase histories
- local_lyapunov: Local exponent for single oscillator
- classify_stability: Categorize stability regime
- adaptive_coupling: Adjust coupling based on stability
- delay_embedding: Phase space reconstruction

The Lyapunov exponent λ characterizes exponential divergence:
- λ < 0: Stable (converging trajectories)
- λ ≈ 0: Marginal/Critical (neutral stability)
- λ > 0: Chaotic (diverging trajectories)
"""

import math
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum, auto


class StabilityRegime(Enum):
    """Classification of dynamical stability."""
    STABLE = auto()      # λ < -0.1 (converging)
    MARGINAL = auto()    # -0.1 ≤ λ ≤ 0.1 (neutral)
    CHAOTIC = auto()     # λ > 0.1 (diverging)
    UNKNOWN = auto()     # Insufficient data


def estimate_lyapunov(
    phase_histories: List[List[float]],
    window_size: int = 20
) -> float:
    """
    Estimate Lyapunov exponent from oscillator phase histories.
    
    Uses the divergence between initially close trajectories
    to estimate the maximal Lyapunov exponent.
    
    Args:
        phase_histories: List of phase history arrays (one per oscillator)
        window_size: Number of time steps to consider
        
    Returns:
        Estimated Lyapunov exponent
    """
    if not phase_histories or len(phase_histories) < 2:
        return 0.0
    
    # Check that we have enough history
    min_length = min(len(h) for h in phase_histories)
    if min_length < window_size:
        return 0.0
    
    sum_log = 0.0
    count = 0
    
    for i in range(len(phase_histories) - 1):
        h1 = phase_histories[i]
        h2 = phase_histories[i + 1]
        
        # Initial separation
        d0 = abs(h1[0] - h2[0])
        # Final separation
        dn = abs(h1[-1] - h2[-1])
        
        if d0 > 1e-10 and dn > 1e-10:
            # Lyapunov exponent is rate of exponential divergence
            sum_log += math.log(dn / d0) / window_size
            count += 1
    
    return sum_log / count if count > 0 else 0.0


def estimate_lyapunov_from_oscillators(
    oscillators: List[Any],
    window_size: int = 20
) -> float:
    """
    Estimate Lyapunov exponent from oscillator objects.
    
    Convenience function that extracts phase histories.
    
    Args:
        oscillators: List of oscillator objects with phase_history attribute
        window_size: Analysis window size
        
    Returns:
        Estimated Lyapunov exponent
    """
    phase_histories = [
        osc.phase_history if hasattr(osc, 'phase_history') else []
        for osc in oscillators
    ]
    return estimate_lyapunov(phase_histories, window_size)


def local_lyapunov(
    phase_history: List[float],
    window_size: int = 20
) -> float:
    """
    Compute local Lyapunov exponent for a single oscillator.
    
    Uses the rate of phase change to estimate local stability.
    
    Args:
        phase_history: Phase values over time
        window_size: Analysis window size
        
    Returns:
        Local Lyapunov exponent
    """
    if len(phase_history) < window_size + 1:
        return 0.0
    
    sum_log = 0.0
    valid_count = 0
    
    for i in range(1, window_size):
        d = abs(phase_history[i] - phase_history[i - 1])
        if d > 1e-10:
            sum_log += math.log(d)
            valid_count += 1
    
    return sum_log / valid_count if valid_count > 0 else 0.0


def classify_stability(lyapunov_exponent: float) -> StabilityRegime:
    """
    Classify stability regime based on Lyapunov exponent.
    
    Args:
        lyapunov_exponent: Estimated exponent value
        
    Returns:
        StabilityRegime enum value
    """
    if lyapunov_exponent < -0.1:
        return StabilityRegime.STABLE
    elif lyapunov_exponent > 0.1:
        return StabilityRegime.CHAOTIC
    else:
        return StabilityRegime.MARGINAL


def classify_stability_string(lyapunov_exponent: float) -> str:
    """
    Classify stability as string (for compatibility).
    
    Args:
        lyapunov_exponent: Estimated exponent value
        
    Returns:
        One of 'STABLE', 'MARGINAL', 'CHAOTIC'
    """
    if lyapunov_exponent < -0.1:
        return 'STABLE'
    elif lyapunov_exponent > 0.1:
        return 'CHAOTIC'
    return 'MARGINAL'


def adaptive_coupling(
    base_coupling: float,
    lyapunov_exponent: float,
    gain: float = 0.5
) -> float:
    """
    Adjust coupling strength based on stability.
    
    Increases coupling in stable regime to maintain activity,
    decreases in chaotic regime to restore order.
    
    Args:
        base_coupling: Base coupling strength
        lyapunov_exponent: Current Lyapunov estimate
        gain: Adjustment magnitude
        
    Returns:
        Adjusted coupling strength
    """
    if lyapunov_exponent < -0.1:
        # Stable: can increase coupling
        return base_coupling * (1 + gain)
    elif lyapunov_exponent > 0.1:
        # Chaotic: reduce coupling
        return base_coupling * (1 - gain)
    else:
        # Marginal: no adjustment
        return base_coupling


def delay_embedding(
    time_series: List[float],
    embedding_dim: int = 3,
    delay: int = 1
) -> List[List[float]]:
    """
    Phase space reconstruction using delay embedding.
    
    Takens' theorem: A time series can reconstruct the
    dynamics of the underlying system using delay coordinates.
    
    Args:
        time_series: 1D time series data
        embedding_dim: Dimension of embedding space
        delay: Time delay between coordinates
        
    Returns:
        List of embedding_dim-dimensional points
    """
    if len(time_series) < (embedding_dim - 1) * delay + 1:
        return []
    
    embedded = []
    for i in range(len(time_series) - (embedding_dim - 1) * delay):
        point = []
        for d in range(embedding_dim):
            point.append(time_series[i + d * delay])
        embedded.append(point)
    
    return embedded


def stability_margin(
    lyapunov_exponent: float,
    threshold: float = 0.1
) -> float:
    """
    Calculate distance from instability threshold.
    
    Positive margin = stable, negative = unstable.
    
    Args:
        lyapunov_exponent: Current exponent
        threshold: Threshold for instability
        
    Returns:
        Stability margin (positive = stable)
    """
    return threshold - lyapunov_exponent


def spectrum_from_jacobian(jacobian: List[List[float]]) -> List[float]:
    """
    Compute Lyapunov spectrum from Jacobian matrix.
    
    For small systems, the full spectrum can be computed
    from the eigenvalues of the Jacobian.
    
    Args:
        jacobian: n x n Jacobian matrix
        
    Returns:
        Lyapunov exponents (sorted descending)
    """
    if not jacobian or not jacobian[0]:
        return []
    
    n = len(jacobian)
    
    # Simple power iteration for largest eigenvalue
    # (Full spectrum would require more sophisticated methods)
    v = [1.0 / math.sqrt(n)] * n
    
    for _ in range(50):  # Power iterations
        # Matrix-vector multiply
        new_v = [0.0] * n
        for i in range(n):
            for j in range(n):
                new_v[i] += jacobian[i][j] * v[j]
        
        # Normalize
        norm = math.sqrt(sum(x * x for x in new_v))
        if norm < 1e-10:
            return [0.0]
        
        v = [x / norm for x in new_v]
    
    # Compute Rayleigh quotient for eigenvalue estimate
    av = [sum(jacobian[i][j] * v[j] for j in range(n)) for i in range(n)]
    eigenvalue = sum(v[i] * av[i] for i in range(n))
    
    return [eigenvalue]


def finite_time_lyapunov(
    trajectory: List[List[float]],
    dt: float = 1.0
) -> float:
    """
    Compute finite-time Lyapunov exponent from trajectory.
    
    Uses the stretching of tangent vectors along the trajectory.
    
    Args:
        trajectory: List of state vectors over time
        dt: Time step
        
    Returns:
        Finite-time Lyapunov exponent
    """
    if len(trajectory) < 2:
        return 0.0
    
    # Compute stretching between consecutive states
    total_stretch = 0.0
    count = 0
    
    for i in range(1, len(trajectory)):
        # Displacement
        disp = [
            trajectory[i][j] - trajectory[i - 1][j]
            for j in range(len(trajectory[i]))
        ]
        
        # Norm
        norm = math.sqrt(sum(d * d for d in disp))
        if norm > 1e-10:
            total_stretch += math.log(norm)
            count += 1
    
    if count == 0:
        return 0.0
    
    # Normalize by total time
    total_time = count * dt
    return total_stretch / total_time


class LyapunovTracker:
    """
    Track Lyapunov exponent over time.
    
    Provides running estimates and stability monitoring.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        smoothing: float = 0.1
    ):
        """
        Initialize tracker.
        
        Args:
            window_size: Size of analysis window
            smoothing: Exponential smoothing factor
        """
        self.window_size = window_size
        self.smoothing = smoothing
        
        self.history: List[float] = []
        self.current_estimate = 0.0
        self.regime = StabilityRegime.UNKNOWN
        
        self.stability_changes: List[Dict[str, Any]] = []
    
    def update(self, lyapunov: float, timestamp: float = 0.0) -> None:
        """
        Update with new Lyapunov estimate.
        
        Args:
            lyapunov: New exponent estimate
            timestamp: Optional timestamp
        """
        # Exponential smoothing
        self.current_estimate = (
            self.smoothing * lyapunov + 
            (1 - self.smoothing) * self.current_estimate
        )
        
        self.history.append(self.current_estimate)
        if len(self.history) > 1000:
            self.history = self.history[-500:]
        
        # Check for regime change
        new_regime = classify_stability(self.current_estimate)
        if new_regime != self.regime:
            self.stability_changes.append({
                'from': self.regime,
                'to': new_regime,
                'value': self.current_estimate,
                'timestamp': timestamp
            })
            self.regime = new_regime
    
    def update_from_oscillators(
        self,
        oscillators: List[Any],
        timestamp: float = 0.0
    ) -> None:
        """
        Update from oscillator array.
        
        Args:
            oscillators: List of oscillator objects
            timestamp: Optional timestamp
        """
        lyapunov = estimate_lyapunov_from_oscillators(
            oscillators, self.window_size
        )
        self.update(lyapunov, timestamp)
    
    def get_trend(self, samples: int = 10) -> float:
        """
        Get trend of Lyapunov exponent.
        
        Positive = becoming more chaotic
        Negative = becoming more stable
        
        Args:
            samples: Number of recent samples to consider
            
        Returns:
            Trend value (rate of change)
        """
        if len(self.history) < samples:
            return 0.0
        
        recent = self.history[-samples:]
        
        # Simple linear regression slope
        n = len(recent)
        sum_x = sum(range(n))
        sum_y = sum(recent)
        sum_xy = sum(i * y for i, y in enumerate(recent))
        sum_x2 = sum(i * i for i in range(n))
        
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return 0.0
        
        return (n * sum_xy - sum_x * sum_y) / denom
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            'current': self.current_estimate,
            'regime': self.regime.name,
            'trend': self.get_trend(),
            'stability_margin': stability_margin(self.current_estimate),
            'regime_changes': len(self.stability_changes),
            'history_length': len(self.history)
        }
    
    def is_stable(self) -> bool:
        """Check if currently stable."""
        return self.regime == StabilityRegime.STABLE
    
    def is_chaotic(self) -> bool:
        """Check if currently chaotic."""
        return self.regime == StabilityRegime.CHAOTIC
    
    def reset(self) -> None:
        """Reset tracker."""
        self.history = []
        self.current_estimate = 0.0
        self.regime = StabilityRegime.UNKNOWN
        self.stability_changes = []