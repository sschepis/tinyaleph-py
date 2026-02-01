"""
Kuramoto coupled oscillator model.

The Kuramoto model describes synchronization in populations of coupled oscillators:
    dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)

where:
- θ_i is the phase of oscillator i
- ω_i is the natural frequency of oscillator i
- K is the coupling strength
- N is the number of oscillators
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class KuramotoModel:
    """
    Kuramoto coupled oscillator model for synchronization dynamics.
    
    The model exhibits a phase transition at critical coupling K_c,
    above which oscillators begin to synchronize.
    
    Attributes:
        n_oscillators: Number of oscillators
        coupling: Coupling strength K
        frequencies: Natural frequencies ω_i (default: standard normal)
        phases: Current phases θ_i (default: uniform random)
    
    Examples:
        >>> model = KuramotoModel(n_oscillators=100, coupling=2.0)
        >>> for _ in range(1000):
        ...     model.step(dt=0.01)
        >>> print(f"Synchronization: {model.synchronization():.3f}")
    """
    
    n_oscillators: int
    coupling: float = 1.0
    frequencies: NDArray = field(default_factory=lambda: np.array([]))
    phases: NDArray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Initialize frequencies and phases if not provided."""
        if len(self.frequencies) == 0:
            self.frequencies = np.random.normal(0, 1, self.n_oscillators)
        if len(self.phases) == 0:
            self.phases = np.random.uniform(0, 2 * np.pi, self.n_oscillators)
    
    @classmethod
    def with_uniform_frequencies(
        cls,
        n_oscillators: int,
        coupling: float = 1.0,
        freq_range: float = 1.0
    ) -> KuramotoModel:
        """Create model with uniformly distributed frequencies."""
        frequencies = np.random.uniform(-freq_range, freq_range, n_oscillators)
        return cls(n_oscillators=n_oscillators, coupling=coupling, frequencies=frequencies)
    
    @classmethod
    def with_lorentzian_frequencies(
        cls,
        n_oscillators: int,
        coupling: float = 1.0,
        gamma: float = 1.0
    ) -> KuramotoModel:
        """Create model with Lorentzian (Cauchy) distributed frequencies."""
        frequencies = np.random.standard_cauchy(n_oscillators) * gamma
        return cls(n_oscillators=n_oscillators, coupling=coupling, frequencies=frequencies)
    
    def step(self, dt: float = 0.01) -> None:
        """
        Advance the system by one time step using Euler method.
        
        dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
        
        Args:
            dt: Time step size
        """
        # Compute pairwise phase differences
        phase_diff = self.phases[:, np.newaxis] - self.phases[np.newaxis, :]
        
        # Mean-field coupling
        coupling_term = np.mean(np.sin(phase_diff), axis=1)
        
        # Euler step
        self.phases += (self.frequencies + self.coupling * coupling_term) * dt
        self.phases = self.phases % (2 * np.pi)
    
    def step_rk4(self, dt: float = 0.01) -> None:
        """
        Advance the system using 4th-order Runge-Kutta method.
        
        More accurate than Euler for larger time steps.
        """
        def deriv(phases: NDArray) -> NDArray:
            phase_diff = phases[:, np.newaxis] - phases[np.newaxis, :]
            coupling_term = np.mean(np.sin(phase_diff), axis=1)
            return self.frequencies + self.coupling * coupling_term
        
        k1 = deriv(self.phases)
        k2 = deriv(self.phases + 0.5 * dt * k1)
        k3 = deriv(self.phases + 0.5 * dt * k2)
        k4 = deriv(self.phases + dt * k3)
        
        self.phases += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        self.phases = self.phases % (2 * np.pi)
    
    def order_parameter(self) -> complex:
        """
        Compute the complex order parameter r*e^(iψ).
        
        r = |<e^(iθ)>| measures synchronization (0 = none, 1 = perfect)
        ψ is the mean phase
        
        Returns:
            Complex order parameter
        """
        return complex(np.mean(np.exp(1j * self.phases)))
    
    def synchronization(self) -> float:
        """
        Return synchronization level r = |order_parameter|.
        
        r ≈ 0: incoherent (no synchronization)
        r ≈ 1: coherent (perfect synchronization)
        """
        return abs(self.order_parameter())
    
    def mean_phase(self) -> float:
        """Return the mean phase ψ from order parameter."""
        return float(np.angle(self.order_parameter()))
    
    def phase_coherence(self) -> float:
        """
        Compute phase coherence (variance-based measure).
        
        Low variance = high coherence
        """
        # Circular variance
        mean_cos = np.mean(np.cos(self.phases))
        mean_sin = np.mean(np.sin(self.phases))
        r = np.sqrt(mean_cos**2 + mean_sin**2)
        return float(r)
    
    def critical_coupling(self) -> float:
        """
        Estimate critical coupling for synchronization transition.
        
        For Gaussian frequency distribution: K_c ≈ 2 / (π * g(0))
        where g(0) is the frequency distribution at center.
        """
        # Estimate from frequency distribution
        std = np.std(self.frequencies)
        if std < 1e-10:
            return 0.0
        # For Gaussian: g(0) = 1 / (√(2π) * σ)
        g0 = 1 / (np.sqrt(2 * np.pi) * std)
        return 2 / (np.pi * g0)
    
    def simulate(
        self,
        duration: float,
        dt: float = 0.01,
        method: str = "euler"
    ) -> List[float]:
        """
        Simulate for given duration and return synchronization history.
        
        Args:
            duration: Total simulation time
            dt: Time step
            method: "euler" or "rk4"
            
        Returns:
            List of synchronization values at each step
        """
        n_steps = int(duration / dt)
        history = []
        
        step_fn = self.step_rk4 if method == "rk4" else self.step
        
        for _ in range(n_steps):
            step_fn(dt)
            history.append(self.synchronization())
        
        return history
    
    def entropy(self) -> float:
        """
        Compute phase entropy (uniformity of phase distribution).
        
        Uses histogram-based estimation.
        """
        n_bins = max(10, self.n_oscillators // 10)
        hist, _ = np.histogram(self.phases, bins=n_bins, range=(0, 2*np.pi), density=True)
        hist = hist[hist > 0]  # Remove zero bins
        # Normalize
        hist = hist / np.sum(hist)
        return float(-np.sum(hist * np.log(hist + 1e-10)))
    
    def frequency_histogram(self, n_bins: int = 50) -> tuple[NDArray, NDArray]:
        """Return histogram of natural frequencies."""
        hist, edges = np.histogram(self.frequencies, bins=n_bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, hist
    
    def reset(self) -> None:
        """Reset phases to random initial conditions."""
        self.phases = np.random.uniform(0, 2 * np.pi, self.n_oscillators)
    
    def __repr__(self) -> str:
        r = self.synchronization()
        return f"KuramotoModel(N={self.n_oscillators}, K={self.coupling:.2f}, r={r:.3f})"