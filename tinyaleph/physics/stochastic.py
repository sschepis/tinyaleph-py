"""
Stochastic Kuramoto Models

Kuramoto oscillators with Langevin noise for robust synchronization:
    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ) + σ·ξᵢ(t)

Features:
- White noise (Wiener process)
- Colored noise (Ornstein-Uhlenbeck process)
- Temperature-dependent coupling
- Noise-induced synchronization detection
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math

import numpy as np
from numpy.typing import NDArray
from tinyaleph.physics.kuramoto import KuramotoModel


# =============================================================================
# STOCHASTIC KURAMOTO
# =============================================================================

class StochasticKuramoto(KuramotoModel):
    """
    StochasticKuramoto - Kuramoto model with Langevin noise
    
    Adds thermal fluctuations to oscillator dynamics:
        dθᵢ = [ωᵢ + K·coupling(i)]dt + σ·dWᵢ
    
    where dWᵢ is a Wiener increment with variance dt.
    """
    
    def __init__(
        self,
        frequencies: List[float],
        coupling: float = 0.3,
        noise_intensity: float = 0.1,
        noise_type: str = "white",
        correlation_time: float = 1.0,
        temperature: float = 1.0,
        temperature_coupling: bool = False
    ):
        """
        Initialize StochasticKuramoto model.
        
        Args:
            frequencies: Natural frequencies ωᵢ
            coupling: Coupling strength K
            noise_intensity: Noise amplitude σ
            noise_type: 'white' or 'colored'
            correlation_time: τ for colored noise
            temperature: Temperature for T-dependent coupling
            temperature_coupling: Enable temperature-dependent coupling
        """
        n = len(frequencies)
        freq_array = np.array(frequencies)
        
        object.__setattr__(self, 'n_oscillators', n)
        object.__setattr__(self, 'coupling', coupling)
        object.__setattr__(self, 'frequencies', freq_array)
        object.__setattr__(self, 'phases', np.random.uniform(0, 2 * np.pi, n))
        
        self.sigma = noise_intensity
        self.noise_type = noise_type
        self.tau = correlation_time
        self.temperature = temperature
        self.use_temperature_coupling = temperature_coupling
        
        # For colored noise (Ornstein-Uhlenbeck): dη = -η/τ dt + σ/√τ dW
        self.colored_noise_state = np.zeros(n)
        
        # Track noise history for analysis
        self.noise_history: List[Dict] = []
        self.max_history_length = 1000
        
        # Statistics
        self.noise_stats = {
            "mean": 0.0,
            "variance": 0.0,
            "sample_count": 0
        }
    
    def set_noise_intensity(self, sigma: float) -> None:
        """Set noise intensity dynamically."""
        self.sigma = sigma
    
    def set_temperature(self, T: float) -> None:
        """Set temperature (affects coupling if temperature_coupling is enabled)."""
        self.temperature = max(0.01, T)
    
    def get_effective_coupling(self) -> float:
        """
        Get effective coupling (temperature-dependent).
        K_eff = K / T (Arrhenius-like)
        """
        if self.use_temperature_coupling:
            return self.coupling / self.temperature
        return self.coupling
    
    def step(self, dt: float = 0.01) -> None:
        """Advance system by one time step with stochastic dynamics."""
        # Compute pairwise phase differences
        phase_diff = self.phases[:, np.newaxis] - self.phases[np.newaxis, :]
        
        # Mean-field coupling
        coupling_term = np.mean(np.sin(phase_diff), axis=1)
        
        # Effective coupling (possibly temperature-dependent)
        K_eff = self.get_effective_coupling()
        
        # Deterministic part
        deterministic = self.frequencies + K_eff * coupling_term
        
        # Stochastic part
        if self.noise_type == "colored":
            # Ornstein-Uhlenbeck update
            decay = np.exp(-dt / self.tau)
            diffusion = self.sigma * np.sqrt((1 - decay * decay) / 2)
            self.colored_noise_state = self.colored_noise_state * decay + diffusion * np.random.randn(self.n_oscillators)
            stochastic = self.colored_noise_state * dt
        else:
            # White noise: σ·√dt·N(0,1)
            stochastic = self.sigma * np.sqrt(dt) * np.random.randn(self.n_oscillators)
        
        # Update phases
        self.phases += deterministic * dt + stochastic
        self.phases = self.phases % (2 * np.pi)
        
        # Update statistics
        self._update_noise_stats(stochastic)
    
    def tick(self, dt: float = 0.01) -> None:
        """Alias for step()."""
        self.step(dt)
    
    def _update_noise_stats(self, noise_values: np.ndarray) -> None:
        """Update running noise statistics."""
        for noise_value in noise_values:
            n = self.noise_stats["sample_count"] + 1
            self.noise_stats["sample_count"] = n
            
            delta = noise_value - self.noise_stats["mean"]
            self.noise_stats["mean"] += delta / n
            delta2 = noise_value - self.noise_stats["mean"]
            self.noise_stats["variance"] += (delta * delta2 - self.noise_stats["variance"]) / n
    
    def evolve(self, steps: int, dt: float = 0.01) -> List[Dict]:
        """
        Run multiple steps.
        
        Args:
            steps: Number of steps
            dt: Time step size
            
        Returns:
            Evolution history
        """
        trajectory = []
        
        for i in range(steps):
            self.step(dt)
            trajectory.append({
                "step": i,
                "order_parameter": float(abs(self.order_parameter())),
                "mean_phase": float(self.mean_phase()),
                "noise_stats": dict(self.noise_stats)
            })
        
        return trajectory
    
    def detect_noise_induced_sync(
        self,
        baseline_steps: int = 100,
        noisy_steps: int = 200,
        dt: float = 0.01
    ) -> Dict:
        """
        Detect noise-induced synchronization.
        
        Phenomenon where noise can actually enhance synchronization
        by helping oscillators escape from metastable states.
        
        Args:
            baseline_steps: Steps to establish baseline
            noisy_steps: Steps with noise
            dt: Time step
            
        Returns:
            Detection result
        """
        # Save current state
        original_sigma = self.sigma
        
        # Baseline (no noise)
        self.sigma = 0
        baseline_trajectory = self.evolve(baseline_steps, dt)
        baseline_order = np.mean([t["order_parameter"] for t in baseline_trajectory[-20:]])
        
        # With noise
        self.sigma = original_sigma
        noisy_trajectory = self.evolve(noisy_steps, dt)
        noisy_order = np.mean([t["order_parameter"] for t in noisy_trajectory[-20:]])
        
        enhancement = noisy_order - baseline_order
        is_noise_induced = enhancement > 0.1
        
        return {
            "baseline_order_parameter": float(baseline_order),
            "noisy_order_parameter": float(noisy_order),
            "enhancement": float(enhancement),
            "is_noise_induced": is_noise_induced,
            "noise_intensity": original_sigma
        }
    
    def order_parameter_with_uncertainty(
        self,
        samples: int = 100,
        dt: float = 0.01
    ) -> Dict:
        """
        Compute stochastic order parameter with error bars.
        
        Args:
            samples: Number of samples for averaging
            dt: Time step between samples
            
        Returns:
            Order parameter with uncertainty
        """
        values = []
        
        for _ in range(samples):
            self.step(dt)
            values.append(abs(self.order_parameter()))
        
        values = np.array(values)
        mean = float(np.mean(values))
        std = float(np.std(values))
        std_error = std / np.sqrt(samples)
        
        return {
            "mean": mean,
            "std_dev": std,
            "std_error": std_error,
            "confidence_95": [mean - 1.96 * std_error, mean + 1.96 * std_error],
            "samples": list(values)
        }
    
    def reset_noise(self) -> None:
        """Reset noise state."""
        self.colored_noise_state = np.zeros(self.n_oscillators)
        self.noise_history = []
        self.noise_stats = {"mean": 0.0, "variance": 0.0, "sample_count": 0}
    
    def get_state(self) -> Dict:
        """Get current state snapshot."""
        return {
            "phases": list(self.phases),
            "frequencies": list(self.frequencies),
            "order_parameter": float(abs(self.order_parameter())),
            "noise_intensity": self.sigma,
            "noise_type": self.noise_type,
            "correlation_time": self.tau,
            "temperature": self.temperature,
            "effective_coupling": self.get_effective_coupling(),
            "noise_stats": dict(self.noise_stats)
        }


# =============================================================================
# COLORED NOISE KURAMOTO
# =============================================================================

class ColoredNoiseKuramoto(StochasticKuramoto):
    """
    ColoredNoiseKuramoto - Specialized class for Ornstein-Uhlenbeck noise
    
    Provides more control over colored noise parameters and analysis.
    """
    
    def __init__(
        self,
        frequencies: List[float],
        coupling: float = 0.3,
        noise_intensity: float = 0.1,
        correlation_time: float = 1.0
    ):
        """
        Initialize ColoredNoiseKuramoto model.
        
        Args:
            frequencies: Natural frequencies
            coupling: Coupling strength
            noise_intensity: Noise amplitude
            correlation_time: OU correlation time
        """
        super().__init__(
            frequencies,
            coupling=coupling,
            noise_intensity=noise_intensity,
            noise_type="colored",
            correlation_time=correlation_time
        )
    
    def set_correlation_time(self, tau: float) -> None:
        """Set correlation time dynamically."""
        self.tau = max(0.01, tau)
    
    def get_stationary_variance(self) -> float:
        """
        Get OU process stationary distribution parameters.
        For OU: variance = σ²/(2/τ) = σ²τ/2
        """
        return (self.sigma ** 2) * self.tau / 2
    
    def is_equilibrated(self, threshold: float = 0.1) -> bool:
        """
        Check if OU processes have equilibrated.
        
        Args:
            threshold: Tolerance for equilibration
        """
        expected_var = self.get_stationary_variance()
        actual_var = float(np.mean(self.colored_noise_state ** 2))
        
        if expected_var == 0:
            return True
        return abs(actual_var - expected_var) / expected_var < threshold
    
    def noise_power_spectrum(
        self,
        max_freq: float = 10.0,
        resolution: float = 0.1
    ) -> List[Dict[str, float]]:
        """
        Get theoretical power spectrum of OU noise.
        
        Args:
            max_freq: Maximum frequency
            resolution: Frequency resolution
            
        Returns:
            Theoretical OU spectrum S(ω) = 2σ²τ / (1 + (ωτ)²)
        """
        spectrum = []
        omega = 0.0
        
        while omega <= max_freq:
            theoretical = 2 * self.sigma ** 2 * self.tau / (1 + (omega * self.tau) ** 2)
            spectrum.append({"omega": omega, "power": theoretical})
            omega += resolution
        
        return spectrum


# =============================================================================
# THERMAL KURAMOTO
# =============================================================================

class ThermalKuramoto(StochasticKuramoto):
    """
    ThermalKuramoto - Temperature-controlled synchronization
    
    Models thermal effects on oscillator synchronization:
    - High temperature: Strong fluctuations, weak effective coupling
    - Low temperature: Weak fluctuations, strong effective coupling
    
    Critical temperature T_c ≈ K (coupling strength)
    """
    
    def __init__(
        self,
        frequencies: List[float],
        coupling: float = 0.3,
        temperature: float = 1.0
    ):
        """
        Initialize ThermalKuramoto model.
        
        Args:
            frequencies: Natural frequencies
            coupling: Coupling strength
            temperature: Initial temperature
        """
        super().__init__(
            frequencies,
            coupling=coupling,
            noise_type="white",
            temperature=temperature,
            temperature_coupling=True
        )
        
        # Noise intensity proportional to √T (fluctuation-dissipation)
        self._update_noise_from_temperature()
    
    def _update_noise_from_temperature(self) -> None:
        """Update noise intensity from temperature."""
        # Fluctuation-dissipation: σ² ∝ T
        self.sigma = np.sqrt(self.temperature) * 0.1
    
    def set_temperature(self, T: float) -> None:
        """Set temperature and update noise."""
        super().set_temperature(T)
        self._update_noise_from_temperature()
    
    def estimate_critical_temperature(self) -> float:
        """
        Estimate critical temperature from current state.
        T_c ≈ K for all-to-all coupling with uniform frequencies
        """
        # Approximate T_c = K * (frequency spread factor)
        freq_spread = float(np.std(self.frequencies))
        return self.coupling * (1 + freq_spread)
    
    def temperature_sweep(
        self,
        T_min: float = 0.1,
        T_max: float = 2.0,
        steps: int = 20,
        equilibration_steps: int = 100
    ) -> List[Dict]:
        """
        Perform temperature sweep to find transition.
        
        Args:
            T_min: Minimum temperature
            T_max: Maximum temperature
            steps: Number of temperature steps
            equilibration_steps: Steps to equilibrate at each T
            
        Returns:
            Results for each temperature
        """
        results = []
        
        for i in range(steps):
            T = T_min + (T_max - T_min) * i / (steps - 1)
            self.set_temperature(T)
            
            # Equilibrate
            self.evolve(equilibration_steps, 0.01)
            
            # Measure
            stats = self.order_parameter_with_uncertainty(50, 0.01)
            
            results.append({
                "temperature": T,
                "order_parameter": stats["mean"],
                "std_dev": stats["std_dev"],
                "confidence_95": stats["confidence_95"]
            })
        
        return results
    
    def is_ordered(self, threshold: float = 0.5) -> bool:
        """
        Check if system is in ordered (synchronized) phase.
        
        Args:
            threshold: Order parameter threshold
        """
        return abs(self.order_parameter()) > threshold
    
    def is_near_critical(self, tolerance: float = 0.2) -> bool:
        """
        Check if system is at or near critical temperature.
        
        Args:
            tolerance: Tolerance factor
        """
        T_c = self.estimate_critical_temperature()
        if T_c == 0:
            return False
        return abs(self.temperature - T_c) / T_c < tolerance


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def gaussian_random() -> float:
    """Generate standard normal random variable."""
    return float(np.random.randn())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "StochasticKuramoto",
    "ColoredNoiseKuramoto",
    "ThermalKuramoto",
    "gaussian_random",
]