"""
Base Oscillator Classes

Provides fundamental phase-amplitude oscillators:
- Oscillator: Base class with phase, frequency, amplitude, and excitation
- OscillatorBank: Collection of coupled oscillators

Oscillators start QUIESCENT (amplitude = 0) and must be EXCITED
by input to become active. This ensures the field response
reflects the input, not a default full-amplitude state.
"""

import math
import random
from typing import List, Optional, Callable, Dict, Any


class Oscillator:
    """
    Base class for phase-amplitude oscillator.
    
    Oscillators start quiescent (amplitude = 0) and must be excited
    to become active.
    
    Attributes:
        freq: Natural frequency (Hz)
        phase: Current phase (radians, 0 to 2π)
        amplitude: Current amplitude (0 to 1)
        base_amplitude: Initial amplitude for reset
        phase_history: History of phase values
    """
    
    def __init__(
        self,
        frequency: float,
        phase: float = 0.0,
        amplitude: float = 0.0  # Start quiescent!
    ):
        """
        Initialize oscillator.
        
        Args:
            frequency: Natural frequency in Hz
            phase: Initial phase in radians
            amplitude: Initial amplitude (0 = quiescent, 1 = fully active)
        """
        self.freq = frequency
        self.phase = phase
        self.amplitude = amplitude
        self.base_amplitude = amplitude
        self.phase_history: List[float] = []
        self._max_history = 100
    
    def tick(self, dt: float, coupling: float = 0.0) -> None:
        """
        Advance oscillator by time step.
        
        Args:
            dt: Time step
            coupling: External coupling force
        """
        self.phase = (self.phase + 2 * math.pi * self.freq * dt + coupling) % (2 * math.pi)
        self.phase_history.append(self.phase)
        if len(self.phase_history) > self._max_history:
            self.phase_history.pop(0)
    
    def excite(self, amount: float = 0.5) -> None:
        """
        Excite oscillator to increase amplitude.
        
        Args:
            amount: Amount to add to amplitude
        """
        self.amplitude = min(1.0, self.amplitude + amount)
    
    def decay(self, rate: float = 0.02, dt: float = 1.0) -> None:
        """
        Apply decay to amplitude.
        
        Args:
            rate: Decay rate
            dt: Time step
        """
        self.amplitude *= (1 - rate * dt)
    
    def get_state(self) -> Dict[str, float]:
        """Get current oscillator state."""
        return {
            'freq': self.freq,
            'phase': self.phase,
            'amplitude': self.amplitude
        }
    
    def reset(self) -> None:
        """Reset oscillator to quiescent state."""
        self.phase = 0.0
        self.amplitude = 0.0  # Reset to quiescent, not full amplitude!
        self.base_amplitude = 0.0
        self.phase_history = []
    
    def output(self) -> float:
        """
        Get current oscillator output value.
        
        Returns:
            amplitude * sin(phase)
        """
        return self.amplitude * math.sin(self.phase)
    
    def cosine_output(self) -> float:
        """
        Get cosine component of output.
        
        Returns:
            amplitude * cos(phase)
        """
        return self.amplitude * math.cos(self.phase)
    
    def __repr__(self) -> str:
        return f"Oscillator(freq={self.freq:.3f}, phase={self.phase:.3f}, amp={self.amplitude:.3f})"


class OscillatorBank:
    """
    Collection of coupled oscillators.
    
    Provides bulk operations on multiple oscillators,
    including coupled evolution and decay.
    """
    
    def __init__(self, frequencies: List[float]):
        """
        Initialize oscillator bank.
        
        Args:
            frequencies: List of oscillator frequencies
        """
        self.oscillators = [Oscillator(f) for f in frequencies]
    
    @property
    def size(self) -> int:
        """Number of oscillators."""
        return len(self.oscillators)
    
    def tick(
        self,
        dt: float,
        coupling_fn: Optional[Callable[['Oscillator', List['Oscillator']], float]] = None
    ) -> None:
        """
        Advance all oscillators by time step.
        
        Args:
            dt: Time step
            coupling_fn: Optional function to compute coupling for each oscillator
        """
        for osc in self.oscillators:
            coupling = coupling_fn(osc, self.oscillators) if coupling_fn else 0.0
            osc.tick(dt, coupling)
    
    def excite_by_indices(self, indices: List[int], amount: float = 0.5) -> None:
        """
        Excite oscillators at specific indices.
        
        Args:
            indices: Indices of oscillators to excite
            amount: Excitation amount
        """
        for idx in indices:
            if 0 <= idx < len(self.oscillators):
                self.oscillators[idx].excite(amount)
    
    def excite_all(self, amount: float = 0.5) -> None:
        """Excite all oscillators."""
        for osc in self.oscillators:
            osc.excite(amount)
    
    def decay_all(self, rate: float = 0.02, dt: float = 1.0) -> None:
        """
        Apply decay to all oscillators.
        
        Args:
            rate: Decay rate
            dt: Time step
        """
        for osc in self.oscillators:
            osc.decay(rate, dt)
    
    def get_state(self) -> List[Dict[str, float]]:
        """Get state of all oscillators."""
        return [o.get_state() for o in self.oscillators]
    
    def get_amplitudes(self) -> List[float]:
        """Get amplitudes of all oscillators."""
        return [o.amplitude for o in self.oscillators]
    
    def get_phases(self) -> List[float]:
        """Get phases of all oscillators."""
        return [o.phase for o in self.oscillators]
    
    def get_outputs(self) -> List[float]:
        """Get output values of all oscillators."""
        return [o.output() for o in self.oscillators]
    
    def reset(self) -> None:
        """Reset all oscillators."""
        for osc in self.oscillators:
            osc.reset()
    
    def active_count(self, threshold: float = 0.1) -> int:
        """Count oscillators with amplitude above threshold."""
        return sum(1 for o in self.oscillators if o.amplitude > threshold)
    
    def mean_amplitude(self) -> float:
        """Calculate mean amplitude across all oscillators."""
        if not self.oscillators:
            return 0.0
        return sum(o.amplitude for o in self.oscillators) / len(self.oscillators)
    
    def mean_phase(self) -> float:
        """Calculate mean phase (circular mean)."""
        if not self.oscillators:
            return 0.0
        sin_sum = sum(math.sin(o.phase) for o in self.oscillators)
        cos_sum = sum(math.cos(o.phase) for o in self.oscillators)
        return math.atan2(sin_sum, cos_sum)
    
    def phase_coherence(self) -> float:
        """
        Calculate phase coherence (order parameter magnitude).
        
        Returns:
            Value in [0, 1], 1 = perfect synchronization
        """
        if not self.oscillators:
            return 0.0
        n = len(self.oscillators)
        sin_sum = sum(math.sin(o.phase) for o in self.oscillators)
        cos_sum = sum(math.cos(o.phase) for o in self.oscillators)
        return math.sqrt(sin_sum**2 + cos_sum**2) / n
    
    def __len__(self) -> int:
        return len(self.oscillators)
    
    def __getitem__(self, idx: int) -> Oscillator:
        return self.oscillators[idx]
    
    def __iter__(self):
        return iter(self.oscillators)
    
    def __repr__(self) -> str:
        return f"OscillatorBank(size={self.size}, active={self.active_count()}, coherence={self.phase_coherence():.3f})"


class DrivenOscillator(Oscillator):
    """
    Oscillator with external driving force.
    
    Adds a sinusoidal driving term at specified frequency and amplitude.
    """
    
    def __init__(
        self,
        frequency: float,
        drive_frequency: float = 1.0,
        drive_amplitude: float = 0.1,
        phase: float = 0.0,
        amplitude: float = 0.0
    ):
        """
        Initialize driven oscillator.
        
        Args:
            frequency: Natural frequency
            drive_frequency: Driving force frequency
            drive_amplitude: Driving force amplitude
            phase: Initial phase
            amplitude: Initial amplitude
        """
        super().__init__(frequency, phase, amplitude)
        self.drive_freq = drive_frequency
        self.drive_amp = drive_amplitude
        self.drive_phase = 0.0
    
    def tick(self, dt: float, coupling: float = 0.0) -> None:
        """
        Advance oscillator with driving force.
        
        Args:
            dt: Time step
            coupling: External coupling
        """
        # Drive force contribution
        drive_force = self.drive_amp * math.sin(self.drive_phase)
        
        # Update phase with natural frequency, coupling, and drive
        self.phase = (self.phase + 2 * math.pi * self.freq * dt + coupling + drive_force * dt) % (2 * math.pi)
        
        # Update drive phase
        self.drive_phase = (self.drive_phase + 2 * math.pi * self.drive_freq * dt) % (2 * math.pi)
        
        self.phase_history.append(self.phase)
        if len(self.phase_history) > self._max_history:
            self.phase_history.pop(0)


class NoisyOscillator(Oscillator):
    """
    Oscillator with stochastic noise term.
    
    Adds random perturbations to phase evolution.
    """
    
    def __init__(
        self,
        frequency: float,
        noise_strength: float = 0.1,
        phase: float = 0.0,
        amplitude: float = 0.0
    ):
        """
        Initialize noisy oscillator.
        
        Args:
            frequency: Natural frequency
            noise_strength: Standard deviation of noise
            phase: Initial phase
            amplitude: Initial amplitude
        """
        super().__init__(frequency, phase, amplitude)
        self.noise_strength = noise_strength
    
    def tick(self, dt: float, coupling: float = 0.0) -> None:
        """
        Advance oscillator with noise.
        
        Args:
            dt: Time step
            coupling: External coupling
        """
        noise = random.gauss(0, self.noise_strength * math.sqrt(dt))
        self.phase = (self.phase + 2 * math.pi * self.freq * dt + coupling + noise) % (2 * math.pi)
        
        self.phase_history.append(self.phase)
        if len(self.phase_history) > self._max_history:
            self.phase_history.pop(0)