"""
AlephEngine - Unified backend-agnostic prime-based computing engine

FIELD-BASED COMPUTATION:
The answer emerges from oscillator dynamics, not symbolic manipulation.
We excite the field, evolve it, and sample at coherent emission moments.

Pipeline: encode → excite → evolve → sample → decode
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Set, Protocol
from abc import ABC, abstractmethod
import time
import math

import numpy as np
from tinyaleph.core.primes import primes_up_to, prime_to_frequency
from tinyaleph.physics.kuramoto import KuramotoModel
from tinyaleph.physics.entropy import state_entropy, shannon_entropy


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Frame:
    """Snapshot of coherent field state."""
    step: int
    order: float
    differential: float
    amplitudes: List[float]
    entropy: float


@dataclass
class HistoryEntry:
    """History entry for tracking computations."""
    timestamp: float
    input: Any
    output: Any
    entropy: float
    field_based: bool


@dataclass
class ReasoningStep:
    """One step in entropy-minimizing reasoning."""
    step: int
    transform: str
    entropy_drop: float
    primes: List[int]


@dataclass
class RunResult:
    """Result of engine run."""
    input: Any
    input_primes: List[int]
    result_primes: List[int]
    output: Any
    entropy: float
    coherence: float
    lyapunov: float
    stability: str
    collapsed: bool
    steps: List[ReasoningStep]
    evolution_steps: int
    frames_collected: int
    best_frame_order: float
    best_differential: float
    field_based: bool
    order_parameter: float


# =============================================================================
# DEFAULT BACKEND
# =============================================================================

class DefaultBackend:
    """
    Default backend using prime factorization.
    
    Encodes integers to prime factors, decodes by multiplication.
    """
    
    def __init__(self, dimension: int = 64, max_prime: int = 500):
        """
        Initialize default backend.
        
        Args:
            dimension: State vector dimension
            max_prime: Maximum prime to use
        """
        self._dimension = dimension
        self._primes = primes_up_to(max_prime)
        
        # Standard transforms: multiply by small primes
        self._transforms = [
            {"n": p, "name": f"mul_{p}"} for p in self._primes[:10]
        ] + [
            {"n": -p, "name": f"div_{p}"} for p in self._primes[:10]
        ]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def get_name(self) -> str:
        return "DefaultBackend"
    
    def get_primes(self) -> List[int]:
        return self._primes
    
    def primes_to_frequencies(self, primes: List[int]) -> List[float]:
        """Convert primes to frequencies using logarithmic spacing."""
        return [prime_to_frequency(p) for p in primes]
    
    def encode(self, data: Any) -> List[int]:
        """
        Encode data to primes.
        - Integer: prime factorization
        - String: character codes to primes
        - List: flatten and encode each element
        """
        if isinstance(data, int):
            return self._factorize(abs(data)) if data != 0 else [2]
        elif isinstance(data, str):
            primes = []
            for char in data:
                code = ord(char)
                primes.extend(self._factorize(code) if code > 1 else [2])
            return primes
        elif isinstance(data, (list, tuple)):
            primes = []
            for item in data:
                primes.extend(self.encode(item))
            return primes
        elif isinstance(data, float):
            return self._factorize(int(abs(data) * 1000)) or [2]
        else:
            # Try to get an integer representation
            return self._factorize(hash(data) % 10000) or [2]
    
    def _factorize(self, n: int) -> List[int]:
        """Prime factorization of n."""
        if n <= 1:
            return [2]
        
        factors = []
        for p in self._primes:
            while n % p == 0:
                factors.append(p)
                n //= p
            if n == 1:
                break
        
        if n > 1:
            factors.append(n)  # n itself is prime
        
        return factors if factors else [2]
    
    def decode(self, primes: List[int]) -> Any:
        """Decode primes by multiplication."""
        if not primes:
            return 1
        
        result = 1
        for p in primes:
            result *= p
            if result > 10**18:
                result %= 10**18  # Prevent overflow
        
        return result
    
    def get_transforms(self) -> List[Dict]:
        return self._transforms
    
    def apply_transform(self, primes: List[int], transform: Dict) -> List[int]:
        """Apply transform: multiply or divide by prime n."""
        n = transform.get("n", 1)
        
        if n > 0:
            # Multiplication: add prime
            return primes + [n]
        elif n < 0:
            # Division: remove prime if present
            n = -n
            result = list(primes)
            if n in result:
                result.remove(n)
            return result
        
        return primes


# =============================================================================
# ALEPH ENGINE
# =============================================================================

class AlephEngine:
    """
    AlephEngine - Unified backend-agnostic prime-based computing engine
    
    FIELD-BASED COMPUTATION:
    The answer emerges from oscillator dynamics, not symbolic manipulation.
    We excite the field with input primes, evolve the oscillators, and sample
    the response at coherent emission moments.
    
    Pipeline: encode → excite → evolve → sample → decode
    """
    
    def __init__(
        self,
        backend: Optional[DefaultBackend] = None,
        base_coupling: float = 0.3,
        max_evolution_steps: int = 100,
        coherence_threshold: float = 0.6,
        amplitude_threshold: float = 0.1,
        sample_window: int = 10,
        dt: float = 0.016
    ):
        """
        Initialize AlephEngine.
        
        Args:
            backend: Compute backend (defaults to DefaultBackend)
            base_coupling: Base Kuramoto coupling strength
            max_evolution_steps: Max timesteps to evolve
            coherence_threshold: Min order parameter for coherent emission
            amplitude_threshold: Min amplitude for active prime
            sample_window: Number of best frames to keep
            dt: Time step size
        """
        self.backend = backend or DefaultBackend()
        
        # Options
        self.base_coupling = base_coupling
        self.max_evolution_steps = max_evolution_steps
        self.coherence_threshold = coherence_threshold
        self.amplitude_threshold = amplitude_threshold
        self.sample_window = sample_window
        self.dt = dt
        
        # Initialize oscillators
        self._initialize_oscillators()
        self._reset_state()
        
        # History tracking
        self.history: List[HistoryEntry] = []
        self.frames: List[Frame] = []
    
    def _initialize_oscillators(self) -> None:
        """Initialize Kuramoto oscillators from backend primes."""
        primes = self.backend.get_primes()[:self.backend.dimension]
        frequencies = self.backend.primes_to_frequencies(primes)
        n = len(frequencies)
        self.oscillators = KuramotoModel(
            n_oscillators=n,
            coupling=self.base_coupling,
            frequencies=np.array(frequencies)
        )
        self.prime_list = primes
        # Track amplitudes (excited oscillators have higher amplitude)
        self.amplitudes = np.ones(n)
    
    def _reset_state(self) -> None:
        """Reset internal state variables."""
        dim = self.backend.dimension
        self.entropy = 0.0
        self.coherence_value = 0.0
        self.lyapunov = 0.0
        self.collapse_integral = 0.0
        self.stability = "MARGINAL"
    
    def excite(self, primes: List[int], intensity: float = 0.5) -> None:
        """
        Excite oscillators corresponding to given primes.
        
        Args:
            primes: List of primes to excite
            intensity: Excitation intensity
        """
        prime_set = set(primes)
        for i, p in enumerate(self.prime_list):
            if p in prime_set:
                # Boost amplitude and set initial phase
                self.amplitudes[i] += intensity
                # Align phases of excited oscillators
                self.oscillators.phases[i] = 0.0
    
    def get_weighted_amplitudes(self) -> np.ndarray:
        """Get amplitudes weighted by phase coherence."""
        # Use amplitude * cos(phase) as weighted amplitude
        return self.amplitudes * np.cos(self.oscillators.phases)
    
    def run(self, input_data: Any) -> RunResult:
        """
        Main processing pipeline: encode → excite → evolve → sample → decode
        
        Args:
            input_data: Any input that the backend can encode
            
        Returns:
            RunResult with output and diagnostics
        """
        # 1. Encode input to primes
        input_primes = self.backend.encode(input_data)
        input_prime_set = set(input_primes)
        
        # 2. Capture baseline before excitation
        baseline_amplitudes = self.get_weighted_amplitudes().copy()
        
        # 3. Excite oscillators corresponding to input primes
        self.excite(input_primes)
        
        # 4. EVOLVE field and collect frames
        self.frames = []
        evolution_steps = 0
        
        max_differential = 0.0
        best_frame: Optional[Frame] = None
        
        for i in range(self.max_evolution_steps):
            # Advance oscillators
            self.oscillators.step(self.dt)
            evolution_steps += 1
            
            # Apply damping to amplitudes
            self.amplitudes *= 0.99
            
            current_amplitudes = self.get_weighted_amplitudes()
            order = float(abs(self.oscillators.order_parameter()))
            
            # Compute input-weighted differential response
            input_response = 0.0
            other_response = 0.0
            
            for j, prime in enumerate(self.prime_list):
                diff = abs(current_amplitudes[j]) - abs(baseline_amplitudes[j])
                if prime in input_prime_set:
                    input_response += diff
                else:
                    other_response += abs(diff)
            
            # Differential: how much MORE the input primes respond
            differential = input_response - other_response * 0.3
            
            # Compute entropy
            probs = np.abs(current_amplitudes) ** 2
            total = np.sum(probs)
            if total > 1e-10:
                probs = probs / total
                self.entropy = -np.sum(probs * np.log2(probs + 1e-10))
            else:
                self.entropy = 0.0
            
            # Sample frames with good differential response
            if differential > 0 and order > self.coherence_threshold * 0.5:
                frame = Frame(
                    step=i,
                    order=order,
                    differential=differential,
                    amplitudes=list(current_amplitudes),
                    entropy=self.entropy
                )
                
                self.frames.append(frame)
                
                if differential > max_differential:
                    max_differential = differential
                    best_frame = frame
                
                # Keep only best frames
                if len(self.frames) > self.sample_window:
                    self.frames.sort(key=lambda f: f.differential, reverse=True)
                    self.frames = self.frames[:self.sample_window]
            
            # Stop if we found a strong coherent input-response
            if differential > 1.0 and order > self.coherence_threshold:
                break
        
        # 5. DECODE from field state
        steps: List[ReasoningStep] = []
        
        if best_frame and best_frame.differential > 0:
            # Field-based: decode from transient amplitudes
            result_primes = self.amplitudes_to_primes(best_frame.amplitudes, input_prime_set)
            output = self.backend.decode(result_primes)
        else:
            # Fallback: return input transformed
            result_primes = input_primes
            output = self.backend.decode(result_primes)
        
        # 6. Update engine state
        self.coherence_value = best_frame.order if best_frame else float(abs(self.oscillators.order_parameter()))
        collapsed = self.coherence_value > 0.9
        
        # 7. Build result
        result = RunResult(
            input=input_data,
            input_primes=input_primes,
            result_primes=result_primes,
            output=output,
            entropy=self.entropy,
            coherence=self.coherence_value,
            lyapunov=self.lyapunov,
            stability=self.stability,
            collapsed=collapsed,
            steps=steps,
            evolution_steps=evolution_steps,
            frames_collected=len(self.frames),
            best_frame_order=best_frame.order if best_frame else 0.0,
            best_differential=best_frame.differential if best_frame else 0.0,
            field_based=best_frame is not None and best_frame.differential > 0,
            order_parameter=float(abs(self.oscillators.order_parameter()))
        )
        
        # 8. Record history
        self.history.append(HistoryEntry(
            timestamp=time.time(),
            input=input_data,
            output=output,
            entropy=self.entropy,
            field_based=result.field_based
        ))
        
        return result
    
    def amplitudes_to_primes(
        self,
        amplitudes: List[float],
        input_primes: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Convert oscillator amplitudes to primes.
        
        Args:
            amplitudes: Current oscillator amplitudes
            input_primes: Set of primes that were excited by input
            
        Returns:
            List of active primes sorted by score
        """
        input_primes = input_primes or set()
        active_primes = []
        
        for i in range(min(len(amplitudes), len(self.prime_list))):
            amplitude = abs(amplitudes[i])
            if amplitude > self.amplitude_threshold:
                prime = self.prime_list[i]
                # Boost score for primes in input
                input_boost = 2.0 if prime in input_primes else 1.0
                
                active_primes.append({
                    "prime": prime,
                    "amplitude": amplitude,
                    "score": amplitude * input_boost
                })
        
        active_primes.sort(key=lambda p: p["score"], reverse=True)
        return [p["prime"] for p in active_primes]
    
    def evolve(self, steps: int = 10) -> List[Dict]:
        """
        Continuously evolve state without new input.
        
        Args:
            steps: Number of evolution steps
            
        Returns:
            List of state snapshots
        """
        states = []
        for i in range(steps):
            self.oscillators.step(self.dt)
            self.amplitudes *= 0.99  # Damping
            
            order = float(abs(self.oscillators.order_parameter()))
            states.append({
                "step": i,
                "order_parameter": order,
                "entropy": self.entropy,
                "stability": self.stability
            })
        return states
    
    def reset(self) -> None:
        """Reset engine state without changing backend."""
        self.oscillators.reset()
        self.amplitudes = np.ones(len(self.prime_list))
        self._reset_state()
        self.history = []
        self.frames = []
    
    def get_history(self, limit: int = 10) -> List[HistoryEntry]:
        """Get processing history."""
        return self.history[-limit:]
    
    def get_backend_info(self) -> Dict:
        """Get backend information."""
        return {
            "name": self.backend.get_name(),
            "dimension": self.backend.dimension,
            "transform_count": len(self.backend.get_transforms()),
            "prime_count": len(self.backend.get_primes())
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AlephEngine",
    "DefaultBackend",
    "Frame",
    "HistoryEntry",
    "ReasoningStep",
    "RunResult",
]