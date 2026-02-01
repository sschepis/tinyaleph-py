"""
Holographic Quantum Encoding (HQE)

Implements the holographic projection mechanism from "A Design for a
Sentient Observer" paper, Section 5.

Key features:
- Discrete Fourier Transform holographic projection
- Spatial interference patterns from prime states
- Pattern reconstruction via inverse DFT
- Similarity metrics between holographic patterns
- Distributed, non-local semantic representation
- Dynamic λ(t) stabilization control (equation 12)
- Tick gating for discrete-time operations
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.complex import Complex
from ..core.primes import first_n_primes
from ..hilbert.state import PrimeState


class TickGate:
    """
    Tick Gate for HQE operations.
    
    From discrete.pdf: HQE computation should only occur on valid tick events
    to prevent continuous CPU usage and maintain discrete-time semantics.
    
    Tick conditions:
    1. Minimum time elapsed since last tick
    2. Coherence threshold crossed
    3. External event trigger
    """
    
    def __init__(
        self,
        min_tick_interval: float = 16.0,  # ~60fps max
        coherence_threshold: float = 0.7,
        max_tick_history: int = 50,
        mode: str = "adaptive",  # 'strict' | 'adaptive' | 'free'
    ):
        """
        Initialize tick gate.
        
        Args:
            min_tick_interval: Minimum ms between ticks
            coherence_threshold: Coherence threshold for auto-tick
            max_tick_history: Maximum tick history size
            mode: Gating mode ('strict', 'adaptive', 'free')
        """
        self.min_tick_interval = min_tick_interval
        self.coherence_threshold = coherence_threshold
        self.max_tick_history = max_tick_history
        self.mode = mode
        
        self.last_tick_time: float = 0.0
        self.tick_count: int = 0
        self.pending_tick: bool = False
        self.tick_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.gated_count: int = 0
        self.passed_count: int = 0
    
    def tick(self) -> None:
        """Register an external tick event."""
        now = time.time() * 1000
        self.pending_tick = True
        self.last_tick_time = now
        self.tick_count += 1
        
        self.tick_history.append({
            "time": now,
            "count": self.tick_count,
        })
        
        if len(self.tick_history) > self.max_tick_history:
            self.tick_history.pop(0)
    
    def should_process(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if an operation should proceed (tick gate).
        
        Args:
            state: Current system state with optional 'coherence'
            
        Returns:
            Gate result with shouldPass, reason, etc.
        """
        state = state or {}
        now = time.time() * 1000
        time_since_last = now - self.last_tick_time
        coherence = state.get("coherence", 0)
        
        should_pass = False
        reason = ""
        
        if self.mode == "free":
            should_pass = True
            reason = "free_mode"
        
        elif self.mode == "strict":
            should_pass = self.pending_tick
            reason = "pending_tick" if should_pass else "no_tick"
        
        else:  # adaptive
            if self.pending_tick:
                should_pass = True
                reason = "pending_tick"
            elif (coherence >= self.coherence_threshold and 
                  time_since_last >= self.min_tick_interval):
                should_pass = True
                reason = "coherence_threshold"
                self.tick()
            elif time_since_last >= self.min_tick_interval * 10:
                should_pass = True
                reason = "timeout_fallback"
                self.tick()
            else:
                reason = "gated"
        
        # Clear pending tick if we're processing
        if should_pass:
            self.pending_tick = False
            self.passed_count += 1
        else:
            self.gated_count += 1
        
        return {
            "shouldPass": should_pass,
            "reason": reason,
            "tickCount": self.tick_count,
            "timeSinceLastTick": time_since_last,
            "coherence": coherence,
            "mode": self.mode,
        }
    
    def get_tick_rate(self) -> float:
        """Get ticks per second."""
        if len(self.tick_history) < 2:
            return 0.0
        
        recent = self.tick_history[-10:]
        duration = recent[-1]["time"] - recent[0]["time"]
        
        if duration <= 0:
            return 0.0
        return ((len(recent) - 1) / duration) * 1000
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gating statistics."""
        total = self.passed_count + self.gated_count
        return {
            "tickCount": self.tick_count,
            "tickRate": self.get_tick_rate(),
            "passedCount": self.passed_count,
            "gatedCount": self.gated_count,
            "gateRatio": self.gated_count / total if total > 0 else 0,
            "mode": self.mode,
            "lastTickTime": self.last_tick_time,
        }
    
    def reset(self) -> None:
        """Reset gate state."""
        self.tick_count = 0
        self.last_tick_time = 0.0
        self.pending_tick = False
        self.tick_history = []
        self.gated_count = 0
        self.passed_count = 0
    
    def set_mode(self, mode: str) -> None:
        """Set gating mode."""
        if mode in ["strict", "adaptive", "free"]:
            self.mode = mode


class StabilizationController:
    """
    Stabilization Controller (equation 12).
    
    Implements dynamic λ(t):
    λ(t) = λ₀ · σ(a_C·C(t) - a_S·S(t) - a_SMF·S_SMF(s(t)))
    
    Controls the "condensation pressure" - balance between
    unitary evolution and dissipative stabilization.
    """
    
    def __init__(
        self,
        lambda0: float = 0.1,
        a_c: float = 1.0,
        a_s: float = 0.8,
        a_smf: float = 0.5,
        steepness: float = 2.0,
        lambda_min: float = 0.01,
        lambda_max: float = 0.5,
        max_history: int = 100,
    ):
        """
        Initialize stabilization controller.
        
        Args:
            lambda0: Base stabilization rate λ₀
            a_c: Coherence weight
            a_s: Entropy weight
            a_smf: SMF entropy weight
            steepness: Sigmoid steepness
            lambda_min: Minimum λ
            lambda_max: Maximum λ
            max_history: History size
        """
        self.lambda0 = lambda0
        self.a_c = a_c
        self.a_s = a_s
        self.a_smf = a_smf
        self.steepness = steepness
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.max_history = max_history
        
        self.history: List[Dict[str, Any]] = []
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid squashing function σ. Maps (-∞, ∞) → (0, 1)."""
        return 1.0 / (1.0 + math.exp(-self.steepness * x))
    
    def compute_lambda(
        self,
        coherence: float,
        entropy: float,
        smf_entropy: float = 0.0,
    ) -> float:
        """
        Compute current λ(t) value.
        
        λ(t) = λ₀ · σ(a_C·C(t) - a_S·S(t) - a_SMF·S_SMF(s(t)))
        
        Args:
            coherence: Global coherence C(t) ∈ [0, 1]
            entropy: System entropy S(t)
            smf_entropy: SMF entropy S_SMF
            
        Returns:
            Stabilization rate λ(t)
        """
        arg = self.a_c * coherence - self.a_s * entropy - self.a_smf * smf_entropy
        
        # Apply sigmoid and scale by λ₀
        lam = self.lambda0 * self.sigmoid(arg)
        
        # Clamp to bounds
        clamped = max(self.lambda_min, min(self.lambda_max, lam))
        
        # Record to history
        self.history.append({
            "timestamp": time.time() * 1000,
            "coherence": coherence,
            "entropy": entropy,
            "smfEntropy": smf_entropy,
            "arg": arg,
            "lambda": clamped,
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return clamped
    
    def interpret(self, lam: float) -> str:
        """Get interpretation of current λ."""
        if lam > 0.3:
            return "high_stabilization"
        elif lam > 0.1:
            return "normal"
        else:
            return "low_stabilization"
    
    def get_trend(self) -> float:
        """Get recent lambda trend."""
        if len(self.history) < 5:
            return 0.0
        
        recent = self.history[-10:]
        mid = len(recent) // 2
        first = recent[:mid]
        second = recent[mid:]
        
        first_avg = sum(h["lambda"] for h in first) / len(first) if first else 0
        second_avg = sum(h["lambda"] for h in second) / len(second) if second else 0
        
        return second_avg - first_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        if not self.history:
            return {"current": self.lambda0, "mean": self.lambda0, "trend": 0.0}
        
        lambdas = [h["lambda"] for h in self.history]
        current = lambdas[-1]
        mean = sum(lambdas) / len(lambdas)
        
        return {
            "current": current,
            "mean": mean,
            "min": min(lambdas),
            "max": max(lambdas),
            "trend": self.get_trend(),
            "interpretation": self.interpret(current),
        }
    
    def reset(self) -> None:
        """Reset controller."""
        self.history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config": {
                "lambda0": self.lambda0,
                "a_c": self.a_c,
                "a_s": self.a_s,
                "a_smf": self.a_smf,
            },
            "stats": self.get_stats(),
        }


class HolographicEncoder:
    """
    Holographic Quantum Encoding system.
    
    Projects prime-amplitude states into spatial interference patterns
    using DFT, enabling distributed, reconstruction-capable memory.
    
    Equations:
    - (13) H(x,y,t) = Σ_p α_p(t) exp(i[k_p·r + φ_p(t)])
    - (14) I(x,y,t) = |H(x,y,t)|²
    - (15) Reconstruction via inverse DFT
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        primes: Optional[List[int]] = None,
        num_primes: int = 64,
        wavelength_scale: float = 10.0,
        phase_offset: float = 0.0,
        stabilization_options: Optional[Dict[str, Any]] = None,
        tick_gate_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize holographic encoder.
        
        Args:
            grid_size: Size of the 2D holographic grid
            primes: Primes to use (or None to generate)
            num_primes: Number of primes to generate if primes is None
            wavelength_scale: Wavelength scaling factor
            phase_offset: Global phase offset
            stabilization_options: Options for StabilizationController
            tick_gate_options: Options for TickGate
        """
        self.grid_size = grid_size
        
        # Handle primes
        if primes is not None:
            self.primes = list(primes)
        else:
            self.primes = first_n_primes(num_primes)
        
        self.prime_to_index = {p: i for i, p in enumerate(self.primes)}
        
        self.wavelength_scale = wavelength_scale
        self.phase_offset = phase_offset
        
        # Stabilization controller for dynamic λ(t)
        self.stabilization = StabilizationController(**(stabilization_options or {}))
        
        # Tick gate for HQE operations
        self.tick_gate = TickGate(**(tick_gate_options or {}))
        
        # Precompute spatial frequencies for each prime
        self.spatial_frequencies = self._compute_spatial_frequencies()
        
        # Holographic field (2D complex array)
        self.field = self._create_field()
    
    def _create_field(self) -> List[List[Complex]]:
        """Create an empty holographic field."""
        return [
            [Complex(0, 0) for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]
    
    def _compute_spatial_frequencies(self) -> List[Dict[str, Any]]:
        """Compute spatial frequencies for each prime using golden ratio spiral."""
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        frequencies = []
        
        for i, p in enumerate(self.primes):
            # Use logarithmic prime mapping for wavelength
            wavelength = self.wavelength_scale * (1 + math.log(p) / math.log(2))
            k = 2 * math.pi / wavelength
            
            # Distribute angles using golden ratio for optimal coverage
            angle = 2 * math.pi * i * phi
            
            frequencies.append({
                "prime": p,
                "kx": k * math.cos(angle),
                "ky": k * math.sin(angle),
                "wavelength": wavelength,
            })
        
        return frequencies
    
    def project(
        self,
        state: Any,
        clear: bool = True,
    ) -> List[List[Complex]]:
        """
        Project a prime state into the holographic field (equation 13).
        
        H(x,y,t) = Σ_p α_p(t) exp(i[k_p·r + φ_p(t)])
        
        Args:
            state: PrimeState or dict mapping primes to Complex amplitudes
            clear: Whether to clear field before projection
            
        Returns:
            The holographic field
        """
        if clear:
            self.clear_field()
        
        # Handle different state formats
        if isinstance(state, PrimeState):
            amplitudes = [state.get(p) or Complex(0, 0) for p in self.primes]
        elif isinstance(state, dict):
            amplitudes = []
            for p in self.primes:
                amp = state.get(p)
                if amp is None:
                    amplitudes.append(Complex(0, 0))
                elif isinstance(amp, Complex):
                    amplitudes.append(amp)
                elif isinstance(amp, (int, float)):
                    amplitudes.append(Complex(amp, 0))
                elif hasattr(amp, "re"):
                    amplitudes.append(Complex(amp.re, getattr(amp, "im", 0)))
                else:
                    amplitudes.append(Complex(0, 0))
        else:
            raise ValueError("Invalid state format")
        
        # Project each prime's contribution
        for i, freq in enumerate(self.spatial_frequencies):
            alpha = amplitudes[i]
            
            if alpha.norm() < 1e-10:
                continue
            
            # Add this prime's plane wave to the field
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    # k·r = kx*x + ky*y
                    phase = freq["kx"] * x + freq["ky"] * y + self.phase_offset
                    wave = Complex.from_polar(1, phase)
                    
                    # H(x,y) += α_p * exp(i*k·r)
                    self.field[x][y] = self.field[x][y].add(alpha.mul(wave))
        
        return self.field
    
    def reconstruct(self) -> Dict[int, Complex]:
        """
        Reconstruct amplitudes from holographic field (equation 15).
        
        Uses inverse DFT to recover prime amplitudes.
        
        Returns:
            Dictionary mapping primes to reconstructed amplitudes
        """
        amplitudes = {}
        N2 = self.grid_size * self.grid_size
        
        for i, freq in enumerate(self.spatial_frequencies):
            prime = self.primes[i]
            
            # Inverse DFT at this frequency
            sum_val = Complex(0, 0)
            
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    # Inverse: exp(-i*k·r)
                    phase = -(freq["kx"] * x + freq["ky"] * y + self.phase_offset)
                    wave = Complex.from_polar(1, phase)
                    
                    sum_val = sum_val.add(self.field[x][y].mul(wave))
            
            # Normalize by grid size
            amplitudes[prime] = Complex(sum_val.re / N2, sum_val.im / N2)
        
        return amplitudes
    
    def reconstruct_to_state(self) -> PrimeState:
        """Reconstruct to PrimeState."""
        amplitudes = self.reconstruct()
        state = PrimeState(self.primes)
        
        for prime, amp in amplitudes.items():
            state.set(prime, amp)
        
        return state
    
    def intensity(self) -> List[List[float]]:
        """
        Compute intensity pattern (equation 14): I(x,y,t) = |H(x,y,t)|².
        
        Returns:
            2D array of intensities
        """
        return [
            [self.field[x][y].norm_sq() for y in range(self.grid_size)]
            for x in range(self.grid_size)
        ]
    
    def real_part(self) -> List[List[float]]:
        """Get real part of field for visualization."""
        return [
            [self.field[x][y].re for y in range(self.grid_size)]
            for x in range(self.grid_size)
        ]
    
    def phase_pattern(self) -> List[List[float]]:
        """Get phase pattern for visualization."""
        return [
            [self.field[x][y].phase() for y in range(self.grid_size)]
            for x in range(self.grid_size)
        ]
    
    def clear_field(self) -> None:
        """Clear the holographic field."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.field[x][y] = Complex(0, 0)
    
    def superpose(self, other_field: List[List[Complex]]) -> None:
        """Add another field to this one (superposition)."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.field[x][y] = self.field[x][y].add(other_field[x][y])
    
    def scale(self, scalar: float) -> None:
        """Multiply field by a scalar."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.field[x][y] = Complex(
                    self.field[x][y].re * scalar,
                    self.field[x][y].im * scalar,
                )
    
    def clone(self) -> "HolographicEncoder":
        """Clone this encoder with its current field."""
        cloned = HolographicEncoder(
            grid_size=self.grid_size,
            primes=self.primes.copy(),
            wavelength_scale=self.wavelength_scale,
            phase_offset=self.phase_offset,
        )
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cloned.field[x][y] = Complex(
                    self.field[x][y].re,
                    self.field[x][y].im,
                )
        
        return cloned
    
    def total_energy(self) -> float:
        """Compute total field energy."""
        energy = 0.0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                energy += self.field[x][y].norm_sq()
        return energy
    
    def field_entropy(self) -> float:
        """Compute field entropy based on intensity distribution."""
        intensities = self.intensity()
        total = self.total_energy()
        
        if total < 1e-10:
            return 0.0
        
        H = 0.0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                p = intensities[x][y] / total
                if p > 1e-10:
                    H -= p * math.log2(p)
        
        return H
    
    def get_state(self) -> Dict[str, Any]:
        """Get compressed state snapshot for storage."""
        cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.field[x][y]
                if cell.norm_sq() > 1e-10:
                    cells.append({
                        "x": x,
                        "y": y,
                        "re": cell.re,
                        "im": cell.im,
                    })
        
        return {
            "gridSize": self.grid_size,
            "cells": cells,
            "totalEnergy": self.total_energy(),
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from snapshot."""
        if state.get("gridSize") != self.grid_size:
            raise ValueError("Grid size mismatch")
        
        self.clear_field()
        
        for cell in state.get("cells", []):
            self.field[cell["x"]][cell["y"]] = Complex(cell["re"], cell["im"])
    
    def evolve(
        self,
        state: Dict[str, Any],
        dt: float = 0.016,
    ) -> Dict[str, Any]:
        """
        Evolve the holographic field with stabilization (equation 11).
        
        d|Ψ(t)⟩/dt = iĤ|Ψ(t)⟩ - λ(t)D̂(Ψ,s)|Ψ(t)⟩
        
        Args:
            state: System state with coherence, entropy, smfEntropy
            dt: Time step
            
        Returns:
            Evolution result with lambda value
        """
        coherence = state.get("coherence", 0.5)
        entropy = state.get("entropy", 0.5)
        smf_entropy = state.get("smfEntropy", 0)
        
        # Check tick gate before expensive HQE operations
        gate_result = self.tick_gate.should_process({"coherence": coherence})
        
        if not gate_result["shouldPass"]:
            return {
                "lambda": 0,
                "interpretation": "gated",
                "totalEnergy": self.total_energy(),
                "fieldEntropy": self.field_entropy(),
                "gated": True,
                "gateReason": gate_result["reason"],
                "tickCount": gate_result["tickCount"],
            }
        
        # Compute dynamic λ(t)
        lam = self.stabilization.compute_lambda(coherence, entropy, smf_entropy)
        
        # Apply damped evolution
        damping_factor = math.exp(-lam * dt)
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.field[x][y]
                intensity = cell.norm_sq()
                local_damping = damping_factor * (1 + lam * intensity * 0.1)
                
                self.field[x][y] = Complex(
                    cell.re * local_damping,
                    cell.im * local_damping,
                )
        
        return {
            "lambda": lam,
            "interpretation": self.stabilization.interpret(lam),
            "totalEnergy": self.total_energy(),
            "fieldEntropy": self.field_entropy(),
            "gated": False,
            "tickCount": gate_result["tickCount"],
        }
    
    def tick(self) -> None:
        """Register a tick event."""
        self.tick_gate.tick()
    
    def get_tick_stats(self) -> Dict[str, Any]:
        """Get tick gate statistics."""
        return self.tick_gate.get_stats()
    
    def set_tick_mode(self, mode: str) -> None:
        """Set tick gate mode."""
        self.tick_gate.set_mode(mode)
    
    def get_stabilization_stats(self) -> Dict[str, Any]:
        """Get stabilization statistics."""
        return self.stabilization.get_stats()


class HolographicMemory:
    """
    Holographic Memory.
    
    Stores and retrieves patterns using holographic interference.
    Enables content-addressable, distributed, fault-tolerant memory.
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        primes: Optional[List[int]] = None,
        num_primes: int = 64,
        max_memories: int = 100,
        decay_rate: float = 0.01,
        **encoder_options,
    ):
        """
        Initialize holographic memory.
        
        Args:
            grid_size: Size of holographic grid
            primes: Primes to use
            num_primes: Number of primes to generate if primes is None
            max_memories: Maximum memories to store
            decay_rate: Memory decay rate
            **encoder_options: Additional options for HolographicEncoder
        """
        self.encoder = HolographicEncoder(
            grid_size=grid_size,
            primes=primes,
            num_primes=num_primes,
            **encoder_options,
        )
        
        self.memories: List[Dict[str, Any]] = []
        self.max_memories = max_memories
        self.decay_rate = decay_rate
    
    def store(
        self,
        state: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Store a pattern in memory.
        
        Args:
            state: State to store
            metadata: Associated metadata
            
        Returns:
            Memory index
        """
        # Create a new encoder for this memory
        encoder = HolographicEncoder(
            grid_size=self.encoder.grid_size,
            primes=self.encoder.primes.copy(),
            wavelength_scale=self.encoder.wavelength_scale,
            phase_offset=self.encoder.phase_offset,
        )
        
        encoder.project(state)
        
        self.memories.append({
            "encoder": encoder,
            "metadata": metadata or {},
            "timestamp": time.time() * 1000,
            "accessCount": 0,
            "strength": 1.0,
        })
        
        # Prune if over capacity
        if len(self.memories) > self.max_memories:
            self.prune()
        
        return len(self.memories) - 1
    
    def recall(
        self,
        cue: Any,
        threshold: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """
        Recall the best matching memory.
        
        Args:
            cue: Retrieval cue
            threshold: Minimum similarity threshold
            
        Returns:
            Best matching memory or None
        """
        if not self.memories:
            return None
        
        # Project cue to holographic form
        cue_encoder = HolographicEncoder(
            grid_size=self.encoder.grid_size,
            primes=self.encoder.primes.copy(),
        )
        cue_encoder.project(cue)
        
        # Find best match
        best_match = None
        best_score = threshold
        
        for memory in self.memories:
            score = self._correlate(cue_encoder, memory["encoder"])
            if score > best_score:
                best_score = score
                best_match = memory
        
        if best_match:
            best_match["accessCount"] += 1
            best_match["strength"] = min(1.0, best_match["strength"] + 0.1)
            
            return {
                "state": best_match["encoder"].reconstruct_to_state(),
                "metadata": best_match["metadata"],
                "score": best_score,
                "strength": best_match["strength"],
            }
        
        return None
    
    def _correlate(
        self,
        enc1: HolographicEncoder,
        enc2: HolographicEncoder,
    ) -> float:
        """Compute correlation between two holographic fields."""
        sum_product = 0.0
        sum_sq1 = 0.0
        sum_sq2 = 0.0
        
        for x in range(enc1.grid_size):
            for y in range(enc1.grid_size):
                v1 = enc1.field[x][y].norm_sq()
                v2 = enc2.field[x][y].norm_sq()
                
                sum_product += v1 * v2
                sum_sq1 += v1 * v1
                sum_sq2 += v2 * v2
        
        norm = math.sqrt(sum_sq1) * math.sqrt(sum_sq2)
        return sum_product / norm if norm > 1e-10 else 0.0
    
    def decay(self) -> None:
        """Apply decay to all memories."""
        for memory in self.memories:
            memory["strength"] *= (1 - self.decay_rate)
        
        # Remove very weak memories
        self.memories = [m for m in self.memories if m["strength"] > 0.1]
    
    def prune(self) -> None:
        """Prune memories to capacity."""
        if len(self.memories) <= self.max_memories:
            return
        
        # Sort by strength * accessCount
        self.memories.sort(
            key=lambda m: m["strength"] * (m["accessCount"] + 1),
            reverse=True,
        )
        
        # Keep top memories
        self.memories = self.memories[:self.max_memories]
    
    def find_similar(
        self,
        cue: Any,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Find all memories above similarity threshold."""
        cue_encoder = HolographicEncoder(
            grid_size=self.encoder.grid_size,
            primes=self.encoder.primes.copy(),
        )
        cue_encoder.project(cue)
        
        results = []
        for memory in self.memories:
            score = self._correlate(cue_encoder, memory["encoder"])
            if score > threshold:
                results.append({
                    "state": memory["encoder"].reconstruct_to_state(),
                    "metadata": memory["metadata"],
                    "score": score,
                    "strength": memory["strength"],
                })
        
        return sorted(results, key=lambda r: r["score"], reverse=True)
    
    @property
    def count(self) -> int:
        """Get number of stored memories."""
        return len(self.memories)
    
    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gridSize": self.encoder.grid_size,
            "primes": self.encoder.primes,
            "memories": [
                {
                    "state": m["encoder"].get_state(),
                    "metadata": m["metadata"],
                    "timestamp": m["timestamp"],
                    "accessCount": m["accessCount"],
                    "strength": m["strength"],
                }
                for m in self.memories
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HolographicMemory":
        """Load from dictionary."""
        memory = cls(
            grid_size=data["gridSize"],
            primes=data["primes"],
        )
        
        for saved in data.get("memories", []):
            encoder = HolographicEncoder(
                grid_size=data["gridSize"],
                primes=data["primes"],
            )
            encoder.load_state(saved["state"])
            
            memory.memories.append({
                "encoder": encoder,
                "metadata": saved["metadata"],
                "timestamp": saved["timestamp"],
                "accessCount": saved["accessCount"],
                "strength": saved["strength"],
            })
        
        return memory


class HolographicSimilarity:
    """Pattern similarity using holographic comparison."""
    
    def __init__(
        self,
        grid_size: int = 32,
        primes: Optional[List[int]] = None,
        num_primes: int = 32,
    ):
        """
        Initialize similarity calculator.
        
        Args:
            grid_size: Holographic grid size
            primes: Primes to use
            num_primes: Number of primes if primes is None
        """
        self.encoder1 = HolographicEncoder(
            grid_size=grid_size,
            primes=primes,
            num_primes=num_primes,
        )
        self.encoder2 = HolographicEncoder(
            grid_size=grid_size,
            primes=primes,
            num_primes=num_primes,
        )
    
    def similarity(self, state1: Any, state2: Any) -> float:
        """
        Compute holographic similarity between two states.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Similarity score (0-1)
        """
        self.encoder1.project(state1)
        self.encoder2.project(state2)
        
        # Compute intensity correlation
        I1 = self.encoder1.intensity()
        I2 = self.encoder2.intensity()
        
        sum_product = 0.0
        sum_sq1 = 0.0
        sum_sq2 = 0.0
        
        for x in range(self.encoder1.grid_size):
            for y in range(self.encoder1.grid_size):
                sum_product += I1[x][y] * I2[x][y]
                sum_sq1 += I1[x][y] * I1[x][y]
                sum_sq2 += I2[x][y] * I2[x][y]
        
        norm = math.sqrt(sum_sq1) * math.sqrt(sum_sq2)
        return sum_product / norm if norm > 1e-10 else 0.0
    
    def difference(self, state1: Any, state2: Any) -> HolographicEncoder:
        """
        Compute difference pattern between two states.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Difference encoder
        """
        self.encoder1.project(state1)
        self.encoder2.project(state2)
        
        diff = self.encoder1.clone()
        
        for x in range(diff.grid_size):
            for y in range(diff.grid_size):
                diff.field[x][y] = Complex(
                    diff.field[x][y].re - self.encoder2.field[x][y].re,
                    diff.field[x][y].im - self.encoder2.field[x][y].im,
                )
        
        return diff