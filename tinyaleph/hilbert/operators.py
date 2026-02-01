"""
Quantum Operators on Prime Hilbert Space H_P

Implements the fundamental operators for quantum computation
in the prime Hilbert space:

1. P̂ (Prime Shift): Shifts prime index: P̂|p_n⟩ = |p_{n+1}⟩
2. F̂ (Fourier Transform): Prime-to-frequency domain transform
3. R̂ (Resonance): Applies prime-resonance interaction
4. Ĉ (Collapse): Measurement operator with Born rule

Mathematical Foundation:
    Let H_P = {|ψ⟩ = Σ_p α_p |p⟩ : Σ|α_p|² = 1, p ∈ P}
    
    These operators preserve or transform states within H_P,
    maintaining unitarity where applicable.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import math
import cmath
from functools import lru_cache

from tinyaleph.core.primes import (
    is_prime, nth_prime, prime_index, next_prime, prev_prime,
    prime_sieve, factorize
)
from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.complex import Complex
from tinyaleph.core.constants import PHI, LAMBDA_STABILITY_THRESHOLD


@dataclass
class PrimeState:
    """
    Quantum state in the prime Hilbert space H_P.
    
    |ψ⟩ = Σ_p α_p |p⟩
    
    Amplitudes are complex numbers satisfying normalization.
    """
    
    amplitudes: Dict[int, Complex] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that all keys are primes."""
        for p in self.amplitudes:
            if not is_prime(p):
                raise ValueError(f"Key {p} is not prime")
    
    @classmethod
    def basis(cls, p: int) -> PrimeState:
        """Create basis state |p⟩."""
        if not is_prime(p):
            raise ValueError(f"{p} is not prime")
        return cls(amplitudes={p: Complex(1.0, 0.0)})
    
    @classmethod
    def superposition(cls, primes: List[int], 
                      phases: Optional[List[float]] = None) -> PrimeState:
        """Create equal superposition with optional phases."""
        n = len(primes)
        if n == 0:
            return cls()
        
        if phases is None:
            phases = [0.0] * n
        
        norm = 1.0 / math.sqrt(n)
        amplitudes = {}
        for p, phase in zip(primes, phases):
            if not is_prime(p):
                raise ValueError(f"{p} is not prime")
            amplitudes[p] = Complex(
                norm * math.cos(phase),
                norm * math.sin(phase)
            )
        
        return cls(amplitudes=amplitudes)
    
    def __add__(self, other: PrimeState) -> PrimeState:
        result = {}
        all_primes = set(self.amplitudes.keys()) | set(other.amplitudes.keys())
        for p in all_primes:
            a1 = self.amplitudes.get(p, Complex(0, 0))
            a2 = other.amplitudes.get(p, Complex(0, 0))
            result[p] = a1 + a2
        return PrimeState(amplitudes=result)
    
    def __mul__(self, scalar: complex) -> PrimeState:
        if isinstance(scalar, Complex):
            c = scalar
        else:
            c = Complex(scalar.real if hasattr(scalar, 'real') else scalar,
                       scalar.imag if hasattr(scalar, 'imag') else 0)
        return PrimeState(
            amplitudes={p: a * c for p, a in self.amplitudes.items()}
        )
    
    def __rmul__(self, scalar: complex) -> PrimeState:
        return self * scalar
    
    def norm_squared(self) -> float:
        return sum(a.magnitude_squared() for a in self.amplitudes.values())
    
    def normalize(self) -> PrimeState:
        n = math.sqrt(self.norm_squared())
        if n < 1e-10:
            return self
        return PrimeState(
            amplitudes={p: a * Complex(1/n, 0) for p, a in self.amplitudes.items()}
        )
    
    def inner_product(self, other: PrimeState) -> Complex:
        """Compute ⟨self|other⟩."""
        result = Complex(0, 0)
        for p in self.amplitudes:
            if p in other.amplitudes:
                result = result + self.amplitudes[p].conjugate() * other.amplitudes[p]
        return result
    
    def probabilities(self) -> Dict[int, float]:
        """Get probability distribution over primes."""
        total = self.norm_squared()
        if total < 1e-10:
            return {}
        return {p: a.magnitude_squared() / total for p, a in self.amplitudes.items()}
    
    def entropy(self) -> float:
        """Compute von Neumann entropy."""
        probs = self.probabilities()
        entropy = 0.0
        for prob in probs.values():
            if prob > 1e-10:
                entropy -= prob * math.log2(prob)
        return entropy


class Operator(ABC):
    """Abstract base class for operators on H_P."""
    
    @abstractmethod
    def apply(self, state: PrimeState) -> PrimeState:
        """Apply operator to state: Ô|ψ⟩."""
        pass
    
    def __call__(self, state: PrimeState) -> PrimeState:
        return self.apply(state)
    
    def compose(self, other: Operator) -> CompositeOperator:
        """Operator composition: (self ∘ other)|ψ⟩ = self(other(|ψ⟩))."""
        return CompositeOperator([self, other])
    
    def __matmul__(self, other: Operator) -> CompositeOperator:
        """@ operator for composition."""
        return self.compose(other)


@dataclass
class CompositeOperator(Operator):
    """Composition of multiple operators."""
    
    operators: List[Operator]
    
    def apply(self, state: PrimeState) -> PrimeState:
        result = state
        # Apply right to left (standard operator composition)
        for op in reversed(self.operators):
            result = op.apply(result)
        return result


@dataclass
class IdentityOperator(Operator):
    """Identity operator: Î|ψ⟩ = |ψ⟩."""
    
    def apply(self, state: PrimeState) -> PrimeState:
        return PrimeState(amplitudes=dict(state.amplitudes))


@dataclass
class PrimeShiftOperator(Operator):
    """
    Prime Shift Operator P̂_k.
    
    Shifts prime indices by k positions:
        P̂_k|p_n⟩ = |p_{n+k}⟩
    
    where p_n is the n-th prime.
    
    For negative k, shifts to smaller primes (clamped at p_1 = 2).
    """
    
    shift: int = 1
    
    def apply(self, state: PrimeState) -> PrimeState:
        result = {}
        
        for p, amp in state.amplitudes.items():
            # Find index of current prime
            idx = prime_index(p)
            
            # Compute new index
            new_idx = idx + self.shift
            
            # Clamp to valid range
            if new_idx < 1:
                new_idx = 1
            
            # Get new prime
            new_p = nth_prime(new_idx)
            
            # Accumulate amplitude (may have collisions from clamping)
            if new_p in result:
                result[new_p] = result[new_p] + amp
            else:
                result[new_p] = amp
        
        return PrimeState(amplitudes=result).normalize()


@dataclass
class PrimeFourierOperator(Operator):
    """
    Prime Fourier Transform F̂.
    
    Maps between prime basis and "frequency" basis using
    prime-weighted phases:
    
        F̂|p⟩ = (1/√N) Σ_q e^(2πi·log(p)·log(q)/log(P)) |q⟩
    
    where P is a normalization prime.
    
    This creates interference patterns based on multiplicative
    structure of primes.
    """
    
    max_primes: int = 100
    
    def apply(self, state: PrimeState) -> PrimeState:
        # Get primes up to limit
        primes = [nth_prime(i) for i in range(1, self.max_primes + 1)]
        n = len(primes)
        norm = 1.0 / math.sqrt(n)
        
        # Normalization constant (use largest prime)
        log_P = math.log(primes[-1])
        
        result = {}
        
        for q in primes:
            log_q = math.log(q)
            amplitude = Complex(0, 0)
            
            for p, amp in state.amplitudes.items():
                # Phase based on multiplicative structure
                log_p = math.log(p)
                phase = 2 * math.pi * log_p * log_q / log_P
                
                # e^(iφ) = cos(φ) + i·sin(φ)
                phase_factor = Complex(math.cos(phase), math.sin(phase))
                amplitude = amplitude + amp * phase_factor
            
            if amplitude.magnitude_squared() > 1e-10:
                result[q] = amplitude * Complex(norm, 0)
        
        return PrimeState(amplitudes=result)
    
    def inverse(self) -> InversePrimeFourierOperator:
        """Get inverse Fourier operator."""
        return InversePrimeFourierOperator(max_primes=self.max_primes)


@dataclass
class InversePrimeFourierOperator(Operator):
    """
    Inverse Prime Fourier Transform F̂†.
    
    Conjugate transpose of F̂.
    """
    
    max_primes: int = 100
    
    def apply(self, state: PrimeState) -> PrimeState:
        primes = [nth_prime(i) for i in range(1, self.max_primes + 1)]
        n = len(primes)
        norm = 1.0 / math.sqrt(n)
        log_P = math.log(primes[-1])
        
        result = {}
        
        for q in primes:
            log_q = math.log(q)
            amplitude = Complex(0, 0)
            
            for p, amp in state.amplitudes.items():
                log_p = math.log(p)
                # Negative phase for inverse
                phase = -2 * math.pi * log_p * log_q / log_P
                phase_factor = Complex(math.cos(phase), math.sin(phase))
                amplitude = amplitude + amp * phase_factor
            
            if amplitude.magnitude_squared() > 1e-10:
                result[q] = amplitude * Complex(norm, 0)
        
        return PrimeState(amplitudes=result)


@dataclass
class ResonanceOperator(Operator):
    """
    Resonance Operator R̂(ω, g).
    
    Applies phase coupling between primes based on resonance:
    
        R̂(ω, g)|ψ⟩ = Σ_p α_p · e^(i·g·Σ_q sin(ω(p-q))) |p⟩
    
    This implements Kuramoto-like coupling in the prime basis.
    
    Parameters:
        frequency: Base frequency ω
        coupling: Coupling strength g
    """
    
    frequency: float = 1.0
    coupling: float = 0.1
    
    def apply(self, state: PrimeState) -> PrimeState:
        primes = list(state.amplitudes.keys())
        result = {}
        
        for p, amp in state.amplitudes.items():
            # Compute total phase shift from resonance with other primes
            phase_shift = 0.0
            
            for q in primes:
                if q != p:
                    phase_shift += math.sin(self.frequency * (p - q))
            
            phase_shift *= self.coupling
            
            # Apply phase
            phase_factor = Complex(
                math.cos(phase_shift),
                math.sin(phase_shift)
            )
            result[p] = amp * phase_factor
        
        return PrimeState(amplitudes=result)


@dataclass
class PhaseOperator(Operator):
    """
    Phase Gate Φ̂(θ).
    
    Applies phase to specific prime:
        Φ̂(p, θ)|p⟩ = e^(iθ) |p⟩
        Φ̂(p, θ)|q⟩ = |q⟩  for q ≠ p
    """
    
    prime: int
    phase: float
    
    def __post_init__(self):
        if not is_prime(self.prime):
            raise ValueError(f"{self.prime} is not prime")
    
    def apply(self, state: PrimeState) -> PrimeState:
        result = dict(state.amplitudes)
        
        if self.prime in result:
            phase_factor = Complex(
                math.cos(self.phase),
                math.sin(self.phase)
            )
            result[self.prime] = result[self.prime] * phase_factor
        
        return PrimeState(amplitudes=result)


@dataclass
class ProjectionOperator(Operator):
    """
    Projection Operator Π̂_S.
    
    Projects onto subspace spanned by primes in set S:
        Π̂_S|ψ⟩ = Σ_{p∈S} α_p |p⟩ / ||...||
    """
    
    primes: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        for p in self.primes:
            if not is_prime(p):
                raise ValueError(f"{p} is not prime")
    
    def apply(self, state: PrimeState) -> PrimeState:
        prime_set = set(self.primes)
        result = {
            p: amp for p, amp in state.amplitudes.items()
            if p in prime_set
        }
        return PrimeState(amplitudes=result).normalize()


@dataclass
class CollapseOperator(Operator):
    """
    Collapse (Measurement) Operator Ĉ.
    
    Performs projective measurement in the prime basis,
    collapsing state to a single eigenstate according to
    Born rule probabilities.
    
    Ĉ|ψ⟩ = |p_k⟩ with probability |α_k|²
    
    If deterministic=True, collapses to highest probability prime.
    """
    
    deterministic: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            import random
            random.seed(self.seed)
    
    def apply(self, state: PrimeState) -> PrimeState:
        probs = state.probabilities()
        
        if not probs:
            return PrimeState()
        
        if self.deterministic:
            # Choose highest probability
            selected = max(probs.keys(), key=lambda p: probs[p])
        else:
            # Sample according to Born rule
            import random
            r = random.random()
            cumulative = 0.0
            selected = list(probs.keys())[-1]
            
            for p, prob in probs.items():
                cumulative += prob
                if r < cumulative:
                    selected = p
                    break
        
        return PrimeState.basis(selected)
    
    def measure_and_result(self, state: PrimeState) -> Tuple[int, PrimeState]:
        """Perform measurement and return (outcome, collapsed_state)."""
        collapsed = self.apply(state)
        outcome = list(collapsed.amplitudes.keys())[0]
        return outcome, collapsed


@dataclass
class HadamardLikeOperator(Operator):
    """
    Hadamard-like operator Ĥ.
    
    Creates superposition from basis state and vice versa.
    For a single prime:
        Ĥ|p⟩ = (1/√2)(|p⟩ + |next_prime(p)⟩)
    
    For superpositions, applies component-wise with interference.
    """
    
    def apply(self, state: PrimeState) -> PrimeState:
        result = {}
        inv_sqrt2 = Complex(1.0 / math.sqrt(2), 0)
        
        for p, amp in state.amplitudes.items():
            # Add component at current prime
            if p in result:
                result[p] = result[p] + amp * inv_sqrt2
            else:
                result[p] = amp * inv_sqrt2
            
            # Add component at next prime
            next_p = next_prime(p)
            if next_p in result:
                result[next_p] = result[next_p] + amp * inv_sqrt2
            else:
                result[next_p] = amp * inv_sqrt2
        
        return PrimeState(amplitudes=result).normalize()


@dataclass  
class EntanglementOperator(Operator):
    """
    Entanglement Operator Ê.
    
    Creates entanglement between prime pairs based on
    their product structure:
    
        Ê|p⟩ = Σ_q α_{p,q} |p*q mod P⟩
    
    where α_{p,q} encodes entanglement strength.
    
    This is a simplified entanglement suitable for prime-based
    distributed computing.
    """
    
    partner_primes: List[int] = field(default_factory=lambda: [2, 3, 5])
    modulus_prime: int = 104729  # 10000th prime
    
    def __post_init__(self):
        for p in self.partner_primes:
            if not is_prime(p):
                raise ValueError(f"Partner {p} is not prime")
    
    def apply(self, state: PrimeState) -> PrimeState:
        result = {}
        n = len(self.partner_primes)
        norm = 1.0 / math.sqrt(n)
        
        for p, amp in state.amplitudes.items():
            for q in self.partner_primes:
                # Product modulo prime
                product = (p * q) % self.modulus_prime
                
                # Find nearest prime to product
                while not is_prime(product) and product > 1:
                    product += 1
                    if product >= self.modulus_prime:
                        product = 2
                
                # Phase from product structure  
                phase = 2 * math.pi * (p * q) / (self.modulus_prime * PHI)
                phase_factor = Complex(
                    math.cos(phase) * norm,
                    math.sin(phase) * norm
                )
                
                new_amp = amp * phase_factor
                
                if product in result:
                    result[product] = result[product] + new_amp
                else:
                    result[product] = new_amp
        
        return PrimeState(amplitudes=result).normalize()


@dataclass
class TimeEvolutionOperator(Operator):
    """
    Time Evolution Operator Û(t) = e^(-iĤt).
    
    Evolves state under Hamiltonian with prime energy levels:
        E_p = ℏω · log(p)
    
    Parameters:
        time: Evolution time t
        frequency: Base frequency ω
    """
    
    time: float
    frequency: float = 1.0
    
    def apply(self, state: PrimeState) -> PrimeState:
        result = {}
        
        for p, amp in state.amplitudes.items():
            # Energy eigenvalue for prime p
            energy = self.frequency * math.log(p)
            
            # Phase from time evolution
            phase = -energy * self.time
            phase_factor = Complex(
                math.cos(phase),
                math.sin(phase)
            )
            
            result[p] = amp * phase_factor
        
        return PrimeState(amplitudes=result)


@dataclass
class GoldenPhaseOperator(Operator):
    """
    Golden Phase Operator Ĝ.
    
    Applies golden-ratio phased rotation across primes:
        Ĝ|p_n⟩ = e^(i·n·2π/Φ²) |p_n⟩
    
    This creates maximum phase spread using the golden angle.
    """
    
    def apply(self, state: PrimeState) -> PrimeState:
        golden_angle = 2 * math.pi / (PHI ** 2)
        result = {}
        
        for p, amp in state.amplitudes.items():
            n = prime_index(p)
            phase = n * golden_angle
            phase_factor = Complex(
                math.cos(phase),
                math.sin(phase)
            )
            result[p] = amp * phase_factor
        
        return PrimeState(amplitudes=result)


# Convenience functions for common operators
def shift(k: int = 1) -> PrimeShiftOperator:
    """Create prime shift operator P̂_k."""
    return PrimeShiftOperator(shift=k)


def fourier(max_primes: int = 100) -> PrimeFourierOperator:
    """Create prime Fourier operator F̂."""
    return PrimeFourierOperator(max_primes=max_primes)


def resonance(frequency: float = 1.0, coupling: float = 0.1) -> ResonanceOperator:
    """Create resonance operator R̂."""
    return ResonanceOperator(frequency=frequency, coupling=coupling)


def collapse(deterministic: bool = False) -> CollapseOperator:
    """Create collapse (measurement) operator Ĉ."""
    return CollapseOperator(deterministic=deterministic)


def phase(prime: int, theta: float) -> PhaseOperator:
    """Create phase gate Φ̂(p, θ)."""
    return PhaseOperator(prime=prime, phase=theta)


def project(primes: List[int]) -> ProjectionOperator:
    """Create projection operator Π̂_S."""
    return ProjectionOperator(primes=primes)


def hadamard() -> HadamardLikeOperator:
    """Create Hadamard-like operator Ĥ."""
    return HadamardLikeOperator()


def evolve(time: float, frequency: float = 1.0) -> TimeEvolutionOperator:
    """Create time evolution operator Û(t)."""
    return TimeEvolutionOperator(time=time, frequency=frequency)


def golden_phase() -> GoldenPhaseOperator:
    """Create golden phase operator Ĝ."""
    return GoldenPhaseOperator()


def identity() -> IdentityOperator:
    """Create identity operator Î."""
    return IdentityOperator()