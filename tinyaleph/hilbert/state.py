"""
Prime Hilbert Space: HP = {|ψ⟩ = Σ αp|p⟩ : Σ|αp|² = 1}

This module implements quantum-inspired states in a Hilbert space
where basis vectors are indexed by prime numbers.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from tinyaleph.core.complex import Complex
from tinyaleph.core.primes import first_n_primes, prime_factorization
from tinyaleph.core.constants import DEFAULT_PRIME_COUNT


class PrimeState:
    """
    Quantum state in Prime Hilbert space.
    
    A PrimeState represents a superposition of prime basis states:
        |ψ⟩ = Σ αp|p⟩
    
    where αp are complex amplitudes and |p⟩ are prime basis states.
    
    Attributes:
        primes: List of prime numbers forming the basis
        amplitudes: Dictionary mapping primes to complex amplitudes
    
    Examples:
        >>> # Uniform superposition
        >>> state = PrimeState.uniform()
        >>> state.entropy()  # High entropy
        
        >>> # Pure basis state |2⟩
        >>> pure = PrimeState.basis(2)
        >>> pure.entropy()  # Zero entropy
        0.0
        
        >>> # State from composite number factorization
        >>> state = PrimeState.composite(60)  # 2² × 3 × 5
    """
    
    __slots__ = ('primes', 'amplitudes', '_norm_cache', '_entropy_cache')
    
    def __init__(self, primes: List[int] | None = None):
        """
        Initialize a PrimeState with given prime basis.
        
        Args:
            primes: List of primes to use as basis (defaults to first 25 primes)
        """
        self.primes = primes if primes is not None else first_n_primes(DEFAULT_PRIME_COUNT)
        self.amplitudes: Dict[int, Complex] = {p: Complex.zero() for p in self.primes}
        self._norm_cache: Optional[float] = None
        self._entropy_cache: Optional[float] = None
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached values when state changes."""
        self._norm_cache = None
        self._entropy_cache = None
    
    @classmethod
    def basis(cls, p: int, primes: List[int] | None = None) -> PrimeState:
        """
        Create a pure basis state |p⟩.
        
        Args:
            p: The prime number for the basis state
            primes: Optional list of primes for the space
            
        Returns:
            State with amplitude 1 at prime p, 0 elsewhere
        """
        state = cls(primes)
        if p in state.amplitudes:
            state.amplitudes[p] = Complex.one()
        else:
            raise ValueError(f"Prime {p} not in basis {state.primes}")
        return state
    
    @classmethod
    def uniform(cls, primes: List[int] | None = None) -> PrimeState:
        """
        Create a uniform superposition over all basis states.
        
        |ψ⟩ = (1/√N) Σ |p⟩
        
        This is the maximum entropy state.
        """
        state = cls(primes)
        n = len(state.primes)
        amp = Complex(1.0 / np.sqrt(n), 0.0)
        for p in state.primes:
            state.amplitudes[p] = amp
        return state
    
    @classmethod
    def composite(cls, n: int, primes: List[int] | None = None) -> PrimeState:
        """
        Create state from prime factorization of composite number.
        
        Amplitudes are proportional to exponents in factorization:
            n = Π p_i^{e_i}  →  |ψ⟩ ∝ Σ e_i |p_i⟩
        
        Args:
            n: Composite number to factorize
            primes: Optional list of primes for the space
            
        Returns:
            Normalized state encoding the factorization
        """
        state = cls(primes)
        factors = prime_factorization(n)
        total = sum(factors.values())
        
        if total == 0:
            return state.uniform() if primes else PrimeState.uniform()
        
        for p, exp in factors.items():
            if p in state.amplitudes:
                state.amplitudes[p] = Complex(float(exp) / total, 0.0)
        
        return state.normalize()
    
    @classmethod
    def random(cls, primes: List[int] | None = None) -> PrimeState:
        """Create random normalized state with complex amplitudes."""
        state = cls(primes)
        for p in state.primes:
            state.amplitudes[p] = Complex(
                np.random.randn(),
                np.random.randn()
            )
        return state.normalize()
    
    @classmethod
    def single_prime(cls, p: int, primes: List[int] | None = None) -> PrimeState:
        """
        Alias for basis() - create pure basis state |p⟩.
        
        Args:
            p: The prime number for the basis state
            primes: Optional list of primes for the space
            
        Returns:
            State with amplitude 1 at prime p
        """
        return cls.basis(p, primes)
    
    @classmethod
    def uniform_superposition(cls, primes_to_superpose: List[int]) -> PrimeState:
        """
        Create uniform superposition over specified primes.
        
        |ψ⟩ = (1/√N) Σ_{p ∈ primes_to_superpose} |p⟩
        
        Args:
            primes_to_superpose: List of primes to include in superposition
            
        Returns:
            Normalized uniform superposition state
        """
        state = cls(primes_to_superpose)
        n = len(primes_to_superpose)
        amp = Complex(1.0 / np.sqrt(n), 0.0)
        for p in primes_to_superpose:
            state.amplitudes[p] = amp
        return state
    
    @classmethod
    def first_n_superposition(cls, n: int) -> PrimeState:
        """
        Create uniform superposition over first n primes.
        
        Args:
            n: Number of primes to include
            
        Returns:
            Uniform superposition over [2, 3, 5, ...]
        """
        primes = first_n_primes(n)
        return cls.uniform_superposition(primes)
    
    @classmethod
    def coherent(cls, alpha: Complex, primes: List[int] | None = None) -> PrimeState:
        """
        Create coherent-like state with Poisson-like distribution.
        
        Similar to quantum optics coherent states but indexed by primes.
        """
        state = cls(primes)
        for i, p in enumerate(state.primes):
            # Exponentially decaying with prime index
            phase = Complex.from_polar(1.0, i * alpha.phase())
            amp = np.exp(-alpha.norm2() / 2) * (alpha.norm() ** i) / np.sqrt(float(np.math.factorial(min(i, 170))))
            state.amplitudes[p] = Complex(amp, 0) * phase
        return state.normalize()
    
    def get(self, p: int) -> Complex:
        """Get amplitude for prime p."""
        return self.amplitudes.get(p, Complex.zero())
    
    def set(self, p: int, amplitude: Complex) -> None:
        """Set amplitude for prime p."""
        if p not in self.amplitudes:
            raise ValueError(f"Prime {p} not in basis")
        self.amplitudes[p] = amplitude
        self._invalidate_cache()
    
    def norm2(self) -> float:
        """Return squared norm: Σ|αp|²."""
        return sum(self.get(p).norm2() for p in self.primes)
    
    def norm(self) -> float:
        """Return norm: √(Σ|αp|²)."""
        if self._norm_cache is None:
            self._norm_cache = np.sqrt(self.norm2())
        return self._norm_cache
    
    def normalize(self) -> PrimeState:
        """
        Normalize the state in-place.
        
        Returns self for chaining.
        """
        n = self.norm()
        if n < 1e-10:
            return self
        
        for p in self.primes:
            self.amplitudes[p] = self.amplitudes[p] * (1.0 / n)
        
        self._invalidate_cache()
        return self
    
    def normalized(self) -> PrimeState:
        """Return a normalized copy of this state."""
        result = self.copy()
        return result.normalize()
    
    def copy(self) -> PrimeState:
        """Create a deep copy of this state."""
        result = PrimeState(list(self.primes))
        for p in self.primes:
            result.amplitudes[p] = Complex(self.amplitudes[p].re, self.amplitudes[p].im)
        return result
    
    def entropy(self) -> float:
        """
        Calculate von Neumann entropy of the state.
        
        S = -Σ |αp|² log₂(|αp|²)
        
        Returns:
            Entropy in bits, range [0, log₂(N)]
        """
        if self._entropy_cache is not None:
            return self._entropy_cache
        
        n2 = self.norm2()
        if n2 < 1e-10:
            self._entropy_cache = 0.0
            return 0.0
        
        h = 0.0
        for p in self.primes:
            prob = self.get(p).norm2() / n2
            if prob > 1e-10:
                h -= prob * np.log2(prob)
        
        self._entropy_cache = h
        return h
    
    def coherence(self) -> float:
        """
        Calculate coherence as normalized inverse entropy.
        
        C = 1 - S / S_max
        
        Returns:
            Coherence in [0, 1]
        """
        max_entropy = np.log2(len(self.primes))
        if max_entropy < 1e-10:
            return 1.0
        return 1.0 - self.entropy() / max_entropy
    
    def measure(self) -> Tuple[int, float]:
        """
        Perform Born measurement (probabilistic collapse).
        
        Returns:
            (prime, probability) - the measured prime and its probability
        """
        n2 = self.norm2()
        if n2 < 1e-10:
            # Uniform random selection if state is zero
            p = np.random.choice(self.primes)
            return (p, 1.0 / len(self.primes))
        
        r = np.random.random() * n2
        cumulative = 0.0
        
        for p in self.primes:
            prob = self.get(p).norm2()
            cumulative += prob
            if r < cumulative:
                return (p, prob / n2)
        
        # Fallback to last prime
        return (self.primes[-1], self.get(self.primes[-1]).norm2() / n2)
    
    def collapse(self, p: int) -> PrimeState:
        """
        Collapse state to basis state |p⟩.
        
        Args:
            p: Prime to collapse to
            
        Returns:
            Self after collapse
        """
        for q in self.primes:
            self.amplitudes[q] = Complex.one() if q == p else Complex.zero()
        self._invalidate_cache()
        return self
    
    def probabilities(self) -> Dict[int, float]:
        """Return dictionary mapping primes to probabilities."""
        n2 = self.norm2()
        if n2 < 1e-10:
            return {p: 1.0 / len(self.primes) for p in self.primes}
        return {p: self.get(p).norm2() / n2 for p in self.primes}
    
    def inner_product(self, other: PrimeState) -> Complex:
        """
        Compute inner product ⟨self|other⟩.
        
        ⟨ψ|φ⟩ = Σ αp* βp
        """
        if set(self.primes) != set(other.primes):
            raise ValueError("States must have same prime basis")
        
        result = Complex.zero()
        for p in self.primes:
            result = result + self.get(p).conj() * other.get(p)
        return result
    
    def overlap(self, other: PrimeState) -> float:
        """
        Compute overlap probability |⟨self|other⟩|².
        """
        return self.inner_product(other).norm2()
    
    def __add__(self, other: PrimeState) -> PrimeState:
        """Add two states (not normalized)."""
        if set(self.primes) != set(other.primes):
            raise ValueError("States must have same prime basis")
        result = PrimeState(list(self.primes))
        for p in self.primes:
            result.amplitudes[p] = self.get(p) + other.get(p)
        return result
    
    def __sub__(self, other: PrimeState) -> PrimeState:
        """Subtract two states (not normalized)."""
        if set(self.primes) != set(other.primes):
            raise ValueError("States must have same prime basis")
        result = PrimeState(list(self.primes))
        for p in self.primes:
            result.amplitudes[p] = self.get(p) - other.get(p)
        return result
    
    def __mul__(self, scalar: float | Complex) -> PrimeState:
        """Multiply state by scalar."""
        result = PrimeState(list(self.primes))
        for p in self.primes:
            if isinstance(scalar, Complex):
                result.amplitudes[p] = self.get(p) * scalar
            else:
                result.amplitudes[p] = self.get(p) * scalar
        return result
    
    def __rmul__(self, scalar: float | Complex) -> PrimeState:
        """Right multiplication by scalar."""
        return self.__mul__(scalar)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array of complex numbers."""
        return np.array([self.get(p).to_numpy() for p in self.primes])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, primes: List[int] | None = None) -> PrimeState:
        """Create state from numpy array."""
        if primes is None:
            primes = first_n_primes(len(vector))
        if len(vector) != len(primes):
            raise ValueError(f"Vector length {len(vector)} != number of primes {len(primes)}")
        
        state = cls(primes)
        for i, p in enumerate(primes):
            z = vector[i]
            state.amplitudes[p] = Complex(float(np.real(z)), float(np.imag(z)))
        return state
    
    def __repr__(self) -> str:
        """String representation."""
        n = self.norm()
        s = self.entropy()
        return f"PrimeState(dim={len(self.primes)}, norm={n:.4g}, entropy={s:.4g})"
    
    def __str__(self) -> str:
        """Human-readable representation showing non-zero amplitudes."""
        parts = []
        for p in self.primes:
            amp = self.get(p)
            if amp.norm() > 1e-4:
                parts.append(f"{amp}|{p}⟩")
        return " + ".join(parts) if parts else "|0⟩"