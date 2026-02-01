"""
Holographic memory fragment from ResoLang.

ResonantFragment provides a holographic memory storage mechanism
where information is encoded in prime-indexed coefficients with
spatial entropy modulation.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, List, Optional
from tinyaleph.core.primes import is_prime, first_n_primes


class ResonantFragment:
    """
    Holographic memory field with prime coefficients.
    
    Information is encoded using the formula:
        A_p * e^(-S) * e^(ipθ)
    
    where:
    - A_p is the base amplitude for prime p
    - S is the spatial entropy
    - θ is the phase modulation
    
    Attributes:
        coeffs: Dictionary mapping primes to amplitudes
        center: Spatial center of the fragment (x, y)
        entropy: Shannon entropy of the coefficient distribution
    
    Examples:
        >>> frag = ResonantFragment.encode("hello world")
        >>> frag.entropy
        2.456...
        >>> collapsed = frag.collapse()
        >>> len(collapsed.coeffs)
        1
    """
    
    __slots__ = ('coeffs', 'center', 'entropy', '_norm_cache')
    
    def __init__(
        self,
        coeffs: Dict[int, float] | None = None,
        center: Tuple[float, float] = (0.0, 0.0),
        entropy: float = 0.0
    ):
        """
        Initialize a ResonantFragment.
        
        Args:
            coeffs: Dictionary mapping primes to amplitudes
            center: Spatial center coordinates
            entropy: Pre-computed entropy value
        """
        self.coeffs = coeffs or {}
        self.center = center
        self.entropy = entropy
        self._norm_cache: Optional[float] = None
    
    @classmethod
    def encode(cls, pattern: str, spatial_entropy: float = 0.5) -> ResonantFragment:
        """
        Encode string pattern into holographic memory.
        
        Each character is mapped to a prime-indexed coefficient
        with amplitude modulated by spatial entropy.
        
        Args:
            pattern: String to encode
            spatial_entropy: Entropy parameter controlling dispersion
            
        Returns:
            ResonantFragment containing the encoded pattern
        """
        if not pattern:
            return cls()
        
        coeffs: Dict[int, float] = {}
        primes = first_n_primes(len(pattern) + 10)
        
        for i, char in enumerate(pattern):
            prime = primes[i]
            
            # Holographic encoding: A_p * e^(-S) * e^(ipθ)
            base_amp = ord(char) / 255.0
            spatial_factor = np.exp(-spatial_entropy)
            phase_factor = np.cos(prime * np.pi / 4)
            
            coeffs[prime] = base_amp * spatial_factor * phase_factor
        
        # Normalize
        total = np.sqrt(sum(a**2 for a in coeffs.values()))
        if total > 0:
            coeffs = {k: v / total for k, v in coeffs.items()}
        
        # Compute Shannon entropy
        entropy = 0.0
        for amp in coeffs.values():
            p = amp ** 2
            if p > 1e-10:
                entropy -= p * np.log(p)
        
        center = (len(pattern) / 2.0, total / max(len(pattern), 1))
        
        return cls(coeffs, center, entropy)
    
    @classmethod
    def from_primes(cls, prime_amps: List[Tuple[int, float]]) -> ResonantFragment:
        """
        Create fragment from list of (prime, amplitude) pairs.
        
        Args:
            prime_amps: List of (prime, amplitude) tuples
            
        Returns:
            Normalized ResonantFragment
        """
        coeffs = {p: a for p, a in prime_amps if is_prime(p)}
        
        # Normalize
        total = np.sqrt(sum(a**2 for a in coeffs.values()))
        if total > 0:
            coeffs = {k: v / total for k, v in coeffs.items()}
        
        # Compute entropy
        entropy = 0.0
        for amp in coeffs.values():
            p = amp ** 2
            if p > 1e-10:
                entropy -= p * np.log(p)
        
        return cls(coeffs, (0.0, 0.0), entropy)
    
    @classmethod
    def random(cls, n_primes: int = 10) -> ResonantFragment:
        """Create random fragment with given number of prime components."""
        primes = first_n_primes(n_primes)
        coeffs = {p: np.random.randn() for p in primes}
        
        # Normalize
        total = np.sqrt(sum(a**2 for a in coeffs.values()))
        if total > 0:
            coeffs = {k: v / total for k, v in coeffs.items()}
        
        entropy = 0.0
        for amp in coeffs.values():
            p = amp ** 2
            if p > 1e-10:
                entropy -= p * np.log(p)
        
        return cls(coeffs, (0.0, 0.0), entropy)
    
    def norm(self) -> float:
        """Return the L2 norm of coefficients."""
        if self._norm_cache is None:
            self._norm_cache = np.sqrt(sum(a**2 for a in self.coeffs.values()))
        return self._norm_cache
    
    def normalize(self) -> ResonantFragment:
        """Return normalized copy of this fragment."""
        n = self.norm()
        if n < 1e-10:
            return ResonantFragment(dict(self.coeffs), self.center, self.entropy)
        
        new_coeffs = {k: v / n for k, v in self.coeffs.items()}
        return ResonantFragment(new_coeffs, self.center, self.entropy)
    
    def tensor(self, other: ResonantFragment) -> ResonantFragment:
        """
        Tensor product: field interaction (interference).
        
        Combines two fragments through coefficient addition,
        creating interference patterns.
        
        Args:
            other: Fragment to combine with
            
        Returns:
            New fragment with combined coefficients
        """
        new_coeffs = dict(self.coeffs)
        for p, amp in other.coeffs.items():
            new_coeffs[p] = new_coeffs.get(p, 0) + amp
        
        # Normalize
        total = np.sqrt(sum(a**2 for a in new_coeffs.values()))
        if total > 0:
            new_coeffs = {k: v / total for k, v in new_coeffs.items()}
        
        # New center (weighted average by entropy)
        total_entropy = self.entropy + other.entropy
        if total_entropy > 0:
            w1 = self.entropy / total_entropy
            w2 = other.entropy / total_entropy
        else:
            w1 = w2 = 0.5
        
        center = (
            self.center[0] * w1 + other.center[0] * w2,
            self.center[1] * w1 + other.center[1] * w2
        )
        
        # New entropy
        new_entropy = 0.0
        for amp in new_coeffs.values():
            p = amp ** 2
            if p > 1e-10:
                new_entropy -= p * np.log(p)
        
        return ResonantFragment(new_coeffs, center, new_entropy)
    
    def collapse(self) -> ResonantFragment:
        """
        Probabilistic collapse to single prime.
        
        Performs a measurement that collapses the fragment
        to a single prime basis state with probability
        proportional to |coefficient|².
        
        Returns:
            Collapsed fragment with single non-zero coefficient
        """
        if not self.coeffs:
            return ResonantFragment()
        
        probs = {p: a**2 for p, a in self.coeffs.items()}
        total = sum(probs.values())
        
        if total < 1e-10:
            # Uniform selection
            selected = list(self.coeffs.keys())[0]
        else:
            r = np.random.random() * total
            cumulative = 0.0
            selected = list(self.coeffs.keys())[0]
            
            for p, prob in probs.items():
                cumulative += prob
                if r < cumulative:
                    selected = p
                    break
        
        return ResonantFragment({selected: 1.0}, self.center, 0.0)
    
    def rotate_phase(self, theta: float) -> ResonantFragment:
        """
        Rotate all coefficients by phase angle.
        
        This implements a global phase rotation, useful for
        phase-sensitive interference operations.
        """
        new_coeffs = {}
        for p, amp in self.coeffs.items():
            # Apply phase rotation
            phase = p * theta
            new_coeffs[p] = amp * np.cos(phase)
        
        return ResonantFragment(new_coeffs, self.center, self.entropy)
    
    def overlap(self, other: ResonantFragment) -> float:
        """
        Compute overlap (dot product) with another fragment.
        
        Returns:
            Dot product of coefficient vectors
        """
        result = 0.0
        for p, a in self.coeffs.items():
            if p in other.coeffs:
                result += a * other.coeffs[p]
        return result
    
    def distance(self, other: ResonantFragment) -> float:
        """
        Compute L2 distance to another fragment.
        """
        all_primes = set(self.coeffs.keys()) | set(other.coeffs.keys())
        dist_sq = 0.0
        for p in all_primes:
            a1 = self.coeffs.get(p, 0.0)
            a2 = other.coeffs.get(p, 0.0)
            dist_sq += (a1 - a2) ** 2
        return np.sqrt(dist_sq)
    
    def dominant_prime(self) -> Optional[int]:
        """Return the prime with largest absolute coefficient."""
        if not self.coeffs:
            return None
        return max(self.coeffs.keys(), key=lambda p: abs(self.coeffs[p]))
    
    def primes(self) -> List[int]:
        """Return sorted list of primes in this fragment."""
        return sorted(self.coeffs.keys())
    
    def to_vector(self, primes: List[int]) -> np.ndarray:
        """
        Convert to numpy vector with given prime ordering.
        
        Args:
            primes: List of primes defining the vector indices
            
        Returns:
            Numpy array with coefficients at corresponding positions
        """
        return np.array([self.coeffs.get(p, 0.0) for p in primes])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, primes: List[int]) -> ResonantFragment:
        """
        Create fragment from numpy vector and prime list.
        """
        coeffs = {p: float(v) for p, v in zip(primes, vector) if abs(v) > 1e-10}
        
        entropy = 0.0
        total = sum(a**2 for a in coeffs.values())
        if total > 0:
            for amp in coeffs.values():
                p = amp**2 / total
                if p > 1e-10:
                    entropy -= p * np.log(p)
        
        return cls(coeffs, (0.0, 0.0), entropy)
    
    def __add__(self, other: ResonantFragment) -> ResonantFragment:
        """Add two fragments (not normalized)."""
        new_coeffs = dict(self.coeffs)
        for p, amp in other.coeffs.items():
            new_coeffs[p] = new_coeffs.get(p, 0) + amp
        return ResonantFragment(new_coeffs, self.center, 0.0)
    
    def __mul__(self, scalar: float) -> ResonantFragment:
        """Multiply by scalar."""
        new_coeffs = {p: a * scalar for p, a in self.coeffs.items()}
        return ResonantFragment(new_coeffs, self.center, self.entropy)
    
    def __rmul__(self, scalar: float) -> ResonantFragment:
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        n = len(self.coeffs)
        return f"ResonantFragment(n_primes={n}, entropy={self.entropy:.4f})"
    
    def __str__(self) -> str:
        if not self.coeffs:
            return "ResonantFragment(empty)"
        
        # Show top 5 primes by amplitude
        sorted_primes = sorted(self.coeffs.items(), key=lambda x: -abs(x[1]))[:5]
        parts = [f"{p}:{a:.3f}" for p, a in sorted_primes]
        return f"ResonantFragment({', '.join(parts)})"