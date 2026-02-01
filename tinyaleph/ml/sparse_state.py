"""
Sparse Prime State: H_Q = H_P ⊗ ℍ

The SparsePrimeState extends PrimeState with quaternionic amplitudes,
creating a tensor product of the prime Hilbert space with the quaternions.
This enables richer geometric transformations while maintaining prime-based
sparse encoding.

Mathematical Foundation:
    H_Q = H_P ⊗ ℍ
    
    |ψ_Q⟩ = Σ_p q_p |p⟩
    
    where q_p ∈ ℍ (quaternions) and |p⟩ is the prime basis state.
    
    Inner product: ⟨ψ|φ⟩ = Σ_p q̄_p · r_p (quaternion conjugate product)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterator, Callable
import math
from collections import defaultdict

from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.primes import is_prime, nth_prime
from tinyaleph.core.constants import LAMBDA_STABILITY_THRESHOLD


@dataclass
class SparsePrimeState:
    """
    Sparse representation of a quantum state in H_Q = H_P ⊗ ℍ.
    
    Uses a dictionary mapping prime indices to quaternion amplitudes.
    Only non-zero amplitudes are stored, enabling efficient operations
    on states with many zero components.
    
    Attributes:
        amplitudes: Dict mapping prime -> quaternion amplitude
        coherence: Current coherence (entropy) measure
    """
    
    amplitudes: Dict[int, Quaternion] = field(default_factory=dict)
    coherence: float = 1.0
    
    def __post_init__(self):
        """Validate that all keys are primes."""
        for p in self.amplitudes:
            if not is_prime(p):
                raise ValueError(f"Key {p} is not prime")
    
    @classmethod
    def from_primes(cls, primes: List[int], 
                    amplitudes: Optional[List[Quaternion]] = None) -> SparsePrimeState:
        """
        Create state from list of primes with optional amplitudes.
        
        If amplitudes not provided, creates equal superposition.
        """
        if amplitudes is None:
            # Equal superposition with real amplitudes
            norm = 1.0 / math.sqrt(len(primes))
            amplitudes = [Quaternion(norm) for _ in primes]
        
        if len(primes) != len(amplitudes):
            raise ValueError("Primes and amplitudes must have same length")
            
        amp_dict = dict(zip(primes, amplitudes))
        state = cls(amplitudes=amp_dict)
        state._normalize()
        return state
    
    @classmethod
    def vacuum(cls) -> SparsePrimeState:
        """Create vacuum state |0⟩ (no primes)."""
        return cls(amplitudes={})
    
    @classmethod
    def single_prime(cls, p: int) -> SparsePrimeState:
        """Create single-prime eigenstate |p⟩."""
        if not is_prime(p):
            raise ValueError(f"{p} is not prime")
        return cls(amplitudes={p: Quaternion(1.0)})
    
    @classmethod 
    def first_n_superposition(cls, n: int) -> SparsePrimeState:
        """Create equal superposition of first n primes."""
        primes = [nth_prime(i) for i in range(1, n + 1)]
        return cls.from_primes(primes)
    
    def __len__(self) -> int:
        """Number of non-zero components."""
        return len(self.amplitudes)
    
    def __iter__(self) -> Iterator[Tuple[int, Quaternion]]:
        """Iterate over (prime, amplitude) pairs."""
        return iter(self.amplitudes.items())
    
    def __getitem__(self, p: int) -> Quaternion:
        """Get amplitude for prime p."""
        return self.amplitudes.get(p, Quaternion(0.0))
    
    def __setitem__(self, p: int, q: Quaternion):
        """Set amplitude for prime p."""
        if not is_prime(p):
            raise ValueError(f"{p} is not prime")
        if q.norm() < 1e-10:
            self.amplitudes.pop(p, None)
        else:
            self.amplitudes[p] = q
    
    def __add__(self, other: SparsePrimeState) -> SparsePrimeState:
        """Add two states (quantum superposition)."""
        result = SparsePrimeState()
        all_primes = set(self.amplitudes.keys()) | set(other.amplitudes.keys())
        
        for p in all_primes:
            q = self[p] + other[p]
            if q.norm() > 1e-10:
                result.amplitudes[p] = q
        
        result._normalize()
        return result
    
    def __sub__(self, other: SparsePrimeState) -> SparsePrimeState:
        """Subtract states."""
        result = SparsePrimeState()
        all_primes = set(self.amplitudes.keys()) | set(other.amplitudes.keys())
        
        for p in all_primes:
            q = self[p] - other[p]
            if q.norm() > 1e-10:
                result.amplitudes[p] = q
        
        result._normalize()
        return result
    
    def __mul__(self, scalar: float) -> SparsePrimeState:
        """Scalar multiplication."""
        result = SparsePrimeState(coherence=self.coherence)
        for p, q in self.amplitudes.items():
            result.amplitudes[p] = q * scalar
        return result
    
    def __rmul__(self, scalar: float) -> SparsePrimeState:
        """Right scalar multiplication."""
        return self * scalar
    
    def quaternion_mul(self, q: Quaternion) -> SparsePrimeState:
        """Multiply all amplitudes by quaternion (right multiplication)."""
        result = SparsePrimeState(coherence=self.coherence)
        for p, amp in self.amplitudes.items():
            result.amplitudes[p] = amp * q
        return result
    
    def norm_squared(self) -> float:
        """
        Compute ⟨ψ|ψ⟩ = Σ_p |q_p|².
        
        Uses real part of quaternion inner product.
        """
        return sum(q.norm() ** 2 for q in self.amplitudes.values())
    
    def norm(self) -> float:
        """Compute ||ψ|| = √⟨ψ|ψ⟩."""
        return math.sqrt(self.norm_squared())
    
    def _normalize(self) -> SparsePrimeState:
        """Normalize state in-place. Returns self for chaining."""
        n = self.norm()
        if n > 1e-10:
            scale = 1.0 / n
            for p in self.amplitudes:
                self.amplitudes[p] = self.amplitudes[p] * scale
        return self
    
    def normalized(self) -> SparsePrimeState:
        """Return normalized copy."""
        result = SparsePrimeState(
            amplitudes=dict(self.amplitudes),
            coherence=self.coherence
        )
        return result._normalize()
    
    def inner_product(self, other: SparsePrimeState) -> Quaternion:
        """
        Compute ⟨self|other⟩ = Σ_p q̄_p · r_p.
        
        Uses quaternion conjugate on left factor.
        """
        result = Quaternion(0.0)
        common_primes = set(self.amplitudes.keys()) & set(other.amplitudes.keys())
        
        for p in common_primes:
            conj = self.amplitudes[p].conjugate()
            result = result + conj * other.amplitudes[p]
        
        return result
    
    def overlap(self, other: SparsePrimeState) -> float:
        """
        Compute |⟨self|other⟩|² (transition probability).
        
        Returns real number suitable for probability.
        """
        ip = self.inner_product(other)
        return ip.norm() ** 2
    
    def prime_spectrum(self) -> Dict[int, float]:
        """Get probability distribution over primes."""
        total = self.norm_squared()
        if total < 1e-10:
            return {}
        return {p: q.norm() ** 2 / total for p, q in self.amplitudes.items()}
    
    def entropy(self) -> float:
        """
        Compute von Neumann entropy of the prime distribution.
        
        S = -Σ_p P(p) log₂ P(p)
        """
        spectrum = self.prime_spectrum()
        entropy = 0.0
        
        for prob in spectrum.values():
            if prob > 1e-10:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def is_coherent(self, threshold: float = LAMBDA_STABILITY_THRESHOLD) -> bool:
        """Check if state entropy is below coherence threshold."""
        return self.entropy() < threshold
    
    def apply_rotation(self, axis: Quaternion, angle: float) -> SparsePrimeState:
        """
        Apply quaternion rotation to all amplitudes.
        
        Uses q' = r q r⁻¹ where r = cos(θ/2) + sin(θ/2) * axis.
        """
        # Construct rotation quaternion
        half_angle = angle / 2
        axis_norm = axis.norm()
        if axis_norm < 1e-10:
            return SparsePrimeState(
                amplitudes=dict(self.amplitudes),
                coherence=self.coherence
            )
        
        # Normalize axis
        axis = axis * (1.0 / axis_norm)
        
        # r = cos(θ/2) + sin(θ/2) * axis
        c = math.cos(half_angle)
        s = math.sin(half_angle)
        r = Quaternion(c, s * axis.i, s * axis.j, s * axis.k)
        r_inv = r.conjugate()  # For unit quaternions, inverse = conjugate
        
        # Apply rotation to each amplitude
        result = SparsePrimeState(coherence=self.coherence)
        for p, q in self.amplitudes.items():
            rotated = r * q * r_inv
            result.amplitudes[p] = rotated
        
        return result
    
    def apply_phase(self, prime: int, phase: float) -> SparsePrimeState:
        """
        Apply phase rotation to specific prime component.
        
        |p⟩ → e^(iφ) |p⟩ implemented as quaternion rotation in i-plane.
        """
        if prime not in self.amplitudes:
            return SparsePrimeState(
                amplitudes=dict(self.amplitudes),
                coherence=self.coherence
            )
        
        result = SparsePrimeState(
            amplitudes=dict(self.amplitudes),
            coherence=self.coherence
        )
        
        # Phase rotation in quaternion form: e^(iφ) = cos(φ) + i*sin(φ)
        c = math.cos(phase)
        s = math.sin(phase)
        phase_q = Quaternion(c, s, 0, 0)
        
        result.amplitudes[prime] = result.amplitudes[prime] * phase_q
        return result
    
    def project_to_primes(self, primes: List[int]) -> SparsePrimeState:
        """Project state onto subspace spanned by given primes."""
        result = SparsePrimeState(coherence=self.coherence)
        for p in primes:
            if p in self.amplitudes:
                result.amplitudes[p] = self.amplitudes[p]
        return result._normalize()
    
    def collapse(self) -> Tuple[int, Quaternion]:
        """
        Perform measurement, collapsing to single prime eigenstate.
        
        Returns (prime, quaternion phase) with probability |q_p|².
        """
        import random
        
        spectrum = self.prime_spectrum()
        if not spectrum:
            raise ValueError("Cannot collapse vacuum state")
        
        # Sample according to probability distribution
        primes = list(spectrum.keys())
        probs = list(spectrum.values())
        
        r = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probs):
            cumulative += prob
            if r < cumulative:
                selected_prime = primes[i]
                break
        else:
            selected_prime = primes[-1]
        
        # Get phase (normalized quaternion)
        q = self.amplitudes[selected_prime]
        phase = q * (1.0 / q.norm()) if q.norm() > 1e-10 else Quaternion(1.0)
        
        # Update state to collapsed form
        self.amplitudes = {selected_prime: Quaternion(1.0)}
        self.coherence = 0.0
        
        return selected_prime, phase
    
    def tensor_product(self, other: SparsePrimeState) -> SparsePrimeState:
        """
        Tensor product of two sparse states.
        
        Creates state in product space where primes multiply.
        Note: product of primes is generally not prime, so we use
        (p, q) pairs encoded as Cantor pairing.
        
        For simplicity, we multiply the primes and keep if result is prime.
        This is a simplified version suitable for small states.
        """
        result = SparsePrimeState(coherence=min(self.coherence, other.coherence))
        
        for p1, q1 in self.amplitudes.items():
            for p2, q2 in other.amplitudes.items():
                # Use Cantor pairing for unique encoding
                paired = ((p1 + p2) * (p1 + p2 + 1)) // 2 + p2
                
                # Find nearest prime to pairing
                while not is_prime(paired):
                    paired += 1
                    if paired > 10000000:  # Safety limit
                        break
                
                if is_prime(paired):
                    new_amp = q1 * q2
                    if paired in result.amplitudes:
                        result.amplitudes[paired] = result.amplitudes[paired] + new_amp
                    else:
                        result.amplitudes[paired] = new_amp
        
        return result._normalize()
    
    def apply_operator(self, 
                       op: Callable[[int, Quaternion], Tuple[List[int], List[Quaternion]]]
                       ) -> SparsePrimeState:
        """
        Apply general operator to state.
        
        Operator takes (prime, amplitude) and returns
        (list of output primes, list of output amplitudes).
        """
        result = SparsePrimeState(coherence=self.coherence)
        
        for p, q in self.amplitudes.items():
            out_primes, out_amps = op(p, q)
            for op_prime, op_amp in zip(out_primes, out_amps):
                if op_prime in result.amplitudes:
                    result.amplitudes[op_prime] = result.amplitudes[op_prime] + op_amp
                else:
                    result.amplitudes[op_prime] = op_amp
        
        # Remove near-zero amplitudes
        result.amplitudes = {
            p: q for p, q in result.amplitudes.items()
            if q.norm() > 1e-10
        }
        
        return result._normalize()
    
    def top_k_primes(self, k: int) -> List[Tuple[int, float]]:
        """Get k primes with highest probability."""
        spectrum = self.prime_spectrum()
        sorted_primes = sorted(spectrum.items(), key=lambda x: -x[1])
        return sorted_primes[:k]
    
    def truncate(self, threshold: float = 1e-6) -> SparsePrimeState:
        """Remove components with probability below threshold."""
        spectrum = self.prime_spectrum()
        kept = {p for p, prob in spectrum.items() if prob >= threshold}
        
        result = SparsePrimeState(coherence=self.coherence)
        for p in kept:
            result.amplitudes[p] = self.amplitudes[p]
        
        return result._normalize()
    
    def to_real_vector(self, max_prime_idx: int = 100) -> List[float]:
        """
        Convert to dense real vector (4 components per prime slot).
        
        Returns vector of length 4 * max_prime_idx.
        """
        vector = [0.0] * (4 * max_prime_idx)
        
        for p, q in self.amplitudes.items():
            # Find index of prime
            idx = 0
            test_p = 2
            while test_p < p:
                idx += 1
                test_p = nth_prime(idx + 1)
            
            if idx < max_prime_idx:
                base = idx * 4
                vector[base] = q.w
                vector[base + 1] = q.i
                vector[base + 2] = q.j
                vector[base + 3] = q.k
        
        return vector
    
    @classmethod
    def from_real_vector(cls, vector: List[float]) -> SparsePrimeState:
        """Create state from dense real vector."""
        result = cls()
        num_primes = len(vector) // 4
        
        for idx in range(num_primes):
            base = idx * 4
            q = Quaternion(
                vector[base],
                vector[base + 1],
                vector[base + 2],
                vector[base + 3]
            )
            if q.norm() > 1e-10:
                p = nth_prime(idx + 1)
                result.amplitudes[p] = q
        
        return result
    
    def __repr__(self) -> str:
        if not self.amplitudes:
            return "SparsePrimeState(|vacuum⟩)"
        
        terms = []
        for p, q in sorted(self.amplitudes.items()):
            terms.append(f"({q.w:.3f},{q.i:.3f}i,{q.j:.3f}j,{q.k:.3f}k)|{p}⟩")
        
        if len(terms) > 5:
            shown = terms[:3] + ["..."] + terms[-2:]
        else:
            shown = terms
        
        return f"SparsePrimeState({' + '.join(shown)}, coherence={self.coherence:.3f})"


def coherent_superposition(primes: List[int], 
                           phases: Optional[List[float]] = None) -> SparsePrimeState:
    """
    Create coherent superposition with specified phases.
    
    |ψ⟩ = (1/√n) Σ e^(iφ_k) |p_k⟩
    """
    n = len(primes)
    if n == 0:
        return SparsePrimeState.vacuum()
    
    if phases is None:
        phases = [0.0] * n
    
    norm = 1.0 / math.sqrt(n)
    amplitudes = {}
    
    for p, phase in zip(primes, phases):
        if not is_prime(p):
            raise ValueError(f"{p} is not prime")
        # e^(iφ) as quaternion
        q = Quaternion(math.cos(phase) * norm, math.sin(phase) * norm, 0, 0)
        amplitudes[p] = q
    
    return SparsePrimeState(amplitudes=amplitudes, coherence=1.0)


def golden_superposition(n_primes: int) -> SparsePrimeState:
    """
    Create golden-ratio phased superposition of first n primes.
    
    Uses golden angle spacing: φ_k = k · 2π/Φ² ≈ k · 137.5°
    
    This creates maximum spread in phase space.
    """
    PHI = (1 + math.sqrt(5)) / 2
    golden_angle = 2 * math.pi / (PHI ** 2)
    
    primes = [nth_prime(i) for i in range(1, n_primes + 1)]
    phases = [k * golden_angle for k in range(n_primes)]
    
    return coherent_superposition(primes, phases)