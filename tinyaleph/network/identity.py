"""
Prime Resonance Identity (PRI) for network nodes.

PRI = (P_G, P_E, P_Q) where:
- P_G: Gaussian prime
- P_E: Eisenstein prime  
- P_Q: Quaternionic prime

This provides unique identity and entanglement matching
for nodes in the Prime Resonance Network.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from tinyaleph.core.constants import (
    PRI_GAUSSIAN_PRIMES,
    PRI_EISENSTEIN_PRIMES,
    PRI_QUATERNIONIC_PRIMES,
)


@dataclass
class PrimeResonanceIdentity:
    """
    Prime Resonance Identity (PRI) for network nodes.
    
    A PRI is a triple of primes from different number fields:
    - Gaussian primes (primes in Z[i])
    - Eisenstein primes (primes in Z[ω] where ω = e^(2πi/3))
    - Quaternionic primes (Hurwitz primes)
    
    The PRI serves as a unique identifier and enables resonance-based
    entanglement between nodes with compatible identities.
    
    Attributes:
        gaussian: Gaussian prime component
        eisenstein: Eisenstein prime component
        quaternionic: Quaternionic prime component
    """
    
    gaussian: int
    eisenstein: int
    quaternionic: int
    
    @classmethod
    def random(cls) -> PrimeResonanceIdentity:
        """Create random PRI from standard prime pools."""
        return cls(
            gaussian=int(np.random.choice(PRI_GAUSSIAN_PRIMES)),
            eisenstein=int(np.random.choice(PRI_EISENSTEIN_PRIMES)),
            quaternionic=int(np.random.choice(PRI_QUATERNIONIC_PRIMES))
        )
    
    @classmethod
    def from_seed(cls, seed: int) -> PrimeResonanceIdentity:
        """Create deterministic PRI from seed."""
        rng = np.random.default_rng(seed)
        return cls(
            gaussian=int(rng.choice(PRI_GAUSSIAN_PRIMES)),
            eisenstein=int(rng.choice(PRI_EISENSTEIN_PRIMES)),
            quaternionic=int(rng.choice(PRI_QUATERNIONIC_PRIMES))
        )
    
    @classmethod
    def from_string(cls, s: str) -> PrimeResonanceIdentity:
        """Create PRI from string hash."""
        h = hash(s)
        return cls.from_seed(abs(h))
    
    @property
    def signature(self) -> Tuple[int, int, int]:
        """Return the PRI as a tuple."""
        return (self.gaussian, self.eisenstein, self.quaternionic)
    
    @property
    def hash(self) -> int:
        """Compute unique hash from prime product."""
        return (self.gaussian * self.eisenstein * self.quaternionic) % 1000000007
    
    @property
    def product(self) -> int:
        """Return the product of all three primes."""
        return self.gaussian * self.eisenstein * self.quaternionic
    
    def entanglement_strength(self, other: PrimeResonanceIdentity) -> float:
        """
        Compute entanglement strength with another PRI.
        
        Based on the number of shared primes in the signatures.
        Two nodes with identical PRIs have strength 1.0.
        Two nodes with no shared primes have strength ~0.33.
        
        Args:
            other: Another PrimeResonanceIdentity
            
        Returns:
            Entanglement strength in [0.33, 1.0]
        """
        sig1 = set(self.signature)
        sig2 = set(other.signature)
        shared = len(sig1 & sig2)
        total = len(sig1 | sig2)
        return (2 * shared) / total if total > 0 else 0.0
    
    def resonance_frequency(self) -> float:
        """
        Compute the natural resonance frequency of this PRI.
        
        Based on logarithmic sum of primes.
        """
        return np.log(self.gaussian) + np.log(self.eisenstein) + np.log(self.quaternionic)
    
    def phase(self) -> float:
        """Compute the phase angle associated with this PRI."""
        return (self.product * np.pi / 1000) % (2 * np.pi)
    
    def is_compatible(self, other: PrimeResonanceIdentity, threshold: float = 0.5) -> bool:
        """Check if two PRIs are compatible for entanglement."""
        return self.entanglement_strength(other) >= threshold
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, PrimeResonanceIdentity):
            return self.signature == other.signature
        return False
    
    def __hash__(self) -> int:
        return self.hash
    
    def __repr__(self) -> str:
        return f"PRI({self.gaussian}, {self.eisenstein}, {self.quaternionic})"


@dataclass
class EntangledNode:
    """
    Network node with entanglement capabilities.
    
    Represents a node in the Prime Resonance Network that can
    establish entanglement relationships with other nodes based
    on PRI compatibility.
    
    Attributes:
        pri: Prime Resonance Identity of this node
        coherence: Current coherence level [0, 1]
        entropy: Current entropy level
        entangled_with: List of entangled node identifiers
    """
    
    pri: PrimeResonanceIdentity
    coherence: float = 1.0
    entropy: float = 0.0
    entangled_with: List[int] = field(default_factory=list)
    _id: Optional[int] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self._id is None:
            self._id = self.pri.hash
    
    @property
    def id(self) -> int:
        """Unique node identifier."""
        return self._id or self.pri.hash
    
    @classmethod
    def random(cls) -> EntangledNode:
        """Create node with random PRI."""
        return cls(pri=PrimeResonanceIdentity.random())
    
    @classmethod
    def from_seed(cls, seed: int) -> EntangledNode:
        """Create node with deterministic PRI from seed."""
        return cls(pri=PrimeResonanceIdentity.from_seed(seed))
    
    def can_entangle(self, other: EntangledNode, threshold: float = 0.3) -> bool:
        """
        Check if this node can establish entanglement with another.
        
        Entanglement requires:
        1. PRI compatibility above threshold
        2. Both nodes having sufficient coherence
        """
        if self.coherence < 0.5 or other.coherence < 0.5:
            return False
        
        strength = self.pri.entanglement_strength(other.pri)
        return strength >= threshold
    
    def phase_difference(self, other: EntangledNode) -> float:
        """
        Compute phase difference with another node.
        
        This determines the interference pattern when
        the nodes interact.
        """
        s1 = self.pri.gaussian + self.pri.eisenstein
        s2 = other.pri.gaussian + other.pri.eisenstein
        return float(np.abs(np.sin((s1 - s2) * np.pi / 13)))
    
    def entangle(self, other: EntangledNode) -> bool:
        """
        Attempt to establish entanglement with another node.
        
        Returns:
            True if entanglement was established
        """
        if not self.can_entangle(other):
            return False
        
        if other.id not in self.entangled_with:
            self.entangled_with.append(other.id)
        if self.id not in other.entangled_with:
            other.entangled_with.append(self.id)
        
        # Entanglement increases entropy and reduces coherence slightly
        strength = self.pri.entanglement_strength(other.pri)
        self.entropy += 0.1 * (1 - strength)
        other.entropy += 0.1 * (1 - strength)
        self.coherence *= 0.99
        other.coherence *= 0.99
        
        return True
    
    def disentangle(self, other: EntangledNode) -> None:
        """Remove entanglement with another node."""
        if other.id in self.entangled_with:
            self.entangled_with.remove(other.id)
        if self.id in other.entangled_with:
            other.entangled_with.remove(self.id)
    
    def is_entangled_with(self, other: EntangledNode) -> bool:
        """Check if currently entangled with another node."""
        return other.id in self.entangled_with
    
    def entanglement_count(self) -> int:
        """Return number of active entanglements."""
        return len(self.entangled_with)
    
    def reset(self) -> None:
        """Reset node to initial state."""
        self.coherence = 1.0
        self.entropy = 0.0
        self.entangled_with = []
    
    def decay(self, rate: float = 0.01) -> None:
        """
        Apply time decay to node state.
        
        Coherence decays and entropy increases over time.
        """
        self.coherence *= (1 - rate)
        self.entropy += rate * 0.1
    
    def __repr__(self) -> str:
        return f"EntangledNode(pri={self.pri}, coherence={self.coherence:.3f}, entangled={len(self.entangled_with)})"