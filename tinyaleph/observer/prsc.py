"""
PRSC: Prime Resonance Semantic Coherence layer.

Maps semantic concepts to prime states and maintains coherence
across concept bindings. Enables compositional semantics through
interference patterns in prime Hilbert space.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tinyaleph.hilbert.state import PrimeState
from tinyaleph.core.complex import Complex


@dataclass
class SemanticBinding:
    """
    Binding between prime state and semantic concept.
    
    Attributes:
        prime_state: The PrimeState representing the concept
        concept: Name of the concept
        strength: Binding strength [0, 1]
        coherence: Coherence level of the binding
        metadata: Optional additional metadata
    """
    prime_state: PrimeState
    concept: str
    strength: float = 1.0
    coherence: float = 1.0
    metadata: Dict = field(default_factory=dict)


class PRSC:
    """
    Prime Resonance Semantic Coherence layer.
    
    Manages semantic bindings between concepts and prime states,
    enabling compositional semantics through prime interference.
    
    Key operations:
    - bind: Associate concept with prime state
    - compose: Combine multiple concepts through interference
    - decompose: Identify constituent concepts in a state
    - coherence: Measure semantic coherence
    
    Attributes:
        bindings: Dictionary of concept -> SemanticBinding
        coherence_threshold: Minimum coherence for valid bindings
        global_coherence: Overall coherence of the system
    
    Examples:
        >>> prsc = PRSC()
        >>> prsc.bind("math", PrimeState.composite(30))
        >>> prsc.bind("logic", PrimeState.composite(42))
        >>> composed = prsc.compose(["math", "logic"])
        >>> components = prsc.decompose(composed)
    """
    
    def __init__(self, coherence_threshold: float = 0.7):
        """
        Initialize the PRSC layer.
        
        Args:
            coherence_threshold: Minimum coherence for valid states
        """
        self.bindings: Dict[str, SemanticBinding] = {}
        self.coherence_threshold = coherence_threshold
        self._global_coherence = 1.0
    
    def bind(
        self,
        concept: str,
        state: PrimeState,
        strength: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> SemanticBinding:
        """
        Bind a concept to a prime state.
        
        Args:
            concept: Name of the concept
            state: PrimeState representing the concept
            strength: Binding strength [0, 1]
            metadata: Optional metadata for the binding
            
        Returns:
            The created SemanticBinding
        """
        coherence = self._compute_coherence(state)
        
        binding = SemanticBinding(
            prime_state=state.normalized(),
            concept=concept,
            strength=min(max(strength, 0.0), 1.0),
            coherence=coherence,
            metadata=metadata or {}
        )
        
        self.bindings[concept] = binding
        self._update_global_coherence()
        
        return binding
    
    def unbind(self, concept: str) -> Optional[SemanticBinding]:
        """
        Remove a concept binding.
        
        Returns:
            The removed binding, or None if not found
        """
        binding = self.bindings.pop(concept, None)
        if binding:
            self._update_global_coherence()
        return binding
    
    def get(self, concept: str) -> Optional[SemanticBinding]:
        """Get binding for a concept."""
        return self.bindings.get(concept)
    
    def get_state(self, concept: str) -> Optional[PrimeState]:
        """Get the prime state for a concept."""
        binding = self.bindings.get(concept)
        return binding.prime_state if binding else None
    
    def compose(self, concepts: List[str], normalize: bool = True) -> PrimeState:
        """
        Compose multiple concepts into unified state.
        
        Uses interference composition:
        |ψ_composed⟩ = Σ s_i |ψ_i⟩
        
        where s_i is the strength of each binding.
        
        Args:
            concepts: List of concept names to compose
            normalize: Whether to normalize the result
            
        Returns:
            Composed PrimeState
        """
        if not concepts:
            return PrimeState.uniform()
        
        # Get first valid concept
        result: Optional[PrimeState] = None
        
        for concept in concepts:
            binding = self.bindings.get(concept)
            if binding is None:
                continue
            
            if result is None:
                # Start with first concept
                result = binding.prime_state.copy()
                for p in result.primes:
                    result.amplitudes[p] = result.amplitudes[p] * binding.strength
            else:
                # Add interference
                for p in binding.prime_state.primes:
                    current = result.get(p)
                    added = binding.prime_state.get(p) * binding.strength
                    result.amplitudes[p] = Complex(
                        re=current.re + added.re,
                        im=current.im + added.im
                    )
        
        if result is None:
            return PrimeState.uniform()
        
        return result.normalize() if normalize else result
    
    def compose_weighted(
        self,
        concept_weights: Dict[str, float],
        normalize: bool = True
    ) -> PrimeState:
        """
        Compose concepts with explicit weights.
        
        Args:
            concept_weights: Dictionary mapping concepts to weights
            normalize: Whether to normalize the result
            
        Returns:
            Weighted composed PrimeState
        """
        if not concept_weights:
            return PrimeState.uniform()
        
        result: Optional[PrimeState] = None
        
        for concept, weight in concept_weights.items():
            binding = self.bindings.get(concept)
            if binding is None:
                continue
            
            effective_weight = weight * binding.strength
            
            if result is None:
                result = binding.prime_state.copy()
                for p in result.primes:
                    result.amplitudes[p] = result.amplitudes[p] * effective_weight
            else:
                for p in binding.prime_state.primes:
                    current = result.get(p)
                    added = binding.prime_state.get(p) * effective_weight
                    result.amplitudes[p] = Complex(
                        re=current.re + added.re,
                        im=current.im + added.im
                    )
        
        if result is None:
            return PrimeState.uniform()
        
        return result.normalize() if normalize else result
    
    def decompose(self, state: PrimeState, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        Decompose state into constituent concepts.
        
        Computes overlap with each bound concept.
        
        Args:
            state: PrimeState to decompose
            threshold: Minimum overlap to include
            
        Returns:
            List of (concept, overlap) tuples, sorted by overlap
        """
        results = []
        
        for concept, binding in self.bindings.items():
            # Compute overlap |⟨ψ|φ⟩|²
            overlap = state.overlap(binding.prime_state)
            
            if overlap > threshold:
                results.append((concept, overlap))
        
        results.sort(key=lambda x: -x[1])
        return results
    
    def similarity(self, concept1: str, concept2: str) -> float:
        """
        Compute similarity between two concepts.
        
        Uses overlap of their prime states.
        
        Returns:
            Similarity in [0, 1]
        """
        b1 = self.bindings.get(concept1)
        b2 = self.bindings.get(concept2)
        
        if b1 is None or b2 is None:
            return 0.0
        
        return np.sqrt(b1.prime_state.overlap(b2.prime_state))
    
    def interference(self, concept1: str, concept2: str) -> PrimeState:
        """
        Compute interference pattern between two concepts.
        
        Returns the composed state of exactly two concepts.
        """
        return self.compose([concept1, concept2])
    
    def _compute_coherence(self, state: PrimeState) -> float:
        """
        Compute coherence of a prime state.
        
        Uses normalized inverse entropy:
        C = 1 - S / S_max
        """
        entropy = state.entropy()
        max_entropy = np.log2(len(state.primes))
        
        if max_entropy < 1e-10:
            return 1.0
        
        return 1.0 - (entropy / max_entropy)
    
    def _update_global_coherence(self) -> None:
        """Update global coherence from all bindings."""
        if not self.bindings:
            self._global_coherence = 1.0
            return
        
        coherences = [b.coherence * b.strength for b in self.bindings.values()]
        self._global_coherence = sum(coherences) / len(coherences)
    
    @property
    def global_coherence(self) -> float:
        """Overall system coherence."""
        return self._global_coherence
    
    @property
    def concepts(self) -> List[str]:
        """List of all bound concepts."""
        return list(self.bindings.keys())
    
    @property
    def size(self) -> int:
        """Number of bound concepts."""
        return len(self.bindings)
    
    def is_coherent(self) -> bool:
        """Check if system is above coherence threshold."""
        return self._global_coherence >= self.coherence_threshold
    
    def find_related(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find concepts related to the given concept.
        
        Args:
            concept: Query concept
            top_k: Number of results
            
        Returns:
            List of (related_concept, similarity) tuples
        """
        if concept not in self.bindings:
            return []
        
        similarities = []
        for other in self.bindings:
            if other != concept:
                sim = self.similarity(concept, other)
                similarities.append((other, sim))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]
    
    def cluster(self, threshold: float = 0.5) -> List[List[str]]:
        """
        Cluster concepts by similarity.
        
        Simple single-linkage clustering.
        
        Args:
            threshold: Similarity threshold for clustering
            
        Returns:
            List of concept clusters
        """
        if not self.bindings:
            return []
        
        concepts = list(self.bindings.keys())
        assigned = set()
        clusters = []
        
        for concept in concepts:
            if concept in assigned:
                continue
            
            cluster = [concept]
            assigned.add(concept)
            
            for other in concepts:
                if other in assigned:
                    continue
                if self.similarity(concept, other) >= threshold:
                    cluster.append(other)
                    assigned.add(other)
            
            clusters.append(cluster)
        
        return clusters
    
    def clear(self) -> None:
        """Remove all bindings."""
        self.bindings.clear()
        self._global_coherence = 1.0
    
    def __repr__(self) -> str:
        return f"PRSC(concepts={self.size}, coherence={self._global_coherence:.3f})"
    
    def __contains__(self, concept: str) -> bool:
        return concept in self.bindings
    
    def __len__(self) -> int:
        return len(self.bindings)