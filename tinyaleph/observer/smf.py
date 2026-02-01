"""
Sedenion Memory Field for 16-dimensional holographic memory.

The SMF provides a 16-dimensional memory system using sedenions,
enabling holographic storage with interference patterns and
temporal decay.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from tinyaleph.core.hypercomplex import Hypercomplex


# SMF Axis definitions from the Sentient Observer paper
# 16 semantic axes for sedenion memory orientation
SMF_AXES: List[Dict[str, Any]] = [
    {"index": 0, "name": "coherence", "description": "Degree of internal consistency"},
    {"index": 1, "name": "identity", "description": "Self-continuity and selfhood"},
    {"index": 2, "name": "duality", "description": "Subject/object distinction"},
    {"index": 3, "name": "harmony", "description": "Balance and integration"},
    {"index": 4, "name": "truth", "description": "Alignment with reality"},
    {"index": 5, "name": "consciousness", "description": "Awareness level"},
    {"index": 6, "name": "beauty", "description": "Aesthetic quality"},
    {"index": 7, "name": "compassion", "description": "Empathic resonance"},
    {"index": 8, "name": "intention", "description": "Goal directedness"},
    {"index": 9, "name": "wisdom", "description": "Integrative understanding"},
    {"index": 10, "name": "creativity", "description": "Novelty generation"},
    {"index": 11, "name": "stability", "description": "Temporal persistence"},
    {"index": 12, "name": "complexity", "description": "Structural richness"},
    {"index": 13, "name": "connection", "description": "Relational binding"},
    {"index": 14, "name": "presence", "description": "Here-now awareness"},
    {"index": 15, "name": "transcendence", "description": "Beyond-self orientation"},
]

# AXIS_INDEX: Maps axis name to index
AXIS_INDEX: Dict[str, int] = {
    axis["name"]: axis["index"] for axis in SMF_AXES
}


@dataclass
class MemoryMoment:
    """
    Single moment in the Sedenion Memory Field.
    
    Attributes:
        sedenion: 16-dimensional hypercomplex representation
        timestamp: Time of creation
        entropy: Shannon entropy of the memory
        coherence: Coherence level [0, 1]
        content: Optional raw content that was encoded
    """
    sedenion: Hypercomplex
    timestamp: float
    entropy: float
    coherence: float
    content: Optional[str] = None


class SedenionMemoryField:
    """
    16-dimensional holographic memory using sedenions.
    
    Memory encoding formula:
        M = Σ_t α_t · S_t · e^(-λ(t_now - t))
    
    where:
    - S_t is a sedenion at time t
    - α_t is the amplitude/importance
    - λ is the decay rate
    
    Attributes:
        decay_rate: Rate of memory decay
        max_moments: Maximum number of memories to retain
        moments: List of stored memory moments
        current_time: Current simulation time
    
    Examples:
        >>> smf = SedenionMemoryField(decay_rate=0.01)
        >>> smf.encode("Important fact", importance=0.9)
        >>> smf.step(dt=1.0)
        >>> memories = smf.recall("Important", top_k=3)
    """
    
    DIM = 16  # Sedenion dimension
    
    def __init__(
        self,
        decay_rate: float = 0.01,
        max_moments: int = 1000
    ):
        """
        Initialize the Sedenion Memory Field.
        
        Args:
            decay_rate: Memory decay rate (higher = faster forgetting)
            max_moments: Maximum memories before pruning
        """
        self.decay_rate = decay_rate
        self.max_moments = max_moments
        self.moments: List[MemoryMoment] = []
        self.current_time = 0.0
    
    def encode(self, content: str, importance: float = 1.0) -> MemoryMoment:
        """
        Encode content into sedenion memory.
        
        The encoding uses character ordinals mapped to sedenion
        components with importance weighting.
        
        Args:
            content: String content to encode
            importance: Importance weight [0, 1]
            
        Returns:
            The created MemoryMoment
        """
        # Map characters to 16 components
        components = np.zeros(self.DIM)
        for i, char in enumerate(content[:self.DIM]):
            components[i] = (ord(char) - 64) / 64.0 * importance
        
        # Handle longer content by folding
        for i in range(self.DIM, len(content)):
            idx = i % self.DIM
            components[idx] += (ord(content[i]) - 64) / 64.0 * importance * 0.5
        
        # Normalize
        norm = np.linalg.norm(components)
        if norm > 0:
            components = components / norm
        
        sedenion = Hypercomplex(self.DIM, components)
        entropy = sedenion.entropy()
        coherence = 1.0 / (1.0 + entropy)
        
        moment = MemoryMoment(
            sedenion=sedenion,
            timestamp=self.current_time,
            entropy=entropy,
            coherence=coherence,
            content=content
        )
        
        self.moments.append(moment)
        self._prune_old_memories()
        
        return moment
    
    def recall(self, query: str, top_k: int = 5) -> List[MemoryMoment]:
        """
        Recall memories similar to query.
        
        Uses dot product similarity with time decay and
        coherence boosting.
        
        Args:
            query: Search query string
            top_k: Number of top matches to return
            
        Returns:
            List of most relevant MemoryMoments
        """
        if not self.moments:
            return []
        
        query_sed = self._string_to_sedenion(query)
        
        similarities = []
        for moment in self.moments:
            # Dot product similarity
            sim = np.dot(query_sed.c, moment.sedenion.c)
            
            # Time decay
            age = self.current_time - moment.timestamp
            decay = np.exp(-self.decay_rate * age)
            
            # Coherence boost
            boosted_sim = sim * decay * moment.coherence
            similarities.append((boosted_sim, moment))
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[0])
        
        return [m for _, m in similarities[:top_k]]
    
    def recall_by_content(self, pattern: str) -> List[MemoryMoment]:
        """
        Recall memories with content matching pattern.
        
        Simple substring matching on stored content.
        """
        return [m for m in self.moments if m.content and pattern in m.content]
    
    def interference(self, m1: MemoryMoment, m2: MemoryMoment) -> Hypercomplex:
        """
        Compute interference pattern between two memories.
        
        Uses sedenion multiplication to create interference.
        """
        return m1.sedenion * m2.sedenion
    
    def consolidate(self) -> Hypercomplex:
        """
        Consolidate all memories into single sedenion.
        
        Creates a weighted average based on recency and coherence.
        
        Returns:
            Single sedenion representing consolidated memory
        """
        if not self.moments:
            return Hypercomplex(self.DIM)
        
        result = np.zeros(self.DIM)
        total_weight = 0.0
        
        for moment in self.moments:
            age = self.current_time - moment.timestamp
            weight = np.exp(-self.decay_rate * age) * moment.coherence
            result += moment.sedenion.c * weight
            total_weight += weight
        
        if total_weight > 0:
            result = result / total_weight
        
        return Hypercomplex(self.DIM, result)
    
    def superpose(self, moments: List[MemoryMoment], weights: Optional[List[float]] = None) -> Hypercomplex:
        """
        Create superposition of multiple memories.
        
        Args:
            moments: List of memories to superpose
            weights: Optional weights for each memory
            
        Returns:
            Superposed sedenion
        """
        if not moments:
            return Hypercomplex(self.DIM)
        
        if weights is None:
            weights = [1.0] * len(moments)
        
        result = np.zeros(self.DIM)
        for moment, weight in zip(moments, weights):
            result += moment.sedenion.c * weight
        
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return Hypercomplex(self.DIM, result)
    
    def _string_to_sedenion(self, s: str) -> Hypercomplex:
        """Convert string to sedenion for comparison."""
        components = np.zeros(self.DIM)
        for i, char in enumerate(s[:self.DIM]):
            components[i] = (ord(char) - 64) / 64.0
        
        for i in range(self.DIM, len(s)):
            idx = i % self.DIM
            components[idx] += (ord(s[i]) - 64) / 64.0 * 0.5
        
        norm = np.linalg.norm(components)
        if norm > 0:
            components = components / norm
        return Hypercomplex(self.DIM, components)
    
    def _prune_old_memories(self) -> None:
        """Prune memories when exceeding max capacity."""
        if len(self.moments) > self.max_moments:
            # Keep most recent and highest coherence
            self.moments.sort(key=lambda m: (-m.coherence, -m.timestamp))
            self.moments = self.moments[:self.max_moments]
    
    def decay_all(self) -> None:
        """
        Apply decay to all memories based on age.
        
        Updates coherence values based on time decay.
        """
        for moment in self.moments:
            age = self.current_time - moment.timestamp
            decay_factor = np.exp(-self.decay_rate * age)
            moment.coherence *= decay_factor
        
        # Remove memories with very low coherence
        self.moments = [m for m in self.moments if m.coherence > 0.01]
    
    def step(self, dt: float = 1.0) -> None:
        """
        Advance time and apply decay.
        
        Args:
            dt: Time step to advance
        """
        self.current_time += dt
    
    def clear(self) -> None:
        """Clear all memories."""
        self.moments = []
    
    def reset_time(self) -> None:
        """Reset time to zero."""
        self.current_time = 0.0
    
    @property
    def size(self) -> int:
        """Number of stored memories."""
        return len(self.moments)
    
    @property
    def total_entropy(self) -> float:
        """Total entropy across all memories."""
        if not self.moments:
            return 0.0
        return sum(m.entropy for m in self.moments)
    
    @property
    def mean_coherence(self) -> float:
        """Mean coherence across all memories."""
        if not self.moments:
            return 1.0
        return sum(m.coherence for m in self.moments) / len(self.moments)
    
    @property
    def s(self) -> NDArray[np.float64]:
        """
        Get the current SMF state as 16-dimensional vector.
        
        Returns consolidated memory representation.
        """
        consolidated = self.consolidate()
        return consolidated.c
    
    @s.setter
    def s(self, value: List[float]) -> None:
        """Set the SMF state from a list."""
        if len(value) != self.DIM:
            raise ValueError(f"Expected {self.DIM} components, got {len(value)}")
        # Create a new memory moment with the given state
        sedenion = Hypercomplex(self.DIM, np.array(value))
        entropy = sedenion.entropy()
        coherence = 1.0 / (1.0 + entropy)
        moment = MemoryMoment(
            sedenion=sedenion,
            timestamp=self.current_time,
            entropy=entropy,
            coherence=coherence,
            content=None
        )
        self.moments = [moment]  # Replace with single moment
    
    def normalize(self) -> None:
        """Normalize the SMF state vector."""
        if self.moments:
            latest = self.moments[-1]
            norm = np.linalg.norm(latest.sedenion.c)
            if norm > 0:
                latest.sedenion.c = latest.sedenion.c / norm
    
    def get_orientation(self) -> Optional[NDArray[np.float64]]:
        """
        Get the current memory field orientation as a 16-dimensional vector.
        
        Returns the consolidated memory state, or None if no memories exist.
        The first component represents overall coherence.
        
        Returns:
            16-dimensional orientation vector or None
        """
        if not self.moments:
            return None
        return self.s
    
    def dominant_axes(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get the dominant axes (highest magnitude components).
        
        Args:
            count: Number of top axes to return
            
        Returns:
            List of {index, name, value} dicts
        """
        state = self.s
        indexed = [(i, abs(state[i]), state[i]) for i in range(self.DIM)]
        indexed.sort(key=lambda x: -x[1])
        
        result = []
        for idx, mag, val in indexed[:count]:
            axis_info = SMF_AXES[idx]
            result.append({
                'index': idx,
                'name': axis_info['name'],
                'value': float(val),
                'magnitude': float(mag)
            })
        return result
    
    def smf_entropy(self) -> float:
        """
        Calculate Shannon entropy of SMF state.
        
        Returns:
            Entropy value (higher = more spread)
        """
        state = self.s
        # Normalize to probabilities
        abs_state = np.abs(state)
        total = np.sum(abs_state)
        if total < 1e-10:
            return 0.0
        probs = abs_state / total
        # Shannon entropy
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * np.log2(p)
        return float(entropy)
    
    def coherence(self, other: 'SedenionMemoryField') -> float:
        """
        Calculate coherence (dot product) with another SMF.
        
        Args:
            other: Another SMF instance
            
        Returns:
            Coherence value [-1, 1]
        """
        return float(np.dot(self.s, other.s))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'components': list(self.s),
            'size': self.size,
            'current_time': self.current_time,
            'mean_coherence': self.mean_coherence,
            'total_entropy': self.total_entropy
        }
    
    def __repr__(self) -> str:
        return f"SedenionMemoryField(size={self.size}, time={self.current_time:.1f}, coherence={self.mean_coherence:.3f})"