"""
Symbolic Temporal Layer

Extends the temporal layer with I-Ching style moment classification
using a 64 attractor codebook.

This integrates:
- Hexagram classification for discrete moment types
- SymbolDatabase for moment-symbol mapping
- PHI-harmonic moment detection
- Narrative pattern recognition
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple

from .temporal import Moment, TemporalLayer, TemporalPatternDetector
from .symbols import symbol_database, SymbolDatabase


# PHI constant for resonance calculations
PHI = (1 + math.sqrt(5)) / 2

# First 64 primes for encoding SMF vectors
FIRST_64_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311
]


# ═══════════════════════════════════════════════════════════════════
# I-Ching Hexagram to Symbol Mapping
# Maps the 64 attractors to archetypal moment types
# ═══════════════════════════════════════════════════════════════════

HEXAGRAM_ARCHETYPES: Dict[int, Dict[str, Any]] = {
    # Primary hexagrams (0-7 - trigram doubles)
    0: {'name': 'creative', 'symbol': 'creation', 'tags': ['beginning', 'potential', 'yang']},
    1: {'name': 'receptive', 'symbol': 'earth', 'tags': ['acceptance', 'yin', 'nurture']},
    2: {'name': 'difficulty', 'symbol': 'chaos', 'tags': ['challenge', 'birth', 'growth']},
    3: {'name': 'youthful', 'symbol': 'child', 'tags': ['inexperience', 'wonder', 'education']},
    4: {'name': 'waiting', 'symbol': 'time', 'tags': ['patience', 'timing', 'preparation']},
    5: {'name': 'conflict', 'symbol': 'duality', 'tags': ['tension', 'opposition', 'resolution']},
    6: {'name': 'army', 'symbol': 'power', 'tags': ['organization', 'discipline', 'force']},
    7: {'name': 'holding', 'symbol': 'unity', 'tags': ['togetherness', 'support', 'cooperation']},
    
    # Transformation hexagrams (8-15)
    8: {'name': 'small_taming', 'symbol': 'balance', 'tags': ['moderation', 'gentle', 'accumulation']},
    9: {'name': 'treading', 'symbol': 'growth', 'tags': ['careful', 'conduct', 'advance']},
    10: {'name': 'peace', 'symbol': 'harmony', 'tags': ['balance', 'prosperity', 'flow']},
    11: {'name': 'standstill', 'symbol': 'void', 'tags': ['blockage', 'pause', 'waiting']},
    12: {'name': 'fellowship', 'symbol': 'love', 'tags': ['community', 'friendship', 'gathering']},
    13: {'name': 'possession', 'symbol': 'power', 'tags': ['success', 'wealth', 'fullness']},
    14: {'name': 'modesty', 'symbol': 'balance', 'tags': ['simplicity', 'grace', 'quietude']},
    15: {'name': 'enthusiasm', 'symbol': 'life', 'tags': ['excitement', 'inspiration', 'movement']},
    
    # Dynamic hexagrams (16-31)
    16: {'name': 'following', 'symbol': 'transformation', 'tags': ['flexibility', 'response', 'alignment']},
    17: {'name': 'work_decay', 'symbol': 'decay', 'tags': ['corruption', 'renewal', 'repair']},
    18: {'name': 'approach', 'symbol': 'growth', 'tags': ['expansion', 'advance', 'spring']},
    19: {'name': 'contemplation', 'symbol': 'consciousness', 'tags': ['observation', 'reflection', 'insight']},
    20: {'name': 'biting', 'symbol': 'truth', 'tags': ['decision', 'clarity', 'action']},
    21: {'name': 'grace', 'symbol': 'beauty', 'tags': ['form', 'elegance', 'appearance']},
    22: {'name': 'splitting', 'symbol': 'duality', 'tags': ['release', 'letting_go', 'autumn']},
    23: {'name': 'return', 'symbol': 'renewal', 'tags': ['cycle', 'rebirth', 'winter_solstice']},
    24: {'name': 'innocence', 'symbol': 'child', 'tags': ['natural', 'spontaneous', 'authentic']},
    25: {'name': 'great_taming', 'symbol': 'power', 'tags': ['power', 'accumulation', 'strength']},
    26: {'name': 'nourishing', 'symbol': 'life', 'tags': ['feeding', 'care', 'nutrition']},
    27: {'name': 'great_exceeding', 'symbol': 'chaos', 'tags': ['pressure', 'breakthrough', 'extreme']},
    28: {'name': 'abysmal', 'symbol': 'water', 'tags': ['water', 'danger', 'unknown']},
    29: {'name': 'clinging', 'symbol': 'light', 'tags': ['fire', 'light', 'awareness']},
    30: {'name': 'influence', 'symbol': 'love', 'tags': ['magnetism', 'courtship', 'feeling']},
    31: {'name': 'duration', 'symbol': 'time', 'tags': ['persistence', 'constancy', 'marriage']},
    
    # Completion hexagrams (32-63)
    32: {'name': 'retreat', 'symbol': 'balance', 'tags': ['strategic', 'preservation', 'timing']},
    33: {'name': 'great_power', 'symbol': 'power', 'tags': ['energy', 'movement', 'spring']},
    34: {'name': 'advancement', 'symbol': 'growth', 'tags': ['promotion', 'recognition', 'success']},
    35: {'name': 'darkening', 'symbol': 'darkness', 'tags': ['concealment', 'protection', 'night']},
    36: {'name': 'family', 'symbol': 'love', 'tags': ['domestic', 'roles', 'structure']},
    37: {'name': 'opposition', 'symbol': 'duality', 'tags': ['difference', 'complement', 'polarity']},
    38: {'name': 'obstruction', 'symbol': 'mountain', 'tags': ['difficulty', 'mountain', 'stillness']},
    39: {'name': 'deliverance', 'symbol': 'transformation', 'tags': ['release', 'freedom', 'thunder']},
    40: {'name': 'decrease', 'symbol': 'decay', 'tags': ['sacrifice', 'simplification', 'reduction']},
    41: {'name': 'increase', 'symbol': 'growth', 'tags': ['benefit', 'growth', 'generosity']},
    42: {'name': 'breakthrough', 'symbol': 'power', 'tags': ['resolve', 'action', 'courage']},
    43: {'name': 'coming', 'symbol': 'love', 'tags': ['encounter', 'temptation', 'choice']},
    44: {'name': 'gathering', 'symbol': 'unity', 'tags': ['collection', 'focus', 'concentration']},
    45: {'name': 'pushing', 'symbol': 'growth', 'tags': ['rising', 'effort', 'climbing']},
    46: {'name': 'oppression', 'symbol': 'decay', 'tags': ['depletion', 'adversity', 'testing']},
    47: {'name': 'well', 'symbol': 'water', 'tags': ['resources', 'depth', 'inexhaustible']},
    48: {'name': 'revolution', 'symbol': 'transformation', 'tags': ['change', 'renewal', 'molting']},
    49: {'name': 'cauldron', 'symbol': 'fire', 'tags': ['refinement', 'nourishment', 'offering']},
    50: {'name': 'arousing', 'symbol': 'power', 'tags': ['thunder', 'awakening', 'initiation']},
    51: {'name': 'keeping_still', 'symbol': 'mountain', 'tags': ['mountain', 'stillness', 'rest']},
    52: {'name': 'development', 'symbol': 'growth', 'tags': ['patience', 'steady', 'tree']},
    53: {'name': 'marrying', 'symbol': 'love', 'tags': ['commitment', 'crossing', 'threshold']},
    54: {'name': 'abundance', 'symbol': 'life', 'tags': ['zenith', 'peak', 'eclipse']},
    55: {'name': 'wanderer', 'symbol': 'explorer', 'tags': ['travel', 'stranger', 'fire_mountain']},
    56: {'name': 'gentle', 'symbol': 'air', 'tags': ['wind', 'subtle', 'influence']},
    57: {'name': 'joyous', 'symbol': 'life', 'tags': ['lake', 'delight', 'exchange']},
    58: {'name': 'dispersion', 'symbol': 'void', 'tags': ['scattering', 'wind_water', 'release']},
    59: {'name': 'limitation', 'symbol': 'balance', 'tags': ['measure', 'moderation', 'structure']},
    60: {'name': 'inner_truth', 'symbol': 'truth', 'tags': ['trust', 'authenticity', 'wind_lake']},
    61: {'name': 'small_exceeding', 'symbol': 'balance', 'tags': ['care', 'attention', 'thunder_mountain']},
    62: {'name': 'after_completion', 'symbol': 'harmony', 'tags': ['success', 'vigilance', 'water_fire']},
    63: {'name': 'before_completion', 'symbol': 'time', 'tags': ['almost', 'potential', 'fire_water']},
}


@dataclass
class SymbolicMoment(Moment):
    """
    Extended moment with symbolic classification.
    
    Adds I-Ching hexagram classification and resonance metrics
    to the base Moment class.
    """
    
    # Symbolic classification from attractor codebook
    hexagram_index: Optional[int] = None  # 0-63
    archetype: Optional[Dict[str, Any]] = None  # From HEXAGRAM_ARCHETYPES
    symbol_id: Optional[str] = None  # From SymbolDatabase
    
    # Resonance metrics
    phi_resonance: float = 0.0  # Golden ratio resonance
    prime_resonance: float = 0.0  # Prime-based resonance
    harmonic_order: int = 0  # Detected harmonic
    
    # Confidence of classification
    classification_confidence: float = 0.0
    
    # Related symbols (from semantic inference)
    related_symbols: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        return {
            **base,
            'hexagram_index': self.hexagram_index,
            'archetype': self.archetype,
            'symbol_id': self.symbol_id,
            'phi_resonance': self.phi_resonance,
            'prime_resonance': self.prime_resonance,
            'harmonic_order': self.harmonic_order,
            'classification_confidence': self.classification_confidence,
            'related_symbols': self.related_symbols,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolicMoment':
        """Create from dictionary."""
        return cls(
            id=data.get('id', ''),
            trigger=data.get('trigger', ''),
            coherence=data.get('coherence', 0.0),
            entropy=data.get('entropy', 0.0),
            phase_transition_rate=data.get('phase_transition_rate', 0.0),
            active_primes=data.get('active_primes', []),
            smf_snapshot=data.get('smf_snapshot'),
            semantic_content=data.get('semantic_content'),
            subjective_duration=data.get('subjective_duration', 0.0),
            previous_moment_id=data.get('previous_moment_id'),
            timestamp=data.get('timestamp', 0.0),
            hexagram_index=data.get('hexagram_index'),
            archetype=data.get('archetype'),
            symbol_id=data.get('symbol_id'),
            phi_resonance=data.get('phi_resonance', 0.0),
            prime_resonance=data.get('prime_resonance', 0.0),
            harmonic_order=data.get('harmonic_order', 0),
            classification_confidence=data.get('classification_confidence', 0.0),
            related_symbols=data.get('related_symbols', []),
        )


class EntropyCollapseHead:
    """
    Entropy-based collapse to discrete attractors.
    
    Uses a 64-attractor codebook (I-Ching hexagrams) to classify
    continuous states into discrete moment types.
    """
    
    def __init__(self, target_entropy: float = 5.99, num_attractors: int = 64):
        """
        Initialize the collapse head.
        
        Args:
            target_entropy: Target entropy for maximum spread (~log2(64) ≈ 5.99)
            num_attractors: Number of attractors (default 64 = hexagrams)
        """
        self.target_entropy = target_entropy
        self.num_attractors = num_attractors
        
        # Initialize attractor centers using primes
        self.attractors: List[List[float]] = []
        for i in range(num_attractors):
            # Use prime-based initialization
            prime = FIRST_64_PRIMES[i]
            center = []
            for j in range(16):
                # Distribute using prime digit expansion
                val = ((prime >> j) & 1) * 2 - 1  # -1 or 1
                center.append(val * 0.5)
            self.attractors.append(center)
    
    def hard_assign(self, state_vector: List[float]) -> Dict[str, Any]:
        """
        Assign state to nearest attractor (hard assignment).
        
        Args:
            state_vector: 16-dimensional state vector
            
        Returns:
            Dict with index, confidence, and probability distribution
        """
        if not state_vector:
            return {'index': 0, 'confidence': 0.0, 'probs': []}
        
        # Compute distances to all attractors
        distances = []
        for attractor in self.attractors:
            dist = 0.0
            for i in range(min(len(state_vector), len(attractor))):
                diff = state_vector[i] - attractor[i]
                dist += diff * diff
            distances.append(math.sqrt(dist))
        
        # Find nearest
        min_dist = min(distances)
        min_idx = distances.index(min_dist)
        
        # Compute softmax probabilities
        temperature = 0.5
        exp_neg_dists = [math.exp(-d / temperature) for d in distances]
        total = sum(exp_neg_dists)
        probs = [e / total if total > 0 else 1.0 / len(distances) for e in exp_neg_dists]
        
        # Confidence is probability of selected attractor
        confidence = probs[min_idx]
        
        return {
            'index': min_idx,
            'confidence': confidence,
            'probs': probs,
            'distance': min_dist
        }
    
    def soft_assign(self, state_vector: List[float]) -> List[float]:
        """
        Get soft assignment probabilities over all attractors.
        
        Args:
            state_vector: 16-dimensional state vector
            
        Returns:
            Probability distribution over attractors
        """
        result = self.hard_assign(state_vector)
        return result['probs']


class SymbolicTemporalLayer(TemporalLayer):
    """
    Enhanced temporal layer with symbolic classification.
    
    Extends TemporalLayer with I-Ching hexagram classification,
    PHI-harmonic resonance detection, and narrative pattern recognition.
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.7,
        entropy_min: float = 0.1,
        entropy_max: float = 0.9,
        target_entropy: float = 5.99,
        harmonic_depth: int = 8,
        on_moment: Optional[Callable] = None,
        on_symbolic_moment: Optional[Callable] = None,
        on_hexagram_transition: Optional[Callable] = None
    ):
        """
        Initialize SymbolicTemporalLayer.
        
        Args:
            coherence_threshold: Threshold for coherence events
            entropy_min: Minimum valid entropy
            entropy_max: Maximum valid entropy
            target_entropy: Target entropy for collapse head
            harmonic_depth: Depth for harmonic analysis
            on_moment: Callback for moment creation
            on_symbolic_moment: Callback for symbolic moment with classification
            on_hexagram_transition: Callback for hexagram changes
        """
        super().__init__(
            coherence_threshold=coherence_threshold,
            entropy_min=entropy_min,
            entropy_max=entropy_max,
            on_moment=on_moment
        )
        
        # Entropy collapse head for moment classification
        self.collapse_head = EntropyCollapseHead(target_entropy)
        
        # Symbol database
        self.symbol_db = symbol_database
        
        # PHI for resonance
        self.phi = PHI
        self.harmonic_depth = harmonic_depth
        
        # Classification history
        self.hexagram_history: List[int] = []
        self.archetype_history: List[str] = []
        
        # Callbacks
        self.on_symbolic_moment = on_symbolic_moment
        self.on_hexagram_transition = on_hexagram_transition
    
    def create_moment(self, trigger: str, state: Dict[str, Any]) -> SymbolicMoment:
        """
        Create a symbolic moment from state.
        
        Args:
            trigger: What triggered this moment
            state: Current system state
            
        Returns:
            New SymbolicMoment
        """
        coherence = state.get('coherence', 0.0)
        entropy = state.get('entropy', 0.0)
        active_primes = state.get('active_primes', [])
        smf = state.get('smf')
        semantic_content = state.get('semantic_content')
        amplitudes = state.get('amplitudes', [])
        
        # Calculate subjective duration
        subjective_duration = self.calculate_subjective_duration(state)
        
        # Get SMF snapshot
        smf_snapshot = None
        smf_vector = None
        if smf and hasattr(smf, 's') and smf.s:
            smf_vector = list(smf.s)
            smf_snapshot = {
                'components': smf_vector,
                'entropy': smf.smf_entropy() if hasattr(smf, 'smf_entropy') else 0.0
            }
        
        # Classify moment
        classification = self._classify_moment(smf_vector, active_primes, amplitudes)
        
        # Calculate resonances
        resonances = self._calculate_resonances(active_primes, smf_vector)
        
        # Find related symbols
        related_symbols = self._find_related_symbols(classification['archetype'], active_primes)
        
        self.moment_counter += 1
        moment = SymbolicMoment(
            id=f'm_{self.moment_counter}',
            trigger=trigger,
            coherence=coherence,
            entropy=entropy,
            phase_transition_rate=self.phase_transition_rate(),
            active_primes=active_primes or [],
            smf_snapshot=smf_snapshot,
            semantic_content=semantic_content,
            subjective_duration=subjective_duration,
            previous_moment_id=self.current_moment.id if self.current_moment else None,
            timestamp=time.time() * 1000,
            
            # Symbolic classification
            hexagram_index=classification['hexagram_index'],
            archetype=classification['archetype'],
            symbol_id=classification['symbol_id'],
            classification_confidence=classification['confidence'],
            
            # Resonances
            phi_resonance=resonances['phi'],
            prime_resonance=resonances['prime'],
            harmonic_order=resonances['harmonic_order'],
            
            # Related symbols
            related_symbols=related_symbols
        )
        
        # Update subjective time
        self.subjective_time += subjective_duration
        
        # Update histories
        self.hexagram_history.append(classification['hexagram_index'])
        if len(self.hexagram_history) > 1000:
            self.hexagram_history = self.hexagram_history[-500:]
        
        if classification['archetype']:
            self.archetype_history.append(classification['archetype']['name'])
            if len(self.archetype_history) > 1000:
                self.archetype_history = self.archetype_history[-500:]
        
        # Detect hexagram transitions
        if self.current_moment and hasattr(self.current_moment, 'hexagram_index'):
            if self.current_moment.hexagram_index != classification['hexagram_index']:
                if self.on_hexagram_transition:
                    prev_archetype = HEXAGRAM_ARCHETYPES.get(self.current_moment.hexagram_index)
                    self.on_hexagram_transition({
                        'from': self.current_moment.hexagram_index,
                        'to': classification['hexagram_index'],
                        'from_archetype': prev_archetype,
                        'to_archetype': classification['archetype'],
                        'moment': moment
                    })
        
        # Store moment
        self.moments.append(moment)
        self.current_moment = moment
        
        # Callbacks
        if self.on_moment:
            self.on_moment(moment)
        if self.on_symbolic_moment:
            self.on_symbolic_moment(moment, classification)
        
        return moment
    
    def _classify_moment(
        self,
        smf_vector: Optional[List[float]],
        active_primes: Optional[List[int]],
        amplitudes: Optional[List[float]]
    ) -> Dict[str, Any]:
        """
        Classify moment using entropy collapse head.
        
        Args:
            smf_vector: SMF state vector
            active_primes: Active prime numbers
            amplitudes: Amplitude values
            
        Returns:
            Classification result
        """
        # Build state vector for collapse head
        state_vector: List[float] = []
        
        if smf_vector and len(smf_vector) >= 16:
            state_vector = list(smf_vector[:16])
        elif active_primes and len(active_primes) > 0:
            # Use log-primes
            state_vector = [0.0] * 16
            for i, prime in enumerate(active_primes[:16]):
                state_vector[i] = math.log(prime) / 10
        elif amplitudes and len(amplitudes) > 0:
            state_vector = list(amplitudes[:16])
            while len(state_vector) < 16:
                state_vector.append(0.0)
        else:
            # Default state
            state_vector = [0.1] * 16
        
        # Collapse to attractor
        collapse_result = self.collapse_head.hard_assign(state_vector)
        hexagram_index = collapse_result['index']
        confidence = collapse_result['confidence']
        
        # Get archetype
        archetype = HEXAGRAM_ARCHETYPES.get(hexagram_index, {
            'name': f'unknown_{hexagram_index}',
            'symbol': 'unknown',
            'tags': []
        })
        
        # Look up symbol in database
        symbol_id = None
        symbol_match = self.symbol_db.get_symbol(archetype.get('symbol', ''))
        if symbol_match:
            symbol_id = symbol_match.id
        
        return {
            'hexagram_index': hexagram_index,
            'archetype': archetype,
            'symbol_id': symbol_id,
            'confidence': confidence,
            'distribution': collapse_result['probs']
        }
    
    def _calculate_resonances(
        self,
        active_primes: Optional[List[int]],
        smf_vector: Optional[List[float]]
    ) -> Dict[str, Any]:
        """
        Calculate PHI and prime resonances.
        
        Args:
            active_primes: Active prime numbers
            smf_vector: SMF state vector
            
        Returns:
            Resonance metrics
        """
        result = {
            'phi': 0.0,
            'prime': 0.0,
            'harmonic_order': 0
        }
        
        if not active_primes or len(active_primes) < 2:
            return result
        
        # Calculate PHI resonance (ratio of consecutive primes)
        phi_sum = 0.0
        prime_sum = 0.0
        
        for i in range(1, len(active_primes)):
            ratio = active_primes[i] / active_primes[i - 1]
            phi_distance = abs(ratio - PHI) / PHI
            phi_sum += math.exp(-phi_distance * 2)
            
            # Check for prime ratios
            if self._is_prime(round(ratio)):
                prime_sum += 1
        
        result['phi'] = phi_sum / (len(active_primes) - 1)
        result['prime'] = prime_sum / (len(active_primes) - 1)
        
        # Detect harmonic order from SMF
        if smf_vector and len(smf_vector) > 0:
            max_val = max(smf_vector)
            dominant = smf_vector.index(max_val)
            result['harmonic_order'] = dominant + 1
        
        return result
    
    def _is_prime(self, n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def _find_related_symbols(
        self,
        archetype: Optional[Dict[str, Any]],
        active_primes: Optional[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Find symbols related to the current archetype.
        
        Args:
            archetype: Current archetype dict
            active_primes: Active prime numbers
            
        Returns:
            List of related symbols
        """
        related: List[Dict[str, Any]] = []
        
        if not archetype:
            return related
        
        # Search by tags
        for tag in archetype.get('tags', []):
            matches = self.symbol_db.get_symbols_by_tag(tag)
            for match in matches[:3]:
                if not any(r.get('id') == match.id for r in related):
                    related.append({
                        'id': match.id,
                        'name': match.name,
                        'match_type': 'tag',
                        'tag': tag
                    })
        
        # Search by prime if available
        if active_primes and len(active_primes) > 0:
            prime_match = self.symbol_db.get_symbol_by_prime(active_primes[0])
            if prime_match and not any(r.get('id') == prime_match.id for r in related):
                related.append({
                    'id': prime_match.id,
                    'name': prime_match.name,
                    'match_type': 'prime',
                    'prime': active_primes[0]
                })
        
        return related[:5]  # Limit to 5
    
    def get_hexagram_distribution(self) -> List[float]:
        """
        Get frequency distribution of hexagram types.
        
        Returns:
            64-element probability distribution
        """
        dist = [0.0] * 64
        for idx in self.hexagram_history:
            if 0 <= idx < 64:
                dist[idx] += 1
        
        total = len(self.hexagram_history) or 1
        return [c / total for c in dist]
    
    def get_dominant_archetypes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get most frequent archetypes.
        
        Args:
            top_n: Number of top archetypes to return
            
        Returns:
            List of {name, count, frequency} dicts
        """
        counts: Dict[str, int] = {}
        for name in self.archetype_history:
            counts[name] = counts.get(name, 0) + 1
        
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        total = len(self.archetype_history) or 1
        
        return [
            {
                'name': name,
                'count': count,
                'frequency': count / total
            }
            for name, count in sorted_items[:top_n]
        ]
    
    def detect_archetype_sequences(
        self,
        min_length: int = 2,
        max_length: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Detect recurring archetype sequences (narrative patterns).
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            
        Returns:
            List of {sequence, count} dicts
        """
        sequences: Dict[str, int] = {}
        
        for length in range(min_length, max_length + 1):
            for i in range(len(self.archetype_history) - length + 1):
                seq = '→'.join(self.archetype_history[i:i + length])
                sequences[seq] = sequences.get(seq, 0) + 1
        
        # Filter sequences that appear at least twice
        filtered = [(seq, count) for seq, count in sequences.items() if count >= 2]
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {'sequence': seq, 'count': count}
            for seq, count in filtered[:10]
        ]
    
    def predict_next_archetype(self) -> Dict[str, Any]:
        """
        Predict next archetype based on sequence analysis.
        
        Returns:
            Dict with predicted archetype and confidence
        """
        if len(self.archetype_history) < 2:
            return {'predicted': None, 'confidence': 0.0}
        
        recent_seq = '→'.join(self.archetype_history[-3:])
        candidates: Dict[str, int] = {}
        
        # Look for sequences that match recent history
        for i in range(len(self.archetype_history) - 3):
            seq = '→'.join(self.archetype_history[i:i + 3])
            if seq == recent_seq and i + 3 < len(self.archetype_history):
                next_arch = self.archetype_history[i + 3]
                candidates[next_arch] = candidates.get(next_arch, 0) + 1
        
        if not candidates:
            return {'predicted': None, 'confidence': 0.0}
        
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        total = sum(count for _, count in sorted_candidates)
        
        return {
            'predicted': sorted_candidates[0][0],
            'confidence': sorted_candidates[0][1] / total,
            'alternatives': [
                {'name': name, 'probability': count / total}
                for name, count in sorted_candidates[1:4]
            ]
        }
    
    def get_iching_reading(self) -> Optional[Dict[str, Any]]:
        """
        Get I-Ching style reading for current moment.
        
        Returns:
            Dict with hexagram, interpretation, and prediction
        """
        if not self.current_moment or not hasattr(self.current_moment, 'hexagram_index'):
            return None
        
        hexagram_idx = self.current_moment.hexagram_index
        if hexagram_idx is None:
            return None
        
        hexagram = HEXAGRAM_ARCHETYPES.get(hexagram_idx)
        prediction = self.predict_next_archetype()
        
        return {
            'current': {
                'number': hexagram_idx + 1,  # Traditional 1-64
                'name': hexagram['name'] if hexagram else 'unknown',
                'symbol': hexagram['symbol'] if hexagram else 'unknown',
                'tags': hexagram['tags'] if hexagram else []
            },
            'confidence': self.current_moment.classification_confidence,
            'resonance': {
                'phi': self.current_moment.phi_resonance,
                'prime': self.current_moment.prime_resonance,
                'harmonic': self.current_moment.harmonic_order
            },
            'prediction': prediction,
            'related_symbols': self.current_moment.related_symbols,
            'history': self.get_dominant_archetypes(3)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extended stats including symbolic metrics."""
        base_stats = super().get_stats()
        
        current_hex = None
        current_arch = None
        if self.current_moment and hasattr(self.current_moment, 'hexagram_index'):
            current_hex = self.current_moment.hexagram_index
            if hasattr(self.current_moment, 'archetype') and self.current_moment.archetype:
                current_arch = self.current_moment.archetype.get('name')
        
        return {
            **base_stats,
            'symbolic': {
                'current_hexagram': current_hex,
                'current_archetype': current_arch,
                'hexagram_distribution': self.get_hexagram_distribution(),
                'dominant_archetypes': self.get_dominant_archetypes(5),
                'sequences': self.detect_archetype_sequences(),
                'prediction': self.predict_next_archetype()
            }
        }
    
    def reset(self) -> None:
        """Reset including symbolic state."""
        super().reset()
        self.hexagram_history = []
        self.archetype_history = []


class SymbolicPatternDetector(TemporalPatternDetector):
    """
    Enhanced pattern detection with archetype awareness.
    
    Extends TemporalPatternDetector to include archetype-based
    pattern recognition and narrative detection.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize the detector.
        
        Args:
            window_size: Size of analysis window
        """
        super().__init__(window_size)
        self.archetype_patterns: List[Dict[str, Any]] = []
    
    def moment_signature(self, moment: Moment) -> Dict[str, Any]:
        """
        Create signature including archetype.
        
        Args:
            moment: Moment to analyze
            
        Returns:
            Signature dict
        """
        base = super().moment_signature(moment)
        
        if hasattr(moment, 'hexagram_index'):
            base['hexagram'] = moment.hexagram_index
        if hasattr(moment, 'archetype') and moment.archetype:
            base['archetype'] = moment.archetype.get('name')
        if hasattr(moment, 'phi_resonance'):
            base['phi_resonance'] = round(moment.phi_resonance * 10) / 10
        
        return base
    
    def signatures_match(
        self,
        sig1: Dict[str, Any],
        sig2: Dict[str, Any]
    ) -> bool:
        """
        Check if signatures match including archetype.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            True if signatures match
        """
        if not super().signatures_match(sig1, sig2):
            return False
        
        # Archetype must match for strong similarity
        arch1 = sig1.get('archetype')
        arch2 = sig2.get('archetype')
        if arch1 and arch2 and arch1 != arch2:
            return False
        
        return True
    
    def detect_narrative_patterns(
        self,
        moments: List[Moment]
    ) -> List[Dict[str, Any]]:
        """
        Detect archetype-based narrative patterns.
        
        Args:
            moments: List of moments to analyze
            
        Returns:
            List of detected narratives
        """
        if len(moments) < 4:
            return []
        
        # Extract archetype sequence
        archetypes = [
            m.archetype['name'] if hasattr(m, 'archetype') and m.archetype else None
            for m in moments
        ]
        archetypes = [a for a in archetypes if a]
        
        narratives: List[Dict[str, Any]] = []
        
        # Hero's journey pattern
        hero_pattern = self._find_sequence(archetypes, ['creative', 'difficulty', 'return'])
        if hero_pattern:
            narratives.append({'type': 'hero_journey', 'occurrences': hero_pattern})
        
        # Transformation pattern
        transform_pattern = self._find_sequence(archetypes, ['standstill', 'revolution', 'renewal'])
        if transform_pattern:
            narratives.append({'type': 'transformation', 'occurrences': transform_pattern})
        
        # Growth pattern
        growth_pattern = self._find_sequence(archetypes, ['youthful', 'development', 'abundance'])
        if growth_pattern:
            narratives.append({'type': 'growth', 'occurrences': growth_pattern})
        
        return narratives
    
    def _find_sequence(
        self,
        arr: List[str],
        pattern: List[str]
    ) -> List[int]:
        """
        Find occurrences of a pattern in array.
        
        Args:
            arr: Array to search
            pattern: Pattern to find
            
        Returns:
            List of starting indices
        """
        occurrences: List[int] = []
        
        for i in range(len(arr) - len(pattern) + 1):
            matches = True
            for j, p in enumerate(pattern):
                if arr[i + j] != p:
                    matches = False
                    break
            if matches:
                occurrences.append(i)
        
        return occurrences