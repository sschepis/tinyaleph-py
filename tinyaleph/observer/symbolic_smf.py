"""
Symbolic SMF Layer

Enhances SedenionMemoryField with symbolic features:
- Symbol database integration (100+ symbols)
- Semantic inference for grounding SMF in archetypes
- Cultural tag mapping to SMF axes
- Compound symbol formation from dominant axes

This module connects abstract 16-dimensional SMF orientation
to culturally-grounded archetypal symbols.
"""

import math
import time
from typing import Dict, List, Optional, Any, Tuple

from .smf import SedenionMemoryField, SMF_AXES, AXIS_INDEX
from .symbols import (
    symbol_database, SymbolDatabase, Symbol, SymbolCategory,
    SemanticInference, ResonanceCalculator, CompoundBuilder,
    CompoundSymbol, SymbolSequence, compound_builder
)


# ═══════════════════════════════════════════════════════════════════
# SMF Axis to Symbol Category Mapping
# ═══════════════════════════════════════════════════════════════════

# Maps SMF axes to symbol categories and archetypal symbols
AXIS_SYMBOL_MAPPING: Dict[int, Dict[str, Any]] = {
    0: {'category': 'abstract', 'archetypes': ['unity', 'light', 'order']},           # coherence
    1: {'category': 'archetype', 'archetypes': ['hero', 'self', 'everyman']},         # identity
    2: {'category': 'abstract', 'archetypes': ['duality', 'yin_yang', 'mirror']},     # duality
    3: {'category': 'abstract', 'archetypes': ['order', 'temple']},                   # structure
    4: {'category': 'abstract', 'archetypes': ['transformation', 'wheel']},           # change
    5: {'category': 'element', 'archetypes': ['tree', 'life', 'flower']},             # life
    6: {'category': 'abstract', 'archetypes': ['harmony', 'balance']},                # harmony
    7: {'category': 'archetype', 'archetypes': ['sage', 'wisdom', 'hermit']},         # wisdom
    8: {'category': 'abstract', 'archetypes': ['infinity', 'void', 'stars']},         # infinity
    9: {'category': 'abstract', 'archetypes': ['creation', 'creator']},               # creation
    10: {'category': 'abstract', 'archetypes': ['truth', 'light', 'sun']},            # truth
    11: {'category': 'abstract', 'archetypes': ['love', 'heart', 'connection']},      # love
    12: {'category': 'abstract', 'archetypes': ['power', 'strength']},                # power
    13: {'category': 'abstract', 'archetypes': ['time', 'hourglass', 'wheel']},       # time
    14: {'category': 'place', 'archetypes': ['space', 'void', 'cosmos']},             # space
    15: {'category': 'abstract', 'archetypes': ['consciousness', 'eye_providence']},  # consciousness
}

# Cultural tag to SMF axis mapping
TAG_TO_AXIS: Dict[str, int] = {
    # Universal tags
    'universal': 0, 'unity': 0, 'oneness': 0,
    
    # Identity tags
    'self': 1, 'individual': 1, 'jungian': 1, 'archetype': 1,
    
    # Duality tags
    'duality': 2, 'opposites': 2, 'polarity': 2, 'eastern': 2,
    
    # Structure tags
    'order': 3, 'form': 3, 'architecture': 3, 'sacred_geometry': 3,
    
    # Change tags
    'transformation': 4, 'alchemy': 4, 'metamorphosis': 4, 'cycle': 4,
    
    # Life tags
    'nature': 5, 'growth': 5, 'organic': 5, 'vitality': 5,
    
    # Harmony tags
    'balance': 6, 'peace': 6, 'music': 6, 'resonance': 6,
    
    # Wisdom tags
    'knowledge': 7, 'wisdom': 7, 'insight': 7, 'philosophy': 7,
    
    # Infinity tags
    'infinity': 8, 'eternal': 8, 'cosmic': 8, 'transcendence': 8,
    
    # Creation tags
    'creation': 9, 'genesis': 9, 'origin': 9, 'birth': 9,
    
    # Truth tags
    'truth': 10, 'reality': 10, 'clarity': 10,
    
    # Love tags
    'love': 11, 'heart': 11, 'connection': 11, 'emotion': 11,
    
    # Power tags
    'power': 12, 'strength': 12, 'force': 12, 'authority': 12,
    
    # Time tags
    'time': 13, 'temporal': 13, 'history': 13, 'future': 13,
    
    # Space tags
    'space': 14, 'place': 14, 'location': 14, 'realm': 14,
    
    # Consciousness tags
    'consciousness': 15, 'awareness': 15, 'mind': 15, 'spirit': 15,
}


class SymbolicSMF(SedenionMemoryField):
    """
    SymbolicSMF extends SedenionMemoryField with symbolic grounding.
    
    Connects the abstract 16-dimensional SMF orientation to
    culturally-grounded archetypal symbols.
    """
    
    def __init__(
        self,
        components: Optional[List[float]] = None,
        symbol_db: Optional[SymbolDatabase] = None,
        inference: Optional[SemanticInference] = None,
        compound_builder_instance: Optional[CompoundBuilder] = None,
        resonance_calc: Optional[ResonanceCalculator] = None,
        max_history: int = 100
    ):
        """
        Initialize SymbolicSMF.
        
        Args:
            components: Initial 16-component vector (optional)
            symbol_db: Symbol database instance
            inference: Semantic inference engine
            compound_builder_instance: Compound symbol builder
            resonance_calc: Resonance calculator
            max_history: Maximum symbol history entries
        """
        super().__init__(components)
        
        self.symbol_db = symbol_db or symbol_database
        self.inference = inference or SemanticInference(self.symbol_db)
        self._compound_builder = compound_builder_instance or compound_builder
        self.resonance_calc = resonance_calc or ResonanceCalculator()
        
        # Active symbols (derived from SMF state)
        self.active_symbols: List[Symbol] = []
        
        # Symbol activation history
        self.symbol_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        
        # Compound representing current state
        self.current_compound: Optional[CompoundSymbol] = None
    
    @classmethod
    def from_smf(cls, smf: SedenionMemoryField, **kwargs) -> 'SymbolicSMF':
        """
        Create from base SedenionMemoryField.
        
        Args:
            smf: Base SMF instance
            **kwargs: Additional arguments for SymbolicSMF
            
        Returns:
            New SymbolicSMF instance
        """
        return cls(components=list(smf.s), **kwargs)
    
    def tag_to_axis(self, tag: str) -> int:
        """
        Map cultural tag to SMF axis index.
        
        Args:
            tag: Cultural tag string
            
        Returns:
            Axis index (0-15) or -1 if not found
        """
        normalized = tag.lower().strip()
        return TAG_TO_AXIS.get(normalized, -1)
    
    def get_axis_archetype(self, axis: Any) -> Optional[Symbol]:
        """
        Get the archetypal symbol for a given axis.
        
        Args:
            axis: Axis index (int) or name (str)
            
        Returns:
            Symbol or None
        """
        if isinstance(axis, str):
            idx = AXIS_INDEX.get(axis, -1)
        else:
            idx = int(axis)
        
        mapping = AXIS_SYMBOL_MAPPING.get(idx)
        if not mapping:
            return None
        
        # Try each archetype until we find one
        for archetype_id in mapping['archetypes']:
            symbol = self.symbol_db.get_symbol(archetype_id)
            if symbol:
                return symbol
        
        return None
    
    def ground_in_symbols(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Ground current SMF orientation in archetypal symbols.
        
        Returns the symbols most aligned with the dominant axes.
        
        Args:
            count: Number of symbols to return
            
        Returns:
            List of {axis, symbol, alignment} dicts
        """
        dominant = self.dominant_axes(count)
        grounded: List[Dict[str, Any]] = []
        
        for axis in dominant:
            symbol = self.get_axis_archetype(axis['index'])
            if symbol:
                grounded.append({
                    'axis': axis['name'],
                    'axis_value': axis['value'],
                    'symbol': symbol,
                    'alignment': abs(axis['value']),
                    'is_positive': axis['value'] >= 0,
                    'contribution': abs(axis['value'])
                })
        
        self.active_symbols = [g['symbol'] for g in grounded]
        return grounded
    
    def excite_from_symbol(self, symbol_id: str, intensity: float = 0.3) -> bool:
        """
        Excite SMF from a symbol activation.
        
        Maps symbol's prime and cultural tags to SMF axes.
        
        Args:
            symbol_id: Symbol ID to activate
            intensity: Activation intensity (0-1)
            
        Returns:
            True if symbol was found and applied
        """
        symbol = self.symbol_db.get_symbol(symbol_id)
        if not symbol:
            return False
        
        # Primary axis: map prime to axis via log2
        primary_axis = int(math.log2(symbol.prime)) % 16
        self.s[primary_axis] = min(1.0, self.s[primary_axis] + intensity)
        
        # Secondary axes from cultural tags
        tag_intensity = intensity * 0.3
        for tag in symbol.cultural_tags[:5]:
            axis_idx = self.tag_to_axis(tag)
            if axis_idx >= 0 and axis_idx != primary_axis:
                self.s[axis_idx] = min(1.0, self.s[axis_idx] + tag_intensity)
        
        # Category-based axis
        category_axis = self._category_to_axis(symbol.category)
        if category_axis >= 0 and category_axis != primary_axis:
            self.s[category_axis] = min(1.0, self.s[category_axis] + tag_intensity)
        
        self.normalize()
        
        # Record to history
        self.symbol_history.append({
            'symbol_id': symbol_id,
            'symbol': symbol,
            'intensity': intensity,
            'timestamp': time.time() * 1000,
            'primary_axis': primary_axis,
            'smf_snapshot': list(self.s)
        })
        
        if len(self.symbol_history) > self.max_history:
            self.symbol_history.pop(0)
        
        return True
    
    def _category_to_axis(self, category: SymbolCategory) -> int:
        """Map symbol category to primary axis."""
        mapping = {
            SymbolCategory.ARCHETYPE: 1,    # identity
            SymbolCategory.ELEMENT: 5,      # life
            SymbolCategory.PLACE: 14,       # space
            SymbolCategory.OBJECT: 3,       # structure
            SymbolCategory.ABSTRACT: 0,     # coherence
            SymbolCategory.MYTHOLOGICAL: 12, # power
            SymbolCategory.TAROT: 4,        # change
            SymbolCategory.ICHING: 2,       # duality
            SymbolCategory.CELESTIAL: 8,    # infinity
            SymbolCategory.CREATURE: 5,     # life
        }
        return mapping.get(category, 0)
    
    def excite_from_symbols(self, symbol_ids: List[str], base_intensity: float = 0.2) -> None:
        """
        Excite SMF from multiple symbols simultaneously.
        
        Uses resonance weighting to determine relative intensities.
        
        Args:
            symbol_ids: List of symbol IDs
            base_intensity: Base intensity for each symbol
        """
        symbols = [
            self.symbol_db.get_symbol(sid)
            for sid in symbol_ids
        ]
        symbols = [s for s in symbols if s is not None]
        
        if not symbols:
            return
        
        # Calculate resonance weights
        weights: List[float] = []
        for i, s1 in enumerate(symbols):
            total_resonance = 0.0
            for j, s2 in enumerate(symbols):
                if i != j:
                    total_resonance += self.resonance_calc.calculate_resonance(
                        s1.prime, s2.prime
                    )
            weights.append(1.0 + total_resonance / len(symbols))
        
        # Normalize weights
        max_weight = max(weights) if weights else 1.0
        normalized_weights = [w / max_weight for w in weights]
        
        # Apply excitations with resonance weighting
        for i, symbol in enumerate(symbols):
            self.excite_from_symbol(symbol.id, base_intensity * normalized_weights[i])
    
    def excite_from_text(self, text: str, intensity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Infer symbols from text and excite SMF.
        
        Args:
            text: Text to process
            intensity: Activation intensity
            
        Returns:
            List of inferred symbols
        """
        results = self.inference.infer_with_resonance(text, {
            'maxCandidates': 10,
            'useAttention': True
        })
        
        for result in results:
            self.excite_from_symbol(
                result['symbol'].id,
                intensity * result['confidence'] * result.get('attentionWeight', 1.0)
            )
        
        return results
    
    def create_compound_from_state(
        self,
        compound_id: str,
        meaning: Optional[str] = None
    ) -> Optional[CompoundSymbol]:
        """
        Create a compound symbol from current SMF state.
        
        Combines the active symbols into a unified concept.
        
        Args:
            compound_id: ID for the compound
            meaning: Optional meaning description
            
        Returns:
            Created compound or None
        """
        grounded = self.ground_in_symbols(4)
        
        if len(grounded) < 2:
            return None
        
        symbols = [g['symbol'] for g in grounded]
        cultural_tags = ['smf-derived'] + [g['axis'] for g in grounded]
        
        parts = [f"{g['axis']}({g['symbol'].id})" for g in grounded]
        default_meaning = "SMF compound: " + " + ".join(parts)
        
        try:
            self.current_compound = self._compound_builder.create_compound_from_symbols(
                compound_id,
                symbols,
                meaning or default_meaning,
                cultural_tags
            )
            return self.current_compound
        except Exception as e:
            print(f'Failed to create compound: {e}')
            return None
    
    def resonance_with_symbol(self, symbol_id: str) -> float:
        """
        Calculate resonance between SMF state and a symbol.
        
        Args:
            symbol_id: Symbol to check
            
        Returns:
            Resonance score 0-1
        """
        symbol = self.symbol_db.get_symbol(symbol_id)
        if not symbol:
            return 0.0
        
        # Get axis activations from symbol
        primary_axis = int(math.log2(symbol.prime)) % 16
        resonance = abs(self.s[primary_axis])
        
        # Add tag-based resonance
        tag_resonance = 0.0
        tag_count = 0
        
        for tag in symbol.cultural_tags:
            axis_idx = self.tag_to_axis(tag)
            if axis_idx >= 0:
                tag_resonance += abs(self.s[axis_idx])
                tag_count += 1
        
        if tag_count > 0:
            resonance = resonance * 0.6 + (tag_resonance / tag_count) * 0.4
        
        return min(1.0, resonance)
    
    def find_resonant_symbols(
        self,
        count: int = 5,
        category: Optional[SymbolCategory] = None
    ) -> List[Dict[str, Any]]:
        """
        Find most resonant symbols with current SMF state.
        
        Args:
            count: Number of symbols to return
            category: Optional category filter
            
        Returns:
            List of {symbol, resonance} dicts
        """
        if category:
            candidates = self.symbol_db.get_symbols_by_category(category)
        else:
            candidates = self.symbol_db.get_all_symbols()
        
        scored = [
            {'symbol': symbol, 'resonance': self.resonance_with_symbol(symbol.id)}
            for symbol in candidates
        ]
        
        scored.sort(key=lambda x: x['resonance'], reverse=True)
        return scored[:count]
    
    def get_symbol_stats(self) -> Dict[str, Any]:
        """Get symbol history statistics."""
        if not self.symbol_history:
            return {
                'total_activations': 0,
                'unique_symbols': 0,
                'most_active': [],
                'recent_symbols': []
            }
        
        symbol_counts: Dict[str, int] = {}
        for entry in self.symbol_history:
            sid = entry['symbol_id']
            symbol_counts[sid] = symbol_counts.get(sid, 0) + 1
        
        sorted_counts = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_activations': len(self.symbol_history),
            'unique_symbols': len(symbol_counts),
            'most_active': [
                {
                    'symbol_id': sid,
                    'symbol': self.symbol_db.get_symbol(sid),
                    'count': count
                }
                for sid, count in sorted_counts[:5]
            ],
            'recent_symbols': [
                {
                    'symbol_id': h['symbol_id'],
                    'intensity': h['intensity'],
                    'timestamp': h['timestamp']
                }
                for h in self.symbol_history[-5:]
            ]
        }
    
    def clear_history(self) -> None:
        """Clear symbol history."""
        self.symbol_history = []
        self.active_symbols = []
        self.current_compound = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Enhanced JSON serialization."""
        base = super().to_dict()
        grounded = self.ground_in_symbols(3)
        
        return {
            **base,
            'symbolic': {
                'grounded_symbols': [
                    {
                        'axis': g['axis'],
                        'symbol_id': g['symbol'].id,
                        'symbol_unicode': g['symbol'].unicode,
                        'alignment': g['alignment']
                    }
                    for g in grounded
                ],
                'current_compound': self.current_compound.to_dict() if self.current_compound else None,
                'history_stats': self.get_symbol_stats()
            }
        }
    
    def __str__(self) -> str:
        """String representation with symbols."""
        grounded = self.ground_in_symbols(3)
        symbols = ''.join(g['symbol'].unicode for g in grounded)
        axes = ', '.join(f"{g['axis']}:{g['axis_value']:.2f}" for g in grounded)
        return f"SymbolicSMF({symbols} | {axes})"


class SMFSymbolMapper:
    """
    Utility class for mapping between symbols and SMF orientations.
    """
    
    def __init__(self, symbol_db: Optional[SymbolDatabase] = None):
        self.symbol_db = symbol_db or symbol_database
        self.resonance_calc = ResonanceCalculator()
    
    def symbol_to_smf(self, symbol: Symbol) -> SymbolicSMF:
        """
        Create SMF orientation from a symbol.
        
        Args:
            symbol: Symbol object
            
        Returns:
            SMF oriented toward symbol
        """
        smf = SymbolicSMF(symbol_db=self.symbol_db)
        smf.s = [0.0] * 16
        smf.excite_from_symbol(symbol.id, 1.0)
        return smf
    
    def symbols_to_smf(self, symbols: List[Symbol]) -> SymbolicSMF:
        """
        Create SMF from multiple symbols with resonance weighting.
        
        Args:
            symbols: List of symbol objects
            
        Returns:
            Combined SMF
        """
        smf = SymbolicSMF(symbol_db=self.symbol_db)
        smf.s = [0.0] * 16
        smf.excite_from_symbols([s.id for s in symbols], 0.5)
        return smf
    
    def compound_to_smf(self, compound: CompoundSymbol) -> SymbolicSMF:
        """
        Create SMF from compound symbol.
        
        Args:
            compound: Compound symbol
            
        Returns:
            SMF from compound components
        """
        return self.symbols_to_smf(compound.components)
    
    def sequence_to_smf(
        self,
        sequence: SymbolSequence,
        recency_bias: float = 0.3
    ) -> SymbolicSMF:
        """
        Create SMF from symbol sequence (with temporal weighting).
        
        Args:
            sequence: Symbol sequence
            recency_bias: How much to weight recent symbols (0-1)
            
        Returns:
            SMF with recency-weighted symbols
        """
        smf = SymbolicSMF(symbol_db=self.symbol_db)
        smf.s = [0.0] * 16
        
        symbols = list(sequence.symbols)
        n = len(symbols)
        
        if n == 0:
            return smf
        
        for i, symbol in enumerate(symbols):
            # Weight increases with position (recency)
            position = i / (n - 1) if n > 1 else 1.0
            weight = (1 - recency_bias) + recency_bias * position
            smf.excite_from_symbol(symbol.id, 0.3 * weight)
        
        return smf
    
    def find_best_match(
        self,
        smf: SedenionMemoryField,
        category: Optional[SymbolCategory] = None
    ) -> Optional[Symbol]:
        """
        Find best matching symbol for an SMF orientation.
        
        Args:
            smf: SMF to match
            category: Optional category filter
            
        Returns:
            Best matching symbol
        """
        symbolic = smf if isinstance(smf, SymbolicSMF) else SymbolicSMF.from_smf(smf)
        resonant = symbolic.find_resonant_symbols(1, category)
        return resonant[0]['symbol'] if resonant else None
    
    def symbolic_distance(
        self,
        smf1: SedenionMemoryField,
        smf2: SedenionMemoryField
    ) -> float:
        """
        Calculate symbolic distance between two SMFs.
        
        Uses symbol-grounded comparison.
        
        Args:
            smf1: First SMF
            smf2: Second SMF
            
        Returns:
            Distance 0-1 (0 = same, 1 = opposite)
        """
        s1 = smf1 if isinstance(smf1, SymbolicSMF) else SymbolicSMF.from_smf(smf1)
        s2 = smf2 if isinstance(smf2, SymbolicSMF) else SymbolicSMF.from_smf(smf2)
        
        g1 = s1.ground_in_symbols(5)
        g2 = s2.ground_in_symbols(5)
        
        # Compare symbol sets
        primes1 = set(g['symbol'].prime for g in g1)
        primes2 = set(g['symbol'].prime for g in g2)
        
        # Jaccard similarity on prime sets
        intersection = primes1 & primes2
        union = primes1 | primes2
        
        jaccard = len(intersection) / len(union) if union else 0
        
        # Also use standard coherence
        coherence_val = s1.coherence(s2)
        
        # Combined metric
        return 1 - (0.5 * jaccard + 0.5 * abs(coherence_val))


# Singleton mapper instance
smf_mapper = SMFSymbolMapper()


# Convenience functions
def create_symbolic_smf(
    components: Optional[List[float]] = None,
    **options
) -> SymbolicSMF:
    """Create a new SymbolicSMF instance."""
    return SymbolicSMF(components, **options)


def from_smf(smf: SedenionMemoryField, **options) -> SymbolicSMF:
    """Create SymbolicSMF from base SMF."""
    return SymbolicSMF.from_smf(smf, **options)


def symbol_to_smf(symbol: Symbol) -> SymbolicSMF:
    """Create SMF from a symbol."""
    return smf_mapper.symbol_to_smf(symbol)


def symbols_to_smf(symbols: List[Symbol]) -> SymbolicSMF:
    """Create SMF from multiple symbols."""
    return smf_mapper.symbols_to_smf(symbols)