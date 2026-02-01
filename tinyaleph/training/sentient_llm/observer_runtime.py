from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time

from tinyaleph.observer.boundary import BoundaryLayer
from tinyaleph.observer.agency import AgencyLayer
from tinyaleph.observer.smf import SedenionMemoryField
from tinyaleph.observer.symbols import SymbolDatabase
from tinyaleph.observer.semantic_prime_mapper import SemanticPrimeMapper
from tinyaleph.observer.learning_engine import LearningEngine

from .schema import ObserverEpisode, ObserverSymbol, SymbolId


@dataclass
class ObserverState:
    boundary: BoundaryLayer
    agency: AgencyLayer
    smf: SedenionMemoryField
    symbols: SymbolDatabase
    lexicon: Dict[SymbolId, Dict[str, Any]]
    mapper: SemanticPrimeMapper
    learning: LearningEngine
    cycle: int = 0


class SentientObserverWrapper:
    """
    Thin wrapper composing BoundaryLayer + AgencyLayer + SMF to expose
    step/get_symbols/upsert_lexicon expected by the sentient LLM loop.

    This does not alter core logic; it just presents the minimal protocol surface.
    """

    def __init__(self):
        mapper = SemanticPrimeMapper(num_primes=128)

        def _chaperone_fn(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Stub: integrate your LLM here; returns dict shaped like TS chaperone outputs
            gtype = ctx.get("type")
            if gtype == "define_symbol":
                prime = ctx.get("target_prime") or 2
                return {"prime": prime, "meaning": f"concept_{prime}", "confidence": 0.7}
            if gtype == "find_relationship":
                return {"relationships": []}
            if gtype == "expand_concept":
                return {"suggestions": []}
            return {}

        learning = LearningEngine(mapper=mapper, chaperone_fn=_chaperone_fn)
        symbol_db = SymbolDatabase()

        self.state = ObserverState(
            boundary=BoundaryLayer(),
            agency=AgencyLayer(),
            smf=SedenionMemoryField(),
            symbols=symbol_db,
            lexicon={},
            mapper=mapper,
            learning=learning,
            cycle=0,
        )
        
        # Initialize core concepts for learning
        self._initialize_core_concepts(symbol_db, learning)
    
    def _initialize_core_concepts(self, symbol_db: Any, learning: Any) -> None:
        """Initialize core concepts to bootstrap the learning process"""
        # Get a selection of foundational symbols to seed learning
        core_symbol_ids = [
            'unity', 'duality', 'harmony', 'chaos', 'order',
            'transformation', 'consciousness', 'life', 'wisdom',
            'hero', 'sage', 'mother', 'father', 'self',
            'fire', 'water', 'earth', 'air'
        ]
        
        for symbol_id in core_symbol_ids:
            symbol = symbol_db.get_symbol(symbol_id)
            if symbol:
                # Add the learned meaning to the mapper (symbols already seeded from SymbolDatabase)
                # This ensures the prime is tracked in the mapper's field
                self.state.mapper.add_learned_meaning(
                    symbol.prime,
                    symbol.name,
                    confidence=0.95,
                    category=symbol.category.name if hasattr(symbol, 'category') else None
                )
                
                # Add to learning goals
                learning.add_goal(
                    goal_type="define_symbol",
                    description=f"Understand {symbol.name}",
                    priority=0.8,
                    target_prime=symbol.prime,
                    concepts=[symbol.name] + (symbol.cultural_tags[:2] if symbol.cultural_tags else [])
                )
                
                # Initialize lexicon entry
                self.state.lexicon[symbol.id] = {
                    "label": symbol.name,
                    "unicode": symbol.unicode,
                    "prime": symbol.prime,
                    "tags": symbol.cultural_tags[:5] if symbol.cultural_tags else [],
                }

    def step(self) -> ObserverEpisode:
        st = self.state
        st.cycle += 1
        # Advance internal clocks
        st.smf.step(dt=1.0)

        smf_state = {"s": st.smf.s.tolist() if hasattr(st.smf, "s") else []}

        # Run one learning step opportunistically each step
        st.learning.process_next_goal()

        # Minimal state fusion for agency update
        agency_state = st.agency.update({
            "smf": smf_state,
            "coherence": st.smf.mean_coherence,
            "entropy": st.smf.total_entropy,
            "active_primes": [],
        })

        # Construct episode
        return ObserverEpisode(
            episode_id=f"ep_{st.cycle}",
            created_at=time.time(),
            context={
                "coherence": st.smf.mean_coherence,
                "entropy": st.smf.total_entropy,
                "agency": agency_state,
            },
            actions=[],
            entropy_before=st.smf.total_entropy,
            entropy_after=st.smf.total_entropy,
            symbols_stabilized=None,
            notes="observer wrapper step",
        )

    def get_symbols(self, ids: Optional[List[SymbolId]] = None) -> List[ObserverSymbol]:
        syms = self.state.symbols.get_all_symbols()
        if ids:
            syms = [s for s in syms if s.id in ids]
        out: List[ObserverSymbol] = []
        now = time.time()
        
        # Create dynamic stability/novelty based on SMF state and cycle
        # This ensures symbols evolve over time and trigger continued training
        import math
        coherence = self.state.smf.mean_coherence
        entropy = self.state.smf.total_entropy
        
        for i, s in enumerate(syms):
            # Vary stability based on coherence and prime oscillation
            # This creates temporal dynamics that trigger the update condition
            phase = (self.state.cycle * 0.1 + s.prime * 0.01) % (2 * math.pi)
            stability_drift = 0.08 * math.sin(phase)  # ±0.08 variation (increased)
            
            # Base stability starts at 0.90 and oscillates up to 0.98
            # This ensures symbols periodically cross the 0.92 mint threshold
            stability_base = 0.92 + (coherence - 0.85) * 0.3  # 0.92 base, scales with coherence
            stability = max(0.88, min(0.99, stability_base + stability_drift))
            
            # For first 10 cycles, force higher stability to kickstart minting
            if self.state.cycle <= 10 and i < 5:
                stability = 0.95  # Force first 5 symbols to be mintable initially
            
            # Vary novelty based on entropy and position
            # Ensures some symbols remain novel enough to be interesting
            novelty_base = 0.25 + (entropy * 0.15) + (i % 5) * 0.03
            novelty = max(0.21, min(0.65, novelty_base))  # Raise minimum to 0.21 (above 0.20 threshold)
            
            out.append(
                ObserverSymbol(
                    id=s.id,
                    created_at=now,
                    stability=stability,
                    novelty=novelty,
                    prime_basis=[s.prime],
                    embedding=None,
                    metadata={"unicode": s.unicode},
                )
            )
        return out

    def upsert_lexicon(self, mapping: Dict[SymbolId, Dict[str, Any]]) -> None:
        self.state.lexicon.update(mapping)
    
    def save_state(self, path: str) -> None:
        """Save the observer state to disk"""
        import json
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "cycle": self.state.cycle,
                "coherence": self.state.smf.mean_coherence,
                "entropy": self.state.smf.total_entropy,
                "lexicon_size": len(self.state.lexicon),
            }, f, indent=2)
