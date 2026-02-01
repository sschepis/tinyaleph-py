from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..schema import ObserverEpisode, ObserverSymbol, SymbolId


@dataclass
class CallableObserverAdapter:
    """
    Wrap your existing observer system with three functions:

    step_fn() -> ObserverEpisode-compatible dict OR ObserverEpisode
    get_symbols_fn(ids) -> list of dicts OR ObserverSymbol
    upsert_lexicon_fn(mapping) -> None
    """
    step_fn: Callable[[], Any]
    get_symbols_fn: Callable[[Optional[List[SymbolId]]], Any]
    upsert_lexicon_fn: Callable[[Dict[SymbolId, Dict[str, Any]]], None]

    def step(self) -> ObserverEpisode:
        x = self.step_fn()
        if isinstance(x, ObserverEpisode):
            return x
        return ObserverEpisode(**x)

    def get_symbols(self, ids: Optional[List[SymbolId]] = None) -> List[ObserverSymbol]:
        xs = self.get_symbols_fn(ids)
        out: List[ObserverSymbol] = []
        for x in xs:
            if isinstance(x, ObserverSymbol):
                out.append(x)
            else:
                out.append(ObserverSymbol(**x))
        return out

    def upsert_lexicon(self, mapping: Dict[SymbolId, Dict[str, Any]]) -> None:
        self.upsert_lexicon_fn(mapping)

