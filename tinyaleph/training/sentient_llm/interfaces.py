from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .schema import ObserverEpisode, ObserverSymbol, TrainingShard, SymbolId


@dataclass
class LoopConfig:
    run_id: str
    out_dir: str

    max_cycles: int = 10_000

    # symbol minting gates
    mint_stability_threshold: float = 0.92
    mint_min_novelty: float = 0.20

    # teacher generation
    shards_per_new_symbol: int = 24
    shards_per_updated_symbol: int = 8

    # replay / continual learning
    replay_max_shards: int = 200_000

    # cadence
    train_every_cycles: int = 1
    eval_every_cycles: int = 10

    # batch sizing (shards; keep simple and token-agnostic)
    max_train_shards_per_step: int = 128
    replay_sample_per_step: int = 96
    
    # optional logging callback: log_fn(message, level="INFO")
    log_fn: Optional[Callable[[str, str], None]] = field(default=None, repr=False)


class SentientObserver(Protocol):
    def step(self) -> ObserverEpisode: ...
    def get_symbols(self, ids: Optional[List[SymbolId]] = None) -> List[ObserverSymbol]: ...
    def upsert_lexicon(self, mapping: Dict[SymbolId, Dict[str, Any]]) -> None: ...


class TeacherModel(Protocol):
    def generate_shards(
        self,
        symbols: List[ObserverSymbol],
        episodes: List[ObserverEpisode],
        shards_per_symbol: int,
    ) -> List[TrainingShard]: ...


class StudentTrainer(Protocol):
    def train_on_shards(self, shards: List[TrainingShard]) -> Dict[str, Any]: ...
    def evaluate(self) -> Dict[str, Any]: ...

