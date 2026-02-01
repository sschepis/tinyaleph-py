from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal
import json
import time

SymbolId = str  # "SYM:000001"


def now_ts() -> float:
    return time.time()


@dataclass
class ObserverSymbol:
    id: SymbolId
    created_at: float
    stability: float
    novelty: float
    prime_basis: Optional[List[int]] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ObserverEpisode:
    episode_id: str
    created_at: float
    context: Dict[str, Any]
    actions: List[Dict[str, Any]]
    entropy_before: Optional[float] = None
    entropy_after: Optional[float] = None
    symbols_stabilized: Optional[List[SymbolId]] = None
    notes: Optional[str] = None


@dataclass
class TrainingShard:
    shard_id: str
    created_at: float
    symbol_ids: List[SymbolId]
    kind: Literal["label", "definition", "example", "qa", "tool", "contrastive"]
    input_text: str
    target_text: str
    symbol_embedding: Optional[List[float]] = None


def dumps_jsonl(obj: Any) -> str:
    if hasattr(obj, "__dataclass_fields__"):
        return json.dumps(asdict(obj), ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False)

