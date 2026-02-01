from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Any

from ..schema import TrainingShard


@dataclass
class CallableStudentTrainer:
    """
    train_fn(shards: List[TrainingShard]) -> metrics dict
    eval_fn() -> metrics dict
    """
    train_fn: Callable[[List[TrainingShard]], Dict[str, Any]]
    eval_fn: Callable[[], Dict[str, Any]]

    def train_on_shards(self, shards: List[TrainingShard]) -> Dict[str, Any]:
        return self.train_fn(shards)

    def evaluate(self) -> Dict[str, Any]:
        return self.eval_fn()

