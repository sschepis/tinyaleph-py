from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, List
import os
import random

from .schema import TrainingShard, dumps_jsonl


@dataclass
class ReplayBuffer:
    max_items: int
    items: Deque[TrainingShard]

    @classmethod
    def create(cls, max_items: int) -> "ReplayBuffer":
        return cls(max_items=max_items, items=deque())

    def add_many(self, shards: List[TrainingShard]) -> None:
        for s in shards:
            self.items.append(s)
            while len(self.items) > self.max_items:
                self.items.popleft()

    def sample(self, n: int) -> List[TrainingShard]:
        if not self.items:
            return []
        n = min(n, len(self.items))
        return random.sample(list(self.items), n)


def append_jsonl(path: str, shards: List[TrainingShard]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for s in shards:
            f.write(dumps_jsonl(s) + "\n")

