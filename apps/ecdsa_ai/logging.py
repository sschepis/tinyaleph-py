"""Simple JSONL logging for explorer runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import json
import time


@dataclass
class JsonlLogger:
    path: str

    def log(self, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("timestamp", time.time())
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
