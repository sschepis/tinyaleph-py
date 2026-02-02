"""
Load/save semantic landscapes.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from .landscape import SemanticLandscape


def load_landscape(path: str) -> SemanticLandscape:
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return SemanticLandscape.from_dict(data)


def save_landscape(landscape: SemanticLandscape, path: str) -> None:
    data = landscape.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
