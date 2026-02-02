"""
Data models for semantic landscape premodels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class FusionRoute:
    p: int
    q: int
    r: int
    score: float
    canonical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "q": self.q,
            "r": self.r,
            "score": self.score,
            "canonical": self.canonical,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionRoute":
        return cls(
            p=int(data["p"]),
            q=int(data["q"]),
            r=int(data["r"]),
            score=float(data.get("score", 0.0)),
            canonical=bool(data.get("canonical", False)),
        )


@dataclass
class PrimeEntry:
    prime: int
    meaning: str
    category: str
    role: Optional[str]
    origin: str
    confidence: float
    coherence: float
    entropy: float
    route_count: int
    components: List[int] = field(default_factory=list)
    routes: List[FusionRoute] = field(default_factory=list)
    classification: Dict[str, Any] = field(default_factory=dict)
    adjective: Optional[str] = None
    refined: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prime": self.prime,
            "meaning": self.meaning,
            "category": self.category,
            "role": self.role,
            "origin": self.origin,
            "confidence": self.confidence,
            "coherence": self.coherence,
            "entropy": self.entropy,
            "route_count": self.route_count,
            "components": list(self.components),
            "routes": [r.to_dict() for r in self.routes],
            "classification": dict(self.classification),
            "adjective": self.adjective,
            "refined": self.refined,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimeEntry":
        return cls(
            prime=int(data["prime"]),
            meaning=str(data.get("meaning", "")),
            category=str(data.get("category", "")),
            role=data.get("role"),
            origin=str(data.get("origin", "")),
            confidence=float(data.get("confidence", 0.0)),
            coherence=float(data.get("coherence", 0.0)),
            entropy=float(data.get("entropy", 0.0)),
            route_count=int(data.get("route_count", 0)),
            components=[int(p) for p in data.get("components", [])],
            routes=[FusionRoute.from_dict(r) for r in data.get("routes", [])],
            classification=dict(data.get("classification", {})),
            adjective=data.get("adjective"),
            refined=bool(data.get("refined", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SemanticLandscape:
    metadata: Dict[str, Any]
    entries: Dict[int, PrimeEntry]
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        entries_sorted = [
            self.entries[p].to_dict() for p in sorted(self.entries.keys())
        ]
        return {
            "metadata": dict(self.metadata),
            "stats": dict(self.stats),
            "entries": entries_sorted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticLandscape":
        entries_list = data.get("entries", [])
        entries = {int(e["prime"]): PrimeEntry.from_dict(e) for e in entries_list}
        return cls(
            metadata=dict(data.get("metadata", {})),
            stats=dict(data.get("stats", {})),
            entries=entries,
        )
