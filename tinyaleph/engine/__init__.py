"""
TinyAleph Engine Module

Unified backend-agnostic prime-based computing engine.
"""

from tinyaleph.engine.aleph import (
    AlephEngine,
    DefaultBackend,
    Frame,
    HistoryEntry,
    ReasoningStep,
    RunResult,
)

__all__ = [
    "AlephEngine",
    "DefaultBackend",
    "Frame",
    "HistoryEntry",
    "ReasoningStep",
    "RunResult",
]