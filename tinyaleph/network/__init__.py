"""
Network Primitives for Prime-Resonant Distributed Computing.

Provides:
- Identity: Prime Resonance Identity (PRI) for network nodes
- Entanglement: Quantum entanglement protocols for distributed nodes
"""

from tinyaleph.network.identity import (
    PrimeResonanceIdentity,
    EntangledNode,
)

from tinyaleph.network.entanglement import (
    BellState,
    EntangledPair,
    EntanglementSource,
    EntanglementSwapper,
    Teleporter,
    EntanglementDistiller,
    EntanglementNetwork,
    create_ghz_state,
    create_w_state,
    entanglement_entropy,
)

__all__ = [
    # Identity
    "PrimeResonanceIdentity",
    "EntangledNode",
    # Entanglement
    "BellState",
    "EntangledPair",
    "EntanglementSource",
    "EntanglementSwapper",
    "Teleporter",
    "EntanglementDistiller",
    "EntanglementNetwork",
    "create_ghz_state",
    "create_w_state",
    "entanglement_entropy",
]