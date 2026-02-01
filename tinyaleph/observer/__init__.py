"""
Observer architecture for sentient-like computation.

Provides:
- SedenionMemoryField: 16-dimensional holographic memory
- PRSC: Prime Resonance Semantic Coherence
- Temporal layer and moments
- Agency layer: attention, goals, actions
- Boundary layer: self/other distinction
- Safety layer: constraints and monitoring
- HQE: Holographic Quantum Encoding
- Symbols: Database of archetypal symbols
- SymbolicSMF: SMF with symbolic grounding
- SymbolicTemporal: I-Ching style moment classification
"""

from tinyaleph.observer.smf import SedenionMemoryField, MemoryMoment, SMF_AXES
from tinyaleph.observer.prsc import PRSC, SemanticBinding

# Temporal layer
from tinyaleph.observer.temporal import (
    Moment,
    TemporalLayer,
    TemporalPatternDetector,
)

# Agency layer
from tinyaleph.observer.agency import (
    AttentionFocus,
    Goal,
    Action,
    Intent,
    AgencyLayer,
)

# Boundary layer
from tinyaleph.observer.boundary import (
    SensoryChannel,
    MotorChannel,
    EnvironmentalModel,
    BoundarySelfModel,
    ObjectivityGate,
    BoundaryLayer,
)

# Safety layer
from tinyaleph.observer.safety import (
    SafetyConstraint,
    ViolationEvent,
    SafetyMonitor,
    SafetyLayer,
)

# HQE - Holographic Quantum Encoding
from tinyaleph.observer.hqe import (
    TickGate,
    StabilizationController,
    HolographicEncoder,
    HolographicMemory,
    HolographicSimilarity,
)

# Symbols
from tinyaleph.observer.symbols import (
    Symbol,
    SymbolCategory,
    SymbolDatabase,
    symbol_database,
    SemanticInference,
    ResonanceCalculator,
    CompoundSymbol,
    SymbolSequence,
    CompoundBuilder,
    compound_builder,
    EntityExtractor,
)

# Symbolic SMF
from tinyaleph.observer.symbolic_smf import (
    SymbolicSMF,
    SMFSymbolMapper,
    smf_mapper,
    AXIS_SYMBOL_MAPPING,
    TAG_TO_AXIS,
    create_symbolic_smf,
    from_smf,
    symbol_to_smf,
    symbols_to_smf,
)

# Symbolic Temporal
from tinyaleph.observer.symbolic_temporal import (
    SymbolicMoment,
    SymbolicTemporalLayer,
    SymbolicPatternDetector,
    EntropyCollapseHead,
    HEXAGRAM_ARCHETYPES,
    FIRST_64_PRIMES,
)

__all__ = [
    # SMF
    "SedenionMemoryField",
    "MemoryMoment",
    "SMF_AXES",
    # PRSC
    "PRSC",
    "SemanticBinding",
    # Temporal
    "Moment",
    "TemporalLayer",
    "TemporalPatternDetector",
    # Agency
    "AttentionFocus",
    "Goal",
    "Action",
    "Intent",
    "AgencyLayer",
    # Boundary
    "SensoryChannel",
    "MotorChannel",
    "EnvironmentalModel",
    "BoundarySelfModel",
    "ObjectivityGate",
    "BoundaryLayer",
    # Safety
    "SafetyConstraint",
    "ViolationEvent",
    "SafetyMonitor",
    "SafetyLayer",
    # HQE
    "TickGate",
    "StabilizationController",
    "HolographicEncoder",
    "HolographicMemory",
    "HolographicSimilarity",
    # Symbols
    "Symbol",
    "SymbolCategory",
    "SymbolDatabase",
    "symbol_database",
    "SemanticInference",
    "ResonanceCalculator",
    "CompoundSymbol",
    "SymbolSequence",
    "CompoundBuilder",
    "compound_builder",
    "EntityExtractor",
    # Symbolic SMF
    "SymbolicSMF",
    "SMFSymbolMapper",
    "smf_mapper",
    "AXIS_SYMBOL_MAPPING",
    "TAG_TO_AXIS",
    "create_symbolic_smf",
    "from_smf",
    "symbol_to_smf",
    "symbols_to_smf",
    # Symbolic Temporal
    "SymbolicMoment",
    "SymbolicTemporalLayer",
    "SymbolicPatternDetector",
    "EntropyCollapseHead",
    "HEXAGRAM_ARCHETYPES",
    "FIRST_64_PRIMES",
]