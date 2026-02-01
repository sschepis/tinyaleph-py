"""
Semantic module for TinyAleph.

Implements formal type systems and reduction semantics for prime-indexed
compositional languages, including:

- Type system: NounTerm, AdjTerm, ChainTerm, FusionTerm, Sentences
- Reduction: Prime operators, reduction system, proof generation
- Lambda calculus: Translation, evaluation, concept interpretation
- CRT-Homology: Residue encoding, CRT reconstruction, Birkhoff projection
- Topology: Knots, gauge symmetry, free energy dynamics
- Inference: Compound building, semantic inference, entity extraction
"""

from .types import (
    Types,
    Term,
    NounTerm,
    AdjTerm,
    ChainTerm,
    FusionTerm,
    SentenceTerm,
    NounSentence,
    SeqSentence,
    ImplSentence,
    TypingContext,
    TypingJudgment,
    TypeChecker,
    N, A, FUSE, CHAIN, SENTENCE, SEQ, IMPL
)

from .reduction import (
    PrimeOperator,
    NextPrimeOperator,
    ModularPrimeOperator,
    ResonancePrimeOperator,
    IdentityPrimeOperator,
    ReductionStep,
    ReductionTrace,
    ReductionSystem,
    FusionCanonicalizer,
    NormalFormVerifier,
    ProofTrace,
    ProofGenerator,
    RouteStatistics,
    is_normal_form,
    is_reducible,
    term_size,
    term_depth,
    extract_primes
)

from .lambda_calc import (
    LambdaExpr,
    VarExpr,
    ConstExpr,
    LamExpr,
    AppExpr,
    PairExpr,
    ImplExpr,
    PrimOpExpr,
    Translator,
    TypeDirectedTranslator,
    LambdaEvaluator,
    Semantics,
    ConceptInterpreter,
    PRQS_LEXICON,
    classify_prime
)

from .crt_homology import (
    extended_gcd,
    mod_inverse,
    are_coprime,
    softmax,
    ResidueEncoding,
    ResidueEncoder,
    CRTReconstructor,
    DoublyStochasticMatrix,
    BirkhoffProjector,
    HomologyLoss,
    CRTModularLayer,
    CRTFusedAttention,
    CoprimeSelector,
    create_semantic_crt_encoder,
    crt_embed_sequence,
    crt_similarity,
    homology_regularizer
)

from .topology import (
    Knot,
    KnotDiagram,
    Crossing,
    CrossingSign,
    PhysicalConstants,
    GaugeGroup,
    GaugeField,
    GaugeSymmetry,
    BeliefState,
    Observation,
    FreeEnergyDynamics,
    TopologicalFeatures,
    create_semantic_knot,
    analyze_semantic_topology,
    derive_physical_constant,
    free_energy_update
)

from .inference import (
    SemanticCategory,
    SemanticPrimitive,
    SEMANTIC_PRIMES,
    CompoundConcept,
    CompoundBuilder,
    InferenceRule,
    SemanticInference,
    ExtractedEntity,
    ExtractedRelation,
    EntityExtractor,
    build_compound,
    analyze_semantic_value,
    infer_from_knowledge,
    extract_semantic_structure,
    semantic_similarity
)

__all__ = [
    # Types
    'Types', 'Term', 'NounTerm', 'AdjTerm', 'ChainTerm', 'FusionTerm',
    'SentenceTerm', 'NounSentence', 'SeqSentence', 'ImplSentence',
    'TypingContext', 'TypingJudgment', 'TypeChecker',
    'N', 'A', 'FUSE', 'CHAIN', 'SENTENCE', 'SEQ', 'IMPL',
    
    # Reduction
    'PrimeOperator', 'NextPrimeOperator', 'ModularPrimeOperator',
    'ResonancePrimeOperator', 'IdentityPrimeOperator',
    'ReductionStep', 'ReductionTrace', 'ReductionSystem',
    'FusionCanonicalizer', 'NormalFormVerifier',
    'ProofTrace', 'ProofGenerator', 'RouteStatistics',
    'is_normal_form', 'is_reducible', 'term_size', 'term_depth', 'extract_primes',
    
    # Lambda calculus
    'LambdaExpr', 'VarExpr', 'ConstExpr', 'LamExpr', 'AppExpr',
    'PairExpr', 'ImplExpr', 'PrimOpExpr',
    'Translator', 'TypeDirectedTranslator', 'LambdaEvaluator',
    'Semantics', 'ConceptInterpreter', 'PRQS_LEXICON', 'classify_prime',
    
    # CRT-Homology
    'extended_gcd', 'mod_inverse', 'are_coprime', 'softmax',
    'ResidueEncoding', 'ResidueEncoder', 'CRTReconstructor',
    'DoublyStochasticMatrix', 'BirkhoffProjector', 'HomologyLoss',
    'CRTModularLayer', 'CRTFusedAttention', 'CoprimeSelector',
    'create_semantic_crt_encoder', 'crt_embed_sequence', 'crt_similarity',
    'homology_regularizer',
    
    # Topology
    'Knot', 'KnotDiagram', 'Crossing', 'CrossingSign',
    'PhysicalConstants', 'GaugeGroup', 'GaugeField', 'GaugeSymmetry',
    'BeliefState', 'Observation', 'FreeEnergyDynamics',
    'TopologicalFeatures',
    'create_semantic_knot', 'analyze_semantic_topology',
    'derive_physical_constant', 'free_energy_update',
    
    # Inference
    'SemanticCategory', 'SemanticPrimitive', 'SEMANTIC_PRIMES',
    'CompoundConcept', 'CompoundBuilder',
    'InferenceRule', 'SemanticInference',
    'ExtractedEntity', 'ExtractedRelation', 'EntityExtractor',
    'build_compound', 'analyze_semantic_value', 'infer_from_knowledge',
    'extract_semantic_structure', 'semantic_similarity',
]