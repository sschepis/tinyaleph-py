"""
Machine learning primitives for TinyAleph.

Provides:
- SparsePrimeState: H_Q = H_P ⊗ ℍ (Prime-quaternion tensor states)
- Resonant attention mechanisms
- Coherence-gated halting (ACT)
- ResoFormer: Complete transformer architecture with quaternion operations
"""

from tinyaleph.ml.sparse_state import SparsePrimeState, coherent_superposition, golden_superposition
from tinyaleph.ml.attention import resonant_attention, golden_ratio
from tinyaleph.ml.resoformer import (
    # Tensor operations
    Tensor,
    zeros,
    ones,
    randn,
    glorot_uniform,
    quaternion_normalize,
    
    # Base classes
    Layer,
    
    # ResoFormer layers
    QuaternionDense,
    SparsePrimeEmbedding,
    ResonantAttentionLayer,
    CoherenceGatingLayer,
    EntropyCollapseLayer,
    ResonanceOperator,
    LayerNorm,
    Dense,
    Dropout,
    ResoFormerBlock,
    
    # Model builders
    ResoFormerConfig,
    ResoFormerModel,
    create_resoformer_model,
    create_resoformer_classifier,
    create_resoformer_embedder,
)

__all__ = [
    # Existing exports
    "SparsePrimeState",
    "coherent_superposition",
    "golden_superposition",
    "resonant_attention",
    "golden_ratio",
    
    # ResoFormer tensor operations
    "Tensor",
    "zeros",
    "ones",
    "randn",
    "glorot_uniform",
    "quaternion_normalize",
    
    # Base classes
    "Layer",
    
    # ResoFormer layers
    "QuaternionDense",
    "SparsePrimeEmbedding",
    "ResonantAttentionLayer",
    "CoherenceGatingLayer",
    "EntropyCollapseLayer",
    "ResonanceOperator",
    "LayerNorm",
    "Dense",
    "Dropout",
    "ResoFormerBlock",
    
    # Model builders
    "ResoFormerConfig",
    "ResoFormerModel",
    "create_resoformer_model",
    "create_resoformer_classifier",
    "create_resoformer_embedder",
]