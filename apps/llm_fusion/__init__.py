"""
LLM Fusion: Resonance-Aware Layer Grafting for Pre-trained Transformers.

This module provides tools to graft resonance-aware fusion layers onto
existing pre-trained language models (GPT-2, Llama, etc.), combining
the strengths of:

1. Standard transformer architectures (proven, scalable, well-understood)
2. Prime resonance dynamics (semantic coherence, stability monitoring)
3. Quaternion geometry (rotation-aware representations)
4. Kuramoto synchronization (emergent coherence through oscillator coupling)

Key Components:
--------------
- FusionConfig: Configuration for fusion layers and components
- PrimeProjection: Project embeddings to prime Hilbert space
- QuaternionAttention: Attention using quaternionic inner products
- KuramotoModule: Coupled oscillator dynamics for synchronization
- ResonanceFusionLayer: Main fusion layer combining all components
- ResonanceWrapper: Hook-based wrapper for HuggingFace models
- ResonanceGenerator: Stability-aware text generation
- FusionTrainer: Training for fusion layers

Usage Example:
-------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from apps.llm_fusion import (
        ResonanceWrapper, 
        FusionConfig,
        ResonanceGenerator,
        GenerationConfig,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Configure fusion
    config = FusionConfig.for_gpt2()
    
    # Wrap with resonance layers
    model = ResonanceWrapper(base_model, config)
    
    # Generate with stability monitoring
    generator = ResonanceGenerator(model, tokenizer)
    result = generator.generate("Once upon a time")
    
    print(result.generated_text)
    print(f"Coherence: {result.stability_metrics.mean_coherence:.3f}")
"""

# Configuration
from .config import (
    FusionConfig,
    GenerationConfig,
    PrimeConfig,
    QuaternionConfig,
    KuramotoConfig,
    PRSCConfig,
    SMFConfig,
    StabilityConfig,
    PHI,
    COHERENCE_THRESHOLD,
)

# Prime embeddings
from .prime_embeddings import (
    PrimeProjection,
    PrimeBackProjection,
    SparsePrimeProjection,
)

# Quaternion operations
from .quaternion_layers import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
    quaternion_inverse,
    quaternion_exp,
    quaternion_log,
    quaternion_slerp,
    quaternion_rotate_vector,
    QuaternionLinear,
    QuaternionAttention,
    QuaternionRotationLayer,
)

# Kuramoto synchronization
from .kuramoto_attention import (
    compute_order_parameter,
    kuramoto_step,
    KuramotoModule,
    KuramotoAttentionModulator,
    GlobalSynchronizationLayer,
    AdaptiveCouplingLayer,
)

# Fusion layers
from .fusion_layers import (
    CoherenceGatingLayer,
    NeuralPRSC,
    SedenionMemoryCell,
    ResonanceFusionLayer,
    MultiLayerFusion,
)

# Model wrappers
from .wrappers import (
    ResonanceWrapper,
    LightweightWrapper,
)

# Inference
from .inference import (
    StabilityMetrics,
    GenerationResult,
    ResonanceGenerator,
    coherence_weighted_sample,
)

# Training
from .train import (
    TrainingConfig,
    ResonanceLoss,
    FusionTrainer,
    create_fusion_dataset,
    quick_train,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "FusionConfig",
    "GenerationConfig",
    "PrimeConfig",
    "QuaternionConfig",
    "KuramotoConfig",
    "PRSCConfig",
    "SMFConfig",
    "StabilityConfig",
    "PHI",
    "COHERENCE_THRESHOLD",
    
    # Prime embeddings
    "PrimeProjection",
    "PrimeBackProjection",
    "SparsePrimeProjection",
    
    # Quaternion
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_normalize",
    "quaternion_inverse",
    "quaternion_exp",
    "quaternion_log",
    "quaternion_slerp",
    "quaternion_rotate_vector",
    "QuaternionLinear",
    "QuaternionAttention",
    "QuaternionRotationLayer",
    
    # Kuramoto
    "compute_order_parameter",
    "kuramoto_step",
    "KuramotoModule",
    "KuramotoAttentionModulator",
    "GlobalSynchronizationLayer",
    "AdaptiveCouplingLayer",
    
    # Fusion
    "CoherenceGatingLayer",
    "NeuralPRSC",
    "SedenionMemoryCell",
    "ResonanceFusionLayer",
    "MultiLayerFusion",
    
    # Wrappers
    "ResonanceWrapper",
    "LightweightWrapper",
    
    # Inference
    "StabilityMetrics",
    "GenerationResult",
    "ResonanceGenerator",
    "coherence_weighted_sample",
    
    # Training
    "TrainingConfig",
    "ResonanceLoss",
    "FusionTrainer",
    "create_fusion_dataset",
    "quick_train",
]
