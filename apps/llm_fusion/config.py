"""
Configuration for LLM Fusion: Resonance-Aware Layer Grafting.

Provides dataclasses for configuring fusion layers, prime projection,
quaternion operations, Kuramoto dynamics, and generation parameters.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import math


# Golden ratio - used for various thresholds and spacing
PHI: float = (1 + math.sqrt(5)) / 2
COHERENCE_THRESHOLD: float = 0.7


@dataclass
class PrimeConfig:
    """Configuration for prime projection layers."""
    num_primes: int = 25  # Number of prime basis states
    use_quaternion: bool = True  # Use quaternionic amplitudes (4D) vs complex (2D)
    normalize: bool = True  # Normalize prime states
    learnable_phases: bool = True  # Learnable phase biases per prime


@dataclass
class QuaternionConfig:
    """Configuration for quaternion operations."""
    use_slerp: bool = True  # Use SLERP for interpolation
    golden_ratio_heads: bool = True  # Use φ-spaced rotation axes
    normalize_output: bool = True  # Normalize quaternion outputs


@dataclass
class KuramotoConfig:
    """Configuration for Kuramoto synchronization module."""
    enabled: bool = True
    coupling: float = 1.0  # Coupling strength K
    dt: float = 0.01  # Time step for integration
    use_learned_frequencies: bool = True  # Learn natural frequencies


@dataclass
class PRSCConfig:
    """Configuration for Prime Resonance Semantic Coherence layer."""
    enabled: bool = True
    coherence_threshold: float = COHERENCE_THRESHOLD
    max_bindings: int = 1000
    composition_weight: float = 0.1  # Weight for composed semantics


@dataclass
class SMFConfig:
    """Configuration for Sedenion Memory Field."""
    enabled: bool = True
    decay_rate: float = 0.01
    max_moments: int = 1000
    memory_weight: float = 0.05  # Weight for memory injection


@dataclass
class StabilityConfig:
    """Configuration for stability monitoring."""
    enabled: bool = True
    lyapunov_threshold: float = 0.1
    coherence_threshold: float = 0.2
    entropy_window: int = 20
    auto_temperature: bool = True
    min_temperature: float = 0.3
    max_temperature: float = 1.5


@dataclass
class FusionConfig:
    """
    Main configuration for resonance-aware LLM fusion.
    
    Controls which layers receive fusion, what components are active,
    and how fusion is performed.
    """
    # Base model configuration
    hidden_dim: Optional[int] = None  # Will be inferred from base model if None
    
    # Fusion positions (layer indices where fusion is applied)
    fusion_positions: List[int] = field(default_factory=lambda: [4, 8, 11])
    
    # Fusion mode
    fusion_mode: str = "residual"  # "parallel", "sequential", "residual"
    fusion_alpha: float = 0.5  # Initial fusion weight (higher for knowledge injection)
    learnable_alpha: bool = True  # Make fusion weight learnable
    fusion_norm: str = "output"  # "output", "fusion", "none"
    adapter_rank: int = 128  # Rank for output adapter
    
    # Component configurations
    prime: PrimeConfig = field(default_factory=PrimeConfig)
    quaternion: QuaternionConfig = field(default_factory=QuaternionConfig)
    kuramoto: KuramotoConfig = field(default_factory=KuramotoConfig)
    prsc: PRSCConfig = field(default_factory=PRSCConfig)
    smf: SMFConfig = field(default_factory=SMFConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    
    # Components to enable (shorthand)
    enabled_components: List[str] = field(default_factory=lambda: [
        "prime", "quaternion", "kuramoto", "coherence_gate"
    ])
    
    def __post_init__(self):
        """Validate configuration."""
        valid_modes = ["parallel", "sequential", "residual"]
        if self.fusion_mode not in valid_modes:
            raise ValueError(f"fusion_mode must be one of {valid_modes}")
        
        valid_norms = ["output", "fusion", "none"]
        if self.fusion_norm not in valid_norms:
            raise ValueError(f"fusion_norm must be one of {valid_norms}")
        
        valid_components = ["prime", "quaternion", "kuramoto", "prsc", "smf", "coherence_gate"]
        for comp in self.enabled_components:
            if comp not in valid_components:
                raise ValueError(f"Unknown component '{comp}'. Valid: {valid_components}")
    
    @classmethod
    def minimal(cls) -> 'FusionConfig':
        """Minimal configuration with only prime projection and gating."""
        return cls(
            fusion_positions=[6],  # Single layer
            enabled_components=["prime", "coherence_gate"],
            prime=PrimeConfig(num_primes=16, use_quaternion=False),
            kuramoto=KuramotoConfig(enabled=False),
            prsc=PRSCConfig(enabled=False),
            smf=SMFConfig(enabled=False),
            fusion_alpha=0.05,
            fusion_norm="fusion",
        )

    @classmethod
    def enrichment(cls, num_layers: Optional[int] = None) -> 'FusionConfig':
        """Semantic enrichment configuration with small, controlled deltas."""
        if num_layers is None:
            positions = [6]
        else:
            mid = max(1, num_layers // 2)
            positions = [mid]
        return cls(
            fusion_positions=positions,
            enabled_components=["prime", "coherence_gate"],
            prime=PrimeConfig(num_primes=16, use_quaternion=False),
            kuramoto=KuramotoConfig(enabled=False),
            prsc=PRSCConfig(enabled=False),
            smf=SMFConfig(enabled=False),
            fusion_alpha=0.05,
            fusion_norm="fusion",
        )
    
    @classmethod
    def standard(cls) -> 'FusionConfig':
        """Standard configuration with core components."""
        return cls(
            fusion_positions=[4, 8, 11],
            enabled_components=["prime", "quaternion", "kuramoto", "coherence_gate"],
        )
    
    @classmethod
    def full(cls) -> 'FusionConfig':
        """Full configuration with all components enabled."""
        return cls(
            fusion_positions=[2, 4, 6, 8, 10, 11],
            enabled_components=["prime", "quaternion", "kuramoto", "prsc", "smf", "coherence_gate"],
            prsc=PRSCConfig(enabled=True),
            smf=SMFConfig(enabled=True),
        )
    
    @classmethod
    def for_gpt2(cls) -> 'FusionConfig':
        """Configuration optimized for GPT-2 models (12 layers)."""
        return cls(
            hidden_dim=768,
            fusion_positions=[3, 6, 9, 11],
            prime=PrimeConfig(num_primes=25),
        )
    
    @classmethod
    def for_llama(cls, num_layers: int = 32) -> 'FusionConfig':
        """Configuration optimized for Llama models."""
        # Fusion at 25%, 50%, 75%, and final layers
        positions = [
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1
        ]
        return cls(
            fusion_positions=positions,
            prime=PrimeConfig(num_primes=32),
        )


@dataclass
class GenerationConfig:
    """Configuration for resonance-aware text generation."""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_length: int = 100
    repetition_penalty: float = 1.1
    
    # Stability-aware generation
    auto_temperature: bool = True  # Adjust temperature based on stability
    stop_on_instability: bool = True  # Stop if divergent
    
    # Memory settings
    update_memory: bool = True  # Update SMF with generated text
    memory_importance: float = 0.8  # Importance for new memories
    
    seed: Optional[int] = None
