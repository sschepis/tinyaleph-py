"""
Configuration for the Resonant LLM (Reso-LLM).

Standard Configuration (Default):
- Matches jupyter2.py architecture exactly for compatibility with Colab-trained models
- Core ResoFormer with RoPE, coherence gating, and Kuramoto dynamics

Extended Configuration (Optional):
- Agency Layer (self-directed attention and goals)
- PRSC (Prime Resonance Semantic Coherence)
- Temporal SMF (episodic memory with decay)
- Entanglement Network (multi-agent coordination)
- Advanced Stability Monitoring (predictive Lyapunov analysis)
- Stochastic Resonance (controlled noise injection)
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import math

# Golden ratio - used for coherence thresholds
PHI: float = (1 + math.sqrt(5)) / 2
COHERENCE_THRESHOLD: float = 0.7


# =============================================================================
# Extended Component Configurations (Optional - Disabled by Default)
# =============================================================================

@dataclass
class AgencyConfig:
    """Configuration for the Agency Layer (optional extension)."""
    enabled: bool = False  # Disabled by default for jupyter2.py compatibility
    max_foci: int = 5  # Maximum simultaneous attention foci
    max_goals: int = 10  # Maximum active goals
    attention_decay_rate: float = 0.02
    novelty_weight: float = 0.4  # Weight for novelty in salience
    relevance_weight: float = 0.4  # Weight for goal-relevance
    intensity_weight: float = 0.2  # Weight for raw intensity
    goal_coherence_threshold: float = 0.3  # Coherence below this triggers corrective goals


@dataclass
class PRSCConfig:
    """Configuration for Prime Resonance Semantic Coherence (optional extension)."""
    enabled: bool = False  # Disabled by default for jupyter2.py compatibility
    coherence_threshold: float = 0.7  # Minimum coherence for valid bindings
    max_bindings: int = 1000  # Maximum semantic bindings
    composition_decay: float = 0.1  # Amplitude decay during composition
    auto_bind_frequent: bool = True  # Auto-bind frequent token patterns


@dataclass
class TemporalSMFConfig:
    """Configuration for Temporal Sedenion Memory Field (optional extension)."""
    enabled: bool = False  # Disabled by default for jupyter2.py compatibility
    smf_dim: int = 16  # Sedenion dimensions
    memory_decay_rate: float = 0.01  # Exponential decay rate
    max_moments: int = 1000  # Maximum stored memories
    temporal_resolution: float = 0.001  # Time step granularity
    importance_threshold: float = 0.1  # Minimum importance to store
    episodic_tagging: bool = True  # Enable temporal phase tagging


@dataclass
class EntanglementConfig:
    """Configuration for Entanglement Network (optional multi-agent extension)."""
    enabled: bool = False  # Disabled by default
    node_id: str = "primary"  # This agent's node ID
    default_primes: tuple = (2, 3)  # Default primes for Bell pairs
    base_fidelity: float = 0.95  # Entanglement source fidelity
    success_probability: float = 0.8  # Pair generation success rate
    swap_success_probability: float = 0.5  # Swapping success rate
    distillation_target: float = 0.99  # Target fidelity for distillation


@dataclass
class StabilityConfig:
    """Configuration for Predictive Stability Monitoring (optional extension)."""
    enabled: bool = False  # Disabled by default for jupyter2.py compatibility
    lyapunov_threshold: float = 0.1  # Lambda threshold for chaos detection
    entropy_window: int = 20  # Window size for entropy tracking
    predictive_horizon: int = 5  # Steps ahead to predict instability
    auto_temperature_adjust: bool = True  # Lower temp when approaching chaos
    min_temperature: float = 0.3  # Minimum temperature during adjustment
    max_temperature: float = 1.5  # Maximum temperature
    bifurcation_sensitivity: float = 0.05  # Sensitivity to attractor splitting


@dataclass
class StochasticResonanceConfig:
    """Configuration for Stochastic Resonance (optional noise injection)."""
    enabled: bool = False  # Disabled by default
    noise_amplitude: float = 0.1  # Base noise level
    signal_threshold: float = 0.3  # Minimum signal strength to boost
    optimal_noise_ratio: float = 0.5  # Noise/signal ratio for resonance
    escape_threshold: float = 0.8  # Repetition score triggering escape


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class ResoLLMConfig:
    """
    Configuration for the Resonant LLM.
    
    Standard Mode (default):
        Matches jupyter2.py architecture exactly for compatibility with 
        Colab-trained models. Uses:
        - ResoFormerBlock with RoPE positional encoding
        - Coherence gating for stable generation
        - Optional Kuramoto dynamics for attention synchronization
        - Optional SMF memory (holographic memory field)
    
    Extended Mode (optional):
        Enable additional physics-based features by setting extension configs:
        - agency.enabled = True: Self-directed reasoning
        - prsc.enabled = True: Compositional semantics
        - temporal_smf.enabled = True: Episodic memory
        - entanglement.enabled = True: Multi-agent coordination
        - stability.enabled = True: Predictive stability monitoring
        - stochastic_resonance.enabled = True: Escape local minima
    """
    
    # =========================================================================
    # Core Model Architecture (matches jupyter2.py exactly)
    # =========================================================================
    vocab_size: int = 50257  # GPT-2 vocab size
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 1024
    
    # =========================================================================
    # Standard Resonant Features (matches jupyter2.py)
    # =========================================================================
    use_resonance: bool = True
    use_coherence_gating: bool = True
    coherence_threshold: float = COHERENCE_THRESHOLD
    
    # Memory (Sedenion Memory Field) - standard jupyter2.py style
    use_smf_memory: bool = True
    smf_dim: int = 16
    memory_decay_rate: float = 0.01
    max_memories: int = 1000
    
    # Dynamics (Kuramoto) - standard jupyter2.py style
    use_kuramoto_dynamics: bool = True
    kuramoto_coupling: float = 1.0
    
    # Stability (Lyapunov) - standard jupyter2.py style
    stability_threshold: float = 0.1
    
    # =========================================================================
    # Training Configuration
    # =========================================================================
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    save_steps: int = 1000
    
    # =========================================================================
    # Extended Components (Optional - All Disabled by Default)
    # =========================================================================
    agency: AgencyConfig = field(default_factory=AgencyConfig)
    prsc: PRSCConfig = field(default_factory=PRSCConfig)
    temporal_smf: TemporalSMFConfig = field(default_factory=TemporalSMFConfig)
    entanglement: EntanglementConfig = field(default_factory=EntanglementConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    stochastic_resonance: StochasticResonanceConfig = field(default_factory=StochasticResonanceConfig)
    
    # =========================================================================
    # Compatibility Mode Flag
    # =========================================================================
    # When True, uses simplified model matching jupyter2.py exactly
    # When False, enables extended architecture (if extensions are enabled)
    standard_mode: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dim % self.num_heads != 0:
            raise ValueError(f"Dimension {self.dim} must be divisible by num_heads {self.num_heads}")
        
        # Ensure phi-based defaults where applicable
        if self.coherence_threshold == 0:
            self.coherence_threshold = 1.0 / PHI  # ≈ 0.618
    
    def has_extensions_enabled(self) -> bool:
        """Check if any extended features are enabled."""
        return (
            self.agency.enabled or
            self.prsc.enabled or
            self.temporal_smf.enabled or
            self.entanglement.enabled or
            self.stability.enabled or
            self.stochastic_resonance.enabled
        )
    
    def enable_extensions(self):
        """Enable all extension features (switches to extended mode)."""
        self.standard_mode = False
        self.agency.enabled = True
        self.prsc.enabled = True
        self.temporal_smf.enabled = True
        self.stability.enabled = True
        # Note: entanglement and stochastic_resonance still disabled (advanced)
    
    def disable_extensions(self):
        """Disable all extension features (switches to standard mode)."""
        self.standard_mode = True
        self.agency.enabled = False
        self.prsc.enabled = False
        self.temporal_smf.enabled = False
        self.entanglement.enabled = False
        self.stability.enabled = False
        self.stochastic_resonance.enabled = False

    # =========================================================================
    # Factory Methods for Common Configurations
    # =========================================================================
    
    @classmethod
    def standard(cls, **kwargs) -> 'ResoLLMConfig':
        """
        Create standard config matching jupyter2.py exactly.
        
        This is the default and ensures compatibility with models
        trained using the Colab notebook.
        """
        config = cls(**kwargs)
        config.disable_extensions()
        return config
    
    @classmethod
    def extended(cls, **kwargs) -> 'ResoLLMConfig':
        """
        Create extended config with all optional features enabled.
        
        Enables Agency, PRSC, Temporal SMF, and Stability monitoring.
        """
        config = cls(**kwargs)
        config.enable_extensions()
        return config

    @classmethod
    def tiny(cls, standard: bool = True) -> 'ResoLLMConfig':
        """Tiny model for quick experiments (~10M params)."""
        config = cls(
            dim=256,
            num_layers=4,
            num_heads=4,
            ffn_dim=1024,
            max_seq_len=512,
            standard_mode=standard
        )
        if standard:
            config.disable_extensions()
        return config

    @classmethod
    def small(cls, standard: bool = True) -> 'ResoLLMConfig':
        """Small model (~125M params): GPT-2 Small equivalent."""
        config = cls(
            dim=768,
            num_layers=12,
            num_heads=12,
            ffn_dim=3072,
            max_seq_len=1024,
            standard_mode=standard
        )
        if standard:
            config.disable_extensions()
        return config
    
    # Alias for backward compatibility
    @classmethod
    def base(cls, standard: bool = True) -> 'ResoLLMConfig':
        """Base model (alias for small)."""
        return cls.small(standard=standard)

    @classmethod
    def medium(cls, standard: bool = True) -> 'ResoLLMConfig':
        """Medium model (~350M params): GPT-2 Medium equivalent."""
        config = cls(
            dim=1024,
            num_layers=24,
            num_heads=16,
            ffn_dim=4096,
            max_seq_len=1024,
            standard_mode=standard
        )
        if standard:
            config.disable_extensions()
        return config

    @classmethod
    def large(cls, standard: bool = True) -> 'ResoLLMConfig':
        """Large model (~760M params): GPT-2 Large equivalent."""
        config = cls(
            dim=1280,
            num_layers=36,
            num_heads=20,
            ffn_dim=5120,
            max_seq_len=2048,
            standard_mode=standard
        )
        if standard:
            config.disable_extensions()
        return config

    @classmethod
    def extra_large(cls, standard: bool = True) -> 'ResoLLMConfig':
        """Extra-large model (~1.3B params): GPT-2 XL equivalent."""
        config = cls(
            dim=1600,
            num_layers=48,
            num_heads=25,
            ffn_dim=6400,
            max_seq_len=2048,
            standard_mode=standard
        )
        if standard:
            config.disable_extensions()
        return config

    @classmethod
    def from_size(cls, size: str = 'small', standard: bool = True) -> 'ResoLLMConfig':
        """
        Create config from size name.
        
        Args:
            size: One of 'tiny', 'small', 'base', 'medium', 'large', 'extra_large' (or 'xl')
            standard: If True, use standard mode (jupyter2.py compatible)
        
        Returns:
            ResoLLMConfig instance
        """
        size_map = {
            'tiny': cls.tiny,
            'small': cls.small,
            'base': cls.base,
            'medium': cls.medium,
            'large': cls.large,
            'extra_large': cls.extra_large,
            'xl': cls.extra_large,
            'xlarge': cls.extra_large,
        }
        if size.lower() not in size_map:
            raise ValueError(f"Unknown size '{size}'. Choose from: {list(size_map.keys())}")
        return size_map[size.lower()](standard=standard)


# =============================================================================
# Generation Configuration
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    repetition_window: int = 50
    seed: Optional[int] = None


# =============================================================================
# Preset Functions (for backward compatibility)
# =============================================================================

def tiny_config(standard: bool = True) -> ResoLLMConfig:
    """Tiny model for quick experiments."""
    return ResoLLMConfig.tiny(standard=standard)


def small_config(standard: bool = True) -> ResoLLMConfig:
    """Small model (~125M params): GPT-2 Small equivalent."""
    return ResoLLMConfig.small(standard=standard)


def medium_config(standard: bool = True) -> ResoLLMConfig:
    """Medium model (~350M params): GPT-2 Medium equivalent."""
    return ResoLLMConfig.medium(standard=standard)


def large_config(standard: bool = True) -> ResoLLMConfig:
    """Large model (~760M params): GPT-2 Large equivalent."""
    return ResoLLMConfig.large(standard=standard)


def multi_agent_config(node_id: str = "agent_0") -> ResoLLMConfig:
    """Configuration for multi-agent scenarios with entanglement."""
    config = ResoLLMConfig.medium(standard=False)
    config.enable_extensions()
    config.entanglement = EntanglementConfig(
        enabled=True,
        node_id=node_id,
    )
    return config
