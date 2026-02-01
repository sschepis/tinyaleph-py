"""
Physics Primitives for TinyAleph.

Provides:
- Kuramoto oscillator model for phase synchronization
- Extended sync models (network, adaptive, Sakaguchi, small-world)
- Stochastic Kuramoto with Langevin noise
- Entropy analysis and Lyapunov stability
- Base oscillator classes with phase-amplitude dynamics
- Quantum-inspired collapse mechanics
- Extended Lyapunov analysis and stability tracking
"""

from tinyaleph.physics.kuramoto import (
    KuramotoModel,
)

from tinyaleph.physics.entropy import (
    StabilityClass,
    classify_stability,
    shannon_entropy,
    relative_entropy,
    joint_entropy,
    conditional_entropy,
    mutual_information,
    EntropyTracker,
    LyapunovAnalyzer,
    prime_entropy,
    golden_entropy_threshold,
    EntropyGradient,
    coherence_from_entropy,
    entropy_production_rate,
)

from tinyaleph.physics.oscillator import (
    Oscillator,
    OscillatorBank,
    DrivenOscillator,
    NoisyOscillator,
)

from tinyaleph.physics.collapse import (
    collapse_probability,
    should_collapse,
    measure_state,
    collapse_to_index,
    born_measurement,
    partial_collapse,
    apply_decoherence,
    continuous_collapse,
    zeno_effect,
    entropy_threshold_collapse,
    CollapseMonitor,
)

from tinyaleph.physics.lyapunov import (
    StabilityRegime,
    estimate_lyapunov,
    estimate_lyapunov_from_oscillators,
    local_lyapunov,
    classify_stability_string,
    adaptive_coupling,
    delay_embedding,
    stability_margin,
    spectrum_from_jacobian,
    finite_time_lyapunov,
    LyapunovTracker,
)

from tinyaleph.physics.sync_models import (
    NetworkKuramoto,
    AdaptiveKuramoto,
    SakaguchiKuramoto,
    SmallWorldKuramoto,
    MultiSystemCoupling,
    create_hierarchical_coupling,
    create_peer_coupling,
)

from tinyaleph.physics.stochastic import (
    StochasticKuramoto,
    ColoredNoiseKuramoto,
    ThermalKuramoto,
    gaussian_random,
)

__all__ = [
    # Kuramoto
    "KuramotoModel",
    # Extended Sync Models
    "NetworkKuramoto",
    "AdaptiveKuramoto",
    "SakaguchiKuramoto",
    "SmallWorldKuramoto",
    "MultiSystemCoupling",
    "create_hierarchical_coupling",
    "create_peer_coupling",
    # Stochastic
    "StochasticKuramoto",
    "ColoredNoiseKuramoto",
    "ThermalKuramoto",
    "gaussian_random",
    # Entropy
    "StabilityClass",
    "classify_stability",
    "shannon_entropy",
    "relative_entropy",
    "joint_entropy",
    "conditional_entropy",
    "mutual_information",
    "EntropyTracker",
    "LyapunovAnalyzer",
    "prime_entropy",
    "golden_entropy_threshold",
    "EntropyGradient",
    "coherence_from_entropy",
    "entropy_production_rate",
    # Oscillator
    "Oscillator",
    "OscillatorBank",
    "DrivenOscillator",
    "NoisyOscillator",
    # Collapse
    "collapse_probability",
    "should_collapse",
    "measure_state",
    "collapse_to_index",
    "born_measurement",
    "partial_collapse",
    "apply_decoherence",
    "continuous_collapse",
    "zeno_effect",
    "entropy_threshold_collapse",
    "CollapseMonitor",
    # Lyapunov
    "StabilityRegime",
    "estimate_lyapunov",
    "estimate_lyapunov_from_oscillators",
    "local_lyapunov",
    "classify_stability_string",
    "adaptive_coupling",
    "delay_embedding",
    "stability_margin",
    "spectrum_from_jacobian",
    "finite_time_lyapunov",
    "LyapunovTracker",
]