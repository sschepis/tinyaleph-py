# LLM Fusion: Resonance-Aware Layer Grafting

Graft physics-inspired resonance layers onto pre-trained transformer models to create hybrid systems that combine the scalability of standard LLMs with the coherence and stability properties of prime resonance dynamics.

## Overview

**LLM Fusion** provides a modular framework for enhancing existing language models with:

- **Prime Hilbert Space Projections**: Map token embeddings to a basis indexed by prime numbers, enabling compositional semantics through prime factorization
- **Quaternion Geometry**: 4D hypercomplex representations with rotation-aware attention
- **Kuramoto Synchronization**: Coupled oscillator dynamics for emergent coherence
- **Coherence Gating**: Stability-based information flow control
- **Lyapunov Monitoring**: Real-time divergence detection for hallucination prevention

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ResonanceWrapper                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐    ┌───────────────────────────────────────┐ │
│   │   Base Model    │    │      Fusion Layers (via hooks)        │ │
│   │   (GPT-2, etc)  │◄───┤                                       │ │
│   │                 │    │  ┌─────────────────────────────────┐  │ │
│   │  Layer 0        │    │  │     ResonanceFusionLayer        │  │ │
│   │  Layer 1        │    │  │  ┌─────────┐  ┌──────────────┐  │  │ │
│   │  ...            │    │  │  │ Prime   │  │ Quaternion   │  │  │ │
│   │  Layer k ◄──────┼────┼──┼─►│ Project │─►│ Rotation     │  │  │ │
│   │  ...            │    │  │  └─────────┘  └──────────────┘  │  │ │
│   │  Layer n        │    │  │       │              │          │  │ │
│   └─────────────────┘    │  │       ▼              ▼          │  │ │
│                          │  │  ┌─────────┐  ┌──────────────┐  │  │ │
│                          │  │  │Kuramoto │  │  Coherence   │  │  │ │
│                          │  │  │ Sync    │─►│   Gate       │  │  │ │
│                          │  │  └─────────┘  └──────────────┘  │  │ │
│                          │  └─────────────────────────────────┘  │ │
│                          └───────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# From the AlephPrime repository root
pip install -e .

# Optional: Install transformers for HuggingFace integration
pip install transformers
```

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from apps.llm_fusion import ResonanceWrapper, FusionConfig

# Load a pre-trained model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure fusion (layers 3, 6, 9, 11 get resonance)
config = FusionConfig.for_gpt2()

# Wrap with resonance layers
model = ResonanceWrapper(base_model, config, freeze_base=True)

# Generate text
inputs = tokenizer("The meaning of life is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))

# Check fusion metrics
metrics = model.get_average_metrics()
print(f"Coherence: {metrics.get('gate_coherence', 'N/A')}")
print(f"Kuramoto Order: {metrics.get('kuramoto_order', 'N/A')}")
```

### Stability-Aware Generation

```python
from apps.llm_fusion import ResonanceGenerator, GenerationConfig

# Configure generation
gen_config = GenerationConfig(
    temperature=0.8,
    auto_temperature=True,  # Adjust based on stability
    stop_on_instability=True,  # Stop if diverging
)

# Create generator
generator = ResonanceGenerator(model, tokenizer, gen_config)

# Generate with monitoring
result = generator.generate(
    "Once upon a time in a kingdom far away",
    max_length=100
)

print(result.generated_text)
print(f"Mean Coherence: {result.stability_metrics.mean_coherence:.3f}")
print(f"Lyapunov Estimate: {result.stability_metrics.lyapunov_estimate:.4f}")
print(f"Stable: {result.stability_metrics.is_stable}")
```

### Training Fusion Layers

```python
from apps.llm_fusion import FusionTrainer, TrainingConfig, create_fusion_dataset

# Prepare training data
texts = [
    "Sample training text 1...",
    "Sample training text 2...",
    # ... more training samples
]

dataset = create_fusion_dataset(texts, tokenizer, max_length=512)

# Configure training
train_config = TrainingConfig(
    learning_rate=1e-4,
    max_steps=5000,
    coherence_loss_weight=0.1,  # Encourage high coherence
    kuramoto_loss_weight=0.05,  # Encourage synchronization
)

# Train
trainer = FusionTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    config=train_config
)

results = trainer.train()

# Save fusion weights only (small file!)
model.save_fusion_weights("fusion_weights.pt")
```

## Configuration Presets

### Minimal (Lowest overhead)
```python
config = FusionConfig.minimal()
# - Single layer fusion
# - Prime projection only (no quaternion)
# - ~100K additional parameters
```

### Standard (Recommended)
```python
config = FusionConfig.standard()
# - 4 fusion layers
# - Prime + Quaternion + Kuramoto + Gating
# - ~500K additional parameters
```

### Full (Maximum features)
```python
config = FusionConfig.full()
# - 6 fusion layers
# - All components including PRSC and SMF
# - ~1M additional parameters
```

### Model-Specific
```python
# For GPT-2 (768 hidden, 12 layers)
config = FusionConfig.for_gpt2()

# For Llama (custom layer count)
config = FusionConfig.for_llama(num_layers=32)
```

## Components

### Prime Projection
Maps hidden states to a prime-indexed Hilbert space:

```
H_P = {|ψ⟩ = Σ αp|p⟩ : p ∈ Primes}
```

With quaternionic amplitudes, each token becomes a superposition over prime basis states with 4D coefficients.

```python
from apps.llm_fusion import PrimeProjection

proj = PrimeProjection(
    input_dim=768,
    num_primes=25,
    use_quaternion=True
)

# (batch, seq, hidden) → (batch, seq, num_primes, 4)
prime_states = proj(hidden_states)
coherence = proj.coherence(prime_states)  # [0, 1]
```

### Quaternion Attention
Attention using quaternionic inner products:

```
⟨q₁, q₂⟩ = Re(q₁* ⊗ q₂)
```

This provides rotation-aware similarity that captures geometric relationships.

```python
from apps.llm_fusion import QuaternionAttention

attn = QuaternionAttention(
    hidden_dim=768,
    num_heads=8,
    golden_ratio_heads=True  # φ-spaced rotation axes
)

output, weights = attn(hidden_states, return_attention=True)
```

### Kuramoto Synchronization
Coupled oscillator dynamics for emergent coherence:

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

Order parameter `r ∈ [0,1]` measures global synchronization.

```python
from apps.llm_fusion import KuramotoModule

kuramoto = KuramotoModule(
    input_dim=768,
    coupling=1.5,
    num_steps=5
)

phases, order_param, _ = kuramoto(hidden_states)
# order_param close to 1 = synchronized (coherent)
# order_param close to 0 = desynchronized (incoherent)
```

### Coherence Gating
Gate information flow based on stability:

```python
from apps.llm_fusion import CoherenceGatingLayer

gate = CoherenceGatingLayer(
    hidden_dim=768,
    threshold=0.7,
    soft_gate=True
)

gated, coherence = gate(hidden_states)
# Low coherence → attenuated output
# High coherence → full pass-through
```

## Advanced Usage

### Custom Fusion Configuration

```python
from apps.llm_fusion import (
    FusionConfig, PrimeConfig, KuramotoConfig
)

config = FusionConfig(
    hidden_dim=1024,
    fusion_positions=[4, 8, 12, 15],  # Which layers
    fusion_mode="residual",  # "parallel", "sequential", "residual"
    fusion_alpha=0.15,  # Initial blending weight
    learnable_alpha=True,  # Learn optimal weight
    
    prime=PrimeConfig(
        num_primes=32,
        use_quaternion=True,
        learnable_phases=True
    ),
    
    kuramoto=KuramotoConfig(
        coupling=2.0,
        use_learned_frequencies=True
    ),
    
    enabled_components=["prime", "quaternion", "kuramoto", "coherence_gate"]
)
```

### Accessing Layer Metrics

```python
# After forward pass
layer_metrics = model.get_metrics()  # Dict[layer_idx, Dict[str, float]]

for layer_idx, metrics in layer_metrics.items():
    print(f"Layer {layer_idx}:")
    print(f"  Prime entropy: {metrics.get('prime_entropy', 'N/A')}")
    print(f"  Coherence: {metrics.get('gate_coherence', 'N/A')}")
    print(f"  Kuramoto order: {metrics.get('kuramoto_order', 'N/A')}")
```

### Standalone Fusion Layer

```python
from apps.llm_fusion import ResonanceFusionLayer

# Use without wrapper
fusion = ResonanceFusionLayer(hidden_dim=768, config=config)

# Apply to hidden states
output, metrics = fusion(hidden_states, return_metrics=True)
```

## Mathematical Background

### Prime Hilbert Space

Standard transformers operate in ℝⁿ. We extend this to a tensor product:

```
H_Q = H_P ⊗ ℍ
```

Where:
- `H_P` = Prime Hilbert space with basis `{|p⟩ : p ∈ Primes}`
- `ℍ` = Quaternion algebra

This enables:
1. **Compositional semantics** via prime multiplication: `|AB⟩ ~ |p_A · p_B⟩`
2. **Rotation geometry** via quaternion operations
3. **Interference patterns** for semantic similarity

### Kuramoto Dynamics

The Kuramoto model describes phase synchronization:

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

The order parameter:
```
r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)
```

- `r → 1`: Full synchronization (coherent state)
- `r → 0`: Incoherence (random phases)

We use `r` as a measure of representational coherence.

### Lyapunov Stability

We estimate the Lyapunov exponent from entropy dynamics:

```
λ ≈ d(entropy)/dt
```

- `λ < 0`: Converging (stable)
- `λ > 0`: Diverging (unstable, potential hallucination)

This enables real-time stability monitoring during generation.

## Performance

Fusion layers add modest overhead:

| Configuration | Parameters | Forward Time | Memory |
|---------------|------------|--------------|--------|
| Minimal       | ~100K      | +5%          | +2%    |
| Standard      | ~500K      | +15%         | +5%    |
| Full          | ~1M        | +25%         | +8%    |

Training only fusion layers is efficient since base model is frozen.

## Files

```
apps/llm_fusion/
├── __init__.py           # Module exports
├── config.py             # Configuration classes
├── prime_embeddings.py   # Prime projection layers
├── quaternion_layers.py  # Quaternion operations
├── kuramoto_attention.py # Kuramoto synchronization
├── fusion_layers.py      # Main fusion layers
├── wrappers.py           # Model wrappers
├── inference.py          # Generation utilities
├── train.py              # Training utilities
└── README.md             # This file
```

## References

- TinyAleph framework for resonance dynamics
- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators
- Hamilton, W.R. (1843). On quaternions
- Coherence gating inspired by mixture-of-experts routing

## License

Part of the AlephPrime project. See repository license.
