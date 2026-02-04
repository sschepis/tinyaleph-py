# LLM Fusion Reference Guide

LLM Fusion provides a framework for grafting physics-inspired resonance layers onto pre-trained transformer models (like GPT-2, Llama, BERT). This allows standard LLMs to benefit from Prime Hilbert Space semantics, Kuramoto synchronization, and stability monitoring without retraining the base model from scratch.

## Architecture

The core concept is the **Resonance Fusion Layer**, which intercepts hidden states from the base model, transforms them into a hypercomplex prime basis, applies resonance dynamics, and injects the result back into the stream (or gates it).

### Fusion Pipeline

1.  **Prime Projection**: Maps hidden states $h \in \mathbb{R}^d$ to a Prime Hilbert Space $H_P \otimes \mathbb{H}$ (quaternionic amplitudes over prime basis).
2.  **Quaternion Rotation**: Applies geometric transformations in 4D space, allowing for rotation-invariant semantic operations.
3.  **Kuramoto Synchronization**: Models attention as a system of coupled oscillators. Tokens that are semantically related synchronize their phases.
4.  **PRSC & SMF (Optional)**: Injects compositional semantics (PRSC) and retrieves episodic memories (SMF).
5.  **Coherence Gating**: Computes the entropy of the state. High-entropy (chaotic) states are gated/attenuated, while coherent states are passed through.

---

## Configuration

Configuration is managed by `FusionConfig` in `apps.llm_fusion.config`.

### Configuration Presets

```python
from apps.llm_fusion.config import FusionConfig

# Minimal: Single layer, low overhead
config = FusionConfig.minimal()

# Standard: 3 fusion layers (recommended)
config = FusionConfig.standard()

# Full: All features enabled (PRSC, SMF, etc.)
config = FusionConfig.full()

# Model-specific optimizations
config = FusionConfig.for_gpt2()
config = FusionConfig.for_llama(num_layers=32)
```

### Custom Configuration

```python
config = FusionConfig(
    hidden_dim=768,
    fusion_positions=[4, 8, 11],  # Inject at these layers
    fusion_mode="residual",       # "parallel", "sequential", or "residual"
    enabled_components=["prime", "quaternion", "kuramoto", "coherence_gate"]
)
```

---

## Python API

### ResonanceWrapper

The primary interface is `ResonanceWrapper`, which wraps a HuggingFace model.

```python
from transformers import AutoModelForCausalLM
from apps.llm_fusion import ResonanceWrapper, FusionConfig

# 1. Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Configure fusion
config = FusionConfig.for_gpt2()

# 3. Wrap model
model = ResonanceWrapper(base_model, config, freeze_base=True)
```

### Generation

Use the wrapped model just like a standard HuggingFace model, or use `ResonanceGenerator` for stability-aware generation.

```python
# Standard generation (hooks apply fusion automatically)
output = model.generate(input_ids, max_length=50)

# Stability-aware generation
from apps.llm_fusion import ResonanceGenerator, GenerationConfig

gen_config = GenerationConfig(stop_on_instability=True)
generator = ResonanceGenerator(model, tokenizer, gen_config)
result = generator.generate("The future of AI is")

print(f"Stability: {result.stability_metrics.is_stable}")
```

### Accessing Metrics

You can inspect the internal state of the fusion layers after a forward pass:

```python
# Run forward pass
outputs = model(input_ids)

# Get metrics
metrics = model.get_metrics()
for layer_idx, data in metrics.items():
    print(f"Layer {layer_idx}:")
    print(f"  Kuramoto Order: {data.get('kuramoto_order')}")
    print(f"  Coherence: {data.get('gate_coherence')}")
```

---

## Components Detail

### Prime Projection
*   **Input**: Hidden states $(B, S, D)$
*   **Operation**: Projects to $N$ prime basis vectors with quaternion coefficients.
*   **Output**: Prime states $(B, S, N, 4)$

### Kuramoto Synchronization
*   **Dynamics**: $\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_j \sin(\theta_j - \theta_i)$
*   **Effect**: Synchronizes the phases of related tokens. The "Order Parameter" $r$ ($0 \le r \le 1$) measures global coherence.

### Coherence Gating
*   **Logic**: $Gate = \sigma(10 \cdot (C - T))$ where $C$ is coherence and $T$ is threshold.
*   **Effect**: Filters out hallucinations (low coherence) before they propagate to the next layer.

---

## Training

You can train *only* the fusion layers while keeping the base model frozen.

```python
from apps.llm_fusion.train import FusionTrainer

trainer = FusionTrainer(model, dataset, tokenizer)
trainer.train()

# Save only the fusion weights (lightweight)
model.save_fusion_weights("fusion_adapter.pt")
```
