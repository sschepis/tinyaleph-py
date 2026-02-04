# Reso-LLM Reference Guide

Reso-LLM is a standalone resonant language model built on the TinyAleph framework. It integrates quantum-inspired physics principles directly into the transformer architecture to enable self-directed reasoning, compositional semantics, and stability monitoring.

## Architecture Overview

The model operates in two modes:
1.  **Standard Mode**: A clean, efficient transformer architecture compatible with models trained via `jupyter2.py`.
2.  **Extended Mode**: Enables advanced physics-based components for agency, semantics, and multi-agent coordination.

### Core Components

*   **ResoFormerBlock**: The fundamental building block, incorporating:
    *   **Resonant Attention**: Attention mechanism scaled by the Golden Ratio ($\phi$).
    *   **RoPE**: Rotary Positional Embeddings for relative position encoding.
    *   **Coherence Gating**: Optional gating mechanism to filter incoherent information flow.
*   **Prime Token Embedding**: Tokens are mapped to prime numbers, enabling unique factorization and compositional semantics.

### Extended Components (Optional)

*   **Agency Layer**: Manages attention foci and high-level goals. It modulates attention weights based on goal relevance and novelty.
*   **PRSC (Prime Resonance Semantic Coherence)**: A semantic layer that binds concepts to prime states. It allows for compositional semantics through prime interference patterns ($|AB\rangle \approx |p_A \cdot p_B\rangle$).
*   **Temporal SMF (Sedenion Memory Field)**: A 16-dimensional holographic memory system that stores episodic information with temporal decay.
*   **Predictive Stability Monitor**: Uses Lyapunov exponents to detect the onset of chaotic generation (hallucination) and adjust sampling temperature dynamically.
*   **Entanglement Network**: Facilitates state sharing between multiple model instances (agents) via simulated quantum entanglement.

---

## Configuration

Configuration is handled by the `ResoLLMConfig` class in `apps.reso_llm.config`.

### Standard Presets

```python
from apps.reso_llm.config import ResoLLMConfig

# Tiny model for testing (4 layers, 256 dim)
config = ResoLLMConfig.tiny()

# GPT-2 Small equivalent (12 layers, 768 dim)
config = ResoLLMConfig.small()

# GPT-2 Medium equivalent (24 layers, 1024 dim)
config = ResoLLMConfig.medium()
```

### Extended Configuration

To enable advanced features, use the `extended` factory or modify specific sub-configs:

```python
config = ResoLLMConfig.small(standard=False)
config.enable_extensions()

# Customize Agency
config.agency.enabled = True
config.agency.max_goals = 5

# Customize Stability
config.stability.enabled = True
config.stability.lyapunov_threshold = 0.15
```

---

## Python API

### Model Initialization

```python
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.config import ResoLLMConfig

config = ResoLLMConfig.small()
model = ResoLLMModel(config)
```

### Inference

The `ResoLLMInference` class provides a high-level interface for text generation and chat.

```python
from apps.reso_llm.inference import ResoLLMInference
from apps.reso_llm.tokenizer import create_default_tokenizer

tokenizer = create_default_tokenizer()
engine = ResoLLMInference(model, tokenizer)

# Simple generation
text = engine.generate_simple("Once upon a time", max_length=50)

# Chat with history
response = engine.chat("Hello!", history=[])
```

### Detailed Generation Results

In extended mode, `generate()` returns a `GenerationResult` object containing physics metrics:

```python
result = engine.generate("Analyze the system stability.", stop_on_instability=True)

print(f"Stability Class: {result.stability}")  # e.g., "stable", "chaotic"
print(f"Lyapunov Exponent: {result.lyapunov}")
print(f"Entropy Trace: {result.entropy_trace}")
```

### Agency Interaction

Interact with the model's agency layer (Extended Mode only):

```python
# Set a high-level goal
model.create_goal("Explain quantum mechanics simply")

# Check active goals
goals = model.get_active_goals()
for goal in goals:
    print(f"{goal.description}: {goal.progress}%")
```

---

## Dataset Handling

Reso-LLM uses a unified chat format:
```
<|user|>
...
<|endofuser|>
<|assistant|>
...
<|endofassistant|>
```

### Loading Datasets

Use `GuanacoDataset` for the recommended training data or `MultiDataset` to combine sources.

```python
from apps.reso_llm.dataset import GuanacoDataset, MultiDataset

# Load single dataset
ds = GuanacoDataset(tokenizer, seq_len=256)

# Combine multiple datasets
multi_ds = MultiDataset(
    dataset_names=["timdettmers/openassistant-guanaco", "databricks/databricks-dolly-15k"],
    tokenizer=tokenizer
)
```

---

## Training

Training is performed using the `ResoLLMTrainer` (not detailed here, see `train.py`). It supports standard PyTorch training loops with added hooks for resonance dynamics.

```bash
# Run standard training
python apps/reso_llm/train.py --config small --batch_size 32
```
