# Reso-LLM v2: Enhanced Resonant Large Language Model

Reso-LLM v2 is a next-generation language model architecture built on **TinyAleph** principles. It extends the original resonant transformer with advanced physics-based features for self-directed reasoning, compositional semantics, and multi-agent coordination.

**Default Training Dataset:** [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

## What's New in v2

| Feature | Description | Benefits |
|---------|-------------|----------|
| **Agency Layer** | Self-directed attention and goal formation | Model can set and track goals during generation |
| **PRSC** | Prime Resonance Semantic Coherence | Compositional semantics through prime interference |
| **Temporal SMF** | Episodic memory with decay | Better long-term context with natural forgetting |
| **Predictive Stability** | Lyapunov-based hallucination prevention | Early warning of instability, auto-correction |
| **Multi-Agent** | Entanglement network coordination | Multiple models can share state non-locally |
| **Stochastic Resonance** | Controlled noise injection | Escape repetitive patterns |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Tokens                              │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Prime Token Embedding (vocab_size → dim)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  PRSC Semantic Layer (compositional bindings)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Kuramoto Synchronization (dynamic attention modulation)         │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  ResoFormer Blocks × N                                           │
│  ├─ Resonant Attention (φ-weighted)                              │
│  ├─ Agency Attention Modulation (goal-guided)                    │
│  ├─ Coherence Gating (adaptive computation)                      │
│  └─ FFN with Stochastic Resonance                                │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Temporal Sedenion Memory Field                                  │
│  (16D holographic memory with episodic tagging)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Predictive Stability Monitor                                    │
│  (Lyapunov analysis, temperature adjustment)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Output Logits + State Metadata                                  │
│  (stability, coherence, goals, warnings)                         │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from apps.reso_llm.config import ResoLLMConfig, small_config
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.inference import ResoLLMInference

# Initialize with small config for testing
config = small_config()
tokenizer = create_default_tokenizer()
model = ResoLLMModel(config)
engine = ResoLLMInference(model, tokenizer, config)

# Generate text
result = engine.generate("Once upon a time", max_length=100)
print(result.text)
print(f"Stability: {result.stability}")
print(f"Lyapunov: {result.lyapunov:.4f}")
```

### Goal-Guided Generation

```python
# Generate with an explicit goal
result = engine.generate_with_goal(
    prompt="Explain quantum computing",
    goal_description="Provide a clear, beginner-friendly explanation",
    max_length=200
)

print(result.text)
print(f"Goals achieved: {result.goals_achieved}")
```

### Interactive Chat

```python
# Start interactive session with metrics display
engine.interactive_session(show_metrics=True)
```

Commands in interactive mode:
- `exit` - End session
- `goal <description>` - Set goal for next response
- `status` - Show model status
- `metrics` - Toggle metrics display

### Multi-Agent Coordination

```python
from apps.reso_llm.config import multi_agent_config

# Create multiple agents
config_a = multi_agent_config("agent_alpha")
config_b = multi_agent_config("agent_beta")

model_a = ResoLLMModel(config_a)
model_b = ResoLLMModel(config_b)

engine_a = ResoLLMInference(model_a, tokenizer, config_a)
engine_b = ResoLLMInference(model_b, tokenizer, config_b)

# Coordinate generation
results = engine_a.generate_multi_agent(
    prompt="Write a story about AI",
    other_agents=[engine_b],
    coordination_strategy="consensus"  # or "parallel", "sequential"
)

for i, result in enumerate(results):
    print(f"Agent {i}: {result.text[:100]}...")
```

## Configuration

### Preset Configurations

```python
from apps.reso_llm.config import small_config, medium_config, large_config

# For development/testing
config = small_config()  # 4 layers, 256 dim

# For production
config = medium_config()  # 8 layers, 512 dim

# For maximum capability
config = large_config()  # 24 layers, 1024 dim
```

### Component Configuration

Each enhanced feature has its own configuration dataclass:

```python
from apps.reso_llm.config import (
    ResoLLMConfig,
    AgencyConfig,
    PRSCConfig,
    TemporalSMFConfig,
    StabilityConfig,
    EntanglementConfig,
    StochasticResonanceConfig,
)

config = ResoLLMConfig(
    dim=512,
    num_layers=8,
    
    # Agency Layer
    agency=AgencyConfig(
        enabled=True,
        max_foci=5,
        max_goals=10,
        novelty_weight=0.4,
    ),
    
    # PRSC Semantics
    prsc=PRSCConfig(
        enabled=True,
        coherence_threshold=0.7,
        max_bindings=1000,
    ),
    
    # Temporal Memory
    temporal_smf=TemporalSMFConfig(
        enabled=True,
        memory_decay_rate=0.01,
        max_moments=1000,
        episodic_tagging=True,
    ),
    
    # Stability Monitoring
    stability=StabilityConfig(
        enabled=True,
        lyapunov_threshold=0.1,
        auto_temperature_adjust=True,
        predictive_horizon=5,
    ),
    
    # Stochastic Resonance (disabled by default)
    stochastic_resonance=StochasticResonanceConfig(
        enabled=False,
        noise_amplitude=0.1,
    ),
)
```

## Enhanced Features

### 1. Agency Layer

The Agency Layer enables self-directed reasoning:

```python
# Create a goal programmatically
goal = model.create_goal(
    description="Generate coherent technical documentation",
    goal_type="exploratory"
)

# Check attention foci
for focus in model.get_attention_foci():
    print(f"{focus.type}:{focus.target} - intensity: {focus.intensity:.2f}")

# Check goal progress
for goal in model.get_active_goals():
    print(f"{goal.description}: {goal.progress:.0%}")
```

### 2. PRSC Semantic Bindings

Bind concepts to prime states for compositional semantics:

```python
# Bind a semantic concept
binding = model.bind_concept("mathematics", token_id=42)

# The PRSC layer enables compositional operations
# e.g., "mathematics" + "logic" → interference pattern
```

### 3. Predictive Stability

The stability monitor provides early warning of instability:

```python
# Get current stability (0-1 scale)
stability = model.get_stability()

# Get suggested temperature based on analysis
suggested_temp = model.get_suggested_temperature(current_temp=0.7)

# Full stability data in generation result
result = engine.generate("...")
print(result.stability)  # "stable", "periodic", "chaotic", "divergent"
print(result.lyapunov)   # Lyapunov exponent
print(result.warnings)   # Any stability warnings
```

### 4. Memory Management

```python
# Update holographic memory
model.update_memory("Important context to remember", importance=1.0)

# Memory automatically decays over time
# Episodic tagging enables "when did I learn this?"
```

### 5. Generation Result

Every generation returns comprehensive metrics:

```python
result = engine.generate("Hello")

# Core output
result.text            # Generated text
result.tokens_generated # Number of tokens

# Stability metrics
result.stability       # Stability class
result.lyapunov        # Lyapunov exponent
result.entropy_trace   # Entropy over time
result.lyapunov_trace  # Lyapunov over time

# Adaptive control
result.temperature_trace  # Temperature adjustments
result.coherence_trace    # Coherence over time

# Agency state
result.goals_achieved     # Completed goals
result.attention_summary  # What model attended to

# Diagnostics
result.warnings           # Any warnings generated
result.generation_time_ms # Generation time
```

## Training

Training works the same as v1, with additional metrics:

```bash
python apps/reso_llm/train.py
```

The enhanced model is fully compatible with PyTorch autograd.

## Running the Demo

```bash
python apps/reso_llm/demo.py
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Configuration dataclasses for all features |
| `model.py` | Enhanced ResoLLMModel with all integrations |
| `inference.py` | Enhanced inference engine with full monitoring |
| `tokenizer.py` | Prime-based tokenization |
| `train.py` | Training loop |
| `evaluate.py` | Evaluation utilities |
| `chat.py` | Chat interface |
| `demo.py` | Demonstration script |

## Key Concepts

### Prime Hilbert Space
Tokens map to primes, enabling unique factorization and compositional semantics.

### Sedenion Memory (16D)
Memory is stored as 16-dimensional sedenions with semantic axes:
- coherence, identity, duality, harmony
- truth, consciousness, beauty, compassion
- intention, wisdom, creativity, stability
- complexity, connection, presence, transcendence

### Kuramoto Synchronization
Phase oscillators model dynamic attention where relevant tokens synchronize.

### Lyapunov Stability
Chaos theory measures when generation is becoming unstable (hallucinating).

### Golden Ratio (φ ≈ 1.618)
Appears in attention weights, learning rates, thresholds, and gradient clipping.

## Performance Notes

- All enhanced features are optional and configurable
- Agency and PRSC add ~10% overhead during inference
- Stability monitoring is lightweight (moving window)
- Entanglement is disabled by default (for single-agent use)
- Stochastic resonance only activates when repetition detected

## References

- TinyAleph: Prime Hilbert Space formalism
- Kuramoto Model: Phase synchronization
- Adaptive Computation Time (ACT): Halting mechanism
- Lyapunov Exponents: Chaos detection
- Sedenions: 16-dimensional hypercomplex numbers
