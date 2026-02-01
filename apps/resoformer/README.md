# Enhanced ResoFormer

A complete, trainable transformer built on TinyAleph's prime-resonant foundations.

## Overview

This application demonstrates all key TinyAleph ML concepts in a practical implementation:

- **Prime-indexed tokenization** - Each character/token maps to a unique prime number
- **Resonant attention** - Multi-head attention with golden ratio (φ) weighting
- **Coherence gating** - ACT-style adaptive computation based on coherence threshold
- **Entropy collapse** - VQ-style quantization to 64 discrete attractors
- **Numerical training** - Pure Python gradient computation for full trainability

## Architecture

```
Input Tokens
     ↓
┌─────────────────────────────┐
│   Prime Token Embedding     │  (vocab_size → dim)
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│   ResoFormer Block × N      │
│  ├─ LayerNorm               │
│  ├─ Resonant Attention      │  (φ-weighted multi-head)
│  ├─ Resonance Operator      │  (R̂(n)|p⟩ = e^(2πi log(n))|p⟩)
│  ├─ LayerNorm               │
│  ├─ FFN (GELU)              │
│  ├─ Coherence Gate          │  (σ(coherence - τ))
│  └─ [Entropy Collapse]      │  (last layer only)
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│   Final LayerNorm           │
│   Output Projection         │  (dim → vocab_size)
└─────────────────────────────┘
     ↓
Output Logits
```

## Components

### PrimeTokenizer (`tokenizer.py`)
Maps text to prime numbers:
- Special tokens: PAD=2, UNK=3, BOS=5, EOS=7
- Character-level or BPE-style tokenization
- Unique factorization enables semantic compositionality

### TrainableResoFormer (`model.py`)
Complete trainable model:
- `TrainableTensor` - Tensor with gradient tracking
- `TrainableLayer` - Base class with parameter management
- Layers: Embedding, Dense, LayerNorm, Attention, Coherence Gate, etc.
- Save/load functionality

### ResonantTrainer (`trainer.py`)
Training loop with:
- Numerical gradient computation (finite differences)
- Golden ratio learning rate scheduling: `lr = base × φ^(-step/warmup)`
- Gradient clipping at `1/φ ≈ 0.618`
- Coherence monitoring for early stopping

### ResonantGenerator (`generator.py`)
Text generation with multiple strategies:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- **Entropy-aware sampling** (adjusts temperature based on entropy)

## Usage

### Quick Demo

```bash
cd apps/resoformer
python demo.py
```

### Full Training

```bash
python demo.py --full --epochs 5 --dim 128 --layers 4
```

### Programmatic Usage

```python
from apps.resoformer import (
    PrimeTokenizer,
    TrainableResoFormer,
    TrainableResoFormerConfig,
    ResonantTrainer,
    TrainingConfig,
    ResonantGenerator,
    GenerationConfig,
)

# Tokenize
tokenizer = PrimeTokenizer.from_corpus(texts)

# Create model
config = TrainableResoFormerConfig(
    vocab_size=tokenizer.vocab_size,
    dim=128,
    num_layers=4,
    num_heads=8,
)
model = TrainableResoFormer(config)

# Train
trainer = ResonantTrainer(model, tokenizer)
trainer.train(corpus, epochs=10)

# Generate
generator = ResonantGenerator(model, tokenizer)
text = generator.generate("To be, or not to be", max_length=100)
```

## Key Concepts

### Prime Hilbert Space (H_P)
States are superpositions over prime basis:
```
|ψ⟩ = Σ_p α_p|p⟩  where p ∈ Primes
```

### Quaternionic Amplitudes (H_Q = H_P ⊗ ℍ)
Extended with 4D quaternion amplitudes for richer geometry.

### Golden Ratio (φ ≈ 1.618)
Appears throughout:
- Attention head weights: `w_h = 1/φ^h`
- Learning rate decay: `lr = base × φ^(-step)`
- Gradient clipping: `clip = 1/φ ≈ 0.618`
- Coherence threshold: `τ = 1/φ`

### Coherence and Entropy
- **Entropy**: `H = -Σ p log(p)` measures uncertainty
- **Coherence**: `C = 1 - H/H_max` inverse of normalized entropy
- High coherence → focused, stable states
- Low coherence → diffuse, uncertain states

### Resonance Operator
Phase rotation based on prime logarithm:
```
R̂(n)|p⟩ = e^(2πi log_p(n))|p⟩
```

## Mathematical Foundations

Based on TinyAleph's core theories:
1. **Fundamental Theorem of Arithmetic** - Unique prime factorization
2. **Kuramoto Synchronization** - Phase coherence via coupling
3. **Adaptive Computation Time** - Coherence-gated halting
4. **Vector Quantization** - Entropy collapse to attractors

## Files

| File | Lines | Description |
|------|-------|-------------|
| `tokenizer.py` | ~250 | Prime tokenization |
| `model.py` | ~850 | Trainable layers and model |
| `trainer.py` | ~450 | Training loop |
| `generator.py` | ~350 | Text generation |
| `demo.py` | ~400 | Complete demonstration |
| **Total** | **~2,300** | Complete implementation |

## Performance Notes

- Pure Python implementation (no NumPy/PyTorch required)
- Numerical gradients are slow but accurate
- For production use, consider porting to PyTorch/JAX
- Suitable for educational and research purposes

## References

- TinyAleph: Prime Hilbert Space formalism
- ResoLang: Resonant fragment protocols
- Kuramoto Model: Phase synchronization
- Adaptive Computation Time (ACT): Halting mechanism
- Vector Quantization: Discrete attractors