"""
Enhanced ResoFormer Application

A complete, trainable transformer built on TinyAleph's prime-resonant foundations.

Features:
- PrimeTokenizer: Text-to-prime mapping with prime-indexed vocabulary
- TrainableResoFormer: Enhanced model with numerical gradient support
- ResonantTrainer: Training loop with coherence monitoring
- ResonantGenerator: Text generation with entropy-aware sampling

Key Concepts:
- Resonant Attention: φ-weighted multi-head attention with prime resonance
- Coherence Gating: ACT-style halting when coherence threshold reached
- Entropy Collapse: VQ-style quantization to 64 discrete attractors
- Golden Ratio: φ-based learning rates, temperature scheduling, phase spacing

Usage:
    from apps.resoformer import (
        PrimeTokenizer,
        TrainableResoFormer,
        ResonantTrainer,
        ResonantGenerator,
    )
    
    # Tokenize
    tokenizer = PrimeTokenizer.from_corpus(texts)
    
    # Create model
    model = TrainableResoFormer(config)
    
    # Train
    trainer = ResonantTrainer(model, tokenizer)
    trainer.train(corpus, epochs=10)
    
    # Generate
    generator = ResonantGenerator(model, tokenizer)
    text = generator.generate("The ", max_length=100)
"""

from apps.resoformer.tokenizer import PrimeTokenizer
from apps.resoformer.model import (
    TrainableResoFormer,
    TrainableResoFormerConfig,
    TrainableTensor,
)
from apps.resoformer.trainer import ResonantTrainer, TrainingConfig
from apps.resoformer.generator import ResonantGenerator, GenerationConfig
from apps.resoformer.demo import run_demo

__all__ = [
    "PrimeTokenizer",
    "TrainableResoFormer",
    "TrainableResoFormerConfig",
    "TrainableTensor",
    "ResonantTrainer",
    "TrainingConfig",
    "ResonantGenerator",
    "GenerationConfig",
    "run_demo",
]