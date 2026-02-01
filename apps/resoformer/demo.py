#!/usr/bin/env python3
"""
ResoFormer Demo: Complete Training and Generation Example

Demonstrates the full pipeline:
1. Tokenization with prime-indexed vocabulary
2. Model initialization with resonance layers
3. Training with coherence monitoring
4. Text generation with entropy-aware sampling

This demo uses a small Shakespeare corpus to train a character-level
language model, showcasing all key TinyAleph features.
"""

from __future__ import annotations
import sys
import time
import math
sys.path.insert(0, '../..')

from apps.resoformer.tokenizer import PrimeTokenizer, create_shakespeare_tokenizer
from apps.resoformer.model import TrainableResoFormer, TrainableResoFormerConfig
from apps.resoformer.trainer import ResonantTrainer, TrainingConfig
from apps.resoformer.generator import ResonantGenerator, GenerationConfig

# Sample Shakespeare text for demo
SAMPLE_CORPUS = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest,
For Brutus is an honourable man;
So are they all, all honourable men,
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.
Grim-visaged war hath smooth'd his wrinkled front;
And now, instead of mounting barbed steeds
To fright the souls of fearful adversaries,
He capers nimbly in a lady's chamber
To the lascivious pleasing of a lute.
"""


def print_header(text: str):
    """Print formatted header."""
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)
    print()


def demonstrate_tokenizer():
    """Demonstrate PrimeTokenizer."""
    print_header("1. PRIME TOKENIZER")
    
    # Create tokenizer
    tokenizer = create_shakespeare_tokenizer()
    print(f"Created Shakespeare tokenizer:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Max prime: {tokenizer.max_prime}")
    print()
    
    # Show some mappings
    print("Sample token → prime mappings:")
    for char in "Hello, World!":
        if char in tokenizer.vocab:
            prime = tokenizer.vocab[char]
            print(f"  '{char}' → {prime}")
    print()
    
    # Encode/decode example
    text = "To be, or not to be"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Encoding example:")
    print(f"  Input: '{text}'")
    print(f"  Primes: {encoded[:10]}... ({len(encoded)} total)")
    print(f"  Decoded: '{decoded}'")
    print()
    
    return tokenizer


def demonstrate_model():
    """Demonstrate TrainableResoFormer model."""
    print_header("2. TRAINABLE RESOFORMER MODEL")
    
    # Create configuration
    config = TrainableResoFormerConfig(
        vocab_size=100,
        dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        max_seq_len=32,
        use_coherence_gate=True,
        use_entropy_collapse=True,
        use_golden_attention=True,
    )
    
    print("Model Configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  dim: {config.dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  ffn_dim: {config.ffn_dim}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  use_coherence_gate: {config.use_coherence_gate}")
    print(f"  use_entropy_collapse: {config.use_entropy_collapse}")
    print(f"  use_golden_attention: {config.use_golden_attention}")
    print()
    
    # Create model
    model = TrainableResoFormer(config)
    
    # Test forward pass
    test_tokens = list(range(16))
    start = time.time()
    logits = model.forward(test_tokens, training=False)
    elapsed = time.time() - start
    
    print(f"Forward pass test:")
    print(f"  Input: {len(test_tokens)} tokens")
    print(f"  Output: {len(logits)} logits")
    print(f"  Time: {elapsed*1000:.2f}ms")
    print(f"  Parameters: {model.num_parameters():,}")
    print()
    
    return model


def demonstrate_training(tokenizer: PrimeTokenizer):
    """Demonstrate training loop."""
    print_header("3. TRAINING WITH COHERENCE MONITORING")
    
    # Create smaller model for demo
    config = TrainableResoFormerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        max_seq_len=32,
        use_coherence_gate=True,
        use_entropy_collapse=False,  # Faster without collapse
        use_golden_attention=True,
    )
    
    model = TrainableResoFormer(config)
    
    # Build model with dummy forward pass to count parameters
    dummy_tokens = [2] * 8  # Use 8 PAD tokens
    _ = model.forward(dummy_tokens, training=False)
    print(f"Created model with {model.num_parameters():,} parameters")
    print()
    
    # Create training config
    train_config = TrainingConfig(
        epochs=1,
        seq_len=32,
        learning_rate=0.01,
        lr_warmup_steps=10,
        lr_decay="golden",
        gradient_clip=0.618,  # 1/φ
        log_interval=5,
        monitor_coherence=True,
    )
    
    print("Training Configuration:")
    print(f"  epochs: {train_config.epochs}")
    print(f"  seq_len: {train_config.seq_len}")
    print(f"  learning_rate: {train_config.learning_rate}")
    print(f"  lr_decay: {train_config.lr_decay} (Golden ratio)")
    print(f"  gradient_clip: {train_config.gradient_clip} (1/φ)")
    print()
    
    # Create trainer
    trainer = ResonantTrainer(model, tokenizer, train_config)
    
    # Use shorter corpus for demo - just demonstrate one training step
    short_corpus = SAMPLE_CORPUS[:300]
    
    print("Starting training (quick demo - 3 steps)...")
    print("-" * 40)
    
    # Manually run a few training steps for quick demo
    batches = trainer.prepare_batches(short_corpus)
    
    if len(batches) > 0:
        start = time.time()
        for i, (inp, tgt) in enumerate(batches[:3]):  # Just 3 steps
            print(f"  Step {i+1}/3: computing gradients...", end=" ", flush=True)
            step_start = time.time()
            logits = model.forward(inp, training=True)
            loss = trainer.cross_entropy_loss(logits, tgt)
            coherence = trainer.compute_coherence(logits)
            step_time = time.time() - step_start
            print(f"loss={loss:.4f}, coherence={coherence:.4f} ({step_time:.2f}s)")
        elapsed = time.time() - start
        print("-" * 40)
        print(f"Demo completed in {elapsed:.2f}s")
    else:
        print("No training batches created")
    
    print()
    return model, trainer


def demonstrate_generation(model: TrainableResoFormer, tokenizer: PrimeTokenizer):
    """Demonstrate text generation."""
    print_header("4. ENTROPY-AWARE GENERATION")
    
    # Create generation config
    gen_config = GenerationConfig(
        max_length=50,
        min_length=10,
        strategy="entropy_aware",
        temperature=1.0,
        temperature_schedule="golden_decay",
        top_p=0.9,
        repetition_penalty=1.1,
    )
    
    print("Generation Configuration:")
    print(f"  strategy: {gen_config.strategy}")
    print(f"  temperature: {gen_config.temperature}")
    print(f"  temperature_schedule: {gen_config.temperature_schedule}")
    print(f"  top_p: {gen_config.top_p}")
    print(f"  repetition_penalty: {gen_config.repetition_penalty}")
    print()
    
    # Create generator
    generator = ResonantGenerator(model, tokenizer, gen_config)
    
    # Generate with different prompts
    prompts = ["To be", "All the", "Friends"]
    
    print("Generated samples:")
    print("-" * 40)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        start = time.time()
        generated = generator.generate(prompt, max_length=30)
        elapsed = time.time() - start
        
        print(f"Output: '{generated}'")
        print(f"Time: {elapsed:.2f}s")
    
    print()
    return generator


def demonstrate_sampling_strategies(model: TrainableResoFormer, 
                                   tokenizer: PrimeTokenizer):
    """Compare different sampling strategies."""
    print_header("5. SAMPLING STRATEGY COMPARISON")
    
    strategies = ["greedy", "temperature", "top_k", "top_p", "entropy_aware"]
    prompt = "To be"
    
    print(f"Prompt: '{prompt}'")
    print("-" * 40)
    
    for strategy in strategies:
        config = GenerationConfig(
            max_length=20,
            min_length=5,
            strategy=strategy,
            temperature=1.0,
            top_k=10,
            top_p=0.9,
            seed=42,  # For reproducibility
        )
        
        generator = ResonantGenerator(model, tokenizer, config)
        generated = generator.generate(prompt)
        
        print(f"{strategy:15s}: '{generated}'")
    
    print()


def demonstrate_coherence_analysis(model: TrainableResoFormer,
                                   tokenizer: PrimeTokenizer):
    """Analyze coherence during generation."""
    print_header("6. COHERENCE ANALYSIS")
    
    config = GenerationConfig(
        max_length=30,
        strategy="entropy_aware",
        seed=42,
    )
    
    generator = ResonantGenerator(model, tokenizer, config)
    
    coherence_log = []
    
    def track_coherence(text: str, step: int, coherence: float):
        coherence_log.append((step, coherence))
    
    prompt = "All the world"
    generated = generator.generate(prompt, callback=track_coherence)
    
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated}'")
    print()
    print("Coherence over generation:")
    print("-" * 40)
    
    for step, coh in coherence_log[:10]:
        bar = '█' * int(coh * 30)
        print(f"Step {step:3d}: {coh:.4f} {bar}")
    
    if len(coherence_log) > 10:
        print(f"... ({len(coherence_log) - 10} more steps)")
    
    avg_coherence = sum(c for _, c in coherence_log) / len(coherence_log) if coherence_log else 0
    print(f"\nAverage coherence: {avg_coherence:.4f}")
    print()


def run_demo():
    """Run complete demonstration."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " ENHANCED RESOFORMER DEMONSTRATION ".center(58) + "║")
    print("║" + " Prime-Resonant Transformer with Training ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("This demo showcases TinyAleph's ML capabilities:")
    print("• Prime-indexed tokenization")
    print("• Resonant attention with golden ratio weighting")
    print("• Coherence gating and entropy collapse")
    print("• Numerical gradient training")
    print("• Entropy-aware text generation")
    print()
    
    # Run demonstrations
    tokenizer = demonstrate_tokenizer()
    demonstrate_model()
    model, trainer = demonstrate_training(tokenizer)
    generator = demonstrate_generation(model, tokenizer)
    demonstrate_sampling_strategies(model, tokenizer)
    demonstrate_coherence_analysis(model, tokenizer)
    
    print_header("DEMO COMPLETE")
    print("Key observations:")
    print("• Primes provide unique token indices via fundamental theorem of arithmetic")
    print("• Golden ratio (φ ≈ 1.618) appears in learning rate decay and attention weights")
    print("• Coherence monitoring enables adaptive computation")
    print("• Entropy-aware sampling balances exploration and exploitation")
    print()
    print("For full training, use larger models and more data.")
    print("See apps/resoformer/ for complete implementation.")
    print()


def run_full_training(epochs: int = 3, 
                      dim: int = 64,
                      num_layers: int = 2,
                      save_path: Optional[str] = None):
    """
    Run full training on Shakespeare corpus.
    
    Args:
        epochs: Number of training epochs
        dim: Model dimension
        num_layers: Number of transformer layers
        save_path: Optional path to save model
    """
    print_header("FULL TRAINING RUN")
    
    # Create tokenizer from corpus
    tokenizer = PrimeTokenizer.from_corpus([SAMPLE_CORPUS])
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size} tokens")
    
    # Create model
    config = TrainableResoFormerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=4,
        ffn_dim=dim * 4,
        max_seq_len=64,
        use_coherence_gate=True,
        use_entropy_collapse=(num_layers > 1),
        use_golden_attention=True,
    )
    
    model = TrainableResoFormer(config)
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Training config
    train_config = TrainingConfig(
        epochs=epochs,
        seq_len=64,
        learning_rate=0.005,
        lr_warmup_steps=50,
        lr_decay="golden",
        gradient_clip=0.618,
        log_interval=20,
        checkpoint_path=save_path,
    )
    
    # Train
    trainer = ResonantTrainer(model, tokenizer, train_config)
    trainer.train(SAMPLE_CORPUS)
    
    # Generate samples
    print_header("GENERATION SAMPLES")
    
    gen_config = GenerationConfig(
        max_length=100,
        strategy="entropy_aware",
        temperature=0.8,
        temperature_schedule="golden_decay",
    )
    
    generator = ResonantGenerator(model, tokenizer, gen_config)
    
    prompts = ["To be, or not", "All the world", "Friends, Romans"]
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        generated = generator.generate(prompt)
        print(f"Output: '{generated}'")
        print()
    
    return model, tokenizer


# Import for optional
from typing import Optional


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ResoFormer Demo")
    parser.add_argument("--full", action="store_true", help="Run full training")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--save", type=str, default=None, help="Save path")
    
    args = parser.parse_args()
    
    if args.full:
        run_full_training(
            epochs=args.epochs,
            dim=args.dim,
            num_layers=args.layers,
            save_path=args.save,
        )
    else:
        run_demo()