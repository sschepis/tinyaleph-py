#!/usr/bin/env python3
"""
Train Resonance Fusion Layers on a Base LLM.

This script demonstrates how to:
1. Load a HuggingFace model (TinyLlama, GPT-2, Llama, etc.)
2. Wrap it with resonance fusion layers
3. Train ONLY the fusion layers (base model stays frozen)
4. Save and load trained fusion weights
5. Generate text with stability monitoring

Usage:
    # Basic training with TinyLlama
    python -m apps.llm_fusion.train_fusion --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --steps 1000

    # Training with custom dataset
    python -m apps.llm_fusion.train_fusion --model gpt2-medium --data data/train.txt --steps 2000

    # Resume training
    python -m apps.llm_fusion.train_fusion --resume runs/fusion_train/checkpoint_final

Example Workflow:
    1. First run: Train fusion layers
       python -m apps.llm_fusion.train_fusion --steps 5000 --output runs/my_fusion

    2. Test generation:
       python -m apps.llm_fusion.train_fusion --eval-only --checkpoint runs/my_fusion/checkpoint_final

    3. Fine-tune more:
       python -m apps.llm_fusion.train_fusion --resume runs/my_fusion/checkpoint_final --steps 2000
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from apps.llm_fusion import (
    ResonanceWrapper,
    FusionConfig,
    ResonanceGenerator,
    GenerationConfig,
    TrainingConfig,
    FusionTrainer,
    create_fusion_dataset,
)


def load_training_texts(data_path: Optional[str] = None) -> List[str]:
    """
    Load training texts from file or return default dataset.
    
    For best results, provide domain-specific texts that demonstrate
    the coherent, structured writing style you want the model to learn.
    """
    if data_path and Path(data_path).exists():
        with open(data_path, 'r') as f:
            text = f.read()
        # Split by double newlines or paragraphs
        texts = [t.strip() for t in text.split('\n\n') if len(t.strip()) > 50]
        print(f"Loaded {len(texts)} texts from {data_path}")
        return texts
    
    # Default: coherent texts about consciousness, physics, math
    return [
        "The relationship between prime numbers and consciousness reveals deep patterns in the structure of awareness. Each prime represents an indivisible unit of mathematical truth.",
        "Quantum coherence in neural systems suggests that consciousness may arise from synchronized oscillatory dynamics across cortical networks.",
        "The golden ratio appears throughout nature because it represents an optimal balance between growth and stability in self-organizing systems.",
        "Semantic meaning emerges from compositional structure where primitive concepts combine through systematic operations to form complex thoughts.",
        "Kuramoto oscillators demonstrate how global order can emerge from local coupling without central coordination, a model for collective behavior.",
        "Quaternion rotations provide singularity-free representations that are essential for smooth orientation interpolation in continuous systems.",
        "The Lyapunov exponent measures the rate of separation of infinitesimally close trajectories, indicating whether a system tends toward chaos or stability.",
        "Prime factorization forms the mathematical backbone of modern cryptographic systems, relying on the difficulty of decomposing large numbers.",
        "Coherent states in physics correspond to configurations that maintain phase relationships over time and space, minimizing uncertainty.",
        "The observer effect in quantum mechanics shows that measurement fundamentally alters the system being observed, collapsing superpositions.",
        "Entropy in information theory quantifies uncertainty and is maximized for uniform probability distributions across possible states.",
        "Self-organizing systems exhibit emergent behavior that cannot be predicted from individual component properties alone.",
        "Synchronization phenomena arise in networks of coupled oscillators when the coupling strength exceeds a critical threshold.",
        "The holographic principle suggests that the information content of a volume is encoded on its boundary surface.",
        "Compositional semantics enables infinite expressivity from finite primitives through systematic combination rules.",
        "Phase transitions occur when small parameter changes cause qualitative shifts in system behavior and structure.",
        "The binding problem asks how distributed neural representations cohere into unified conscious experiences.",
        "Attractor dynamics in recurrent networks can implement associative memory and pattern completion.",
        "The free energy principle proposes that biological systems minimize surprise by predicting sensory inputs.",
        "Resonance occurs when a system's natural frequency matches an external driving frequency, amplifying oscillations.",
    ]


def get_model_config(model_name: str) -> FusionConfig:
    """Get appropriate FusionConfig for different model types."""
    model_name_lower = model_name.lower()
    
    if "gpt2" in model_name_lower:
        return FusionConfig.for_gpt2()
    elif "llama" in model_name_lower or "tinyllama" in model_name_lower:
        # TinyLlama has 22 layers
        return FusionConfig.for_llama(num_layers=22)
    elif "mistral" in model_name_lower:
        return FusionConfig.for_llama(num_layers=32)
    else:
        # Generic config
        return FusionConfig.standard()


def main():
    parser = argparse.ArgumentParser(
        description="Train resonance fusion layers on a pre-trained LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model arguments
    parser.add_argument(
        "--model", 
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to training data (text file, one paragraph per double-newline)"
    )
    
    # Training arguments
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    
    # Loss weights
    parser.add_argument("--coherence-weight", type=float, default=0.15, help="Coherence loss weight")
    parser.add_argument("--kuramoto-weight", type=float, default=0.05, help="Kuramoto loss weight")
    parser.add_argument("--entropy-weight", type=float, default=0.05, help="Entropy regularization weight")
    
    # Fusion configuration
    parser.add_argument("--fusion-layers", type=int, nargs="+", default=None, 
                        help="Layer indices for fusion (e.g., 5 11 16 21)")
    parser.add_argument("--fusion-alpha", type=float, default=0.1, help="Initial fusion weight")
    
    # Output
    parser.add_argument("--output", default="runs/fusion_train", help="Output directory")
    parser.add_argument("--log-interval", type=int, default=25, help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=500, help="Save every N steps")
    
    # Resume/eval
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Load checkpoint for eval")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, no training")
    
    # Hardware
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    base_model = base_model.to(device)
    
    # Get fusion config
    print("\nConfiguring fusion layers...")
    config = get_model_config(args.model)
    
    # Override fusion positions if specified
    if args.fusion_layers:
        config.fusion_positions = args.fusion_layers
    
    config.fusion_alpha = args.fusion_alpha
    
    print(f"  Hidden dim: {base_model.config.hidden_size}")
    print(f"  Num layers: {base_model.config.num_hidden_layers}")
    print(f"  Fusion at: {config.fusion_positions}")
    print(f"  Fusion alpha: {config.fusion_alpha}")
    
    # Wrap model
    model = ResonanceWrapper(base_model, config, freeze_base=True)
    
    print(f"\n  Base parameters: {model.num_base_parameters():,}")
    print(f"  Fusion parameters: {model.num_fusion_parameters():,}")
    print(f"  Overhead: {100 * model.num_fusion_parameters() / model.num_base_parameters():.2f}%")
    
    # Load checkpoint if specified
    if args.checkpoint or args.resume:
        checkpoint_path = args.checkpoint or args.resume
        fusion_weights = Path(checkpoint_path) / "fusion_weights.pt"
        if fusion_weights.exists():
            model.load_fusion_weights(str(fusion_weights))
            print(f"\nLoaded fusion weights from {fusion_weights}")
    
    # Eval only mode
    if args.eval_only:
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        evaluate_model(model, tokenizer, device)
        return
    
    # Load training data
    print("\nPreparing training data...")
    training_texts = load_training_texts(args.data)
    print(f"  {len(training_texts)} training samples")
    
    # Repeat if too few
    if len(training_texts) < 50:
        multiplier = max(1, 50 // len(training_texts))
        training_texts = training_texts * multiplier
        print(f"  Repeated to {len(training_texts)} samples")
    
    dataset = create_fusion_dataset(training_texts, tokenizer, max_length=args.max_length)
    
    # Training config
    train_config = TrainingConfig(
        learning_rate=args.lr,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.save_interval,
        coherence_loss_weight=args.coherence_weight,
        kuramoto_loss_weight=args.kuramoto_weight,
        entropy_loss_weight=args.entropy_weight,
        output_dir=args.output,
    )
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"  Steps: {args.steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {args.output}")
    print()
    
    trainer = FusionTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        config=train_config,
    )
    
    # Resume optimizer state if resuming
    if args.resume:
        trainer_state = Path(args.resume) / "trainer_state.pt"
        if trainer_state.exists():
            trainer.load_checkpoint(args.resume)
            print(f"Resumed from step {trainer.global_step}")
    
    results = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Final step: {results['final_step']}")
    print(f"  Checkpoints saved to: {args.output}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    evaluate_model(model, tokenizer, device)


def evaluate_model(model, tokenizer, device):
    """Generate sample outputs with stability monitoring."""
    gen_config = GenerationConfig(
        temperature=0.8,
        max_length=100,
        top_k=50,
        top_p=0.9,
        auto_temperature=True,
    )
    
    generator = ResonanceGenerator(model, tokenizer, gen_config)
    
    prompts = [
        "The Schrödinger equation describes",
        "Heisenberg uncertainty principle states that",
        "The Born rule connects",
        "The hydrogen atom energy levels",
    ]
    
    for prompt in prompts:
        print(f"\n{'─'*60}")
        print(f"PROMPT: {prompt}")
        print('─'*60)
        
        result = generator.generate(prompt, max_length=60)
        
        print(f"\nGENERATED:")
        print(result.generated_text)
        
        print(f"\nMETRICS:")
        print(f"  Coherence: {result.stability_metrics.mean_coherence:.3f}")
        print(f"  Kuramoto: {result.stability_metrics.kuramoto_order:.3f}")
        print(f"  Lyapunov: {result.stability_metrics.lyapunov_estimate:.4f}")
        print(f"  Stable: {result.stability_metrics.is_stable}")


if __name__ == "__main__":
    main()
