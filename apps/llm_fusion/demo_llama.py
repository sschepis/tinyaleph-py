#!/usr/bin/env python3
"""
Demo: Resonance Fusion with TinyLlama.

This script demonstrates:
1. Loading a HuggingFace Llama-style model
2. Wrapping it with resonance fusion layers
3. Training the fusion layers on a small dataset
4. Generating text with stability monitoring
5. Comparing base vs fused model outputs

Usage:
    python -m apps.llm_fusion.demo_llama
"""
import os
import sys
import argparse
from pathlib import Path

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


def load_model_and_tokenizer(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load model and tokenizer from HuggingFace."""
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    return model, tokenizer


def wrap_with_fusion(model, tokenizer, num_layers: int = None):
    """Wrap model with resonance fusion layers."""
    print("\nConfiguring resonance fusion...")
    
    # Detect number of layers
    if num_layers is None:
        if hasattr(model.config, "num_hidden_layers"):
            num_layers = model.config.num_hidden_layers
        else:
            num_layers = 22  # TinyLlama default
    
    # Configure fusion for Llama architecture
    config = FusionConfig(
        hidden_dim=model.config.hidden_size,
        fusion_positions=[
            num_layers // 4,      # 25% depth
            num_layers // 2,      # 50% depth  
            3 * num_layers // 4,  # 75% depth
            num_layers - 1,       # Final layer
        ],
        fusion_mode="residual",
        fusion_alpha=0.1,
        learnable_alpha=True,
        enabled_components=["prime", "quaternion", "kuramoto", "coherence_gate"],
    )
    
    print(f"  Model hidden dim: {config.hidden_dim}")
    print(f"  Model layers: {num_layers}")
    print(f"  Fusion at layers: {config.fusion_positions}")
    print(f"  Components: {config.enabled_components}")
    
    # Wrap model
    wrapped = ResonanceWrapper(model, config, freeze_base=True)
    
    print(f"\n  Base parameters: {wrapped.num_base_parameters():,}")
    print(f"  Fusion parameters: {wrapped.num_fusion_parameters():,}")
    print(f"  Overhead: {100 * wrapped.num_fusion_parameters() / wrapped.num_base_parameters():.2f}%")
    
    return wrapped


def generate_comparison(
    base_model,
    fused_model,
    tokenizer,
    prompts: list,
    max_length: int = 100,
):
    """Compare generation from base and fused models."""
    print("\n" + "="*60)
    print("GENERATION COMPARISON")
    print("="*60)
    
    gen_config = GenerationConfig(
        temperature=0.7,
        max_length=max_length,
        top_k=50,
        top_p=0.9,
        auto_temperature=True,
    )
    
    # Create generator for fused model
    fused_generator = ResonanceGenerator(fused_model, tokenizer, gen_config)
    
    for prompt in prompts:
        print(f"\n{'─'*60}")
        print(f"PROMPT: {prompt}")
        print('─'*60)
        
        # Base model generation
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(base_model, "device"):
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            base_output = base_model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
        print(f"\n[BASE MODEL]:")
        print(base_text[len(prompt):])
        
        # Fused model generation with stability monitoring
        result = fused_generator.generate(prompt, max_length=max_length)
        
        print(f"\n[FUSED MODEL]:")
        print(result.generated_text[len(prompt):])
        
        # Show stability metrics
        if result.stability_metrics:
            print(f"\n[STABILITY METRICS]:")
            print(f"  Mean Coherence: {result.stability_metrics.mean_coherence:.4f}")
            print(f"  Mean Entropy: {result.stability_metrics.mean_entropy:.4f}")
            print(f"  Lyapunov Estimate: {result.stability_metrics.lyapunov_estimate:.4f}")
            print(f"  Kuramoto Order: {result.stability_metrics.kuramoto_order:.4f}")
            print(f"  Stable: {result.stability_metrics.is_stable}")
        
        # Show layer metrics
        metrics = fused_model.get_average_metrics()
        if metrics:
            print(f"\n[FUSION LAYER METRICS]:")
            for k, v in sorted(metrics.items()):
                print(f"  {k}: {v:.4f}")


def quick_train_fusion(
    model,
    tokenizer,
    training_texts: list = None,
    steps: int = 100,
    learning_rate: float = 1e-4,
):
    """Quick training of fusion layers."""
    print("\n" + "="*60)
    print("TRAINING FUSION LAYERS")
    print("="*60)
    
    # Default training data
    if training_texts is None:
        training_texts = [
            "The quantum nature of consciousness suggests that awareness arises from coherent oscillations in neural microtubules.",
            "Prime numbers form the basis of cryptographic systems because their distribution follows patterns that are computationally hard to predict.",
            "Synchronization phenomena in coupled oscillators demonstrate how order can emerge from local interactions without central control.",
            "The golden ratio appears in nature because it represents an optimal packing strategy for biological systems.",
            "Semantic meaning arises from the compositional structure of concepts, where primitives combine through systematic operations.",
            "Coherence in physical systems corresponds to low entropy configurations that persist under perturbation.",
            "The relationship between observer and observed is fundamental to understanding measurement in quantum mechanics.",
            "Language models learn to predict patterns by building internal representations of semantic relationships.",
            "Stability analysis using Lyapunov methods can detect when dynamical systems are approaching chaotic regimes.",
            "Quaternion rotations provide a singularity-free representation of 3D orientation useful for graphics and robotics.",
        ]
    
    print(f"  Training samples: {len(training_texts)}")
    print(f"  Steps: {steps}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create dataset
    dataset = create_fusion_dataset(training_texts, tokenizer, max_length=256)
    
    # Configure training
    config = TrainingConfig(
        learning_rate=learning_rate,
        max_steps=steps,
        warmup_steps=10,
        log_interval=10,
        eval_interval=steps + 1,  # No eval during quick train
        save_interval=steps + 1,  # No save during quick train
        coherence_loss_weight=0.1,
        kuramoto_loss_weight=0.05,
        entropy_loss_weight=0.05,
        output_dir="runs/fusion_demo",
    )
    
    # Train
    trainer = FusionTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        config=config,
    )
    
    results = trainer.train()
    
    print(f"\n  Final step: {results['final_step']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Resonance Fusion Demo with TinyLlama")
    parser.add_argument(
        "--model", 
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=50,
        help="Number of training steps"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=80,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (test with random weights)"
    )
    args = parser.parse_args()
    
    # Load model
    base_model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Wrap with fusion
    fused_model = wrap_with_fusion(base_model, tokenizer)
    
    # Train fusion layers
    if not args.skip_train:
        quick_train_fusion(
            fused_model,
            tokenizer,
            steps=args.train_steps,
        )
    
    # Test prompts
    prompts = [
        "Explain the relationship between consciousness and",
        "The fundamental nature of reality is",
        "In the future, artificial intelligence will",
    ]
    
    # Generate and compare
    generate_comparison(
        base_model,
        fused_model,
        tokenizer,
        prompts,
        max_length=args.max_length,
    )
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nThe fused model adds resonance-aware processing that:")
    print("  1. Projects to prime Hilbert space for semantic coherence")
    print("  2. Applies quaternion rotations for geometric transformation")
    print("  3. Uses Kuramoto dynamics for synchronization")
    print("  4. Gates output based on coherence for stability")
    print("\nWith more training data and steps, the fusion layers learn to")
    print("enhance the base model's representations while maintaining stability.")


if __name__ == "__main__":
    main()
