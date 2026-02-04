"""
Benchmark script to compare Reso-LLM against a baseline equivalent model.

This script:
1. Sets up a 'Reso-LLM' (standard mode with coherence gating and Kuramoto dynamics).
2. Sets up a 'Baseline' model (standard mode but with resonance features disabled).
3. Trains both on a 'Noisy Pattern' synthetic dataset designed to test coherence.
4. Compares training speed and final loss.
"""
import sys
import os
import time
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import asdict

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from apps.reso_llm.config import ResoLLMConfig
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.dataset import TextDataset
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.train import ResoLLMTrainer

def generate_noisy_data(path, tokenizer, num_examples=500, seq_len=128):
    """
    Generates data with a hidden structured pattern amidst random noise.
    Reso-LLM's coherence gating should theoretically help filter the noise.
    """
    print(f"Generating noisy pattern data to {path}...")
    
    # Pattern: "The secret code is Alpha Omega."
    pattern = "The secret code is Alpha Omega. "
    pattern_tokens = tokenizer.encode(pattern)
    
    # Vocabulary range (approximate for GPT-2)
    vocab_size = tokenizer.vocab_size
    
    all_tokens = []
    
    for _ in range(num_examples):
        # Create a sequence of random noise
        seq = [random.randint(0, vocab_size-1) for _ in range(seq_len)]
        
        # Inject pattern at random position
        if len(pattern_tokens) < seq_len:
            insert_pos = random.randint(0, seq_len - len(pattern_tokens))
            for i, token in enumerate(pattern_tokens):
                seq[insert_pos + i] = token
        
        all_tokens.extend(seq)
        
    # Write to file (TextDataset expects a file, though it loads into memory)
    # We can't easily write raw tokens to text file that TextDataset reads back perfectly 
    # without decoding. So we'll just return the tokens and patch the dataset.
    return all_tokens

def run_benchmark():
    print("=" * 60)
    print("Reso-LLM vs Baseline Benchmark (Noisy Pattern Task)")
    print("=" * 60)
    print("Task: Find the pattern 'The secret code is Alpha Omega' hidden in random noise.")

    # 1. Setup Data
    tokenizer = create_default_tokenizer()
    
    # Generate data directly
    data_tokens = generate_noisy_data("benchmark_noisy.txt", tokenizer)
    
    # Create dataset wrapper
    # We pass a dummy path but manually inject data
    dataset = TextDataset("benchmark_dummy_data.txt", tokenizer, seq_len=128, batch_size=16)
    dataset.data = data_tokens # Override with our noisy data
    print(f"Dataset size: {len(dataset.data)} tokens")
    
    # 2. Configure Models
    # We use 'tiny' size for speed
    
    # Model A: Reso-LLM (Standard features enabled)
    reso_config = ResoLLMConfig.tiny(standard=True)
    reso_config.use_coherence_gating = True
    reso_config.use_kuramoto_dynamics = True
    reso_config.use_resonance = True
    
    # Model B: Baseline (Standard features disabled -> Vanilla Transformer)
    baseline_config = ResoLLMConfig.tiny(standard=True)
    baseline_config.use_coherence_gating = False
    baseline_config.use_kuramoto_dynamics = False
    baseline_config.use_resonance = False
    
    print("\nConfigurations:")
    print(f"Reso-LLM: Gating={reso_config.use_coherence_gating}, Kuramoto={reso_config.use_kuramoto_dynamics}")
    print(f"Baseline: Gating={baseline_config.use_coherence_gating}, Kuramoto={baseline_config.use_kuramoto_dynamics}")
    
    # 3. Train Models
    epochs = 10 # Increased epochs to allow pattern finding
    results = {}
    
    for name, config in [("Reso-LLM", reso_config), ("Baseline", baseline_config)]:
        print(f"\n" + "-" * 40)
        print(f"Training {name}...")
        print("-" * 40)
        
        # Initialize model
        model = ResoLLMModel(config)
        
        # Count params
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")
        
        # Initialize trainer
        trainer = ResoLLMTrainer(
            model=model,
            dataset=dataset,
            learning_rate=5e-4, 
            warmup_steps=20
        )
        
        # Train
        start_time = time.time()
        trainer.train(epochs=epochs, save_path=f"apps/reso_llm/checkpoints/benchmark_{name.lower()}.pt")
        duration = time.time() - start_time
        
        # Collect metrics
        final_loss = np.mean(trainer.losses[-10:]) if trainer.losses else float('inf')
        
        results[name] = {
            "duration": duration,
            "final_loss": final_loss,
            "losses": trainer.losses,
            "params": params
        }
        
        print(f"\n{name} Results:")
        print(f"  Time: {duration:.2f}s")
        print(f"  Final Loss: {final_loss:.4f}")

    # 4. Compare
    print("\n" + "=" * 60)
    print("Benchmark Comparison Report")
    print("=" * 60)
    
    reso_res = results["Reso-LLM"]
    base_res = results["Baseline"]
    
    print(f"{'Metric':<20} {'Reso-LLM':<15} {'Baseline':<15} {'Diff':<15}")
    print("-" * 65)
    
    # Time
    t_diff = reso_res["duration"] - base_res["duration"]
    t_pct = (t_diff / base_res["duration"]) * 100
    print(f"{'Training Time (s)':<20} {reso_res['duration']:<15.2f} {base_res['duration']:<15.2f} {t_diff:+.2f} ({t_pct:+.1f}%)")
    
    # Loss
    l_diff = reso_res["final_loss"] - base_res["final_loss"]
    l_pct = (l_diff / base_res["final_loss"]) * 100
    print(f"{'Final Loss':<20} {reso_res['final_loss']:<15.4f} {base_res['final_loss']:<15.4f} {l_diff:+.4f} ({l_pct:+.1f}%)")
    
    # Params
    p_diff = reso_res["params"] - base_res["params"]
    print(f"{'Parameters':<20} {reso_res['params']:<15,} {base_res['params']:<15,} {p_diff:+,}")

    # Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(reso_res["losses"], label="Reso-LLM")
        plt.plot(base_res["losses"], label="Baseline")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Comparison (Noisy Pattern)")
        plt.legend()
        plt.grid(True)
        plot_path = "apps/reso_llm/benchmark_plot.png"
        plt.savefig(plot_path)
        print(f"\nLoss plot saved to {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

if __name__ == "__main__":
    run_benchmark()
