"""
Training and Evaluation Script for Reso-LLM Extensions.

Trains two models:
1. Standard model (no extensions)
2. Extended model (with all extensions enabled)

Then evaluates both and compares their metrics.

Usage:
    PYTHONPATH=. .venv/bin/python apps/reso_llm/train_and_evaluate.py
    
Options:
    --epochs N      Number of epochs (default: 1)
    --batch-size N  Batch size (default: 8)
    --seq-len N     Sequence length (default: 64)
"""
import sys
import os
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim

from apps.reso_llm.config import ResoLLMConfig
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.dataset import TextDataset
from apps.reso_llm.evaluation_harness import ResoEvaluationHarness
from apps.resoformer.torch_backend import get_device


RESULTS_DIR = "apps/reso_llm/evaluation_results"


def create_simple_dataset(tokenizer, text_path: str = "data/tinyshakespeare.txt",
                          seq_len: int = 64, batch_size: int = 8):
    """Create a simple text dataset for quick training."""
    return TextDataset(text_path, tokenizer, seq_len=seq_len, batch_size=batch_size)


def train_model(model, dataset, epochs: int = 1, lr: float = 1e-3):
    """Train a model for the specified number of epochs."""
    device = get_device()
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    steps_per_epoch = min(len(dataset), 100)  # Cap for quick training
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            x_batch, y_batch = dataset.get_batch()
            
            x = torch.tensor(x_batch.data, dtype=torch.long).view(
                dataset.batch_size, dataset.seq_len
            ).to(device)
            y = torch.tensor(y_batch.data, dtype=torch.long).view(
                dataset.batch_size, dataset.seq_len
            ).to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(
                logits.view(-1, model.config.vocab_size),
                y.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            losses.append(loss.item())
            
            if step % 20 == 0:
                print(f"    Step {step}/{steps_per_epoch} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / steps_per_epoch
        print(f"  Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")
    
    return losses


def run_comparison(args):
    """Run the full comparison experiment."""
    print("=" * 70)
    print("Reso-LLM Extension Evaluation Experiment")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.seq_len}")
    print()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = create_default_tokenizer()
    
    # Create dataset
    print("Loading dataset...")
    dataset = create_simple_dataset(
        tokenizer, 
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
    print(f"Dataset: {len(dataset.data):,} tokens")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len
        },
        "models": {}
    }
    
    # =========================================================================
    # Train Standard Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING: Standard Model (No Extensions)")
    print("=" * 70)
    
    config_standard = ResoLLMConfig.tiny(standard=True)
    config_standard.vocab_size = tokenizer.vocab_size
    config_standard.disable_extensions()
    
    print(f"Config: dim={config_standard.dim}, layers={config_standard.num_layers}")
    
    model_standard = ResoLLMModel(config_standard)
    params = sum(p.numel() for p in model_standard.parameters())
    print(f"Parameters: {params:,}")
    
    start_time = time.time()
    losses_standard = train_model(model_standard, dataset, epochs=args.epochs)
    train_time_standard = time.time() - start_time
    
    print(f"\nTraining completed in {train_time_standard:.2f}s")
    
    # Evaluate standard model
    print("\nEvaluating Standard Model...")
    harness_standard = ResoEvaluationHarness(model_standard, tokenizer)
    metrics_standard = harness_standard.run_all()
    metrics_standard["final_train_loss"] = losses_standard[-1] if losses_standard else None
    metrics_standard["train_time_seconds"] = train_time_standard
    
    results["models"]["standard"] = {
        "config": {
            "dim": config_standard.dim,
            "layers": config_standard.num_layers,
            "extensions_enabled": False
        },
        "metrics": metrics_standard,
        "losses": losses_standard[-20:] if len(losses_standard) > 20 else losses_standard  # Last 20
    }
    
    print("\nStandard Model Metrics:")
    for k, v in metrics_standard.items():
        print(f"  {k}: {v}")
    
    # =========================================================================
    # Train Extended Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING: Extended Model (All Extensions Enabled)")
    print("=" * 70)
    
    config_extended = ResoLLMConfig.tiny(standard=False)
    config_extended.vocab_size = tokenizer.vocab_size
    config_extended.enable_extensions()
    
    print(f"Config: dim={config_extended.dim}, layers={config_extended.num_layers}")
    print(f"Extensions: Agency={config_extended.agency.enabled}, "
          f"PRSC={config_extended.prsc.enabled}, "
          f"SMF={config_extended.temporal_smf.enabled}, "
          f"Stability={config_extended.stability.enabled}")
    
    model_extended = ResoLLMModel(config_extended)
    params = sum(p.numel() for p in model_extended.parameters())
    print(f"Parameters: {params:,}")
    
    start_time = time.time()
    losses_extended = train_model(model_extended, dataset, epochs=args.epochs)
    train_time_extended = time.time() - start_time
    
    print(f"\nTraining completed in {train_time_extended:.2f}s")
    
    # Evaluate extended model
    print("\nEvaluating Extended Model...")
    harness_extended = ResoEvaluationHarness(model_extended, tokenizer)
    metrics_extended = harness_extended.run_all()
    metrics_extended["final_train_loss"] = losses_extended[-1] if losses_extended else None
    metrics_extended["train_time_seconds"] = train_time_extended
    
    results["models"]["extended"] = {
        "config": {
            "dim": config_extended.dim,
            "layers": config_extended.num_layers,
            "extensions_enabled": True,
            "agency": config_extended.agency.enabled,
            "prsc": config_extended.prsc.enabled,
            "temporal_smf": config_extended.temporal_smf.enabled,
            "stability": config_extended.stability.enabled
        },
        "metrics": metrics_extended,
        "losses": losses_extended[-20:] if len(losses_extended) > 20 else losses_extended
    }
    
    print("\nExtended Model Metrics:")
    for k, v in metrics_extended.items():
        print(f"  {k}: {v}")
    
    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    comparison = {}
    
    # Compare each metric
    for metric in set(list(metrics_standard.keys()) + list(metrics_extended.keys())):
        std_val = metrics_standard.get(metric)
        ext_val = metrics_extended.get(metric)
        
        if std_val is not None and ext_val is not None:
            if isinstance(std_val, (int, float)) and isinstance(ext_val, (int, float)):
                diff = ext_val - std_val
                pct = (diff / std_val * 100) if std_val != 0 else 0
                comparison[metric] = {
                    "standard": std_val,
                    "extended": ext_val,
                    "difference": diff,
                    "percent_change": pct
                }
    
    results["comparison"] = comparison
    
    print("\n{:<30} {:>12} {:>12} {:>12}".format(
        "Metric", "Standard", "Extended", "Diff"
    ))
    print("-" * 70)
    
    for metric, vals in comparison.items():
        std = vals["standard"]
        ext = vals["extended"]
        diff = vals["difference"]
        
        if isinstance(std, float):
            print("{:<30} {:>12.4f} {:>12.4f} {:>+12.4f}".format(metric, std, ext, diff))
        else:
            print("{:<30} {:>12} {:>12} {:>12}".format(metric, str(std)[:12], str(ext)[:12], str(diff)[:12]))
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Loss comparison
    loss_std = metrics_standard.get("final_train_loss", float('inf'))
    loss_ext = metrics_extended.get("final_train_loss", float('inf'))
    
    if loss_ext < loss_std:
        print(f"✓ Extended model achieved LOWER loss ({loss_ext:.4f} vs {loss_std:.4f})")
    else:
        print(f"✗ Standard model achieved lower loss ({loss_std:.4f} vs {loss_ext:.4f})")
    
    # Stability comparison
    lyap_std = metrics_standard.get("final_lyapunov_exponent", 0)
    lyap_ext = metrics_extended.get("final_lyapunov_exponent", 0)
    
    if lyap_ext < lyap_std:
        print(f"✓ Extended model is MORE STABLE (λ={lyap_ext:.4f} vs λ={lyap_std:.4f})")
    else:
        print(f"  Extended model has similar stability (λ={lyap_ext:.4f} vs λ={lyap_std:.4f})")
    
    # Memory comparison
    mem_std = metrics_standard.get("memory_recall_success", 0)
    mem_ext = metrics_extended.get("memory_recall_success", 0)
    
    if mem_ext > 0 and mem_std == 0:
        print(f"✓ Extended model has WORKING MEMORY (recall={mem_ext:.1f} vs {mem_std:.1f})")
    elif mem_ext >= mem_std:
        print(f"  Memory recall comparable ({mem_ext:.1f} vs {mem_std:.1f})")
    
    # Training time
    time_overhead = (train_time_extended - train_time_standard) / train_time_standard * 100
    print(f"\nTraining time overhead: {time_overhead:+.1f}%")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"comparison_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Reso-LLM extensions")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    
    args = parser.parse_args()
    
    run_comparison(args)


if __name__ == "__main__":
    main()
