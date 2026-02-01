"""
Evaluation script for Reso-LLM.

Computes perplexity and performs qualitative analysis.
"""
import sys
import os
import math
from typing import List

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.dataset import TextDataset
from apps.reso_llm.train import ResoLLMTrainer

def evaluate_model(model_path: str, data_path: str):
    """Evaluate a trained model."""
    print(f"Loading model from {model_path}...")
    model = ResoLLMModel.load(model_path)
    
    tokenizer = create_default_tokenizer()
    
    print(f"Loading evaluation data from {data_path}...")
    dataset = TextDataset(data_path, tokenizer, seq_len=32, batch_size=4)
    
    trainer = ResoLLMTrainer(model, dataset) # Re-use trainer for loss calc
    
    print("Computing perplexity...")
    # Loss is avg log prob. Perplexity = exp(loss)
    # Note: Our trainer computes loss per batch.
    
    total_loss = 0.0
    count = 0
    
    for x, y in dataset.iterate_batches():
        logits = model.forward(x.data, training=False)
        logits = logits.reshape((dataset.batch_size, dataset.seq_len, model.config.vocab_size))
        loss = trainer.cross_entropy_loss(logits, y)
        total_loss += loss
        count += 1
        
    avg_loss = total_loss / max(1, count)
    perplexity = math.exp(avg_loss)
    
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    return perplexity

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <model_path> <data_path>")
    else:
        evaluate_model(sys.argv[1], sys.argv[2])
