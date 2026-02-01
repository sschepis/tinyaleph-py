import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List

# Import ResoFormer components
import sys
sys.path.append('.')
from apps.resoformer.pytorch_model import ResoFormerConfig, PyTorchResoFormer

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CONFIG = {
    'vocab_size': 100,      # Small vocab for associative recall
    'seq_len': 64,          # Sequence length
    'hidden_dim': 128,      # Model dimension
    'num_layers': 2,        # Shallow models
    'num_heads': 4,         # 4 heads
    'ffn_dim': 256,         # FFN expansion
    'batch_size': 32,
    'num_steps': 500,       # Training steps
    'learning_rate': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on device: {CONFIG['device']}")

# =============================================================================
# 2. DATASET GENERATION (Associative Recall)
# =============================================================================

def generate_associative_recall_data(
    num_samples: int, 
    vocab_size: int, 
    seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates data for Associative Recall task.
    
    Task:
    The sequence consists of key-value pairs.
    Input:  [k1, v1, k2, v2, ..., query_key, ???]
    Target: [..., query_value]
    
    The model must find 'query_key' earlier in the sequence and output the associated value.
    """
    # Reserve 0 for padding/special tokens if needed (not used here for simplicity)
    # Keys and values are from 1 to vocab_size-1
    
    inputs = torch.randint(1, vocab_size, (num_samples, seq_len))
    targets = torch.zeros_like(inputs)
    
    for i in range(num_samples):
        # Create key-value pairs
        # We ensure keys are unique to avoid ambiguity
        keys = torch.randperm(vocab_size - 1)[:seq_len // 2] + 1
        values = torch.randint(1, vocab_size, (seq_len // 2,))
        
        # Interleave keys and values: k1, v1, k2, v2...
        seq = torch.zeros(seq_len, dtype=torch.long)
        seq[0::2] = keys
        seq[1::2] = values
        
        # Pick a random key to query (from the first half of pairs)
        # We put the query at the end
        query_idx = torch.randint(0, (seq_len // 2) - 1, (1,)).item()
        query_key = keys[query_idx]
        query_val = values[query_idx]
        
        # Overwrite last token with query key
        seq[-2] = query_key
        # The target for the last position is the value
        seq[-1] = 0 # Mask input for last position (optional, but let's keep it simple)
        
        inputs[i] = seq
        
        # Targets: we only care about the last prediction
        # But standard training predicts next token everywhere.
        # We'll create a full target sequence.
        # Target at pos t is input at t+1.
        target_seq = torch.cat([seq[1:], torch.tensor([0])])
        target_seq[-2] = query_val # The answer to the query
        
        targets[i] = target_seq

    return inputs, targets

# =============================================================================
# 3. BASELINE MODEL (Standard Transformer)
# =============================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True # Pre-norm usually stabilizes training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        b, s = x.shape
        positions = torch.arange(s, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(s).to(x.device)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)

# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_model(model, name, train_loader, num_steps, lr, device):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    step = 0
    model.train()
    
    while step < num_steps:
        for x, y in train_loader:
            if step >= num_steps: break
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            if isinstance(logits, dict): # ResoFormer returns dict
                logits = logits['logits']
            
            # We only care about the LAST token prediction for accuracy/loss in this specific task
            # (The associative recall query is at the end)
            last_logits = logits[:, -2, :] # Prediction for the last token (based on query at -2)
            last_targets = y[:, -2]
            
            loss = criterion(last_logits, last_targets)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            preds = last_logits.argmax(dim=-1)
            acc = (preds == last_targets).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(acc)
            
            if step % 50 == 0:
                print(f"Step {step:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
            
            step += 1
            
    total_time = time.time() - start_time
    print(f"Finished {name} in {total_time:.2f}s. Final Acc: {accuracies[-1]:.4f}")
    
    return losses, accuracies

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    # 1. Generate Data
    print("Generating data...")
    inputs, targets = generate_associative_recall_data(
        num_samples=2000, 
        vocab_size=CONFIG['vocab_size'], 
        seq_len=CONFIG['seq_len']
    )
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 2. Initialize Models
    
    # Baseline
    baseline = StandardTransformer(
        vocab_size=CONFIG['vocab_size'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        ffn_dim=CONFIG['ffn_dim'],
        seq_len=CONFIG['seq_len']
    )
    
    # ResoFormer
    reso_config = ResoFormerConfig(
        vocab_size=CONFIG['vocab_size'],
        max_seq_len=CONFIG['seq_len'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        ffn_dim=CONFIG['ffn_dim'],
        dropout=0.1,
        use_golden_attention=True,
        use_resonance_rotation=True,
        use_coherence_gate=True
    )
    resoformer = PyTorchResoFormer(reso_config)
    
    # Count params
    base_params = sum(p.numel() for p in baseline.parameters())
    reso_params = sum(p.numel() for p in resoformer.parameters())
    print(f"\nBaseline Params: {base_params:,}")
    print(f"ResoFormer Params: {reso_params:,}")
    
    # 3. Train
    
    # Run 1: Baseline
    base_losses, base_accs = train_model(
        baseline, "Baseline", loader, CONFIG['num_steps'], CONFIG['learning_rate'], CONFIG['device']
    )
    
    # Run 2: ResoFormer (Full)
    reso_losses, reso_accs = train_model(
        resoformer, "ResoFormer (Full)", loader, CONFIG['num_steps'], CONFIG['learning_rate'], CONFIG['device']
    )
    
    # Run 3: ResoFormer (Ablated - No Fancy Features)
    # This checks if the base implementation is sound without the resonant extras
    ablated_config = ResoFormerConfig(
        vocab_size=CONFIG['vocab_size'],
        max_seq_len=CONFIG['seq_len'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        ffn_dim=CONFIG['ffn_dim'],
        dropout=0.1,
        use_golden_attention=False,     # OFF
        use_resonance_rotation=False,   # OFF
        use_coherence_gate=False        # OFF
    )
    ablated_model = PyTorchResoFormer(ablated_config)
    
    ablated_losses, ablated_accs = train_model(
        ablated_model, "ResoFormer (Ablated)", loader, CONFIG['num_steps'], CONFIG['learning_rate'], CONFIG['device']
    )
    
    # 4. Results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Full Reso':<10} | {'Ablated':<10}")
    print("-" * 60)
    print(f"{'Final Accuracy':<15} | {base_accs[-1]:.4f}     | {reso_accs[-1]:.4f}     | {ablated_accs[-1]:.4f}")
    print(f"{'Min Loss':<15} | {min(base_losses):.4f}     | {min(reso_losses):.4f}     | {min(ablated_losses):.4f}")
    
    # Simple smoothing for visualization
    def smooth(data, window=10):
        if not data: return []
        return [sum(data[i:i+window])/window for i in range(len(data)-window)]

    print("\nSmoothed Accuracy Trajectory (last 5 points):")
    print("Baseline:       ", [f"{x:.2f}" for x in smooth(base_accs)[-5:]])
    print("Full Reso:      ", [f"{x:.2f}" for x in smooth(reso_accs)[-5:]])
    print("Ablated Reso:   ", [f"{x:.2f}" for x in smooth(ablated_accs)[-5:]])

if __name__ == "__main__":
    main()
