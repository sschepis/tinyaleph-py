#!/usr/bin/env python3
"""
Production ResoFormer Training with TinyStories Dataset
========================================================

This script trains the ResoFormer model on the TinyStories dataset from HuggingFace,
implementing proper data loading, regularization, and generation strategies.

Features:
- Real dataset: TinyStories (4.5M training examples)
- Nucleus sampling (top-p) for coherent generation
- Dropout and weight decay for regularization
- Gradient accumulation for larger effective batch sizes
- Model checkpointing
- WandB logging (optional)
"""

import os
import sys
import math
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import our PyTorch ResoFormer
from apps.resoformer.pytorch_model import (
    ResoFormerConfig,
    PyTorchResoFormer,
    PHI,
)

# Try to import transformers for tokenizer
try:
    from transformers import GPT2TokenizerFast
    HAS_TOKENIZER = True
except ImportError:
    print("Warning: transformers not installed, will use basic tokenizer")
    HAS_TOKENIZER = False

# Try to import datasets for TinyStories
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    print("Warning: datasets not installed, will use synthetic data")
    HAS_DATASETS = False

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model architecture
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Generation settings
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 0.8
    max_gen_len: int = 100
    
    # Data settings
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    max_train_samples: Optional[int] = 100000  # Limit for faster training
    
    # System settings
    device: str = "auto"
    dtype: str = "float32"  # float32, float16, bfloat16
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)
    seed: int = 42
    
    # Paths
    output_dir: str = "checkpoints"
    checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


class TinyStoriesDataset(Dataset):
    """Dataset wrapper for TinyStories or similar text data."""
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 256,
        split: str = "train",
        max_samples: Optional[int] = None,
        dataset_name: str = "roneneldan/TinyStories",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Loading dataset: {dataset_name}...")
        
        if HAS_DATASETS:
            try:
                # Load TinyStories from HuggingFace
                dataset = load_dataset(dataset_name, split=split, streaming=False)
                
                # Process samples
                samples_processed = 0
                for item in tqdm(dataset, desc="Tokenizing"):
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    text = item.get("text", "")
                    if len(text) < 50:  # Skip very short texts
                        continue
                    
                    # Tokenize
                    tokens = tokenizer.encode(text)
                    
                    # Create training examples with sliding window
                    for i in range(0, len(tokens) - max_length, max_length // 2):
                        chunk = tokens[i:i + max_length + 1]
                        if len(chunk) == max_length + 1:
                            self.data.append(chunk)
                    
                    samples_processed += 1
                
                print(f"Created {len(self.data)} training examples from {samples_processed} stories")
                
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self._create_synthetic_data()
        else:
            self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic training data if dataset unavailable."""
        print("Creating synthetic TinyStories-style data...")
        
        # Story templates for generating synthetic data
        templates = [
            "Once upon a time, there was a {adj} {animal} named {name}. {Name} lived in a {place}. One day, {name} decided to go on an adventure. {He_she} walked through the forest and met a friendly {animal2}. They became best friends and played together all day. When the sun set, {name} went home and told {his_her} family about the wonderful day.",
            
            "There was a little {child} who loved to {verb}. Every morning, {he_she} would wake up early and {verb} until breakfast. {His_her} {parent} always said, \"{Name}, you are so good at {verb_ing}!\" This made the {child} very happy.",
            
            "In a small village, there lived a {adj} {person}. The {person} had a {adj2} {object} that was very special. One day, a {creature} came and took the {object}! The {person} was very sad. But then, a brave {helper} came to help. Together, they found the {object} and brought it back. Everyone was happy.",
            
            "The {adj} {animal} looked up at the sky. The sun was shining bright. {Name} felt happy today. {He_she} ran through the meadow, jumping over flowers. A butterfly flew by, and {name} tried to catch it. But the butterfly was too fast. {Name} laughed and kept playing.",
            
            "Mom said, \"Time for bed!\" But {name} wasn't tired yet. {He_she} wanted to play more. \"Just five more minutes?\" asked {name}. Mom smiled and said, \"Okay, five more minutes.\" {Name} played with {his_her} toys happily. Then it was time to sleep. {Name} had sweet dreams.",
            
            "It was a rainy day. {Name} looked out the window and felt bored. \"What can I do?\" {he_she} wondered. Then {name} had an idea! {He_she} got paper and crayons and started to draw. {Name} drew a {adj} {animal} and a {adj2} {object}. When the rain stopped, {name} showed the drawing to {parent}. \"{That}'s beautiful!\" said {parent}.",
            
            "The {adj} {animal} was very hungry. {He_she} looked everywhere for food. First, {name} looked under a rock. Nothing there. Then, {name} looked behind a tree. Still nothing. Finally, {name} looked in the garden. There were so many {food}! {Name} ate and ate until {his_her} tummy was full.",
            
            "{Name} was going to school for the first time. {He_she} felt a little scared. \"What if no one wants to be my friend?\" {name} thought. But when {he_she} got to class, a {adj} {child2} smiled at {him_her}. \"Hi! I'm {name2}. Do you want to play?\" {Name} nodded happily. They played together all day and became best friends.",
        ]
        
        names = ["Lily", "Max", "Emma", "Jack", "Sophie", "Tom", "Mia", "Sam", "Ella", "Ben"]
        animals = ["bunny", "puppy", "kitten", "bird", "bear", "fox", "deer", "owl", "duck", "mouse"]
        adjectives = ["little", "happy", "friendly", "curious", "brave", "kind", "smart", "gentle", "playful", "silly"]
        places = ["forest", "garden", "meadow", "house", "castle", "village", "farm", "mountain", "beach", "park"]
        verbs = ["play", "sing", "dance", "draw", "read", "run", "jump", "swim", "climb", "explore"]
        objects = ["ball", "book", "flower", "star", "toy", "blanket", "hat", "key", "lamp", "stone"]
        parents = ["Mom", "Dad", "Grandma", "Grandpa"]
        children = ["boy", "girl", "child"]
        creatures = ["dragon", "giant", "wizard", "witch", "troll"]
        helpers = ["knight", "fairy", "prince", "princess", "hero"]
        foods = ["berries", "apples", "carrots", "nuts", "flowers", "seeds"]
        
        # Generate stories
        num_stories = 5000
        for _ in range(num_stories):
            template = random.choice(templates)
            name = random.choice(names)
            name2 = random.choice([n for n in names if n != name])
            is_male = name in ["Max", "Jack", "Tom", "Sam", "Ben"]
            
            story = template.format(
                name=name,
                Name=name,
                name2=name2,
                adj=random.choice(adjectives),
                adj2=random.choice(adjectives),
                animal=random.choice(animals),
                animal2=random.choice(animals),
                place=random.choice(places),
                verb=random.choice(verbs),
                verb_ing=random.choice(verbs) + "ing",
                object=random.choice(objects),
                parent=random.choice(parents),
                child=random.choice(children),
                child2=random.choice(children),
                creature=random.choice(creatures),
                helper=random.choice(helpers),
                food=random.choice(foods),
                person=random.choice(["farmer", "baker", "teacher", "king", "queen"]),
                he_she="he" if is_male else "she",
                He_she="He" if is_male else "She",
                his_her="his" if is_male else "her",
                His_her="His" if is_male else "Her",
                him_her="him" if is_male else "her",
                That="That",
            )
            
            # Tokenize and create chunks
            tokens = self.tokenizer.encode(story)
            if len(tokens) >= self.max_length + 1:
                self.data.append(tokens[:self.max_length + 1])
            elif len(tokens) >= 32:  # Pad shorter stories
                padding = [self.tokenizer.pad_token_id or 0] * (self.max_length + 1 - len(tokens))
                self.data.append(tokens + padding)
        
        print(f"Created {len(self.data)} synthetic training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class SimpleTokenizer:
    """Simple character-level tokenizer fallback."""
    
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        
        # Initialize with common characters
        chars = list(" \n\t") + list("abcdefghijklmnopqrstuvwxyz") + \
                list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789") + \
                list(".,!?;:'\"()-") + ["<pad>", "<eos>", "<bos>"]
        
        for i, c in enumerate(chars):
            self.char_to_id[c] = i
            self.id_to_char[i] = c
    
    def encode(self, text: str) -> List[int]:
        tokens = [self.bos_token_id]
        for c in text:
            if c in self.char_to_id:
                tokens.append(self.char_to_id[c])
            else:
                tokens.append(self.char_to_id.get(' ', 3))
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join(self.id_to_char.get(t, ' ') for t in tokens)


def nucleus_sampling(
    logits: torch.Tensor,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 1.0,
) -> int:
    """
    Nucleus (top-p) sampling with top-k filtering.
    
    Args:
        logits: Raw logits for next token (vocab_size,)
        top_p: Cumulative probability threshold
        top_k: Maximum number of tokens to consider
        temperature: Sampling temperature
    
    Returns:
        Selected token index
    """
    # Apply temperature
    logits = logits / max(temperature, 1e-8)
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
        probs = probs / probs.sum()
    
    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff index
        cutoff_mask = cumsum_probs > top_p
        cutoff_mask[1:] = cutoff_mask[:-1].clone()
        cutoff_mask[0] = False
        
        # Zero out tokens beyond cutoff
        sorted_probs[cutoff_mask] = 0
        
        # Scatter back and renormalize
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        probs = probs / probs.sum()
    
    # Sample from distribution
    token_id = torch.multinomial(probs, 1).item()
    return token_id


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 0.8,
    device: str = "cpu",
    repetition_penalty: float = 1.2,
) -> str:
    """
    Generate text using nucleus sampling with repetition penalty.
    """
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        if tokens[0] != tokenizer.bos_token_id:
            tokens = [tokenizer.bos_token_id] + tokens
    
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    generated_tokens = []
    
    # Track token frequencies for repetition penalty
    token_counts = {}
    
    for _ in range(max_length):
        # Get model predictions
        with torch.no_grad():
            outputs = model(tokens)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
        
        # Get logits for last position
        next_logits = logits[0, -1, :].clone()
        
        # Apply repetition penalty
        if repetition_penalty > 1.0:
            for token_id, count in token_counts.items():
                if token_id < next_logits.size(0):
                    penalty = repetition_penalty ** count
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= penalty
                    else:
                        next_logits[token_id] *= penalty
        
        # Sample next token
        next_token = nucleus_sampling(
            next_logits,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        
        # Check for EOS
        if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
            break
        
        # Update token counts
        token_counts[next_token] = token_counts.get(next_token, 0) + 1
        
        # Append token
        generated_tokens.append(next_token)
        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
        
        # Truncate context if too long
        max_ctx = model.config.max_seq_len if hasattr(model, 'config') else 512
        if tokens.size(1) > max_ctx:
            tokens = tokens[:, -max_ctx:]
    
    # Decode generated tokens
    all_tokens = tokenizer.encode(prompt)[1:] + generated_tokens  # Skip BOS
    result = tokenizer.decode(all_tokens)
    
    return result


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    config: TrainingConfig,
    accumulation_step: int,
) -> Dict[str, float]:
    """Perform a single training step with gradient accumulation."""
    
    x, y = batch
    x = x.to(config.device)
    y = y.to(config.device)
    
    # Forward pass
    outputs = model(x)
    if isinstance(outputs, dict):
        logits = outputs['logits']
    elif isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs
    
    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=-100,
    )
    
    # Scale loss for gradient accumulation
    loss = loss / config.gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Update weights if accumulation complete
    metrics = {}
    if (accumulation_step + 1) % config.gradient_accumulation_steps == 0:
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0 / PHI,  # Golden ratio clipping
        )
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        
        metrics["grad_norm"] = grad_norm.item()
    
    # Compute metrics
    with torch.no_grad():
        perplexity = torch.exp(loss * config.gradient_accumulation_steps)
    
    metrics.update({
        "loss": loss.item() * config.gradient_accumulation_steps,
        "perplexity": perplexity.item(),
    })
    
    return metrics


def main():
    """Main training loop."""
    
    print("=" * 70)
    print("Production ResoFormer Training")
    print("=" * 70)
    
    # Parse config (could add argparse here)
    config = TrainingConfig()
    
    # Set random seeds
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    print(f"\nDevice: {config.device}")
    print(f"Model: {config.num_layers} layers, dim={config.hidden_dim}")
    print(f"Training: {config.max_steps} steps, batch={config.batch_size}")
    
    # Initialize tokenizer
    print("\n" + "-" * 70)
    print("Initializing tokenizer...")
    
    if HAS_TOKENIZER:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
    else:
        tokenizer = SimpleTokenizer()
        vocab_size = tokenizer.vocab_size
    
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Load dataset
    print("\n" + "-" * 70)
    dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        split=config.dataset_split,
        max_samples=config.max_train_samples,
        dataset_name=config.dataset_name,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.device == "cuda",
    )
    
    # Initialize model
    print("\n" + "-" * 70)
    print("Initializing model...")
    
    model_config = ResoFormerConfig(
        vocab_size=vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        use_golden_attention=True,
        use_resonance_rotation=True,
        use_coherence_gate=True,
        use_entropy_collapse=True,
    )
    
    model = PyTorchResoFormer(model_config)
    
    # Count parameters
    dummy_input = torch.zeros(1, config.max_seq_len, dtype=torch.long)
    _ = model(dummy_input)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Move to device
    model = model.to(config.device)
    
    # Compile model if requested (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Learning rate scheduler with warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.max_steps // 3,
        T_mult=1,
        eta_min=config.learning_rate / 10,
    )
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    model.train()
    step = 0
    accumulation_step = 0
    epoch = 0
    total_loss = 0.0
    best_loss = float('inf')
    start_time = time.time()
    
    while step < config.max_steps:
        epoch += 1
        epoch_loss = 0.0
        epoch_steps = 0
        
        for batch in dataloader:
            # Training step
            metrics = train_step(
                model, batch, optimizer, scheduler, config, accumulation_step
            )
            accumulation_step += 1
            
            # Only count as step when gradient is applied
            if accumulation_step % config.gradient_accumulation_steps == 0:
                step += 1
                total_loss += metrics["loss"]
                epoch_loss += metrics["loss"]
                epoch_steps += 1
                
                # Log progress
                if step % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = total_loss / step
                    lr = optimizer.param_groups[0]["lr"]
                    
                    print(f"Step {step:5d}/{config.max_steps} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"PPL: {metrics['perplexity']:.2f} | "
                          f"LR: {lr:.2e} | "
                          f"Time: {elapsed:.1f}s")
                
                # Evaluate and generate
                if step % config.eval_interval == 0:
                    model.eval()
                    
                    print("\n" + "=" * 70)
                    print("Sample Generations:")
                    print("=" * 70)
                    
                    prompts = [
                        "Once upon a time",
                        "The little dog",
                        "There was a",
                        "One day, the",
                        "Mom said",
                    ]
                    
                    for prompt in prompts:
                        generated = generate(
                            model,
                            tokenizer,
                            prompt,
                            max_length=config.max_gen_len,
                            top_p=config.top_p,
                            top_k=config.top_k,
                            temperature=config.temperature,
                            device=config.device,
                            repetition_penalty=1.3,
                        )
                        print(f"\nPrompt: '{prompt}'")
                        print(f"Generated: '{generated}'")
                    
                    print("=" * 70 + "\n")
                    model.train()
                
                # Save checkpoint
                if step % config.save_interval == 0:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    checkpoint_path = output_dir / f"checkpoint_step{step}.pt"
                    
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": model_config,
                        "loss": avg_loss,
                    }, checkpoint_path)
                    
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_path = output_dir / "best_model.pt"
                        torch.save({
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "config": model_config,
                            "loss": avg_loss,
                        }, best_path)
                        print(f"Saved best model (loss: {avg_loss:.4f})")
                
                if step >= config.max_steps:
                    break
        
        # End of epoch
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"\n=== Epoch {epoch} complete | Avg loss: {avg_epoch_loss:.4f} ===\n")
    
    # Final evaluation
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    
    # Final generation examples
    model.eval()
    print("\nFinal Generation Examples:")
    print("-" * 70)
    
    prompts = [
        "Once upon a time, there was a",
        "The little rabbit hopped through",
        "One sunny morning, the children",
        "In a magical forest, there lived",
        "Mom and Dad took the kids to",
    ]
    
    for prompt in prompts:
        generated = generate(
            model,
            tokenizer,
            prompt,
            max_length=150,
            top_p=0.92,
            top_k=40,
            temperature=0.7,
            device=config.device,
            repetition_penalty=1.4,
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": model_config,
        "loss": total_loss / max(step, 1),
    }, final_path)
    print(f"\nSaved final model to {final_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()