#!/usr/bin/env python3
"""
PyTorch Training Script for ResoFormer

Full training pipeline with:
- HuggingFace datasets (tiny_shakespeare)
- HuggingFace tokenizers
- Proper gradient descent with AdamW
- Learning rate scheduling with golden ratio decay
- Coherence monitoring during training
- Text generation samples
"""

from __future__ import annotations
import os
import sys
import time
import math
from typing import Optional, Dict, Any
import dotenv

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# Set HuggingFace token
if('HF_TOKEN' not in os.environ):
    dotenv.load_dotenv('.env')
    os.environ["HF_TOKEN"] = dotenv.get_key('.env', 'HF_TOKEN')

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch required. Install with: pip install torch")
    sys.exit(1)

try:
    from datasets import load_dataset
    HF_DATASETS = True
except ImportError:
    HF_DATASETS = False
    print("HuggingFace datasets not found. Install with: pip install datasets")

try:
    from transformers import AutoTokenizer
    HF_TOKENIZERS = True
except ImportError:
    HF_TOKENIZERS = False
    print("HuggingFace transformers not found. Using character tokenizer.")

from pytorch_model import PyTorchResoFormer, ResoFormerConfig
from tinyaleph.core.constants import PHI


# =============================================================================
# DATASET
# =============================================================================

class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, text: str, tokenizer, seq_len: int = 128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Tokenize entire text
        if hasattr(tokenizer, 'encode'):
            self.tokens = tokenizer.encode(text)
            if hasattr(self.tokens, 'ids'):
                self.tokens = self.tokens.ids
        else:
            # Character-level fallback
            self.tokens = [ord(c) % 32000 for c in text]
        
        # Create examples
        self.examples = []
        for i in range(0, len(self.tokens) - seq_len - 1, seq_len // 2):
            self.examples.append(self.tokens[i:i + seq_len + 1])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class CharTokenizer:
    """Simple character-level tokenizer - only keeps printable ASCII."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
    
    def fit(self, text: str):
        # Only keep common printable characters
        allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:\'"()-\n')
        chars = sorted([c for c in set(text) if c in allowed])
        
        for i, c in enumerate(chars[:self.vocab_size - 4]):
            self.char_to_id[c] = i + 4  # Reserve 0-3 for special tokens
            self.id_to_char[i + 4] = c
        
        # Special tokens
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.id_to_char[0] = ''
        self.id_to_char[1] = ' '  # Replace UNK with space
        self.id_to_char[2] = ''
        self.id_to_char[3] = ''
        
        # Update vocab_size to actual size
        self.actual_vocab_size = len(self.char_to_id) + 4
        
        return self
    
    def encode(self, text: str):
        return [self.char_to_id.get(c, self.unk_id) for c in text]
    
    def decode(self, ids, skip_special=True):
        result = []
        for i in ids:
            if skip_special and i < 4:
                continue
            char = self.id_to_char.get(i, ' ')
            result.append(char)
        return ''.join(result)


# =============================================================================
# TRAINER
# =============================================================================

class ResoFormerTrainer:
    """Trainer for PyTorch ResoFormer."""
    
    def __init__(self, 
                 model: PyTorchResoFormer,
                 train_dataset: Dataset,
                 tokenizer,
                 val_dataset: Optional[Dataset] = None,
                 learning_rate: float = 1e-4,
                 batch_size: int = 16,
                 num_epochs: int = 10,
                 warmup_steps: int = 100,
                 use_golden_decay: bool = True,
                 device: str = 'auto'):
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.use_golden_decay = use_golden_decay
        
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                       'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        if val_dataset:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Metrics
        self.step = 0
        self.best_loss = float('inf')
        self.loss_history = []
    
    def get_lr(self) -> float:
        """Get learning rate with warmup and golden ratio decay."""
        if self.step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * (self.step + 1) / self.warmup_steps
        
        if self.use_golden_decay:
            # Golden ratio decay: lr = base * φ^(-steps/warmup)
            decay_steps = self.step - self.warmup_steps
            decay_factor = 1.0 / (PHI ** (decay_steps / self.warmup_steps / 10))
            return max(1e-6, self.learning_rate * decay_factor)
        
        return self.learning_rate
    
    def train_step(self, batch) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward
        outputs = self.model(x, labels=y)
        loss = outputs['loss']
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (1/φ ≈ 0.618)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0 / PHI)
        
        # Update LR
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.step()
        self.step += 1
        
        # Metrics
        metrics = {'loss': loss.item(), 'lr': lr}
        
        # Add coherence metrics if available
        for key, val in outputs.items():
            if 'coherence' in key:
                metrics[key] = val.item() if torch.is_tensor(val) else val
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if not self.val_dataset:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        for batch in self.val_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            outputs = self.model(x, labels=y)
            total_loss += outputs['loss'].item()
            count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def train(self, log_interval: int = 50, eval_interval: int = 200):
        """Full training loop."""
        print(f"Training for {self.num_epochs} epochs")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Total steps: ~{len(self.train_loader) * self.num_epochs}")
        print("-" * 70)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                epoch_steps += 1
                
                self.loss_history.append(metrics['loss'])
                
                # Logging
                if self.step % log_interval == 0:
                    avg_loss = epoch_loss / epoch_steps
                    ppl = math.exp(min(avg_loss, 20))
                    elapsed = time.time() - start_time
                    
                    coherence_str = ""
                    for key, val in metrics.items():
                        if 'coherence' in key:
                            coherence_str = f" | Coh: {val:.4f}"
                            break
                    
                    print(f"Step {self.step:5d} | Epoch {epoch+1}/{self.num_epochs} | "
                          f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                          f"LR: {metrics['lr']:.2e}{coherence_str} | "
                          f"Time: {elapsed:.1f}s")
                
                # Evaluation
                if self.val_dataset and self.step % eval_interval == 0:
                    val_loss = self.evaluate()
                    print(f"  >> Validation loss: {val_loss:.4f}")
                    
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        print(f"  >> New best model!")
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"=== Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f} ===")
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Final loss: {self.loss_history[-1]:.4f}")
        
        return self.loss_history
    
    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 0.5) -> str:
        """Generate text from prompt with nucleus sampling."""
        self.model.eval()
        
        # Encode prompt
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt)
            if hasattr(tokens, 'ids'):
                tokens = tokens.ids
        else:
            tokens = [ord(c) % self.model.config.vocab_size for c in prompt]
        
        # Manual generation with better sampling
        generated = list(tokens)
        
        for step in range(max_length - len(tokens)):
            # Forward pass with last 128 tokens
            context = generated[-128:] if len(generated) > 128 else generated
            curr_input = torch.tensor([context], dtype=torch.long, device=self.device)
            outputs = self.model(curr_input)
            logits = outputs['logits'][0, -1, :]  # Last position
            
            # Temperature - lower is more deterministic
            logits = logits / max(temperature, 0.01)
            
            # Light repetition penalty - only penalize very recent repeats
            recent = set(generated[-5:])
            for token in recent:
                if 0 <= token < len(logits):
                    logits[token] *= 0.9
            
            # Top-k sampling (keep top 40)
            top_k = 40
            top_values, top_indices = torch.topk(logits, top_k)
            
            # Softmax over top-k
            probs = F.softmax(top_values, dim=-1)
            
            # Sample from top-k
            idx = torch.multinomial(probs, num_samples=1).item()
            next_token = top_indices[idx].item()
            
            generated.append(next_token)
            
            # Stop on EOS
            if next_token == 3:
                break
        
        # Decode
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(generated, skip_special=True)
        else:
            return ''.join(chr(min(i, 127)) for i in generated)


# =============================================================================
# MAIN
# =============================================================================

def load_shakespeare():
    """Load Shakespeare dataset."""
    # Use built-in corpus for clean character-level training
    print("Using clean Shakespeare corpus for character-level LM...")
    text = """
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
And one man in his time plays many parts.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones.

Now is the winter of our discontent
Made glorious summer by this sun of York.

O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time.

Double, double toil and trouble;
Fire burn and caldron bubble.

The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven.

If music be the food of love, play on.

Out, out, brief candle! Life's but a walking shadow,
A poor player that struts and frets his hour upon the stage
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury, signifying nothing.

What a piece of work is a man! how noble in reason!
how infinite in faculty! in form and moving how
express and admirable! in action how like an angel!
in apprehension how like a god! the beauty of the world!
the paragon of animals! And yet, to me,
what is this quintessence of dust?

Now cracks a noble heart. Good night sweet prince:
And flights of angels sing thee to thy rest!

This above all: to thine own self be true,
And it must follow, as the night the day,
Thou canst not then be false to any man.

There are more things in heaven and earth, Horatio,
Than are dreamt of in your philosophy.

Though this be madness, yet there is method in it.

Brevity is the soul of wit.

Something is rotten in the state of Denmark.

Frailty, thy name is woman!

The lady doth protest too much, methinks.

Neither a borrower nor a lender be.

Good night, good night! Parting is such sweet sorrow,
That I shall say good night till it be morrow.

A horse! a horse! my kingdom for a horse!

Off with his head!

Et tu, Brute?

Beware the ides of March.

The fault, dear Brutus, is not in our stars,
But in ourselves, that we are underlings.

Cowards die many times before their deaths;
The valiant never taste of death but once.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.

Cry havoc, and let slip the dogs of war.

Some are born great, some achieve greatness,
and some have greatness thrust upon them.

If music be the food of love, play on.

Lord, what fools these mortals be!

The course of true love never did run smooth.

Love looks not with the eyes, but with the mind,
And therefore is winged Cupid painted blind.

All that glitters is not gold.

The better part of valour is discretion.

Uneasy lies the head that wears a crown.

Once more unto the breach, dear friends, once more.

We few, we happy few, we band of brothers.

I am not bound to please thee with my answer.

The evil that men do lives after them;
The good is oft interred with their bones.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.

For never was a story of more woe
Than this of Juliet and her Romeo.
""" * 10
    return text, None


def main():
    print("=" * 70)
    print("PyTorch ResoFormer Training")
    print("=" * 70)
    print()
    
    # Load data
    train_text, val_text = load_shakespeare()
    
    # Limit size for demo
    if len(train_text) > 100000:
        train_text = train_text[:100000]
        print(f"Limited to {len(train_text):,} characters")
    print()
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = CharTokenizer(vocab_size=256)
    tokenizer.fit(train_text)
    vocab_size = len(tokenizer.char_to_id) + 4  # +4 for special tokens
    print(f"Vocabulary size: {vocab_size}")
    print()
    
    # Create datasets
    seq_len = 128
    train_dataset = TextDataset(train_text, tokenizer, seq_len=seq_len)
    val_dataset = TextDataset(val_text, tokenizer, seq_len=seq_len) if val_text else None
    print(f"Training examples: {len(train_dataset)}")
    print()
    
    # Create model
    config = ResoFormerConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len + 16,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        use_golden_attention=True,
        use_resonance_rotation=True,
        use_coherence_gate=True,
        use_entropy_collapse=False,
    )
    
    model = PyTorchResoFormer(config)
    print(f"Model: {model.num_parameters:,} parameters")
    print(f"Architecture: {config.num_layers} layers, dim={config.hidden_dim}, heads={config.num_heads}")
    print()
    
    # Create trainer
    trainer = ResoFormerTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        learning_rate=3e-4,
        batch_size=32,
        num_epochs=35,
        warmup_steps=100,
        use_golden_decay=True,
    )
    
    # Train
    print("Starting training...")
    print("-" * 70)
    trainer.train(log_interval=25, eval_interval=100)
    print()
    
    # Generate samples
    print("=" * 70)
    print("Generation Samples")
    print("=" * 70)
    print()
    
    prompts = [
        "To be, or not",
        "All the world",
        "Friends, Romans",
        "Tomorrow, and",
        "O Romeo",
    ]
    
    print("--- With temperature 0.3 ---")
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        generated = trainer.generate(prompt, max_length=80, temperature=0.3)
        print(f"Generated: '{generated}'")
        print()
    
    print("\n--- Greedy decoding (temperature 0.01) ---")
    for prompt in prompts[:3]:
        print(f"Prompt: '{prompt}'")
        generated = trainer.generate(prompt, max_length=60, temperature=0.01)
        print(f"Generated: '{generated}'")
        print()
    
    print("Training complete!")


if __name__ == "__main__":
    main()