#!/usr/bin/env python3
"""
Scaled ResoFormer Training with BPE Tokenizer

Uses GPT-2 tokenizer for proper word-level modeling.
Trains on TinyStories or similar for coherent generation.
"""

from __future__ import annotations
import os
import sys
import time
import math
from typing import Optional, Dict, Any
import dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

if('HF_TOKEN' not in os.environ):
    dotenv.load_dotenv('.env')
    os.environ["HF_TOKEN"] = dotenv.get_key('.env', 'HF_TOKEN')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pytorch_model import PyTorchResoFormer, ResoFormerConfig
from tinyaleph.core.constants import PHI


# =============================================================================
# DATASET WITH BPE TOKENIZER
# =============================================================================

class BPEDataset(Dataset):
    """Dataset using GPT-2 BPE tokenizer."""
    
    def __init__(self, texts, tokenizer, seq_len: int = 256):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        self.tokens = all_tokens
        
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


# =============================================================================
# TRAINER
# =============================================================================

class ScaledTrainer:
    """Trainer for scaled ResoFormer."""
    
    def __init__(self, model, train_dataset, tokenizer, 
                 learning_rate=3e-4, batch_size=16, num_epochs=10,
                 warmup_steps=500, device='auto'):
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                       'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        self.step = 0
        self.loss_history = []
    
    def get_lr(self):
        if self.step < self.warmup_steps:
            return self.learning_rate * (self.step + 1) / self.warmup_steps
        decay_steps = self.step - self.warmup_steps
        decay_factor = 1.0 / (PHI ** (decay_steps / self.warmup_steps / 10))
        return max(1e-6, self.learning_rate * decay_factor)
    
    def train_step(self, batch):
        self.model.train()
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        outputs = self.model(x, labels=y)
        loss = outputs['loss']
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0 / PHI)
        
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        self.optimizer.step()
        self.step += 1
        
        return {'loss': loss.item(), 'lr': lr}
    
    def train(self, log_interval=50):
        print(f"Training for {self.num_epochs} epochs")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print("-" * 70)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch in self.train_loader:
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                epoch_steps += 1
                self.loss_history.append(metrics['loss'])
                
                if self.step % log_interval == 0:
                    avg_loss = epoch_loss / epoch_steps
                    ppl = math.exp(min(avg_loss, 20))
                    elapsed = time.time() - start_time
                    print(f"Step {self.step:5d} | Epoch {epoch+1}/{self.num_epochs} | "
                          f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                          f"LR: {metrics['lr']:.2e} | Time: {elapsed:.1f}s")
            
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"=== Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f} ===")
        
        print(f"\nTraining complete in {time.time() - start_time:.1f}s")
    
    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7):
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        generated = list(tokens)
        
        for step in range(max_length):
            context = generated[-256:] if len(generated) > 256 else generated
            input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
            
            outputs = self.model(input_ids)
            logits = outputs['logits'][0, -1, :].clone()
            
            # Strong repetition penalty - penalize tokens that appeared recently
            for i, token in enumerate(generated[-20:]):
                if token < len(logits):
                    # Stronger penalty for more recent tokens
                    penalty = 1.5 + (20 - i) * 0.1  # 3.5x for most recent, 1.5x for oldest
                    logits[token] /= penalty
            
            # Extra penalty for tokens that repeat more than twice in last 30
            from collections import Counter
            recent_counts = Counter(generated[-30:])
            for token, count in recent_counts.items():
                if count >= 2 and token < len(logits):
                    logits[token] /= (count * 2.0)
            
            # Apply temperature
            logits = logits / max(temperature, 0.01)
            
            # Top-k sampling with smaller k for more coherence
            top_k = 40
            top_values, top_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_values, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            next_token = top_indices[idx].item()
            
            generated.append(next_token)
            
            # Stop on EOS or period after minimum length
            if next_token == self.tokenizer.eos_token_id:
                break
            if step > 30 and self.tokenizer.decode([next_token]).strip().endswith('.'):
                break
        
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# STORIES DATASET
# =============================================================================

def get_stories():
    """Get simple stories for training."""
    stories = [
        "Once upon a time there was a little girl named Lucy. She lived in a small house by the forest. One day she went for a walk and found a magic flower. The flower could talk! It said hello to Lucy and they became best friends.",
        
        "Tom was a brave knight who lived in a big castle. He had a shiny sword and a beautiful horse. One day a dragon came to the kingdom. Tom rode his horse and faced the dragon. He was very brave and saved everyone.",
        
        "There was a little dog named Max. Max loved to play in the park. He would run and jump and chase the birds. His best friend was a cat named Whiskers. They played together every day.",
        
        "Sally wanted to bake a cake for her mom. She got flour and eggs and sugar. She mixed them all together. The cake was delicious! Her mom was so happy.",
        
        "The sun was shining bright. The birds were singing in the trees. It was a beautiful summer day. The children went outside to play. They had so much fun!",
        
        "There was once a wise old owl who lived in an oak tree. All the animals would come to ask him questions. He always gave them good advice. Everyone loved the wise owl.",
        
        "A little fish named Nemo swam in the big blue ocean. He had many friends - a starfish, a seahorse, and a crab. They explored the coral reef together. It was their underwater home.",
        
        "The princess lived in a tall tower. She had long golden hair. A prince came to rescue her. He climbed up her hair and they ran away together. They lived happily ever after.",
        
        "Jack planted a magic bean in his garden. It grew into a giant beanstalk! Jack climbed up and found a castle in the clouds. Inside lived a friendly giant who shared his treasure.",
        
        "There was a little engine that could. Everyone said he was too small. But he tried his best and climbed the big mountain. He said I think I can, I think I can. And he did!",
        
        "Mary had a little lamb. Its fleece was white as snow. Everywhere that Mary went the lamb was sure to go. It followed her to school one day and all the children laughed.",
        
        "The three little pigs built three houses. One of straw, one of sticks, one of bricks. The big bad wolf huffed and puffed. Only the brick house stood strong. The pigs were safe inside.",
        
        "Goldilocks went into the bears house. She tried their porridge - too hot, too cold, just right! She tried their chairs. She tried their beds. Then the bears came home!",
        
        "The ugly duckling was sad. The other ducks were mean to him. But he grew up to be a beautiful swan! He was the most beautiful bird on the lake.",
        
        "Red Riding Hood walked through the forest. She was bringing food to her grandmother. But a wolf was following her! Luckily a woodsman came and saved them both.",
        
        "Cinderella had to do all the chores. Her stepsisters were so mean. But a fairy godmother helped her go to the ball. The prince fell in love with her glass slipper.",
        
        "The gingerbread man ran away from everyone. Run run as fast as you can! But a clever fox tricked him. The gingerbread man learned to be more careful.",
        
        "Hansel and Gretel got lost in the woods. They found a house made of candy! But a witch lived there. They were very brave and escaped safely.",
        
        "The tortoise and the hare had a race. The hare was very fast but he stopped to rest. The tortoise kept going slowly and steadily. The tortoise won the race!",
        
        "Little Bo Peep lost her sheep. She looked everywhere for them. Finally she found them sleeping by the hill. She was so happy to have them back.",
    ] * 50  # Repeat for more data
    
    return stories


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Scaled ResoFormer Training with BPE Tokenizer")
    print("=" * 70)
    print()
    
    # Load GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary size: {vocab_size:,}")
    except Exception as e:
        print(f"Failed to load GPT-2 tokenizer: {e}")
        print("Install with: pip install transformers")
        return
    
    # Get training data
    print("\nLoading training data...")
    stories = get_stories()
    print(f"Training stories: {len(stories)}")
    
    # Create dataset
    seq_len = 128
    train_dataset = BPEDataset(stories, tokenizer, seq_len=seq_len)
    print(f"Training examples: {len(train_dataset)}")
    print()
    
    # Create model
    config = ResoFormerConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len + 16,
        hidden_dim=384,
        num_layers=6,
        num_heads=6,
        ffn_dim=1536,
        dropout=0.1,
        use_golden_attention=True,
        use_resonance_rotation=True,
        use_coherence_gate=True,
        use_entropy_collapse=False,
    )
    
    model = PyTorchResoFormer(config)
    print(f"Model: {model.num_parameters:,} parameters")
    print(f"Architecture: {config.num_layers} layers, dim={config.hidden_dim}")
    print()
    
    # Train
    trainer = ScaledTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        learning_rate=5e-4,
        batch_size=16,
        num_epochs=15,
        warmup_steps=200,
    )
    
    print("Starting training...")
    print("-" * 70)
    trainer.train(log_interval=50)
    print()
    
    # Generate
    print("=" * 70)
    print("Text Generation")
    print("=" * 70)
    print()
    
    prompts = [
        "Once upon a time",
        "The little dog",
        "There was a",
        "The princess",
        "One day",
    ]
    
    print("--- Temperature 0.7 ---")
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = trainer.generate(prompt, max_length=50, temperature=0.7)
        print(f"Generated: '{generated}'")
    
    print("\n\n--- Temperature 0.3 (more focused) ---")
    for prompt in prompts[:3]:
        print(f"\nPrompt: '{prompt}'")
        generated = trainer.generate(prompt, max_length=40, temperature=0.3)
        print(f"Generated: '{generated}'")
    
    print("\n\nTraining complete!")


if __name__ == "__main__":
    main()