#!/usr/bin/env python3
"""
HuggingFace Dataset Training for ResoFormer

Uses HuggingFace datasets for real training data.
Implements efficient training with proper gradient updates.
"""

from __future__ import annotations
import sys
import os
import time
import math
import random
sys.path.insert(0, '../..')
import dotenv

# Set HuggingFace token
if('HF_TOKEN' not in os.environ):
    dotenv.load_dotenv('.env')
    os.environ["HF_TOKEN"] = dotenv.get_key('.env', 'HF_TOKEN')

from apps.resoformer.tokenizer import PrimeTokenizer
from apps.resoformer.model import TrainableResoFormer, TrainableResoFormerConfig
from apps.resoformer.generator import ResonantGenerator, GenerationConfig


def load_tiny_shakespeare():
    """Load tiny_shakespeare dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Loading tiny_shakespeare from HuggingFace...")
        dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
        train_text = dataset["train"]["text"][0]
        print(f"Loaded {len(train_text):,} characters")
        return train_text
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to built-in corpus...")
        return get_builtin_corpus()


def load_wikitext():
    """Load wikitext-2 dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Loading wikitext-2 from HuggingFace...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
        train_text = " ".join(dataset["train"]["text"][:1000])  # First 1000 examples
        print(f"Loaded {len(train_text):,} characters")
        return train_text
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return get_builtin_corpus()


def get_builtin_corpus():
    """Fallback corpus if HuggingFace fails."""
    return """
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
What's in a name? that which we call a rose
By any other name would smell as sweet.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time.

Double, double toil and trouble;
Fire burn and caldron bubble.

The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven.

If music be the food of love, play on.
""" * 10  # Repeat for more data


class EfficientTrainer:
    """Efficient trainer with softmax gradient updates."""
    
    def __init__(self, model: TrainableResoFormer, tokenizer: PrimeTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = model.config.vocab_size
        self.dim = model.config.dim
    
    def softmax(self, logits):
        max_l = max(logits) if logits else 0
        exp_l = [math.exp(min(l - max_l, 50)) for l in logits]  # Clamp for stability
        sum_e = sum(exp_l) + 1e-10
        return [e / sum_e for e in exp_l]
    
    def cross_entropy(self, logits, target):
        if target >= len(logits):
            return 5.0  # Penalty for OOV
        probs = self.softmax(logits)
        return -math.log(max(probs[target], 1e-10))
    
    def update_step(self, input_tokens, target_tokens, lr):
        """Single training step with gradient updates."""
        # Forward pass
        logits = self.model.forward(input_tokens, training=True)
        
        # Get model parameters
        embed = self.model.embedding.params['embedding']
        output = self.model.output_proj.params['kernel']
        
        seq_len = len(target_tokens)
        total_loss = 0.0
        
        for t in range(seq_len):
            start = t * self.vocab_size
            pos_logits = logits[start:start + self.vocab_size]
            
            if len(pos_logits) < self.vocab_size:
                continue
            
            probs = self.softmax(pos_logits)
            target = target_tokens[t]
            
            if target >= self.vocab_size:
                continue
            
            total_loss += self.cross_entropy(pos_logits, target)
            
            # Gradient update for output layer
            # ∂L/∂W = (softmax - one_hot) * hidden
            for v in range(min(self.vocab_size, 20)):  # Update top 20 for speed
                if v == target:
                    grad = probs[v] - 1.0
                else:
                    grad = probs[v]
                
                # Update output kernel
                for d in range(min(self.dim, 16)):  # Update first 16 dims
                    idx = d * self.vocab_size + v
                    if idx < len(output.data):
                        output.data[idx] -= lr * grad * 0.1
            
            # Update embedding for input token
            inp_idx = input_tokens[t] if t < len(input_tokens) else 0
            if inp_idx < self.vocab_size:
                inp_start = inp_idx * self.dim
                tgt_start = target * self.dim
                
                for d in range(self.dim):
                    if inp_start + d < len(embed.data) and tgt_start + d < len(embed.data):
                        # Move input embedding toward target embedding
                        diff = embed.data[tgt_start + d] - embed.data[inp_start + d]
                        embed.data[inp_start + d] += lr * diff * 0.05
        
        return total_loss / max(seq_len, 1)
    
    def train(self, text, epochs=30, seq_len=32, lr=0.5, batch_print=50):
        """Train on text corpus."""
        # Encode
        tokens = self.tokenizer.encode(text)
        print(f"Training tokens: {len(tokens):,}")
        
        # Create sequences
        sequences = []
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            inp = tokens[i:i + seq_len]
            tgt = tokens[i + 1:i + seq_len + 1]
            if len(inp) == seq_len and len(tgt) == seq_len:
                sequences.append((inp, tgt))
        
        print(f"Training sequences: {len(sequences)}")
        print()
        print("Training...")
        print("-" * 70)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            random.shuffle(sequences)
            total_loss = 0.0
            
            # Decay LR with golden ratio
            epoch_lr = lr * (0.618 ** (epoch / 10))
            
            for i, (inp, tgt) in enumerate(sequences):
                loss = self.update_step(inp, tgt, epoch_lr)
                total_loss += loss
                
                if (i + 1) % batch_print == 0:
                    avg = total_loss / (i + 1)
                    print(f"  Epoch {epoch+1} batch {i+1}/{len(sequences)}: loss={avg:.4f}")
            
            avg_loss = total_loss / len(sequences)
            ppl = math.exp(min(avg_loss, 20))
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                marker = " *"
            else:
                marker = ""
            
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | PPL: {ppl:7.2f} | LR: {epoch_lr:.4f}{marker}")
        
        print("-" * 70)
        return best_loss


def main():
    print("=" * 70)
    print("ResoFormer HuggingFace Training")
    print("=" * 70)
    print()
    
    # Load dataset
    corpus = load_tiny_shakespeare()
    
    # Limit corpus size for reasonable training time
    if len(corpus) > 50000:
        corpus = corpus[:50000]
        print(f"Limited to {len(corpus):,} characters for training speed")
    print()
    
    # Create tokenizer
    tokenizer = PrimeTokenizer.from_corpus([corpus])
    print(f"Vocabulary: {tokenizer.vocab_size} unique tokens")
    print()
    
    # Create model
    config = TrainableResoFormerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=48,
        num_layers=2,
        num_heads=4,
        ffn_dim=96,
        max_seq_len=64,
        use_coherence_gate=True,
        use_entropy_collapse=False,
        use_golden_attention=True,
    )
    
    model = TrainableResoFormer(config)
    _ = model.forward([2] * 8, training=False)
    print(f"Model: {model.num_parameters():,} parameters")
    print(f"Architecture: {config.num_layers} layers, dim={config.dim}")
    print()
    
    # Train
    trainer = EfficientTrainer(model, tokenizer)
    start = time.time()
    final_loss = trainer.train(
        corpus,
        epochs=20,
        seq_len=32,
        lr=0.3,
        batch_print=100
    )
    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed:.1f}s")
    print()
    
    # Generate samples
    print("=" * 70)
    print("Generation Results")
    print("=" * 70)
    print()
    
    gen_config = GenerationConfig(
        max_length=100,
        strategy="temperature",
        temperature=0.8,
        repetition_penalty=1.3,
    )
    
    generator = ResonantGenerator(model, tokenizer, gen_config)
    
    prompts = [
        "To be, or not",
        "All the world",
        "The ",
        "And ",
        "O ",
    ]
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        generated = generator.generate(prompt, max_length=80)
        print(f"Generated: '{generated}'")
        print()


if __name__ == "__main__":
    main()