"""
ResonantTrainer: Training Loop with Coherence Monitoring

Provides a complete training pipeline for TrainableResoFormer:
- Numerical gradient computation (pure Python)
- Golden ratio learning rate scheduling
- Coherence-based early stopping
- Entropy tracking for convergence monitoring
- Checkpoint saving/loading

Key Features:
- Cross-entropy loss for language modeling
- Gradient clipping with φ-based threshold
- Learning rate warmup with golden ratio
- Coherence monitoring during training
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
import math
import random
import time
import sys
sys.path.insert(0, '../..')

from tinyaleph.core.constants import PHI, ENTROPY_THRESHOLD
from apps.resoformer.model import TrainableResoFormer, TrainableResoFormerConfig, TrainableTensor
from apps.resoformer.tokenizer import PrimeTokenizer


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Basic training
    epochs: int = 10
    batch_size: int = 8
    seq_len: int = 64
    
    # Learning rate
    learning_rate: float = 1e-3
    lr_warmup_steps: int = 100
    lr_decay: str = "golden"  # "golden", "cosine", "linear"
    min_lr: float = 1e-5
    
    # Gradient
    gradient_clip: float = 1.0 / PHI  # ≈ 0.618
    use_numerical_grad: bool = True
    grad_epsilon: float = 1e-5
    
    # Regularization
    weight_decay: float = 0.01
    
    # Coherence monitoring
    monitor_coherence: bool = True
    coherence_check_interval: int = 10
    early_stop_coherence: float = 0.95
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    checkpoint_path: Optional[str] = None
    
    # Reproducibility
    seed: Optional[int] = None


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    
    step: int = 0
    epoch: int = 0
    loss: float = float('inf')
    perplexity: float = float('inf')
    learning_rate: float = 0.0
    coherence: float = 0.0
    entropy: float = float('inf')
    gradient_norm: float = 0.0
    tokens_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'epoch': self.epoch,
            'loss': self.loss,
            'perplexity': self.perplexity,
            'lr': self.learning_rate,
            'coherence': self.coherence,
            'entropy': self.entropy,
            'grad_norm': self.gradient_norm,
            'tps': self.tokens_per_second,
        }


class ResonantTrainer:
    """
    Trainer for ResoFormer with resonance-aware optimization.
    
    Features:
    - Numerical gradient computation
    - Golden ratio learning rate schedule
    - Coherence-based convergence monitoring
    - Entropy-aware training dynamics
    """
    
    def __init__(self, 
                 model: TrainableResoFormer,
                 tokenizer: PrimeTokenizer,
                 config: Optional[TrainingConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.loss_history: List[float] = []
        self.coherence_history: List[float] = []
        
        # Set seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    def get_learning_rate(self) -> float:
        """Compute learning rate with warmup and decay."""
        base_lr = self.config.learning_rate
        min_lr = self.config.min_lr
        warmup = self.config.lr_warmup_steps
        
        # Warmup
        if self.step < warmup:
            return base_lr * (self.step + 1) / warmup
        
        # Decay
        if self.config.lr_decay == "golden":
            # Golden ratio decay: lr = base * φ^(-step/warmup)
            decay_steps = self.step - warmup
            decay_factor = 1.0 / (PHI ** (decay_steps / warmup / 10))
            return max(min_lr, base_lr * decay_factor)
        
        elif self.config.lr_decay == "cosine":
            # Cosine annealing
            total_steps = self.config.epochs * 1000  # Rough estimate
            progress = min(1.0, (self.step - warmup) / (total_steps - warmup))
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        
        elif self.config.lr_decay == "linear":
            # Linear decay
            total_steps = self.config.epochs * 1000
            progress = min(1.0, (self.step - warmup) / (total_steps - warmup))
            return base_lr * (1 - progress) + min_lr * progress
        
        return base_lr
    
    def cross_entropy_loss(self, logits: List[float], 
                           targets: List[int]) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Model output logits (seq_len * vocab_size)
            targets: Target token indices
            
        Returns:
            Average cross-entropy loss
        """
        vocab_size = self.model.config.vocab_size
        seq_len = len(targets)
        
        total_loss = 0.0
        
        for t in range(seq_len):
            # Extract logits for this position
            start = t * vocab_size
            pos_logits = logits[start:start + vocab_size]
            
            # Softmax
            max_logit = max(pos_logits)
            exp_logits = [math.exp(l - max_logit) for l in pos_logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            
            # Cross-entropy: -log(p[target])
            target = targets[t]
            if 0 <= target < vocab_size:
                prob = max(probs[target], 1e-10)
                total_loss -= math.log(prob)
        
        return total_loss / seq_len
    
    def compute_numerical_gradients(self, 
                                    input_tokens: List[int],
                                    target_tokens: List[int]) -> Dict[str, List[float]]:
        """
        Compute gradients numerically via finite differences.
        
        This is slow but works without autodiff infrastructure.
        
        Args:
            input_tokens: Input sequence
            target_tokens: Target sequence
            
        Returns:
            Dictionary mapping parameter names to gradients
        """
        eps = self.config.grad_epsilon
        gradients = {}
        
        all_params = self.model.get_all_params()
        
        for name, param in all_params.items():
            grad = []
            
            # Sample subset of parameters for efficiency
            n_params = len(param.data)
            if n_params > 100:
                # Stochastic gradient: sample 10% of parameters
                indices = random.sample(range(n_params), max(1, n_params // 10))
            else:
                indices = list(range(n_params))
            
            full_grad = [0.0] * n_params
            
            for i in indices:
                # f(θ + ε)
                original = param.data[i]
                param.data[i] = original + eps
                logits_plus = self.model.forward(input_tokens, training=True)
                loss_plus = self.cross_entropy_loss(logits_plus, target_tokens)
                
                # f(θ - ε)
                param.data[i] = original - eps
                logits_minus = self.model.forward(input_tokens, training=True)
                loss_minus = self.cross_entropy_loss(logits_minus, target_tokens)
                
                # Restore
                param.data[i] = original
                
                # Central difference
                full_grad[i] = (loss_plus - loss_minus) / (2 * eps)
            
            gradients[name] = full_grad
        
        return gradients
    
    def clip_gradients(self, gradients: Dict[str, List[float]]) -> float:
        """
        Clip gradients by global norm.
        
        Returns the gradient norm before clipping.
        """
        # Compute global norm
        global_norm = 0.0
        for grad in gradients.values():
            global_norm += sum(g ** 2 for g in grad)
        global_norm = math.sqrt(global_norm)
        
        # Clip if necessary
        clip = self.config.gradient_clip
        if global_norm > clip:
            scale = clip / (global_norm + 1e-10)
            for name in gradients:
                gradients[name] = [g * scale for g in gradients[name]]
        
        return global_norm
    
    def update_parameters(self, gradients: Dict[str, List[float]], lr: float):
        """Apply gradient updates with optional weight decay."""
        all_params = self.model.get_all_params()
        weight_decay = self.config.weight_decay
        
        for name, param in all_params.items():
            if name in gradients:
                grad = gradients[name]
                for i in range(len(param.data)):
                    # Weight decay
                    if weight_decay > 0:
                        param.data[i] *= (1 - lr * weight_decay)
                    
                    # Gradient descent
                    if i < len(grad):
                        param.data[i] -= lr * grad[i]
    
    def compute_coherence(self, logits: List[float]) -> float:
        """
        Compute coherence from output distribution.
        
        Coherence is inverse of entropy, normalized.
        """
        vocab_size = self.model.config.vocab_size
        
        # Average probabilities across sequence
        avg_probs = [0.0] * vocab_size
        n_positions = len(logits) // vocab_size
        
        for t in range(n_positions):
            start = t * vocab_size
            pos_logits = logits[start:start + vocab_size]
            
            # Softmax
            max_logit = max(pos_logits)
            exp_logits = [math.exp(l - max_logit) for l in pos_logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            
            for i in range(vocab_size):
                avg_probs[i] += probs[i] / n_positions
        
        # Entropy
        entropy = 0.0
        for p in avg_probs:
            if p > 1e-10:
                entropy -= p * math.log(p)
        
        # Max entropy
        max_entropy = math.log(vocab_size)
        
        # Coherence = 1 - normalized_entropy
        coherence = 1.0 - (entropy / max_entropy)
        
        return coherence
    
    def prepare_batches(self, text: str) -> List[Tuple[List[int], List[int]]]:
        """
        Prepare training batches from text.
        
        Returns list of (input, target) tuples.
        """
        # Encode full text
        tokens = self.tokenizer.encode(text)
        
        batches = []
        seq_len = self.config.seq_len
        
        # Create sequences
        for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            input_seq = tokens[i:i + seq_len]
            target_seq = tokens[i + 1:i + seq_len + 1]
            
            if len(input_seq) == seq_len and len(target_seq) == seq_len:
                batches.append((input_seq, target_seq))
        
        return batches
    
    def train_step(self, 
                   input_tokens: List[int],
                   target_tokens: List[int]) -> TrainingMetrics:
        """
        Perform single training step.
        
        Returns metrics for this step.
        """
        start_time = time.time()
        
        # Forward pass
        logits = self.model.forward(input_tokens, training=True)
        
        # Compute loss
        loss = self.cross_entropy_loss(logits, target_tokens)
        
        # Compute gradients
        gradients = self.compute_numerical_gradients(input_tokens, target_tokens)
        
        # Clip gradients
        grad_norm = self.clip_gradients(gradients)
        
        # Get learning rate
        lr = self.get_learning_rate()
        
        # Update parameters
        self.update_parameters(gradients, lr)
        
        # Compute metrics
        coherence = self.compute_coherence(logits) if self.config.monitor_coherence else 0.0
        perplexity = math.exp(min(loss, 100))  # Cap for numerical stability
        
        elapsed = time.time() - start_time
        tokens_per_sec = len(input_tokens) / elapsed if elapsed > 0 else 0
        
        self.step += 1
        
        metrics = TrainingMetrics(
            step=self.step,
            epoch=self.epoch,
            loss=loss,
            perplexity=perplexity,
            learning_rate=lr,
            coherence=coherence,
            entropy=-math.log(max(coherence, 1e-10)),
            gradient_norm=grad_norm,
            tokens_per_second=tokens_per_sec,
        )
        
        self.loss_history.append(loss)
        self.coherence_history.append(coherence)
        
        return metrics
    
    def train(self, 
              corpus: str,
              callback: Optional[Callable[[TrainingMetrics], None]] = None) -> List[TrainingMetrics]:
        """
        Train model on corpus.
        
        Args:
            corpus: Training text
            callback: Optional callback called after each step
            
        Returns:
            List of training metrics
        """
        print(f"Training ResoFormer on {len(corpus):,} characters")
        print(f"Model parameters: {self.model.num_parameters():,}")
        print(f"Config: epochs={self.config.epochs}, seq_len={self.config.seq_len}")
        print()
        
        # Prepare batches
        batches = self.prepare_batches(corpus)
        print(f"Created {len(batches)} training sequences")
        print()
        
        all_metrics = []
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Shuffle batches
            random.shuffle(batches)
            
            for batch_idx, (input_seq, target_seq) in enumerate(batches):
                # Train step
                metrics = self.train_step(input_seq, target_seq)
                all_metrics.append(metrics)
                epoch_losses.append(metrics.loss)
                
                # Callback
                if callback is not None:
                    callback(metrics)
                
                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = sum(epoch_losses[-self.config.log_interval:]) / min(len(epoch_losses), self.config.log_interval)
                    print(f"Step {self.step:5d} | Epoch {epoch+1}/{self.config.epochs} | "
                          f"Loss {avg_loss:.4f} | PPL {math.exp(min(avg_loss, 100)):.2f} | "
                          f"LR {metrics.learning_rate:.6f} | Coh {metrics.coherence:.4f}")
                
                # Checkpoint
                if self.config.checkpoint_path and self.step % self.config.checkpoint_interval == 0:
                    path = f"{self.config.checkpoint_path}_step{self.step}.json"
                    self.model.save(path)
                    print(f"  Saved checkpoint to {path}")
                
                # Early stopping on coherence
                if (self.config.monitor_coherence and 
                    metrics.coherence > self.config.early_stop_coherence):
                    print(f"\nEarly stopping: coherence {metrics.coherence:.4f} > {self.config.early_stop_coherence}")
                    return all_metrics
            
            # Epoch summary
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\n=== Epoch {epoch+1} Complete ===")
            print(f"Average Loss: {epoch_avg_loss:.4f}")
            print(f"Perplexity: {math.exp(min(epoch_avg_loss, 100)):.2f}")
            print()
            
            # Update best loss
            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss
                if self.config.checkpoint_path:
                    path = f"{self.config.checkpoint_path}_best.json"
                    self.model.save(path)
                    print(f"New best model saved to {path}")
        
        print("Training complete!")
        return all_metrics
    
    def evaluate(self, text: str) -> Dict[str, float]:
        """
        Evaluate model on text.
        
        Returns evaluation metrics.
        """
        batches = self.prepare_batches(text)
        
        total_loss = 0.0
        total_coherence = 0.0
        
        for input_seq, target_seq in batches:
            logits = self.model.forward(input_seq, training=False)
            loss = self.cross_entropy_loss(logits, target_seq)
            coherence = self.compute_coherence(logits)
            
            total_loss += loss
            total_coherence += coherence
        
        n_batches = len(batches)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_coherence = total_coherence / n_batches if n_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'perplexity': math.exp(min(avg_loss, 100)),
            'coherence': avg_coherence,
            'entropy': -math.log(max(avg_coherence, 1e-10)),
        }


def create_trainer(model: TrainableResoFormer,
                   tokenizer: PrimeTokenizer,
                   **kwargs) -> ResonantTrainer:
    """Create trainer with custom configuration."""
    config = TrainingConfig(**kwargs)
    return ResonantTrainer(model, tokenizer, config)