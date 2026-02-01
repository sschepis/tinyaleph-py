"""
ResonantGenerator: Text Generation with Entropy-Aware Sampling

Provides text generation capabilities for TrainableResoFormer:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Entropy-aware adaptive sampling
- Coherence-based early stopping

Key Features:
- Golden ratio temperature scheduling
- Entropy monitoring during generation
- Coherence gating for output quality
- Prime-resonant token selection
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
import math
import random
import sys
sys.path.insert(0, '../..')

from tinyaleph.core.constants import PHI, ENTROPY_THRESHOLD
from apps.resoformer.model import TrainableResoFormer
from apps.resoformer.tokenizer import PrimeTokenizer


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    # Length
    max_length: int = 256
    min_length: int = 10
    
    # Sampling strategy
    strategy: str = "entropy_aware"  # "greedy", "temperature", "top_k", "top_p", "entropy_aware"
    
    # Temperature
    temperature: float = 1.0
    temperature_schedule: str = "constant"  # "constant", "golden_decay", "adaptive"
    min_temperature: float = 0.1
    
    # Top-k
    top_k: int = 50
    
    # Top-p (nucleus)
    top_p: float = 0.9
    
    # Entropy-aware settings
    target_entropy: float = ENTROPY_THRESHOLD
    entropy_weight: float = 0.5
    
    # Coherence
    coherence_threshold: float = 0.8
    stop_on_coherence: bool = False
    
    # Special tokens
    stop_tokens: List[str] = field(default_factory=lambda: ["\n\n", ".", "!", "?"])
    
    # Repetition penalty
    repetition_penalty: float = 1.1
    repetition_window: int = 32
    
    # Reproducibility
    seed: Optional[int] = None


class ResonantGenerator:
    """
    Text generator with entropy-aware sampling.
    
    Uses TinyAleph's prime-resonant foundations for:
    - Golden ratio temperature scheduling
    - Entropy-based adaptive sampling
    - Coherence monitoring
    """
    
    def __init__(self,
                 model: TrainableResoFormer,
                 tokenizer: PrimeTokenizer,
                 config: Optional[GenerationConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    def get_temperature(self, step: int, max_steps: int) -> float:
        """Compute temperature with optional scheduling."""
        base_temp = self.config.temperature
        min_temp = self.config.min_temperature
        
        if self.config.temperature_schedule == "constant":
            return base_temp
        
        elif self.config.temperature_schedule == "golden_decay":
            # Golden ratio decay
            progress = step / max_steps
            decay = 1.0 / (PHI ** (progress * 2))
            return max(min_temp, base_temp * decay)
        
        elif self.config.temperature_schedule == "adaptive":
            # Adaptive based on recent entropy
            return base_temp
        
        return base_temp
    
    def apply_repetition_penalty(self, 
                                  logits: List[float],
                                  generated_tokens: List[int]) -> List[float]:
        """Apply repetition penalty to logits."""
        if self.config.repetition_penalty == 1.0:
            return logits
        
        # Get recent tokens
        window = self.config.repetition_window
        recent = set(generated_tokens[-window:])
        
        # Apply penalty
        penalized = list(logits)
        for token_idx in recent:
            if 0 <= token_idx < len(penalized):
                if penalized[token_idx] > 0:
                    penalized[token_idx] /= self.config.repetition_penalty
                else:
                    penalized[token_idx] *= self.config.repetition_penalty
        
        return penalized
    
    def sample_greedy(self, logits: List[float]) -> int:
        """Greedy decoding: select highest probability token."""
        return logits.index(max(logits))
    
    def sample_temperature(self, logits: List[float], temperature: float) -> int:
        """Temperature sampling."""
        if temperature <= 0:
            return self.sample_greedy(logits)
        
        # Apply temperature
        scaled = [l / temperature for l in logits]
        
        # Softmax
        max_logit = max(scaled)
        exp_logits = [math.exp(l - max_logit) for l in scaled]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        # Sample
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return i
        
        return len(probs) - 1
    
    def sample_top_k(self, logits: List[float], k: int, temperature: float) -> int:
        """Top-k sampling: sample from k highest probability tokens."""
        if k <= 0 or k >= len(logits):
            return self.sample_temperature(logits, temperature)
        
        # Get top-k indices
        indexed = list(enumerate(logits))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_k = indexed[:k]
        
        # Apply temperature to top-k
        if temperature <= 0:
            return top_k[0][0]
        
        top_k_logits = [l / temperature for _, l in top_k]
        max_logit = max(top_k_logits)
        exp_logits = [math.exp(l - max_logit) for l in top_k_logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        # Sample
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return top_k[i][0]
        
        return top_k[-1][0]
    
    def sample_top_p(self, logits: List[float], p: float, temperature: float) -> int:
        """Top-p (nucleus) sampling: sample from smallest set with cumulative prob >= p."""
        if p >= 1.0:
            return self.sample_temperature(logits, temperature)
        
        # Apply temperature and get probabilities
        if temperature > 0:
            scaled = [l / temperature for l in logits]
        else:
            scaled = logits
        
        max_logit = max(scaled)
        exp_logits = [math.exp(l - max_logit) for l in scaled]
        sum_exp = sum(exp_logits)
        probs = [(i, e / sum_exp) for i, e in enumerate(exp_logits)]
        
        # Sort by probability descending
        probs.sort(key=lambda x: x[1], reverse=True)
        
        # Find nucleus
        cumsum = 0.0
        nucleus = []
        for idx, prob in probs:
            cumsum += prob
            nucleus.append((idx, prob))
            if cumsum >= p:
                break
        
        # Renormalize
        nucleus_sum = sum(prob for _, prob in nucleus)
        normalized = [(idx, prob / nucleus_sum) for idx, prob in nucleus]
        
        # Sample
        r = random.random()
        cumsum = 0.0
        for idx, prob in normalized:
            cumsum += prob
            if r < cumsum:
                return idx
        
        return normalized[-1][0]
    
    def sample_entropy_aware(self, 
                             logits: List[float],
                             temperature: float,
                             current_entropy: float) -> int:
        """
        Entropy-aware sampling.
        
        Adjusts temperature based on current entropy vs target.
        High entropy -> lower temperature (more focused)
        Low entropy -> higher temperature (more exploration)
        """
        target = self.config.target_entropy
        weight = self.config.entropy_weight
        
        # Compute entropy of current distribution
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * math.log(p)
        
        # Adjust temperature
        entropy_ratio = entropy / max(target, 1e-10)
        if entropy_ratio > 1.0:
            # High entropy: reduce temperature
            adjusted_temp = temperature * (1.0 / (PHI * entropy_ratio))
        else:
            # Low entropy: increase temperature slightly
            adjusted_temp = temperature * (1.0 + weight * (1.0 - entropy_ratio))
        
        adjusted_temp = max(self.config.min_temperature, adjusted_temp)
        
        # Use top-p with adjusted temperature
        return self.sample_top_p(logits, self.config.top_p, adjusted_temp)
    
    def compute_entropy(self, logits: List[float]) -> float:
        """Compute entropy of logit distribution."""
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * math.log(p)
        
        return entropy
    
    def compute_coherence(self, logits: List[float]) -> float:
        """Compute coherence (inverse of normalized entropy)."""
        entropy = self.compute_entropy(logits)
        max_entropy = math.log(len(logits))
        return 1.0 - (entropy / max_entropy)
    
    def sample(self, logits: List[float], step: int, 
               max_steps: int, current_entropy: float) -> int:
        """Sample next token based on strategy."""
        temperature = self.get_temperature(step, max_steps)
        strategy = self.config.strategy
        
        if strategy == "greedy":
            return self.sample_greedy(logits)
        
        elif strategy == "temperature":
            return self.sample_temperature(logits, temperature)
        
        elif strategy == "top_k":
            return self.sample_top_k(logits, self.config.top_k, temperature)
        
        elif strategy == "top_p":
            return self.sample_top_p(logits, self.config.top_p, temperature)
        
        elif strategy == "entropy_aware":
            return self.sample_entropy_aware(logits, temperature, current_entropy)
        
        else:
            return self.sample_temperature(logits, temperature)
    
    def generate(self, 
                 prompt: str = "",
                 max_length: Optional[int] = None,
                 callback: Optional[Callable[[str, int, float], None]] = None) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Starting text
            max_length: Maximum tokens to generate
            callback: Optional callback(text, step, coherence)
            
        Returns:
            Generated text
        """
        max_length = max_length or self.config.max_length
        
        # Encode prompt
        if prompt:
            tokens = self.tokenizer.encode(prompt)
        else:
            # Start with BOS
            tokens = [self.tokenizer.special_tokens.get("<BOS>", 5)]
        
        generated_tokens = []
        current_entropy = ENTROPY_THRESHOLD
        
        vocab_size = self.model.config.vocab_size
        
        for step in range(max_length):
            # Prepare input (use last seq_len tokens)
            input_tokens = (tokens + generated_tokens)[-self.model.config.max_seq_len:]
            
            # Forward pass
            all_logits = self.model.forward(input_tokens, training=False)
            
            # Get logits for last position
            last_pos = len(input_tokens) - 1
            start = last_pos * vocab_size
            
            # Handle both PyTorch tensors and TinyAleph Tensors
            try:
                import torch
                if isinstance(all_logits, torch.Tensor):
                    # PyTorch tensor - convert to list
                    logits = all_logits[start:start + vocab_size].detach().cpu().tolist()
                else:
                    # TinyAleph Tensor - access data directly
                    logits = all_logits.data[start:start + vocab_size]
            except ImportError:
                # No PyTorch - use TinyAleph format
                logits = all_logits.data[start:start + vocab_size]
            
            # Apply repetition penalty
            logits = self.apply_repetition_penalty(logits, generated_tokens)
            
            # Compute entropy
            current_entropy = self.compute_entropy(logits)
            coherence = self.compute_coherence(logits)
            
            # Sample next token
            next_token = self.sample(logits, step, max_length, current_entropy)
            generated_tokens.append(next_token)
            
            # Callback
            if callback is not None:
                current_text = self.tokenizer.decode(generated_tokens)
                callback(current_text, step, coherence)
            
            # Check stopping conditions
            if step >= self.config.min_length:
                # Stop on EOS
                eos_prime = self.tokenizer.special_tokens.get("<EOS>", 7)
                if next_token == self.tokenizer.prime_to_index(eos_prime):
                    break
                
                # Stop on coherence
                if self.config.stop_on_coherence and coherence > self.config.coherence_threshold:
                    break
                
                # Stop on stop tokens
                current_text = self.tokenizer.decode(generated_tokens)
                for stop in self.config.stop_tokens:
                    if current_text.endswith(stop):
                        break
        
        # Decode
        return self.tokenizer.decode(generated_tokens)
    
    def generate_batch(self, 
                       prompts: List[str],
                       max_length: Optional[int] = None) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, max_length) for prompt in prompts]
    
    def sample_continuations(self,
                             prompt: str,
                             n: int = 5,
                             max_length: int = 50) -> List[str]:
        """
        Generate multiple continuations for the same prompt.
        
        Useful for exploring the model's distribution.
        """
        continuations = []
        for i in range(n):
            # Use different seed for each sample
            if self.config.seed is not None:
                random.seed(self.config.seed + i)
            
            text = self.generate(prompt, max_length)
            continuations.append(text)
        
        return continuations
    
    def interactive_generate(self, 
                             prompt: str = "",
                             stream: bool = True) -> str:
        """
        Interactive generation with streaming output.
        
        Prints tokens as they are generated.
        """
        def print_callback(text: str, step: int, coherence: float):
            if stream:
                # Clear line and print current text
                print(f"\r{text}", end="", flush=True)
        
        result = self.generate(prompt, callback=print_callback if stream else None)
        
        if stream:
            print()  # New line after streaming
        
        return result


def create_generator(model: TrainableResoFormer,
                     tokenizer: PrimeTokenizer,
                     **kwargs) -> ResonantGenerator:
    """Create generator with custom configuration."""
    config = GenerationConfig(**kwargs)
    return ResonantGenerator(model, tokenizer, config)