"""
Stability-Aware Inference for LLM Fusion.

Provides generation utilities with real-time stability monitoring,
adaptive temperature control, and coherence-based stopping.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GenerationConfig, StabilityConfig


@dataclass
class StabilityMetrics:
    """Metrics tracking generation stability."""
    entropy_history: List[float] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)
    lyapunov_estimate: float = 0.0
    kuramoto_order: float = 0.0
    is_stable: bool = True
    suggested_temperature: float = 1.0
    
    def update(
        self,
        entropy: float,
        coherence: float,
        kuramoto_order: float = 0.0,
        lyapunov_threshold: float = 0.1,
        coherence_threshold: float = 0.3,
    ):
        """Update metrics with new values."""
        self.entropy_history.append(entropy)
        self.coherence_history.append(coherence)
        self.kuramoto_order = kuramoto_order
        
        # Estimate Lyapunov exponent from entropy changes
        if len(self.entropy_history) >= 2:
            delta = self.entropy_history[-1] - self.entropy_history[-2]
            # Exponential moving average
            self.lyapunov_estimate = 0.9 * self.lyapunov_estimate + 0.1 * delta
        
        # Stability based on Lyapunov and coherence
        self.is_stable = self.lyapunov_estimate < lyapunov_threshold and coherence > coherence_threshold
        
        # Suggest temperature based on stability
        if self.is_stable:
            self.suggested_temperature = 0.7 + 0.3 * coherence
        else:
            # Reduce temperature when unstable
            self.suggested_temperature = max(0.3, 0.7 - self.lyapunov_estimate)
    
    def divergence_detected(self, threshold: float = 0.1) -> bool:
        """Check if generation is diverging."""
        return self.lyapunov_estimate > threshold
    
    @property
    def mean_coherence(self) -> float:
        if not self.coherence_history:
            return 0.0
        return sum(self.coherence_history) / len(self.coherence_history)
    
    @property
    def mean_entropy(self) -> float:
        if not self.entropy_history:
            return 0.0
        return sum(self.entropy_history) / len(self.entropy_history)


@dataclass
class GenerationResult:
    """Result from resonance-aware generation."""
    generated_ids: torch.Tensor
    generated_text: Optional[str] = None
    stability_metrics: Optional[StabilityMetrics] = None
    layer_metrics: Optional[Dict[int, Dict[str, float]]] = None
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "generated_ids": self.generated_ids.tolist(),
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }
        if self.generated_text:
            result["generated_text"] = self.generated_text
        if self.stability_metrics:
            result["mean_coherence"] = self.stability_metrics.mean_coherence
            result["mean_entropy"] = self.stability_metrics.mean_entropy
            result["lyapunov_estimate"] = self.stability_metrics.lyapunov_estimate
            result["final_stability"] = self.stability_metrics.is_stable
        return result


class ResonanceGenerator:
    """
    Stability-aware text generator for resonance-wrapped models.
    
    Features:
    - Real-time stability monitoring via Lyapunov estimates
    - Adaptive temperature based on coherence
    - Early stopping on divergence detection
    - Coherence-weighted sampling
    
    Args:
        model: ResonanceWrapper or compatible model
        tokenizer: Tokenizer for encoding/decoding
        config: GenerationConfig
        stability_config: StabilityConfig for monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[GenerationConfig] = None,
        stability_config: Optional[StabilityConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.stability_config = stability_config or StabilityConfig()
        
        # Check if model has metrics
        self.has_metrics = hasattr(model, "get_average_metrics")
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_stability: bool = True,
        callback: Optional[Callable[[str, StabilityMetrics], bool]] = None,
    ) -> GenerationResult:
        """
        Generate text with stability monitoring.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (auto-adjusted if config.auto_temperature)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            return_stability: Whether to track stability metrics
            callback: Optional function called each step; return False to stop
            
        Returns:
            GenerationResult with generated text and metrics
        """
        # Get config values or use provided
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        
        # Set seed if specified
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)
        elif hasattr(self.model, "base_model"):
            device = next(self.model.base_model.parameters()).device
            input_ids = input_ids.to(device)
        
        # Initialize tracking
        stability = StabilityMetrics() if return_stability else None
        generated = input_ids.clone()
        stopped_early = False
        stop_reason = None
        
        self.model.eval()
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass
                outputs = self.model(input_ids=generated)
                
                # Get logits
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                # Get last token logits
                next_logits = logits[:, -1, :]  # (batch, vocab)
                
                # Get stability metrics
                if return_stability and self.has_metrics:
                    metrics = self.model.get_average_metrics()
                    entropy = metrics.get("prime_entropy", 0.0)
                    coherence = metrics.get("gate_coherence", metrics.get("prime_coherence", 0.5))
                    order = metrics.get("kuramoto_order", 0.0)
                    
                    stability.update(
                        entropy,
                        coherence,
                        order,
                        lyapunov_threshold=self.stability_config.lyapunov_threshold,
                        coherence_threshold=self.stability_config.coherence_threshold,
                    )
                    
                    # Check for divergence
                    if self.config.stop_on_instability:
                        if stability.divergence_detected(self.stability_config.lyapunov_threshold):
                            stopped_early = True
                            stop_reason = "divergence_detected"
                            break
                    
                    # Adaptive temperature
                    if self.config.auto_temperature:
                        temperature = self._adjust_temperature(
                            temperature,
                            stability,
                        )
                
                # Apply temperature
                next_logits = next_logits / temperature
                
                # Apply repetition penalty
                if self.config.repetition_penalty != 1.0:
                    next_logits = self._apply_repetition_penalty(
                        next_logits,
                        generated,
                        self.config.repetition_penalty,
                    )
                
                # Sample next token
                next_token = self._sample_token(next_logits, top_k, top_p)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Callback
                if callback is not None:
                    partial_text = self.tokenizer.decode(generated[0])
                    if not callback(partial_text, stability):
                        stopped_early = True
                        stop_reason = "callback_stop"
                        break
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode output
        generated_text = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )
        
        # Get layer metrics
        layer_metrics = None
        if self.has_metrics:
            layer_metrics = self.model.get_metrics()
        
        return GenerationResult(
            generated_ids=generated,
            generated_text=generated_text,
            stability_metrics=stability,
            layer_metrics=layer_metrics,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )
    
    def _adjust_temperature(
        self,
        base_temperature: float,
        stability: StabilityMetrics,
    ) -> float:
        """Adjust temperature based on stability metrics."""
        # Use suggested temperature with smoothing
        suggested = stability.suggested_temperature
        
        # Clamp to configured range
        suggested = max(
            self.stability_config.min_temperature,
            min(self.stability_config.max_temperature, suggested)
        )
        
        # Smooth transition
        return 0.7 * base_temperature + 0.3 * suggested
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        # Get unique tokens in generated sequence
        for token_id in set(generated[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample next token with top-k and nucleus sampling."""
        # Top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def beam_search(
        self,
        prompt: str,
        num_beams: int = 4,
        max_length: Optional[int] = None,
        length_penalty: float = 1.0,
        coherence_bonus: float = 0.1,
    ) -> GenerationResult:
        """
        Beam search with coherence-based scoring.
        
        Each beam's score is augmented by coherence metrics:
            score = log_prob + coherence_bonus * mean_coherence
            
        Args:
            prompt: Input prompt
            num_beams: Number of beams
            max_length: Maximum generation length
            length_penalty: Length penalty for beam scores
            coherence_bonus: Weight for coherence in beam scoring
            
        Returns:
            GenerationResult with best beam
        """
        max_length = max_length or self.config.max_length
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(self.model, "base_model"):
            device = next(self.model.base_model.parameters()).device
            input_ids = input_ids.to(device)
        
        batch_size = input_ids.shape[0]
        
        # Initialize beams
        beams = [(input_ids, 0.0, StabilityMetrics())]  # (ids, score, stability)
        
        self.model.eval()
        
        with torch.no_grad():
            for step in range(max_length):
                all_candidates = []
                
                for beam_ids, beam_score, beam_stability in beams:
                    # Check for EOS
                    if beam_ids[0, -1].item() == self.tokenizer.eos_token_id:
                        all_candidates.append((beam_ids, beam_score, beam_stability))
                        continue
                    
                    # Forward pass
                    outputs = self.model(input_ids=beam_ids)
                    
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    else:
                        logits = outputs[0]
                    
                    next_logits = logits[:, -1, :]
                    log_probs = F.log_softmax(next_logits, dim=-1)
                    
                    # Get top-k candidates
                    top_log_probs, top_indices = torch.topk(log_probs[0], num_beams)
                    
                    # Get coherence for scoring
                    coherence = 0.5
                    if self.has_metrics:
                        metrics = self.model.get_average_metrics()
                        coherence = metrics.get("gate_coherence", 0.5)
                    
                    for log_prob, token_id in zip(top_log_probs, top_indices):
                        new_ids = torch.cat([
                            beam_ids,
                            token_id.unsqueeze(0).unsqueeze(0)
                        ], dim=-1)
                        
                        # Score with coherence bonus
                        new_score = beam_score + log_prob.item()
                        new_score += coherence_bonus * coherence
                        
                        # Length penalty
                        length = new_ids.shape[1]
                        new_score = new_score / (length ** length_penalty)
                        
                        # Update stability
                        new_stability = StabilityMetrics()
                        new_stability.coherence_history = beam_stability.coherence_history.copy()
                        new_stability.coherence_history.append(coherence)
                        
                        all_candidates.append((new_ids, new_score, new_stability))
                
                # Select top beams
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                beams = all_candidates[:num_beams]
                
                # Check if all beams ended
                all_ended = all(
                    b[0][0, -1].item() == self.tokenizer.eos_token_id
                    for b in beams
                )
                if all_ended:
                    break
        
        # Return best beam
        best_ids, best_score, best_stability = beams[0]
        generated_text = self.tokenizer.decode(best_ids[0], skip_special_tokens=True)
        
        return GenerationResult(
            generated_ids=best_ids,
            generated_text=generated_text,
            stability_metrics=best_stability,
        )


def coherence_weighted_sample(
    logits: torch.Tensor,
    coherence: float,
    base_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample with temperature weighted by coherence.
    
    High coherence → lower temperature → more deterministic
    Low coherence → higher temperature → more random
    
    Args:
        logits: (batch, vocab) logits
        coherence: Coherence value in [0, 1]
        base_temperature: Base temperature
        
    Returns:
        Sampled token IDs
    """
    # Temperature inversely proportional to coherence
    temperature = base_temperature * (2.0 - coherence)
    temperature = max(0.1, min(2.0, temperature))
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Sample
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
