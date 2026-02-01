"""
PyTorch ResoFormer: TinyAleph-Inspired Transformer on PyTorch

Integrates TinyAleph's mathematical concepts with PyTorch's autodiff:
- Prime-indexed tokenization
- Golden ratio attention weighting
- Resonance phase rotations
- Coherence gating mechanisms
- Entropy collapse (VQ-VAE style)

This provides real GPU acceleration and proper gradient computation.
"""

from __future__ import annotations
import math
import sys
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, '../..')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

# Import TinyAleph constants
from tinyaleph.core.constants import PHI, COHERENCE_THRESHOLD
from tinyaleph.core.primes import nth_prime, prime_index


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ResoFormerConfig:
    """Configuration for PyTorch ResoFormer."""
    
    # Vocabulary
    vocab_size: int = 32000
    max_seq_len: int = 512
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 1024
    
    # Regularization
    dropout: float = 0.1
    
    # TinyAleph features
    use_golden_attention: bool = True
    use_resonance_rotation: bool = True
    use_coherence_gate: bool = True
    use_entropy_collapse: bool = False
    
    # Coherence/Entropy
    coherence_threshold: float = COHERENCE_THRESHOLD
    num_attractors: int = 64
    
    # Training
    use_flash_attention: bool = False  # Set True if flash-attn installed


# =============================================================================
# PRIME TOKENIZATION MODULE
# =============================================================================

class PrimeEmbedding(nn.Module):
    """
    Embedding layer with prime-indexed position encoding.
    
    Each position gets a unique prime number, enabling:
    - Unique factorization of position combinations
    - Log-prime positional features
    """
    
    def __init__(self, config: ResoFormerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Learned position embeddings
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)
        
        # Prime position features (log of prime at each position)
        primes = [nth_prime(i + 1) for i in range(config.max_seq_len)]
        log_primes = torch.tensor([math.log(p) for p in primes], dtype=torch.float)
        self.register_buffer('log_primes', log_primes)
        
        # Project log-prime features
        self.prime_proj = nn.Linear(1, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            
        Returns:
            embeddings: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        tok_emb = self.token_embed(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.pos_embed(positions)
        
        # Prime position features
        log_primes = self.log_primes[:seq_len].unsqueeze(-1)  # (seq_len, 1)
        prime_features = self.prime_proj(log_primes)  # (seq_len, hidden_dim)
        
        # Combine: tokens + positions + prime features
        embeddings = tok_emb + pos_emb.unsqueeze(0) + prime_features.unsqueeze(0) * 0.1
        
        return self.dropout(self.layer_norm(embeddings))


# =============================================================================
# RESONANCE ROTATION
# =============================================================================

class ResonanceRotation(nn.Module):
    """
    Resonance phase rotation operator.
    
    Applies R̂(n)|x⟩ = e^(2πi log(n)/log(p_max)) * rotation
    
    This creates position-dependent phase rotations based on prime structure.
    """
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        half_dim = hidden_dim // 2
        
        # Learnable rotation frequencies
        # Initialize to zeros so we start with Identity rotation (no scrambling)
        self.frequencies = nn.Parameter(torch.zeros(half_dim))
        
        # Pre-compute log-prime phases
        primes = [nth_prime(i + 1) for i in range(max_seq_len)]
        max_log = math.log(primes[-1]) if primes else 1.0
        phases = torch.tensor([math.log(p) / max_log for p in primes])
        self.register_buffer('log_phases', phases)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply resonance rotation.
        
        Args:
            x: (batch, seq_len, hidden_dim)
            
        Returns:
            rotated: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, dim = x.shape
        half_dim = dim // 2
        
        # Split into real/imaginary parts
        x_real = x[..., :half_dim]
        x_imag = x[..., half_dim:]
        
        # Compute rotation angles: 2π * log_phase * learned_frequency
        phases = self.log_phases[:seq_len].unsqueeze(-1)  # (seq_len, 1)
        angles = 2 * math.pi * phases * self.frequencies  # (seq_len, half_dim)
        
        cos_theta = torch.cos(angles).unsqueeze(0)  # (1, seq_len, half_dim)
        sin_theta = torch.sin(angles).unsqueeze(0)
        
        # Complex multiplication: (a + bi)(cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
        out_real = x_real * cos_theta - x_imag * sin_theta
        out_imag = x_real * sin_theta + x_imag * cos_theta
        
        return torch.cat([out_real, out_imag], dim=-1)


# =============================================================================
# GOLDEN RATIO ATTENTION
# =============================================================================

class GoldenMultiHeadAttention(nn.Module):
    """
    Multi-head attention with golden ratio head weighting.
    
    Each head h is weighted by w_h = 1/φ^h (normalized).
    This creates a natural hierarchy: first heads dominate.
    """
    
    def __init__(self, config: ResoFormerConfig):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        assert self.hidden_dim % self.num_heads == 0
        
        # QKV projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Golden ratio head weights
        if config.use_golden_attention:
            head_weights = torch.tensor([1.0 / (PHI ** h) for h in range(self.num_heads)])
            head_weights = head_weights / head_weights.mean()  # Preserve average head scale
            # Make it a learnable parameter initialized to golden ratio
            self.head_weights = nn.Parameter(head_weights)
        else:
            self.register_buffer('head_weights', torch.ones(self.num_heads))
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            attention_mask: optional mask
            is_causal: use causal masking
            
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Causal mask
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Additional attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum
        attn_output = torch.matmul(attn_probs, v)  # (batch, heads, seq, head_dim)
        
        # Apply golden ratio head weights
        head_weights = self.head_weights.view(1, self.num_heads, 1, 1)
        attn_output = attn_output * head_weights
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(attn_output)


# =============================================================================
# COHERENCE GATE
# =============================================================================

class CoherenceGate(nn.Module):
    """
    Coherence-based gating mechanism.
    
    Computes coherence as inverse normalized entropy and gates
    information flow based on coherence exceeding threshold.
    
    Similar to ACT (Adaptive Computation Time) halting.
    """
    
    def __init__(self, hidden_dim: int, threshold: float = COHERENCE_THRESHOLD):
        super().__init__()
        
        self.threshold = threshold
        
        # Project to scalar coherence score
        self.coherence_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize last layer bias to positive so gate starts OPEN
        # sigmoid(2.0) ~= 0.88, which is > threshold (0.8)
        nn.init.constant_(self.coherence_proj[2].bias, 2.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            
        Returns:
            gated: (batch, seq_len, hidden_dim)
            coherence: (batch, seq_len, 1) coherence scores
        """
        # Compute coherence per position
        coherence = self.coherence_proj(x)  # (batch, seq_len, 1)
        
        # Soft gate based on threshold
        gate = torch.sigmoid(10.0 * (coherence - self.threshold))
        
        # Apply gate
        gated = x * gate
        
        return gated, coherence


# =============================================================================
# ENTROPY COLLAPSE (VQ-VAE STYLE)
# =============================================================================

class EntropyCollapse(nn.Module):
    """
    Entropy collapse layer using vector quantization.
    
    Collapses continuous representations to discrete attractors,
    reducing entropy while preserving semantic structure.
    """
    
    def __init__(self, hidden_dim: int, num_attractors: int = 64, 
                 temperature: float = 1.0):
        super().__init__()
        
        self.num_attractors = num_attractors
        self.temperature = temperature
        
        # Learnable codebook (attractors)
        self.codebook = nn.Parameter(torch.randn(num_attractors, hidden_dim) * 0.1)
    
    def forward(self, x: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            hard: use hard assignment (argmax) vs soft (weighted sum)
            
        Returns:
            quantized: (batch, seq_len, hidden_dim)
            distances: (batch, seq_len, num_attractors) 
        """
        # Compute distances to each attractor
        # x: (batch, seq_len, dim), codebook: (num_attractors, dim)
        x_flat = x.reshape(-1, x.shape[-1])  # (batch*seq, dim)
        
        # L2 distance
        distances = torch.cdist(x_flat, self.codebook)  # (batch*seq, num_attractors)
        
        # Soft assignment weights
        weights = F.softmax(-distances / self.temperature, dim=-1)
        
        if hard or not self.training:
            # Hard quantization
            indices = weights.argmax(dim=-1)  # (batch*seq,)
            quantized = self.codebook[indices]  # (batch*seq, dim)
            
            # Straight-through estimator for gradients
            if self.training:
                quantized = x_flat + (quantized - x_flat).detach()
        else:
            # Soft quantization (weighted average of attractors)
            quantized = torch.matmul(weights, self.codebook)  # (batch*seq, dim)
        
        # Reshape back
        quantized = quantized.reshape(x.shape)
        distances = distances.reshape(x.shape[0], x.shape[1], -1)
        
        return quantized, distances


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class ResoFormerBlock(nn.Module):
    """ResoFormer transformer block."""
    
    def __init__(self, config: ResoFormerConfig, layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-norm architecture
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        
        # Attention
        self.attention = GoldenMultiHeadAttention(config)
        
        # Resonance rotation (optional)
        if config.use_resonance_rotation:
            self.resonance = ResonanceRotation(config.hidden_dim, config.max_seq_len)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        
        # Coherence gate (optional)
        if config.use_coherence_gate:
            self.coherence_gate = CoherenceGate(config.hidden_dim, config.coherence_threshold)
        
        # Entropy collapse (optional, typically only last layer)
        if config.use_entropy_collapse and layer_idx == config.num_layers - 1:
            self.entropy_collapse = EntropyCollapse(
                config.hidden_dim, 
                config.num_attractors
            )
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            
        Returns:
            output: (batch, seq_len, hidden_dim)
            metrics: dict of coherence, etc.
        """
        metrics = {}
        
        # Attention with residual
        residual = x
        x = self.ln1(x)
        x = self.attention(x, attention_mask)
        x = residual + x
        
        # Resonance rotation
        if self.config.use_resonance_rotation:
            x = self.resonance(x)
        
        # FFN with residual
        residual = x
        x = self.ln2(x)
        ffn_out = self.ffn(x)
        
        # Coherence gate (gate update, not the residual stream)
        if self.config.use_coherence_gate:
            gated, coherence = self.coherence_gate(ffn_out)
            x = residual + gated
            metrics['coherence'] = coherence.mean()
        else:
            x = residual + ffn_out
        
        # Entropy collapse
        if hasattr(self, 'entropy_collapse'):
            x, distances = self.entropy_collapse(x)
            metrics['entropy_collapse_dist'] = distances.mean()
        
        return x, metrics


# =============================================================================
# FULL MODEL
# =============================================================================

class PyTorchResoFormer(nn.Module):
    """
    Complete PyTorch ResoFormer model.
    
    Combines TinyAleph concepts with PyTorch efficiency:
    - Prime-indexed embeddings
    - Golden ratio attention
    - Resonance rotations
    - Coherence gating
    - Entropy collapse
    """
    
    def __init__(self, config: ResoFormerConfig):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.embeddings = PrimeEmbedding(config)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            ResoFormerBlock(config, layer_idx=i) 
            for i in range(config.num_layers)
        ])
        
        # Output
        self.final_ln = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embeddings.token_embed.weight
        
        # Initialize
        self.apply(self._init_weights)
        self._init_coherence_gates()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _init_coherence_gates(self) -> None:
        # Keep coherence gates open at init; _init_weights zeroes their biases.
        for module in self.modules():
            if isinstance(module, CoherenceGate):
                nn.init.constant_(module.coherence_proj[2].bias, 2.0)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: optional
            labels: optional, for computing loss
            
        Returns:
            dict with logits, loss (if labels), and metrics
        """
        # Embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Collect metrics from all layers
        all_metrics = {}
        
        # Transformer blocks
        for i, layer in enumerate(self.layers):
            hidden_states, metrics = layer(hidden_states, attention_mask)
            for key, val in metrics.items():
                all_metrics[f'layer_{i}_{key}'] = val
        
        # Output
        hidden_states = self.final_ln(hidden_states)
        logits = self.lm_head(hidden_states)
        
        result = {'logits': logits, **all_metrics}
        
        # Loss
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result['loss'] = loss
        
        return result
    
    def generate(self, 
                 input_ids: torch.Tensor, 
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :]  # Last position
                
                # Temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Truncate if too long
                if input_ids.shape[1] > self.config.max_seq_len:
                    input_ids = input_ids[:, -self.config.max_seq_len:]
        
        return input_ids
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_resoformer(
    vocab_size: int = 32000,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    **kwargs
) -> PyTorchResoFormer:
    """Create a ResoFormer model with specified configuration."""
    config = ResoFormerConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs
    )
    return PyTorchResoFormer(config)


def create_small_resoformer(vocab_size: int = 32000) -> PyTorchResoFormer:
    """Create a small ResoFormer (suitable for debugging)."""
    return create_resoformer(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
    )


def create_medium_resoformer(vocab_size: int = 32000) -> PyTorchResoFormer:
    """Create a medium ResoFormer."""
    return create_resoformer(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ffn_dim=2048,
    )


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch required. Install with: pip install torch")
        sys.exit(1)
    
    print("PyTorch ResoFormer Test")
    print("=" * 60)
    
    # Create small model
    model = create_small_resoformer(vocab_size=1000)
    print(f"Model parameters: {model.num_parameters:,}")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    outputs = model(input_ids, labels=labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_length=20, temperature=1.0)
    print(f"Generated shape: {generated.shape}")
    
    print("\nSuccess!")
