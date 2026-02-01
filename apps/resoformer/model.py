"""
TrainableResoFormer: Enhanced Model with Gradient Support

A complete, trainable transformer built on TinyAleph's prime-resonant foundations.
Uses numerical gradients for pure Python training.

Key Components:
- TrainableTensor: Tensor with gradient accumulation
- TrainableLayer: Base class with parameter management
- TrainableResoFormer: Full model with forward/backward passes

Mathematical Foundation:
- State space: H_Q = H_P ⊗ ℍ (Prime Hilbert ⊗ Quaternions)
- Attention: R(Q,K) = Σ_p α_p(Q) · β_p(K) · resonance(p)
- Coherence gate: h_t = σ(coherence(s_t) - τ)
- Entropy collapse: softmax over 64 discrete attractors
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
import math
import random
import sys
sys.path.insert(0, '../..')

from tinyaleph.core.constants import PHI, COHERENCE_THRESHOLD


# =============================================================================
# TRAINABLE TENSOR
# =============================================================================

class TrainableTensor:
    """
    Tensor with gradient tracking for training.
    
    Supports:
    - Gradient accumulation
    - Zero-grad reset
    - In-place updates for SGD
    """
    
    def __init__(self, data: List[float], shape: Tuple[int, ...], 
                 requires_grad: bool = True):
        self.data = list(data)
        self.shape = shape
        self.requires_grad = requires_grad
        self.grad: Optional[List[float]] = None
        
        if requires_grad:
            self.grad = [0.0] * len(data)
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], requires_grad: bool = True) -> TrainableTensor:
        size = 1
        for d in shape:
            size *= d
        return cls([0.0] * size, shape, requires_grad)
    
    @classmethod
    def ones(cls, shape: Tuple[int, ...], requires_grad: bool = True) -> TrainableTensor:
        size = 1
        for d in shape:
            size *= d
        return cls([1.0] * size, shape, requires_grad)
    
    @classmethod
    def randn(cls, shape: Tuple[int, ...], requires_grad: bool = True) -> TrainableTensor:
        size = 1
        for d in shape:
            size *= d
        data = []
        for _ in range(size):
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)
            data.append(z)
        return cls(data, shape, requires_grad)
    
    @classmethod
    def glorot_uniform(cls, shape: Tuple[int, ...], requires_grad: bool = True) -> TrainableTensor:
        fan_in = shape[0] if len(shape) > 0 else 1
        fan_out = shape[1] if len(shape) > 1 else 1
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        size = 1
        for d in shape:
            size *= d
        data = [random.uniform(-limit, limit) for _ in range(size)]
        return cls(data, shape, requires_grad)
    
    def size(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result
    
    def zero_grad(self):
        """Reset gradients to zero."""
        if self.grad is not None:
            self.grad = [0.0] * len(self.data)
    
    def accumulate_grad(self, gradient: List[float]):
        """Add gradient to accumulated gradients."""
        if self.grad is not None:
            for i in range(len(self.grad)):
                self.grad[i] += gradient[i] if i < len(gradient) else 0.0
    
    def update(self, lr: float):
        """Apply SGD update: θ = θ - lr * grad."""
        if self.grad is not None:
            for i in range(len(self.data)):
                self.data[i] -= lr * self.grad[i]
    
    def clone(self) -> TrainableTensor:
        t = TrainableTensor(list(self.data), self.shape, self.requires_grad)
        if self.grad is not None:
            t.grad = list(self.grad)
        return t
    
    def __repr__(self) -> str:
        preview = self.data[:5]
        return f"TrainableTensor(shape={self.shape}, data={preview}{'...' if len(self.data) > 5 else ''})"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainableResoFormerConfig:
    """Configuration for TrainableResoFormer model."""
    
    # Vocabulary
    vocab_size: int = 1000
    max_prime: int = 1000
    
    # Architecture
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 512
    
    # Sequence
    max_seq_len: int = 128
    
    # Regularization
    dropout: float = 0.1
    
    # Coherence/Entropy
    coherence_threshold: float = COHERENCE_THRESHOLD
    num_attractors: int = 64
    use_coherence_gate: bool = True
    use_entropy_collapse: bool = True
    
    # Golden ratio features
    use_golden_attention: bool = True
    golden_lr_schedule: bool = True


# =============================================================================
# TRAINABLE LAYERS
# =============================================================================

class TrainableLayer(ABC):
    """Base class for trainable layers."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.params: Dict[str, TrainableTensor] = {}
        self._built = False
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> None:
        pass
    
    @abstractmethod
    def forward(self, x: List[float], shape: Tuple[int, ...], 
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        pass
    
    def __call__(self, x: List[float], shape: Tuple[int, ...],
                 training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        if not self._built:
            self.build(shape)
            self._built = True
        return self.forward(x, shape, training)
    
    def get_params(self) -> Dict[str, TrainableTensor]:
        return self.params
    
    def zero_grad(self):
        for param in self.params.values():
            param.zero_grad()


class TrainableEmbedding(TrainableLayer):
    """Learnable embedding layer."""
    
    def __init__(self, vocab_size: int, dim: int, name: str = "embedding"):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.dim = dim
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        self.params['embedding'] = TrainableTensor.glorot_uniform(
            (self.vocab_size, self.dim)
        )
    
    def forward(self, x: List[float], shape: Tuple[int, ...],
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        """
        x contains token indices (as floats).
        Returns embeddings of shape (batch, seq, dim).
        """
        batch_size = shape[0] if len(shape) >= 1 else 1
        seq_len = shape[1] if len(shape) >= 2 else len(x) // batch_size
        
        result = []
        embed = self.params['embedding']
        
        for i in range(batch_size * seq_len):
            idx = int(x[i]) if i < len(x) else 0
            idx = max(0, min(idx, self.vocab_size - 1))
            
            start = idx * self.dim
            result.extend(embed.data[start:start + self.dim])
        
        return result, (batch_size, seq_len, self.dim)


class TrainableDense(TrainableLayer):
    """Trainable dense layer."""
    
    def __init__(self, units: int, activation: Optional[str] = None,
                 use_bias: bool = True, name: str = "dense"):
        super().__init__(name)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        input_dim = input_shape[-1]
        self.params['kernel'] = TrainableTensor.glorot_uniform((input_dim, self.units))
        if self.use_bias:
            self.params['bias'] = TrainableTensor.zeros((self.units,))
    
    def forward(self, x: List[float], shape: Tuple[int, ...],
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        dim = shape[-1]
        batch_size = len(x) // dim
        
        kernel = self.params['kernel']
        
        result = []
        for b in range(batch_size):
            for u in range(self.units):
                val = 0.0
                for d in range(dim):
                    val += x[b * dim + d] * kernel.data[d * self.units + u]
                
                if self.use_bias:
                    val += self.params['bias'].data[u]
                
                # Activation
                if self.activation == 'relu':
                    val = max(0, val)
                elif self.activation == 'gelu':
                    val = 0.5 * val * (1 + math.tanh(
                        math.sqrt(2 / math.pi) * (val + 0.044715 * val ** 3)
                    ))
                elif self.activation == 'sigmoid':
                    val = 1.0 / (1.0 + math.exp(-max(-700, min(700, val))))
                elif self.activation == 'tanh':
                    val = math.tanh(val)
                
                result.append(val)
        
        new_shape = shape[:-1] + (self.units,)
        return result, new_shape


class TrainableLayerNorm(TrainableLayer):
    """Trainable layer normalization."""
    
    def __init__(self, eps: float = 1e-6, name: str = "layernorm"):
        super().__init__(name)
        self.eps = eps
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        dim = input_shape[-1]
        self.params['gamma'] = TrainableTensor.ones((dim,))
        self.params['beta'] = TrainableTensor.zeros((dim,))
    
    def forward(self, x: List[float], shape: Tuple[int, ...],
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        dim = shape[-1]
        batch_size = len(x) // dim
        
        gamma = self.params['gamma'].data
        beta = self.params['beta'].data
        
        result = []
        for b in range(batch_size):
            vec = x[b * dim:(b + 1) * dim]
            mean = sum(vec) / dim
            var = sum((v - mean) ** 2 for v in vec) / dim
            std = math.sqrt(var + self.eps)
            
            for d in range(dim):
                normalized = (vec[d] - mean) / std
                result.append(normalized * gamma[d] + beta[d])
        
        return result, shape


class TrainableAttention(TrainableLayer):
    """Trainable multi-head attention with resonance weighting."""
    
    def __init__(self, num_heads: int, key_dim: int, 
                 use_golden_weights: bool = True,
                 name: str = "attention"):
        super().__init__(name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.use_golden_weights = use_golden_weights
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        input_dim = input_shape[-1]
        total_dim = self.num_heads * self.key_dim
        
        self.params['query'] = TrainableTensor.glorot_uniform((input_dim, total_dim))
        self.params['key'] = TrainableTensor.glorot_uniform((input_dim, total_dim))
        self.params['value'] = TrainableTensor.glorot_uniform((input_dim, total_dim))
        self.params['output'] = TrainableTensor.glorot_uniform((total_dim, input_dim))
    
    def forward(self, x: List[float], shape: Tuple[int, ...],
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        dim = shape[-1]
        seq_len = shape[-2] if len(shape) >= 2 else 1
        batch_size = len(x) // (seq_len * dim)
        
        total_dim = self.num_heads * self.key_dim
        scale = 1.0 / math.sqrt(self.key_dim)
        
        query_w = self.params['query'].data
        key_w = self.params['key'].data
        value_w = self.params['value'].data
        output_w = self.params['output'].data
        
        # Golden ratio head weights
        if self.use_golden_weights:
            head_weights = [1.0 / (PHI ** h) for h in range(self.num_heads)]
            hw_sum = sum(head_weights)
            head_weights = [w / hw_sum for w in head_weights]
        else:
            head_weights = [1.0 / self.num_heads] * self.num_heads
        
        outputs = []
        
        for b in range(batch_size):
            # Project Q, K, V
            q_proj, k_proj, v_proj = [], [], []
            
            for s in range(seq_len):
                idx = (b * seq_len + s) * dim
                vec = x[idx:idx + dim]
                
                # Q projection
                for t in range(total_dim):
                    val = sum(vec[d] * query_w[d * total_dim + t] for d in range(dim))
                    q_proj.append(val)
                
                # K projection
                for t in range(total_dim):
                    val = sum(vec[d] * key_w[d * total_dim + t] for d in range(dim))
                    k_proj.append(val)
                
                # V projection
                for t in range(total_dim):
                    val = sum(vec[d] * value_w[d * total_dim + t] for d in range(dim))
                    v_proj.append(val)
            
            # Compute attention per head
            attended = [0.0] * (seq_len * total_dim)
            
            for h in range(self.num_heads):
                head_start = h * self.key_dim
                head_end = (h + 1) * self.key_dim
                
                for i in range(seq_len):
                    # Compute attention scores
                    scores = []
                    for j in range(seq_len):
                        score = 0.0
                        for k in range(self.key_dim):
                            q_idx = i * total_dim + head_start + k
                            k_idx = j * total_dim + head_start + k
                            score += q_proj[q_idx] * k_proj[k_idx]
                        scores.append(score * scale)
                    
                    # Softmax
                    max_score = max(scores)
                    exp_scores = [math.exp(s - max_score) for s in scores]
                    sum_exp = sum(exp_scores)
                    attn_weights = [e / sum_exp for e in exp_scores]
                    
                    # Weighted sum of values
                    for k in range(self.key_dim):
                        val = 0.0
                        for j in range(seq_len):
                            v_idx = j * total_dim + head_start + k
                            val += attn_weights[j] * v_proj[v_idx]
                        
                        out_idx = i * total_dim + head_start + k
                        attended[out_idx] += val * head_weights[h]
            
            # Output projection
            for s in range(seq_len):
                for d in range(dim):
                    val = 0.0
                    for t in range(total_dim):
                        val += attended[s * total_dim + t] * output_w[t * dim + d]
                    outputs.append(val)
        
        return outputs, shape


class TrainableCoherenceGate(TrainableLayer):
    """Trainable coherence gating layer."""
    
    def __init__(self, threshold: float = COHERENCE_THRESHOLD, name: str = "coherence_gate"):
        super().__init__(name)
        self.threshold = threshold
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        dim = input_shape[-1]
        self.params['coherence_weight'] = TrainableTensor.glorot_uniform((dim, 1))
    
    def forward(self, x: List[float], shape: Tuple[int, ...],
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        dim = shape[-1]
        batch_size = len(x) // dim
        
        weight = self.params['coherence_weight'].data
        
        result = []
        coherences = []
        
        for b in range(batch_size):
            vec = x[b * dim:(b + 1) * dim]
            
            # Compute coherence score
            coherence_raw = sum(vec[d] * weight[d] for d in range(dim))
            coherence = 1.0 / (1.0 + math.exp(-coherence_raw))  # sigmoid
            coherences.append(coherence)
            
            # Compute gate
            gate = 1.0 / (1.0 + math.exp(-10.0 * (coherence - self.threshold)))
            
            # Apply gate
            for d in range(dim):
                result.append(vec[d] * gate)
        
        return result, shape


class TrainableEntropyCollapse(TrainableLayer):
    """Trainable entropy collapse layer (VQ-style)."""
    
    def __init__(self, num_attractors: int = 64, 
                 temperature: float = 1.0,
                 name: str = "entropy_collapse"):
        super().__init__(name)
        self.num_attractors = num_attractors
        self.temperature = temperature
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        dim = input_shape[-1]
        self.params['codebook'] = TrainableTensor.glorot_uniform(
            (self.num_attractors, dim)
        )
    
    def forward(self, x: List[float], shape: Tuple[int, ...],
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        dim = shape[-1]
        batch_size = len(x) // dim
        
        codebook = self.params['codebook'].data
        
        result = []
        
        for b in range(batch_size):
            vec = x[b * dim:(b + 1) * dim]
            
            # Compute distances to codebook entries
            distances = []
            for a in range(self.num_attractors):
                dist = sum(
                    (vec[d] - codebook[a * dim + d]) ** 2
                    for d in range(dim)
                )
                distances.append(-dist / self.temperature)
            
            # Softmax over attractors
            max_dist = max(distances)
            exp_dists = [math.exp(d - max_dist) for d in distances]
            sum_exp = sum(exp_dists)
            probs = [e / sum_exp for e in exp_dists]
            
            if training:
                # Soft assignment
                for d in range(dim):
                    val = sum(
                        probs[a] * codebook[a * dim + d]
                        for a in range(self.num_attractors)
                    )
                    result.append(val)
            else:
                # Hard assignment
                best_a = probs.index(max(probs))
                for d in range(dim):
                    result.append(codebook[best_a * dim + d])
        
        return result, shape


class TrainableResonanceOperator(TrainableLayer):
    """Trainable resonance phase rotation operator."""
    
    def __init__(self, name: str = "resonance"):
        super().__init__(name)
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        dim = input_shape[-1]
        self.params['rotation'] = TrainableTensor.ones((dim // 2,))
    
    def forward(self, x: List[float], shape: Tuple[int, ...],
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        dim = shape[-1]
        half_dim = dim // 2
        batch_size = len(x) // dim
        
        rotation = self.params['rotation'].data
        
        result = []
        
        for b in range(batch_size):
            vec = x[b * dim:(b + 1) * dim]
            real = vec[:half_dim]
            imag = vec[half_dim:]
            
            for d in range(half_dim):
                n = abs(rotation[d]) + 1
                phase = math.log(n) * 2 * math.pi
                cos_p = math.cos(phase)
                sin_p = math.sin(phase)
                
                new_real = real[d] * cos_p - imag[d] * sin_p
                new_imag = real[d] * sin_p + imag[d] * cos_p
                
                result.append(new_real)
            
            for d in range(half_dim):
                n = abs(rotation[d]) + 1
                phase = math.log(n) * 2 * math.pi
                cos_p = math.cos(phase)
                sin_p = math.sin(phase)
                
                new_imag = real[d] * sin_p + imag[d] * cos_p
                result.append(new_imag)
        
        return result, shape


# =============================================================================
# RESOFORMER BLOCK
# =============================================================================

class TrainableResoFormerBlock(TrainableLayer):
    """Complete ResoFormer transformer block."""
    
    def __init__(self, dim: int, num_heads: int, ffn_dim: int,
                 use_coherence_gate: bool = True,
                 use_entropy_collapse: bool = False,
                 use_golden_attention: bool = True,
                 dropout_rate: float = 0.1,
                 name: str = "block"):
        super().__init__(name)
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_coherence_gate = use_coherence_gate
        self.use_entropy_collapse = use_entropy_collapse
        self.dropout_rate = dropout_rate
        
        # Sub-layers
        self.ln1 = TrainableLayerNorm(name=f"{name}_ln1")
        self.ln2 = TrainableLayerNorm(name=f"{name}_ln2")
        self.attention = TrainableAttention(
            num_heads, dim // num_heads,
            use_golden_weights=use_golden_attention,
            name=f"{name}_attn"
        )
        self.resonance = TrainableResonanceOperator(name=f"{name}_res")
        self.ffn1 = TrainableDense(ffn_dim, activation='gelu', name=f"{name}_ffn1")
        self.ffn2 = TrainableDense(dim, name=f"{name}_ffn2")
        
        if use_coherence_gate:
            self.coherence_gate = TrainableCoherenceGate(name=f"{name}_cg")
        
        if use_entropy_collapse:
            self.entropy_collapse = TrainableEntropyCollapse(name=f"{name}_ec")
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        
        self.ln1.build(input_shape)
        self.ln2.build(input_shape)
        self.attention.build(input_shape)
        self.resonance.build(input_shape)
        self.ffn1.build(input_shape)
        
        ffn_shape = input_shape[:-1] + (self.ffn_dim,)
        self.ffn2.build(ffn_shape)
        
        if self.use_coherence_gate:
            self.coherence_gate.build(input_shape)
        
        if self.use_entropy_collapse:
            self.entropy_collapse.build(input_shape)
        
        # Collect parameters
        self.params.update(self.ln1.params)
        self.params.update(self.ln2.params)
        self.params.update(self.attention.params)
        self.params.update(self.resonance.params)
        self.params.update(self.ffn1.params)
        self.params.update(self.ffn2.params)
        
        if self.use_coherence_gate:
            self.params.update(self.coherence_gate.params)
        
        if self.use_entropy_collapse:
            self.params.update(self.entropy_collapse.params)
    
    def forward(self, x: List[float], shape: Tuple[int, ...],
                training: bool = False) -> Tuple[List[float], Tuple[int, ...]]:
        residual = list(x)
        
        # Pre-norm attention
        normed, _ = self.ln1(x, shape, training)
        attn_out, _ = self.attention(normed, shape, training)
        
        # Residual
        x = [a + b for a, b in zip(residual, attn_out)]
        
        # Resonance
        x, shape = self.resonance(x, shape, training)
        
        # Pre-norm FFN
        residual = list(x)
        normed, _ = self.ln2(x, shape, training)
        ffn_out, ffn_shape = self.ffn1(normed, shape, training)
        ffn_out, _ = self.ffn2(ffn_out, ffn_shape, training)
        
        # Residual
        x = [a + b for a, b in zip(residual, ffn_out)]
        
        # Coherence gate
        if self.use_coherence_gate:
            x, shape = self.coherence_gate(x, shape, training)
        
        # Entropy collapse
        if self.use_entropy_collapse:
            x, shape = self.entropy_collapse(x, shape, training)
        
        return x, shape


# =============================================================================
# FULL MODEL
# =============================================================================

class TrainableResoFormer:
    """
    Complete trainable ResoFormer model.
    
    Architecture:
    1. Token embedding (vocabulary → dense vectors)
    2. N x ResoFormerBlock
    3. Final LayerNorm
    4. Output projection (for language modeling)
    """
    
    def __init__(self, config: TrainableResoFormerConfig):
        self.config = config
        self._built = False
        
        # Layers
        self.embedding = TrainableEmbedding(
            config.vocab_size, config.dim, name="embedding"
        )
        
        self.blocks = []
        for i in range(config.num_layers):
            block = TrainableResoFormerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                use_coherence_gate=config.use_coherence_gate,
                use_entropy_collapse=(i == config.num_layers - 1) and config.use_entropy_collapse,
                use_golden_attention=config.use_golden_attention,
                dropout_rate=config.dropout,
                name=f"block_{i}"
            )
            self.blocks.append(block)
        
        self.final_ln = TrainableLayerNorm(name="final_ln")
        self.output_proj = TrainableDense(config.vocab_size, name="output")
    
    def _build(self, input_shape: Tuple[int, ...]):
        """Build all layers."""
        if self._built:
            return
        
        self.embedding.build(input_shape)
        
        embed_shape = input_shape[:-1] + (self.config.dim,) if len(input_shape) > 1 else (input_shape[0], self.config.dim)
        
        for block in self.blocks:
            block.build(embed_shape)
        
        self.final_ln.build(embed_shape)
        self.output_proj.build(embed_shape)
        
        self._built = True
    
    def forward(self, tokens: List[int], training: bool = False) -> List[float]:
        """
        Forward pass.
        
        Args:
            tokens: List of token indices (flattened batch x seq)
            training: Whether in training mode
            
        Returns:
            Logits of shape (batch * seq, vocab_size)
        """
        batch_size = 1
        seq_len = len(tokens)
        shape = (batch_size, seq_len)
        
        x = [float(t) for t in tokens]
        
        self._build(shape)
        
        # Embedding
        x, shape = self.embedding(x, shape, training)
        
        # Transformer blocks
        for block in self.blocks:
            x, shape = block(x, shape, training)
        
        # Final layer norm
        x, shape = self.final_ln(x, shape, training)
        
        # Output projection
        logits, _ = self.output_proj(x, shape, training)
        
        return logits
    
    def get_all_params(self) -> Dict[str, TrainableTensor]:
        """Get all trainable parameters."""
        params = {}
        params.update(self.embedding.params)
        for block in self.blocks:
            params.update(block.params)
        params.update(self.final_ln.params)
        params.update(self.output_proj.params)
        return params
    
    def zero_grad(self):
        """Zero all gradients."""
        self.embedding.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.final_ln.zero_grad()
        self.output_proj.zero_grad()
    
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0
        for param in self.get_all_params().values():
            total += len(param.data)
        return total
    
    def save(self, path: str):
        """Save model parameters."""
        import json
        params = {}
        for name, tensor in self.get_all_params().items():
            params[name] = {
                'data': tensor.data,
                'shape': tensor.shape,
            }
        
        with open(path, 'w') as f:
            json.dump({
                'config': {
                    'vocab_size': self.config.vocab_size,
                    'dim': self.config.dim,
                    'num_layers': self.config.num_layers,
                    'num_heads': self.config.num_heads,
                    'ffn_dim': self.config.ffn_dim,
                    'max_seq_len': self.config.max_seq_len,
                    'dropout': self.config.dropout,
                },
                'params': params,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> TrainableResoFormer:
        """Load model from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = TrainableResoFormerConfig(**data['config'])
        model = cls(config)
        
        # Build model structure
        dummy_tokens = [0] * 10
        model.forward(dummy_tokens, training=False)
        
        # Load parameters
        for name, param_data in data['params'].items():
            if name in model.get_all_params():
                param = model.get_all_params()[name]
                param.data = param_data['data']
        
        return model
    
    def __repr__(self) -> str:
        return (f"TrainableResoFormer("
                f"layers={self.config.num_layers}, "
                f"dim={self.config.dim}, "
                f"heads={self.config.num_heads}, "
                f"params={self.num_parameters():,})")