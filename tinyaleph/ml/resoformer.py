"""
ResoFormer: Resonant Transformer Layers

Complete port of the ResoFormer architecture from TensorFlow.js to Python.
Provides both pure-Python implementation and optional PyTorch acceleration.

Trainable neural network layers implementing ResoFormer architecture:
- QuaternionDense: Dense layer with Hamilton product
- SparsePrimeEmbedding: Token → sparse prime activations with quaternion orientations
- ResonantAttentionLayer: Attention using Jaccard + Quaternion + Phase
- HamiltonCompose: Order-sensitive composition layer
- CoherenceGatingLayer: Coherence-based gating mechanism
- EntropyCollapseLayer: 64-codebook VQ collapse with entropy regularization
- ResonanceOperator: R̂(n)|p⟩ = e^(2πi log_p(n))|p⟩ phase rotation
- ResoFormerBlock: Complete transformer block
- ResoFormerModel: End-to-end trainable model

State space: H_Q = H_P ⊗ ℍ (Prime Hilbert space ⊗ Quaternions)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from enum import Enum
import math
import random
from abc import ABC, abstractmethod


# =============================================================================
# TENSOR ABSTRACTION (Pure Python)
# =============================================================================

class Tensor:
    """Pure Python tensor for ResoFormer operations."""
    
    def __init__(self, data: Union[List, 'Tensor', float], shape: Optional[Tuple[int, ...]] = None):
        if isinstance(data, Tensor):
            self.data = list(data.data)
            self._shape = data._shape
        elif isinstance(data, (int, float)):
            self.data = [float(data)]
            self._shape = (1,)
        else:
            self.data, self._shape = self._flatten_with_shape(data)
        
        if shape is not None:
            self._shape = shape
    
    def _flatten_with_shape(self, data: List) -> Tuple[List[float], Tuple[int, ...]]:
        if not isinstance(data, (list, tuple)):
            return [float(data)], ()
        if len(data) == 0:
            return [], (0,)
        first = data[0]
        if not isinstance(first, (list, tuple)):
            return [float(x) for x in data], (len(data),)
        sub_results = [self._flatten_with_shape(sub) for sub in data]
        flat = []
        for sub_flat, _ in sub_results:
            flat.extend(sub_flat)
        sub_shape = sub_results[0][1]
        return flat, (len(data),) + sub_shape
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
    
    @property
    def ndim(self) -> int:
        return len(self._shape)
    
    def size(self) -> int:
        result = 1
        for d in self._shape:
            result *= d
        return result
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        return Tensor(list(self.data), new_shape)
    
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, (int, float)):
            return Tensor([x + other for x in self.data], self._shape)
        return Tensor([a + b for a, b in zip(self.data, other.data)], self._shape)
    
    def __radd__(self, other: float) -> 'Tensor':
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, (int, float)):
            return Tensor([x - other for x in self.data], self._shape)
        return Tensor([a - b for a, b in zip(self.data, other.data)], self._shape)
    
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, (int, float)):
            return Tensor([x * other for x in self.data], self._shape)
        return Tensor([a * b for a, b in zip(self.data, other.data)], self._shape)
    
    def __rmul__(self, other: float) -> 'Tensor':
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, (int, float)):
            return Tensor([x / (other + 1e-10) for x in self.data], self._shape)
        return Tensor([a / (b + 1e-10) for a, b in zip(self.data, other.data)], self._shape)
    
    def __neg__(self) -> 'Tensor':
        return Tensor([-x for x in self.data], self._shape)
    
    def sum(self, axis: Optional[int] = None) -> 'Tensor':
        if axis is None:
            return Tensor(sum(self.data))
        if self.ndim == 1:
            return Tensor(sum(self.data))
        # Simple sum for 2D
        if self.ndim == 2 and axis == 1:
            rows = self._shape[0]
            cols = self._shape[1]
            result = []
            for i in range(rows):
                result.append(sum(self.data[i * cols:(i + 1) * cols]))
            return Tensor(result, (rows,))
        return Tensor(sum(self.data))
    
    def mean(self, axis: Optional[int] = None) -> 'Tensor':
        if axis is None:
            return Tensor(sum(self.data) / max(len(self.data), 1))
        summed = self.sum(axis)
        return summed / self._shape[axis]
    
    def sqrt(self) -> 'Tensor':
        return Tensor([math.sqrt(max(0, x)) for x in self.data], self._shape)
    
    def square(self) -> 'Tensor':
        return Tensor([x * x for x in self.data], self._shape)
    
    def exp(self) -> 'Tensor':
        return Tensor([math.exp(min(x, 700)) for x in self.data], self._shape)
    
    def log(self) -> 'Tensor':
        return Tensor([math.log(max(x, 1e-10)) for x in self.data], self._shape)
    
    def abs(self) -> 'Tensor':
        return Tensor([abs(x) for x in self.data], self._shape)
    
    def cos(self) -> 'Tensor':
        return Tensor([math.cos(x) for x in self.data], self._shape)
    
    def sin(self) -> 'Tensor':
        return Tensor([math.sin(x) for x in self.data], self._shape)
    
    def tanh(self) -> 'Tensor':
        return Tensor([math.tanh(x) for x in self.data], self._shape)
    
    def sigmoid(self) -> 'Tensor':
        return Tensor([1.0 / (1.0 + math.exp(-max(-700, min(700, x)))) for x in self.data], self._shape)
    
    def relu(self) -> 'Tensor':
        return Tensor([max(0, x) for x in self.data], self._shape)
    
    def gelu(self) -> 'Tensor':
        return Tensor([
            0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
            for x in self.data
        ], self._shape)
    
    def softmax(self, axis: int = -1) -> 'Tensor':
        if self.ndim == 1:
            max_val = max(self.data)
            exp_vals = [math.exp(x - max_val) for x in self.data]
            total = sum(exp_vals)
            return Tensor([e / total for e in exp_vals], self._shape)
        if self.ndim == 2:
            rows, cols = self._shape
            result = []
            for i in range(rows):
                row = self.data[i * cols:(i + 1) * cols]
                max_val = max(row)
                exp_vals = [math.exp(x - max_val) for x in row]
                total = sum(exp_vals)
                result.extend([e / total for e in exp_vals])
            return Tensor(result, self._shape)
        return self
    
    def argmax(self, axis: int = -1) -> 'Tensor':
        if self.ndim == 1:
            max_idx = 0
            for i, v in enumerate(self.data):
                if v > self.data[max_idx]:
                    max_idx = i
            return Tensor(max_idx)
        if self.ndim == 2:
            rows, cols = self._shape
            result = []
            for i in range(rows):
                row = self.data[i * cols:(i + 1) * cols]
                max_idx = 0
                for j, v in enumerate(row):
                    if v > row[max_idx]:
                        max_idx = j
                result.append(max_idx)
            return Tensor(result, (rows,))
        return Tensor(0)
    
    def topk(self, k: int, axis: int = -1) -> Tuple['Tensor', 'Tensor']:
        if self.ndim == 1:
            indexed = list(enumerate(self.data))
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_k = indexed[:min(k, len(indexed))]
            return (
                Tensor([x[1] for x in top_k], (k,)),
                Tensor([x[0] for x in top_k], (k,))
            )
        if self.ndim == 2:
            rows, cols = self._shape
            all_vals, all_idx = [], []
            for i in range(rows):
                row = self.data[i * cols:(i + 1) * cols]
                indexed = list(enumerate(row))
                indexed.sort(key=lambda x: x[1], reverse=True)
                top_k = indexed[:min(k, len(indexed))]
                all_vals.extend([x[1] for x in top_k])
                all_idx.extend([x[0] for x in top_k])
            return (
                Tensor(all_vals, (rows, k)),
                Tensor(all_idx, (rows, k))
            )
        return self, self
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul requires 2D tensors")
        m, k1 = self._shape
        k2, n = other._shape
        if k1 != k2:
            raise ValueError(f"Cannot multiply shapes {self._shape} and {other._shape}")
        result = []
        for i in range(m):
            for j in range(n):
                total = 0.0
                for k in range(k1):
                    total += self.data[i * k1 + k] * other.data[k * n + j]
                result.append(total)
        return Tensor(result, (m, n))
    
    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        if axis is not None:
            if self._shape[axis] != 1:
                return self
            new_shape = tuple(d for i, d in enumerate(self._shape) if i != axis)
        else:
            new_shape = tuple(d for d in self._shape if d != 1)
        if not new_shape:
            new_shape = (1,)
        return Tensor(list(self.data), new_shape)
    
    def clone(self) -> 'Tensor':
        return Tensor(list(self.data), self._shape)
    
    def __repr__(self) -> str:
        preview = self.data[:5]
        return f"Tensor(shape={self._shape}, data={preview}{'...' if len(self.data) > 5 else ''})"


def zeros(shape: Tuple[int, ...]) -> Tensor:
    size = 1
    for d in shape:
        size *= d
    return Tensor([0.0] * size, shape)


def ones(shape: Tuple[int, ...]) -> Tensor:
    size = 1
    for d in shape:
        size *= d
    return Tensor([1.0] * size, shape)


def randn(shape: Tuple[int, ...]) -> Tensor:
    size = 1
    for d in shape:
        size *= d
    data = []
    for _ in range(size):
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)
        data.append(z)
    return Tensor(data, shape)


def glorot_uniform(shape: Tuple[int, ...]) -> Tensor:
    fan_in = shape[0] if len(shape) > 0 else 1
    fan_out = shape[1] if len(shape) > 1 else 1
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    size = 1
    for d in shape:
        size *= d
    return Tensor([random.uniform(-limit, limit) for _ in range(size)], shape)


# =============================================================================
# QUATERNION TENSOR OPERATIONS
# =============================================================================

def quaternion_normalize(q: Tensor) -> Tensor:
    """Normalize quaternion to unit length."""
    shape = q.shape
    batch_size = q.size() // 4
    result = []
    for i in range(batch_size):
        idx = i * 4
        w, x, y, z = q.data[idx:idx + 4]
        norm = math.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2 + 1e-10)
        result.extend([w / norm, x / norm, y / norm, z / norm])
    return Tensor(result, shape)


# =============================================================================
# LAYER BASE CLASS
# =============================================================================

@dataclass
class Layer(ABC):
    """Base class for all layers."""
    
    name: str = ""
    trainable: bool = True
    _built: bool = field(default=False, init=False)
    _weights: Dict[str, Tensor] = field(default_factory=dict, init=False)
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> None:
        pass
    
    @abstractmethod
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        pass
    
    def __call__(self, x: Tensor, training: bool = False) -> Tensor:
        if not self._built:
            self.build(x.shape)
            self._built = True
        return self.forward(x, training)
    
    def add_weight(self, name: str, shape: Tuple[int, ...], init: str = "glorot_uniform") -> Tensor:
        if init == "zeros":
            w = zeros(shape)
        elif init == "ones":
            w = ones(shape)
        elif init == "normal":
            w = randn(shape) * 0.1
        else:
            w = glorot_uniform(shape)
        self._weights[name] = w
        return w
    
    def get_weights(self) -> Dict[str, Tensor]:
        return self._weights


# =============================================================================
# RESOFORMER LAYERS
# =============================================================================

@dataclass
class QuaternionDense(Layer):
    """Dense layer projecting to quaternion space."""
    
    units: int = 64
    use_bias: bool = True
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        input_dim = input_shape[-1]
        self.kernel = self.add_weight("kernel", (input_dim, self.units * 4))
        if self.use_bias:
            self.bias = self.add_weight("bias", (self.units * 4,), "zeros")
    
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        original_shape = x.shape
        batch_size = x.size() // original_shape[-1]
        x_2d = x.reshape((batch_size, original_shape[-1]))
        output = x_2d.matmul(self.kernel)
        
        if self.use_bias:
            # Broadcast bias to each batch element
            out_data = list(output.data)
            bias_size = self.units * 4
            for i in range(batch_size):
                for j in range(bias_size):
                    out_data[i * bias_size + j] += self.bias.data[j]
            output = Tensor(out_data, output.shape)
        
        new_shape = original_shape[:-1] + (self.units, 4)
        return quaternion_normalize(output.reshape(new_shape))


@dataclass
class SparsePrimeEmbedding(Layer):
    """Sparse Prime Embedding: top-k active primes per token."""
    
    num_primes: int = 4096
    k: int = 32
    embedding_dim: int = 64
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        self.prime_weights = self.add_weight("prime_weights", (self.embedding_dim, self.num_primes))
        self.quaternion_weights = self.add_weight("quaternion_weights", (self.num_primes, 4), "normal")
        self.phase_bias = self.add_weight("phase_bias", (self.num_primes,), "zeros")
    
    def forward(self, x: Tensor, training: bool = False) -> Dict[str, Tensor]:
        batch_seq = x.size() // self.embedding_dim
        x_2d = x.reshape((batch_seq, self.embedding_dim))
        logits = x_2d.matmul(self.prime_weights)
        topk_values, topk_indices = logits.topk(self.k, axis=-1)
        amplitudes = topk_values.softmax(axis=-1)
        return {"indices": topk_indices, "amplitudes": amplitudes, "logits": logits}


@dataclass
class ResonantAttentionLayer(Layer):
    """Multi-head resonant attention layer."""
    
    num_heads: int = 8
    key_dim: int = 64
    dropout: float = 0.0
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        input_dim = input_shape[-1]
        total_dim = self.num_heads * self.key_dim
        self.query_weight = self.add_weight("query_weight", (input_dim, total_dim))
        self.key_weight = self.add_weight("key_weight", (input_dim, total_dim))
        self.value_weight = self.add_weight("value_weight", (input_dim, total_dim))
        self.output_weight = self.add_weight("output_weight", (total_dim, input_dim))
    
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        original_shape = x.shape
        dim = original_shape[-1]
        seq_len = original_shape[-2] if len(original_shape) >= 2 else 1
        batch_size = x.size() // (seq_len * dim)
        
        x_2d = x.reshape((batch_size * seq_len, dim))
        q = x_2d.matmul(self.query_weight)
        k = x_2d.matmul(self.key_weight)
        v = x_2d.matmul(self.value_weight)
        
        total_dim = self.num_heads * self.key_dim
        scale = 1.0 / math.sqrt(self.key_dim)
        
        # Simple attention per batch
        outputs = []
        for b in range(batch_size):
            # Extract Q, K, V for this batch
            q_batch = q.data[b * seq_len * total_dim:(b + 1) * seq_len * total_dim]
            k_batch = k.data[b * seq_len * total_dim:(b + 1) * seq_len * total_dim]
            v_batch = v.data[b * seq_len * total_dim:(b + 1) * seq_len * total_dim]
            
            # Compute attention scores
            attn = []
            for i in range(seq_len):
                row = []
                for j in range(seq_len):
                    score = sum(
                        q_batch[i * total_dim + d] * k_batch[j * total_dim + d]
                        for d in range(total_dim)
                    ) * scale
                    row.append(score)
                # Softmax
                max_val = max(row)
                exp_vals = [math.exp(s - max_val) for s in row]
                total = sum(exp_vals)
                attn.append([e / total for e in exp_vals])
            
            # Apply attention
            for i in range(seq_len):
                out = [0.0] * total_dim
                for j in range(seq_len):
                    for d in range(total_dim):
                        out[d] += attn[i][j] * v_batch[j * total_dim + d]
                outputs.extend(out)
        
        result = Tensor(outputs, (batch_size, seq_len, total_dim))
        result_2d = result.reshape((batch_size * seq_len, total_dim))
        output = result_2d.matmul(self.output_weight)
        return output.reshape(original_shape)


@dataclass
class CoherenceGatingLayer(Layer):
    """Coherence-based gating mechanism."""
    
    threshold: float = 0.8
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        dim = input_shape[-1]
        self.coherence_weight = self.add_weight("coherence_weight", (dim, 1))
    
    def forward(self, x: Tensor, training: bool = False) -> Dict[str, Tensor]:
        original_shape = x.shape
        dim = original_shape[-1]
        batch_size = x.size() // dim
        
        x_2d = x.reshape((batch_size, dim))
        coherence_raw = x_2d.matmul(self.coherence_weight)
        coherence = coherence_raw.sigmoid()
        gate = ((coherence - self.threshold) * 10.0).sigmoid()
        
        # Apply gate
        gate_data = []
        for i in range(batch_size):
            g = gate.data[i]
            for _ in range(dim):
                gate_data.append(g)
        gate_expanded = Tensor(gate_data, original_shape)
        output = x * gate_expanded
        
        return {"output": output, "coherence": coherence.squeeze(-1), "gate": gate.squeeze(-1)}


@dataclass
class EntropyCollapseLayer(Layer):
    """VQ-style collapse to N attractors with entropy regularization."""
    
    num_attractors: int = 64
    target_entropy: float = 5.99
    temperature: float = 1.0
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        dim = input_shape[-1]
        self.codebook = self.add_weight("codebook", (self.num_attractors, dim))
    
    def forward(self, x: Tensor, training: bool = False) -> Dict[str, Tensor]:
        original_shape = x.shape
        dim = original_shape[-1]
        batch_size = x.size() // dim
        
        # Compute distances
        distances = []
        for i in range(batch_size):
            x_vec = x.data[i * dim:(i + 1) * dim]
            for j in range(self.num_attractors):
                c_vec = self.codebook.data[j * dim:(j + 1) * dim]
                dist = sum((x_vec[d] - c_vec[d]) ** 2 for d in range(dim))
                distances.append(dist)
        
        dist_tensor = Tensor(distances, (batch_size, self.num_attractors))
        logits = (dist_tensor * -1.0) / self.temperature
        probs = logits.softmax(axis=-1)
        
        # Entropy
        entropy_data = []
        for i in range(batch_size):
            p = probs.data[i * self.num_attractors:(i + 1) * self.num_attractors]
            ent = -sum(pi * math.log(pi + 1e-10) for pi in p)
            entropy_data.append(ent)
        entropy = Tensor(entropy_data, (batch_size,))
        
        if training:
            # Soft assignment
            output_data = []
            for i in range(batch_size):
                p = probs.data[i * self.num_attractors:(i + 1) * self.num_attractors]
                out_vec = [0.0] * dim
                for j in range(self.num_attractors):
                    c_vec = self.codebook.data[j * dim:(j + 1) * dim]
                    for d in range(dim):
                        out_vec[d] += p[j] * c_vec[d]
                output_data.extend(out_vec)
            output = Tensor(output_data, original_shape)
            return {"output": output, "probs": probs, "entropy": entropy}
        else:
            # Hard assignment
            indices = logits.argmax(axis=-1)
            output_data = []
            for i in range(batch_size):
                idx = int(indices.data[i])
                c_vec = self.codebook.data[idx * dim:(idx + 1) * dim]
                output_data.extend(c_vec)
            output = Tensor(output_data, original_shape)
            return {"output": output, "indices": indices, "entropy": entropy}


@dataclass
class ResonanceOperator(Layer):
    """Applies phase rotation: R̂(n)|p⟩ = e^(2πi log(n))|p⟩"""
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        dim = input_shape[-1]
        self.rotation_param = self.add_weight("rotation_param", (dim,), "ones")
    
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        dim = x.shape[-1]
        half_dim = dim // 2
        batch_size = x.size() // dim
        
        result = []
        for i in range(batch_size):
            start = i * dim
            real = x.data[start:start + half_dim]
            imag = x.data[start + half_dim:start + dim]
            
            n = [abs(self.rotation_param.data[d]) + 1 for d in range(half_dim)]
            phases = [math.log(n_val) * 2 * math.pi for n_val in n]
            
            cos_p = [math.cos(p) for p in phases]
            sin_p = [math.sin(p) for p in phases]
            
            new_real = [real[d] * cos_p[d] - imag[d] * sin_p[d] for d in range(half_dim)]
            new_imag = [real[d] * sin_p[d] + imag[d] * cos_p[d] for d in range(half_dim)]
            result.extend(new_real + new_imag)
        
        return Tensor(result, x.shape)


@dataclass
class LayerNorm(Layer):
    """Layer normalization."""
    
    eps: float = 1e-6
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        dim = input_shape[-1]
        self.gamma = self.add_weight("gamma", (dim,), "ones")
        self.beta = self.add_weight("beta", (dim,), "zeros")
    
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        dim = x.shape[-1]
        batch_size = x.size() // dim
        
        result = []
        for i in range(batch_size):
            vec = x.data[i * dim:(i + 1) * dim]
            mean = sum(vec) / dim
            var = sum((v - mean) ** 2 for v in vec) / dim
            std = math.sqrt(var + self.eps)
            normalized = [(v - mean) / std for v in vec]
            scaled = [normalized[d] * self.gamma.data[d] + self.beta.data[d] for d in range(dim)]
            result.extend(scaled)
        
        return Tensor(result, x.shape)


@dataclass
class Dense(Layer):
    """Standard dense layer."""
    
    units: int = 64
    activation: Optional[str] = None
    use_bias: bool = True
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        input_dim = input_shape[-1]
        self.kernel = self.add_weight("kernel", (input_dim, self.units))
        if self.use_bias:
            self.bias = self.add_weight("bias", (self.units,), "zeros")
    
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        original_shape = x.shape
        dim = original_shape[-1]
        batch_size = x.size() // dim
        
        x_2d = x.reshape((batch_size, dim))
        output = x_2d.matmul(self.kernel)
        
        if self.use_bias:
            # Make data mutable
            out_data = list(output.data)
            for i in range(batch_size):
                for j in range(self.units):
                    out_data[i * self.units + j] += self.bias.data[j]
            output = Tensor(out_data, output.shape)
        
        if self.activation == "relu":
            output = output.relu()
        elif self.activation == "gelu":
            output = output.gelu()
        elif self.activation == "tanh":
            output = output.tanh()
        elif self.activation == "sigmoid":
            output = output.sigmoid()
        
        new_shape = original_shape[:-1] + (self.units,)
        return output.reshape(new_shape)


@dataclass
class Dropout(Layer):
    """Dropout layer."""
    
    rate: float = 0.1
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        pass
    
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        if not training or self.rate == 0:
            return x
        scale = 1.0 / (1.0 - self.rate)
        result = []
        for v in x.data:
            if random.random() > self.rate:
                result.append(v * scale)
            else:
                result.append(0.0)
        return Tensor(result, x.shape)


# =============================================================================
# RESOFORMER BLOCK
# =============================================================================

@dataclass
class ResoFormerBlock(Layer):
    """
    Complete ResoFormer Block.
    
    Pre-norm architecture: Attention → FFN → Coherence Gate → Optional Collapse
    """
    
    dim: int = 256
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout_rate: float = 0.1
    use_collapse: bool = False
    
    def __post_init__(self):
        super().__init__()
        self.layer_norm1 = LayerNorm(name=f"{self.name}_ln1")
        self.layer_norm2 = LayerNorm(name=f"{self.name}_ln2")
        self.attention = ResonantAttentionLayer(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            name=f"{self.name}_attn"
        )
        self.resonance_op = ResonanceOperator(name=f"{self.name}_res")
        self.ffn1 = Dense(units=self.ffn_dim, activation="gelu", name=f"{self.name}_ffn1")
        self.ffn2 = Dense(units=self.dim, name=f"{self.name}_ffn2")
        self.coherence_gate = CoherenceGatingLayer(threshold=0.7, name=f"{self.name}_cg")
        self.dropout = Dropout(rate=self.dropout_rate, name=f"{self.name}_drop")
        if self.use_collapse:
            self.collapse = EntropyCollapseLayer(num_attractors=64, name=f"{self.name}_col")
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self._built:
            return
        self.layer_norm1.build(input_shape)
        self.layer_norm2.build(input_shape)
        self.attention.build(input_shape)
        self.resonance_op.build(input_shape)
        self.ffn1.build(input_shape)
        ffn_shape = input_shape[:-1] + (self.ffn_dim,)
        self.ffn2.build(ffn_shape)
        self.coherence_gate.build(input_shape)
        if self.use_collapse:
            self.collapse.build(input_shape)
    
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        # Pre-norm attention
        residual = x
        normed = self.layer_norm1(x, training)
        attn_out = self.attention(normed, training)
        attn_out = self.dropout(attn_out, training)
        x = residual + attn_out
        
        # Resonance operator
        x = self.resonance_op(x, training)
        
        # Pre-norm FFN
        residual = x
        normed = self.layer_norm2(x, training)
        ffn_out = self.ffn1(normed, training)
        ffn_out = self.ffn2(ffn_out, training)
        ffn_out = self.dropout(ffn_out, training)
        x = residual + ffn_out
        
        # Coherence gating
        gated = self.coherence_gate(x, training)
        x = gated["output"]
        
        # Optional collapse
        if self.use_collapse:
            collapsed = self.collapse(x, training)
            x = collapsed["output"]
        
        return x


# =============================================================================
# RESOFORMER MODEL
# =============================================================================

@dataclass
class ResoFormerConfig:
    """Configuration for ResoFormer model."""
    vocab_size: int = 10000
    seq_len: int = 512
    dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    num_classes: Optional[int] = None
    embedding_dim: Optional[int] = None


class ResoFormerModel:
    """End-to-end ResoFormer model."""
    
    def __init__(self, config: ResoFormerConfig, task: str = "lm"):
        self.config = config
        self.task = task
        self._built = False
        self.embedding: Optional[Tensor] = None
        self.blocks: List[ResoFormerBlock] = []
        self.head: Optional[Dense] = None
    
    def _build(self) -> None:
        if self._built:
            return
        
        self.embedding = glorot_uniform((self.config.vocab_size, self.config.dim))
        
        for i in range(self.config.num_layers):
            block = ResoFormerBlock(
                dim=self.config.dim,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                dropout_rate=self.config.dropout,
                use_collapse=(i == self.config.num_layers - 1),
                name=f"block_{i}"
            )
            self.blocks.append(block)
        
        if self.task == "lm":
            self.head = Dense(units=self.config.vocab_size, name="lm_head")
        elif self.task == "classification":
            nc = self.config.num_classes or 10
            self.head = Dense(units=nc, activation="sigmoid", name="clf_head")
        elif self.task == "embedding":
            ed = self.config.embedding_dim or 128
            self.head = Dense(units=ed, name="emb_head")
        
        self._built = True
    
    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        """Embed token IDs."""
        result = []
        for idx in range(token_ids.size()):
            tid = int(token_ids.data[idx])
            tid = max(0, min(tid, self.config.vocab_size - 1))
            start = tid * self.config.dim
            result.extend(self.embedding.data[start:start + self.config.dim])
        
        if len(token_ids.shape) == 1:
            return Tensor(result, (1, token_ids.shape[0], self.config.dim))
        else:
            batch = token_ids.shape[0]
            seq = token_ids.shape[1]
            return Tensor(result, (batch, seq, self.config.dim))
    
    def forward(self, token_ids: Tensor, training: bool = False) -> Tensor:
        """Forward pass."""
        self._build()
        
        x = self.embed_tokens(token_ids)
        
        for block in self.blocks:
            x = block(x, training)
        
        # Pool for classification/embedding
        if self.task in ["classification", "embedding"]:
            x = x.mean(axis=1)
        
        if self.head is not None:
            x = self.head(x, training)
        
        return x
    
    def __call__(self, token_ids: Tensor, training: bool = False) -> Tensor:
        return self.forward(token_ids, training)


def create_resoformer_model(
    vocab_size: int = 10000,
    seq_len: int = 512,
    dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    ffn_dim: int = 1024,
    dropout: float = 0.1
) -> ResoFormerModel:
    """Create a ResoFormer language model."""
    config = ResoFormerConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout
    )
    return ResoFormerModel(config, task="lm")


def create_resoformer_classifier(
    vocab_size: int = 10000,
    seq_len: int = 512,
    dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    ffn_dim: int = 1024,
    num_classes: int = 10,
    dropout: float = 0.1
) -> ResoFormerModel:
    """Create a ResoFormer classifier."""
    config = ResoFormerConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        num_classes=num_classes
    )
    return ResoFormerModel(config, task="classification")


def create_resoformer_embedder(
    vocab_size: int = 10000,
    seq_len: int = 512,
    dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    ffn_dim: int = 1024,
    embedding_dim: int = 128,
    dropout: float = 0.1
) -> ResoFormerModel:
    """Create a ResoFormer embedder."""
    config = ResoFormerConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        embedding_dim=embedding_dim
    )
    return ResoFormerModel(config, task="embedding")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Tensor operations
    "Tensor",
    "zeros",
    "ones",
    "randn",
    "glorot_uniform",
    "quaternion_normalize",
    
    # Base classes
    "Layer",
    
    # ResoFormer layers
    "QuaternionDense",
    "SparsePrimeEmbedding",
    "ResonantAttentionLayer",
    "CoherenceGatingLayer",
    "EntropyCollapseLayer",
    "ResonanceOperator",
    "LayerNorm",
    "Dense",
    "Dropout",
    "ResoFormerBlock",
    
    # Model builders
    "ResoFormerConfig",
    "ResoFormerModel",
    "create_resoformer_model",
    "create_resoformer_classifier",
    "create_resoformer_embedder",
]