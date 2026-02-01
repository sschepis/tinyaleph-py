"""
PyTorch backend for ResoFormer.

Provides PyTorch implementations of ResoFormer layers to enable autograd and GPU acceleration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class QuaternionDense(nn.Module):
    """Dense layer projecting to quaternion space."""
    
    def __init__(self, input_dim: int, units: int, use_bias: bool = True):
        super().__init__()
        self.units = units
        self.kernel = nn.Linear(input_dim, units * 4, bias=use_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.kernel(x)
        
        # Normalize quaternions
        shape = output.shape
        output = output.view(*shape[:-1], self.units, 4)
        output = F.normalize(output, p=2, dim=-1)
        
        return output

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x: torch.Tensor, seq_len: int):
        if self.cached_cos is None or self.cached_cos.shape[0] < seq_len:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()[None, None, :, :]
            self.cached_sin = emb.sin()[None, None, :, :]
        
        return self.cached_cos[:, :, :seq_len, :], self.cached_sin[:, :, :seq_len, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ResonantAttentionLayer(nn.Module):
    """Multi-head resonant attention layer."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(v, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal masking
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.o_proj(attn_output)

class CoherenceGatingLayer(nn.Module):
    """Coherence-based gating mechanism."""
    
    def __init__(self, dim: int, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.coherence_proj = nn.Linear(dim, 1)
        
        # Initialize to be open by default
        # Bias = 1.0 -> sigmoid(1.0) ~= 0.73
        # If threshold is 0.5, gate input is (0.73 - 0.5) * 10 = 2.3
        # Gate activation is sigmoid(2.3) ~= 0.91 (Open)
        nn.init.constant_(self.coherence_proj.bias, 1.0)
        nn.init.xavier_uniform_(self.coherence_proj.weight, gain=0.01)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        coherence_raw = self.coherence_proj(x)
        coherence = torch.sigmoid(coherence_raw)
        # Reduced slope from 10.0 to 2.0 to prevent gradient spikes
        gate = torch.sigmoid((coherence - self.threshold) * 2.0)
        
        output = x * gate
        return {"output": output, "coherence": coherence.squeeze(-1), "gate": gate.squeeze(-1)}

class ResoFormerBlock(nn.Module):
    """
    Complete ResoFormer Block.
    
    Pre-norm architecture: Attention → FFN → Coherence Gate
    """
    
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, 
                 use_gating: bool = True, coherence_threshold: float = 0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ResonantAttentionLayer(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.use_gating = use_gating
        if use_gating:
            self.gate = CoherenceGatingLayer(dim, threshold=coherence_threshold)
        else:
            self.gate = None
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        
        # Coherence gating applied to the update, preserving the residual stream
        if self.use_gating and self.gate is not None:
            gated = self.gate(x)
            x = gated["output"]
            
        x = residual + x
        
        return x

def get_device() -> torch.device:
    """Get the best available device (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
