"""
Quaternion Operations for LLM Fusion.

Implements quaternion algebra and neural network layers operating
in quaternion space (ℍ). Quaternions provide a natural 4D extension
of complex numbers with geometric rotation properties.

Mathematical Foundation:
    q = w + xi + yj + zk
    i² = j² = k² = ijk = -1
    
Hamilton Product:
    q1 * q2 = (w1w2 - x1x2 - y1y2 - z1z2) +
              (w1x2 + x1w2 + y1z2 - z1y2)i +
              (w1y2 - x1z2 + y1w2 + z1x2)j +
              (w1z2 + x1y2 - y1x2 + z1w2)k
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Golden ratio for geometric operations
PHI: float = (1 + math.sqrt(5)) / 2


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternions.
    
    Args:
        q1: (..., 4) first quaternion (w, x, y, z)
        q2: (..., 4) second quaternion
        
    Returns:
        product: (..., 4) Hamilton product q1 * q2
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Conjugate of a quaternion: q* = w - xi - yj - zk.
    
    Args:
        q: (..., 4) quaternion
        
    Returns:
        conjugate: (..., 4)
    """
    return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)


def quaternion_norm(q: torch.Tensor) -> torch.Tensor:
    """
    Norm of a quaternion: |q| = sqrt(w² + x² + y² + z²).
    
    Args:
        q: (..., 4) quaternion
        
    Returns:
        norm: (...,) scalar norm
    """
    return torch.sqrt((q ** 2).sum(dim=-1))


def quaternion_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize quaternion to unit quaternion.
    
    Args:
        q: (..., 4) quaternion
        eps: Small value for numerical stability
        
    Returns:
        normalized: (..., 4) unit quaternion
    """
    norm = quaternion_norm(q).unsqueeze(-1)
    return q / (norm + eps)


def quaternion_inverse(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Inverse of a quaternion: q⁻¹ = q* / |q|².
    
    Args:
        q: (..., 4) quaternion
        
    Returns:
        inverse: (..., 4)
    """
    conj = quaternion_conjugate(q)
    norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
    return conj / (norm_sq + eps)


def quaternion_exp(q: torch.Tensor) -> torch.Tensor:
    """
    Exponential of a quaternion: exp(q).
    
    For q = w + v (scalar + vector):
        exp(q) = e^w * (cos|v| + v/|v| * sin|v|)
        
    Args:
        q: (..., 4) quaternion
        
    Returns:
        exp_q: (..., 4)
    """
    w = q[..., 0:1]  # scalar part
    v = q[..., 1:4]  # vector part
    
    v_norm = torch.sqrt((v ** 2).sum(dim=-1, keepdim=True) + 1e-8)
    exp_w = torch.exp(w)
    
    cos_v = torch.cos(v_norm)
    sin_v = torch.sin(v_norm) / (v_norm + 1e-8)
    
    return torch.cat([exp_w * cos_v, exp_w * sin_v * v], dim=-1)


def quaternion_log(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Logarithm of a quaternion: log(q).
    
    For unit quaternion q = cos(θ) + sin(θ)n:
        log(q) = θ * n
        
    Args:
        q: (..., 4) quaternion (should be unit quaternion)
        
    Returns:
        log_q: (..., 4)
    """
    q_norm = quaternion_norm(q).unsqueeze(-1)
    q = q / (q_norm + eps)  # normalize
    
    w = q[..., 0:1]
    v = q[..., 1:4]
    
    v_norm = torch.sqrt((v ** 2).sum(dim=-1, keepdim=True) + eps)
    theta = torch.acos(torch.clamp(w, -1 + eps, 1 - eps))
    
    # When v_norm is small, use Taylor expansion
    n = v / (v_norm + eps) * theta
    
    return torch.cat([torch.log(q_norm + eps), n], dim=-1)


def quaternion_slerp(
    q1: torch.Tensor,
    q2: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Spherical linear interpolation between quaternions.
    
    SLERP provides the shortest path interpolation on the unit sphere.
    
    Args:
        q1: (..., 4) starting quaternion
        q2: (..., 4) ending quaternion
        t: (...,) or scalar interpolation parameter in [0, 1]
        
    Returns:
        interpolated: (..., 4)
    """
    # Normalize inputs
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)
    
    # Compute dot product
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    
    # If negative, negate q2 to take shorter path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    
    # If quaternions are close, use linear interpolation
    linear_threshold = 0.9995
    linear_mask = dot > linear_threshold
    
    # Compute angles
    theta = torch.acos(torch.clamp(dot, -1, 1))
    sin_theta = torch.sin(theta)
    
    # Expand t if necessary
    if t.dim() < q1.dim():
        t = t.unsqueeze(-1)
    
    # SLERP formula
    w1 = torch.sin((1 - t) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(t * theta) / (sin_theta + 1e-8)
    
    result = w1 * q1 + w2 * q2
    
    # Use linear interpolation where angles are small
    linear_result = (1 - t) * q1 + t * q2
    linear_result = quaternion_normalize(linear_result)
    
    result = torch.where(linear_mask, linear_result, result)
    return quaternion_normalize(result)


def quaternion_rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate 3D vector by unit quaternion: v' = q * v * q*.
    
    Args:
        q: (..., 4) unit quaternion
        v: (..., 3) vector to rotate
        
    Returns:
        rotated: (..., 3) rotated vector
    """
    # Convert v to pure quaternion (0, vx, vy, vz)
    v_quat = F.pad(v, (1, 0), value=0)  # (..., 4)
    
    # Rotate: q * v * q*
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    
    return rotated[..., 1:4]  # Extract vector part


class QuaternionLinear(nn.Module):
    """
    Quaternion linear layer.
    
    Performs linear transformation in quaternion space using
    Hamilton product with learnable weight quaternions.
    
    Each output quaternion is:
        y_j = Σ_i W_{ij} * x_i  (Hamilton product)
        
    Args:
        in_features: Number of input quaternions
        out_features: Number of output quaternions
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight is a quaternion matrix: (out, in, 4)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 4))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, 4))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using quaternion Xavier."""
        # Quaternion Xavier: scale by 1/sqrt(4 * in_features)
        std = 1.0 / math.sqrt(4 * self.in_features)
        nn.init.normal_(self.weight, mean=0, std=std)
        
        # Initialize scalar part slightly larger for stability
        with torch.no_grad():
            self.weight[..., 0] += 0.5
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (..., in_features, 4) input quaternions
            
        Returns:
            y: (..., out_features, 4) output quaternions
        """
        # x: (..., in, 4)
        # weight: (out, in, 4)
        
        # Expand for broadcasting
        # x_expanded: (..., 1, in, 4)
        x_expanded = x.unsqueeze(-3)
        
        # Hamilton product for each output
        # result: (..., out, in, 4)
        products = quaternion_multiply(
            self.weight,  # (out, in, 4)
            x_expanded    # (..., 1, in, 4)
        )
        
        # Sum over input dimension
        # y: (..., out, 4)
        y = products.sum(dim=-2)
        
        if self.bias is not None:
            y = y + self.bias
        
        return y


class QuaternionAttention(nn.Module):
    """
    Attention mechanism using quaternion inner products.
    
    Attention scores are computed from the quaternionic inner product:
        ⟨q1, q2⟩ = Re(q1* ⊗ q2)
        
    This provides geometric awareness through rotation-sensitive similarity.
    
    Args:
        hidden_dim: Hidden dimension (per quaternion = hidden_dim / 4)
        num_heads: Number of attention heads
        dropout: Attention dropout probability
        golden_ratio_heads: Use golden ratio spacing for head rotations
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        golden_ratio_heads: bool = True,
    ):
        super().__init__()
        
        assert hidden_dim % (4 * num_heads) == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by 4 * num_heads ({4 * num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.quat_per_head = self.head_dim // 4
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Golden ratio rotation axes for heads
        if golden_ratio_heads:
            angles = torch.arange(num_heads) * math.pi / PHI
            axes = torch.stack([
                torch.cos(angles),
                torch.sin(angles),
                torch.zeros_like(angles)
            ], dim=-1)  # (num_heads, 3)
            self.register_buffer("head_axes", axes)
        else:
            self.head_axes = None
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            attention_mask: (batch, seq) or (batch, 1, seq, seq)
            return_attention: Whether to return attention weights
            
        Returns:
            output: (batch, seq, hidden_dim)
            attention_weights: (batch, num_heads, seq, seq) if return_attention
        """
        batch, seq, _ = hidden_states.shape
        
        # Project to Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Reshape to (batch, num_heads, seq, head_dim)
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape to quaternions: (batch, heads, seq, quat_per_head, 4)
        Q_quat = Q.view(batch, self.num_heads, seq, self.quat_per_head, 4)
        K_quat = K.view(batch, self.num_heads, seq, self.quat_per_head, 4)
        
        # Quaternion inner product: ⟨q1, q2⟩ = Re(q1* · q2)
        K_conj = quaternion_conjugate(K_quat)  # (batch, heads, seq, quat, 4)
        
        # Compute attention scores via inner product
        # For each pair of positions, sum over quaternion dimensions
        # scores[i,j] = Σ_q Re(Q[i,q]* · K[j,q])
        
        # Expand for pairwise computation
        Q_exp = Q_quat.unsqueeze(3)  # (batch, heads, seq_q, 1, quat, 4)
        K_exp = K_conj.unsqueeze(2)  # (batch, heads, 1, seq_k, quat, 4)
        
        # Hamilton product
        prod = quaternion_multiply(K_exp, Q_exp)  # (batch, heads, seq_q, seq_k, quat, 4)
        
        # Take real part and sum over quaternions
        scores = prod[..., 0].sum(dim=-1)  # (batch, heads, seq_q, seq_k)
        scores = scores * self.scale
        
        # Apply mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # attn_weights: (batch, heads, seq_q, seq_k)
        # V: (batch, heads, seq_k, head_dim)
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class QuaternionRotationLayer(nn.Module):
    """
    Apply learned quaternion rotations to hidden states.
    
    Each position in the sequence receives a rotation based on
    its content, enabling geometric transformations of the
    representation space.
    
    Args:
        hidden_dim: Hidden dimension
        use_position_rotation: Add position-dependent rotation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        use_position_rotation: bool = True,
    ):
        super().__init__()
        
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4"
        
        self.hidden_dim = hidden_dim
        self.num_quaternions = hidden_dim // 4
        
        # Learn rotation axis and angle from content
        self.axis_proj = nn.Linear(hidden_dim, 3)  # 3D rotation axis
        self.angle_proj = nn.Linear(hidden_dim, 1)  # rotation angle
        
        self.use_position_rotation = use_position_rotation
        if use_position_rotation:
            # Learnable position rotations
            self.position_rotation = nn.Parameter(
                torch.randn(1000, 4) * 0.1  # max 1000 positions
            )
            # Initialize close to identity
            with torch.no_grad():
                self.position_rotation[:, 0] = 1.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply quaternion rotations.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            positions: (batch, seq) position indices
            
        Returns:
            rotated: (batch, seq, hidden_dim)
        """
        batch, seq, _ = hidden_states.shape
        
        # Compute rotation quaternion from content
        axis = self.axis_proj(hidden_states)  # (batch, seq, 3)
        axis = F.normalize(axis, p=2, dim=-1)  # unit vector
        
        angle = self.angle_proj(hidden_states)  # (batch, seq, 1)
        angle = torch.tanh(angle) * math.pi  # limit to [-π, π]
        
        # Create rotation quaternion: q = cos(θ/2) + sin(θ/2)(xi + yj + zk)
        half_angle = angle / 2
        w = torch.cos(half_angle)  # (batch, seq, 1)
        xyz = torch.sin(half_angle) * axis  # (batch, seq, 3)
        rotation_quat = torch.cat([w, xyz], dim=-1)  # (batch, seq, 4)
        
        # Add position rotation if enabled
        if self.use_position_rotation and positions is not None:
            pos_indices = positions.clamp(0, self.position_rotation.size(0) - 1)
            pos_rot = self.position_rotation[pos_indices]  # (batch, seq, 4)
            pos_rot = quaternion_normalize(pos_rot)
            rotation_quat = quaternion_multiply(pos_rot, rotation_quat)
            rotation_quat = quaternion_normalize(rotation_quat)
        
        # Reshape hidden states to quaternions
        quats = hidden_states.view(batch, seq, self.num_quaternions, 4)
        
        # Apply rotation: q' = r * q * r*
        r = rotation_quat.unsqueeze(2)  # (batch, seq, 1, 4)
        r_conj = quaternion_conjugate(r)
        
        rotated = quaternion_multiply(quaternion_multiply(r, quats), r_conj)
        
        # Reshape back
        return rotated.view(batch, seq, self.hidden_dim)
