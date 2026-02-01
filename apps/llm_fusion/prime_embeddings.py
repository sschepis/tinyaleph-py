"""
Prime Projection Layer for LLM Fusion.

Maps standard transformer embeddings into Prime Hilbert Space,
where basis vectors are indexed by prime numbers.

Mathematical Foundation:
    |ψ⟩ = Σ αp|p⟩ where p ∈ Primes
    
With quaternionic amplitudes:
    |ψ⟩ = Σ qp|p⟩ where qp ∈ ℍ (quaternion)
"""
import sys
import os
import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tinyaleph.core.primes import first_n_primes


class PrimeProjection(nn.Module):
    """
    Project embeddings into Prime Hilbert Space.
    
    Maps hidden states from standard transformer space to prime basis
    with optional quaternionic amplitudes.
    
    State representation:
        Complex (use_quaternion=False): (batch, seq, num_primes, 2)
        Quaternion (use_quaternion=True): (batch, seq, num_primes, 4)
    
    Args:
        input_dim: Dimension of input hidden states
        num_primes: Number of prime basis states (default 25)
        use_quaternion: Use quaternionic (4D) vs complex (2D) amplitudes
        normalize: Whether to normalize output states
        learnable_phases: Whether to learn phase biases per prime
    """
    
    def __init__(
        self,
        input_dim: int,
        num_primes: int = 25,
        use_quaternion: bool = True,
        normalize: bool = True,
        learnable_phases: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_primes = num_primes
        self.use_quaternion = use_quaternion
        self.normalize = normalize
        self.amp_dim = 4 if use_quaternion else 2
        
        # Get actual prime numbers for reference
        self.register_buffer(
            "primes",
            torch.tensor(first_n_primes(num_primes), dtype=torch.long)
        )
        
        # Output dimension
        self.output_dim = num_primes * self.amp_dim
        
        # Projection to prime amplitudes
        self.prime_proj = nn.Linear(input_dim, self.output_dim)
        
        # Learnable phase biases per prime
        if learnable_phases:
            self.phase_bias = nn.Parameter(torch.zeros(num_primes))
        else:
            self.register_buffer("phase_bias", torch.zeros(num_primes))
        
        # Prime weighting (log-spaced based on prime values)
        log_primes = torch.log(self.primes.float())
        prime_weights = 1.0 / (1.0 + log_primes - log_primes.min())
        self.register_buffer("prime_weights", prime_weights)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with prime-aware scaling."""
        # Xavier initialization scaled by prime structure
        nn.init.xavier_uniform_(self.prime_proj.weight, gain=0.5)
        nn.init.zeros_(self.prime_proj.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_entropy: bool = False
    ) -> torch.Tensor:
        """
        Project hidden states to prime space.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            return_entropy: Whether to also return entropy values
            
        Returns:
            prime_states: (batch, seq, num_primes, amp_dim)
            entropy (optional): (batch, seq) entropy values
        """
        batch, seq, _ = hidden_states.shape
        
        # Project to prime space
        projected = self.prime_proj(hidden_states)  # (batch, seq, num_primes * amp_dim)
        
        # Reshape to (batch, seq, num_primes, amp_dim)
        prime_states = projected.view(batch, seq, self.num_primes, self.amp_dim)
        
        # Apply phase bias (rotation in complex/quaternion space)
        prime_states = self._apply_phase_bias(prime_states)
        
        # Apply prime weighting
        prime_states = prime_states * self.prime_weights.view(1, 1, -1, 1)
        
        # Compute entropy BEFORE normalization (to get meaningful distribution)
        if return_entropy:
            entropy = self._compute_entropy(prime_states)
        
        # Normalize if requested (after entropy computation)
        if self.normalize:
            prime_states = self._normalize_amplitudes(prime_states)
        
        if return_entropy:
            return prime_states, entropy
        
        return prime_states
    
    def _apply_phase_bias(self, prime_states: torch.Tensor) -> torch.Tensor:
        """Apply learnable phase rotation to each prime."""
        if self.use_quaternion:
            # Quaternion rotation: q' = r * q where r = (cos(θ/2), sin(θ/2), 0, 0)
            half_phase = self.phase_bias / 2  # (num_primes,)
            cos_hp = torch.cos(half_phase).view(1, 1, -1, 1)
            sin_hp = torch.sin(half_phase).view(1, 1, -1, 1)
            
            # Simple rotation in w-i plane
            w = prime_states[..., 0:1]
            i = prime_states[..., 1:2]
            j = prime_states[..., 2:3]
            k = prime_states[..., 3:4]
            
            # Rotate (w, i) by phase
            new_w = cos_hp * w - sin_hp * i
            new_i = sin_hp * w + cos_hp * i
            
            return torch.cat([new_w, new_i, j, k], dim=-1)
        else:
            # Complex rotation: z' = e^(iθ) * z
            phase = self.phase_bias.view(1, 1, -1, 1)
            cos_p = torch.cos(phase)
            sin_p = torch.sin(phase)
            
            real = prime_states[..., 0:1]
            imag = prime_states[..., 1:2]
            
            new_real = cos_p * real - sin_p * imag
            new_imag = sin_p * real + cos_p * imag
            
            return torch.cat([new_real, new_imag], dim=-1)
    
    def _normalize_amplitudes(self, prime_states: torch.Tensor) -> torch.Tensor:
        """Normalize amplitudes to unit total norm per position (across all primes)."""
        # Flatten primes and amplitude dimensions for normalization
        batch, seq, num_primes, amp_dim = prime_states.shape
        flat = prime_states.view(batch, seq, -1)  # (batch, seq, num_primes * amp_dim)
        
        # Normalize to unit total norm (preserves relative magnitudes between primes)
        normalized = F.normalize(flat, p=2, dim=-1)
        
        return normalized.view(batch, seq, num_primes, amp_dim)
    
    def _compute_entropy(self, prime_states: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the prime state distribution.
        
        S = -Σ |αp|² log₂(|αp|²)
        
        Returns:
            entropy: (batch, seq) entropy values in bits
        """
        # Compute probability distribution |αp|²
        probs = (prime_states ** 2).sum(dim=-1)  # (batch, seq, num_primes)
        
        # Normalize to probability distribution
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Shannon entropy
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq)
        
        return entropy
    
    def to_hidden(self, prime_states: torch.Tensor) -> torch.Tensor:
        """
        Project prime states back to hidden dimension.
        
        This is a convenience method; typically you'd use a separate
        back-projection layer.
        
        Args:
            prime_states: (batch, seq, num_primes, amp_dim)
            
        Returns:
            hidden: (batch, seq, output_dim) where output_dim = num_primes * amp_dim
        """
        batch, seq, _, _ = prime_states.shape
        return prime_states.view(batch, seq, -1)
    
    def coherence(self, prime_states: torch.Tensor) -> torch.Tensor:
        """
        Compute coherence from entropy.
        
        C = 1 - S / S_max where S_max = log₂(num_primes)
        
        Note: This method works best with un-normalized states.
        After normalization, all primes have equal weight which
        maximizes entropy and minimizes coherence.
        
        Returns:
            coherence: (batch, seq) values in [0, 1]
        """
        # If normalized (each quaternion is unit length), we need to
        # look at the magnitude before the last dimension normalization
        # Compute squared magnitudes per prime
        prime_magnitudes = (prime_states ** 2).sum(dim=-1)  # (batch, seq, num_primes)
        
        # Normalize to probability distribution across primes
        probs = prime_magnitudes / (prime_magnitudes.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Shannon entropy
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq)
        
        max_entropy = math.log2(self.num_primes)
        coherence = 1.0 - entropy / max_entropy
        
        # Clamp to [0, 1]
        return torch.clamp(coherence, 0.0, 1.0)


class PrimeBackProjection(nn.Module):
    """
    Project prime states back to transformer hidden space.
    
    Args:
        num_primes: Number of prime basis states
        output_dim: Target hidden dimension
        use_quaternion: Whether input uses quaternionic amplitudes
    """
    
    def __init__(
        self,
        num_primes: int,
        output_dim: int,
        use_quaternion: bool = True,
    ):
        super().__init__()
        
        self.num_primes = num_primes
        self.output_dim = output_dim
        self.amp_dim = 4 if use_quaternion else 2
        
        # Input dimension
        input_dim = num_primes * self.amp_dim
        
        # Back projection
        self.back_proj = nn.Linear(input_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.back_proj.weight, gain=0.5)
        nn.init.zeros_(self.back_proj.bias)
    
    def forward(self, prime_states: torch.Tensor) -> torch.Tensor:
        """
        Project prime states to hidden dimension.
        
        Args:
            prime_states: (batch, seq, num_primes, amp_dim)
            
        Returns:
            hidden: (batch, seq, output_dim)
        """
        batch, seq, _, _ = prime_states.shape
        flat = prime_states.view(batch, seq, -1)
        return self.back_proj(flat)


class SparsePrimeProjection(nn.Module):
    """
    Sparse prime projection that only activates k primes per token.
    
    Uses top-k selection to maintain sparsity in prime space,
    which is more efficient for large num_primes.
    
    Args:
        input_dim: Dimension of input hidden states
        num_primes: Total number of prime basis states
        active_k: Number of primes to activate per token
        use_quaternion: Use quaternionic amplitudes
    """
    
    def __init__(
        self,
        input_dim: int,
        num_primes: int = 64,
        active_k: int = 8,
        use_quaternion: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_primes = num_primes
        self.active_k = active_k
        self.use_quaternion = use_quaternion
        self.amp_dim = 4 if use_quaternion else 2
        
        # Get prime numbers
        self.register_buffer(
            "primes",
            torch.tensor(first_n_primes(num_primes), dtype=torch.long)
        )
        
        # Score each prime for activation
        self.score_proj = nn.Linear(input_dim, num_primes)
        
        # Amplitude projection (only for top-k)
        self.amp_proj = nn.Linear(input_dim, active_k * self.amp_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project to sparse prime representation.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            
        Returns:
            prime_indices: (batch, seq, active_k) - which primes are active
            prime_amps: (batch, seq, active_k, amp_dim) - amplitudes
        """
        batch, seq, _ = hidden_states.shape
        
        # Score primes
        scores = self.score_proj(hidden_states)  # (batch, seq, num_primes)
        
        # Select top-k
        topk_scores, topk_indices = torch.topk(scores, self.active_k, dim=-1)
        
        # Compute amplitudes
        amps = self.amp_proj(hidden_states)  # (batch, seq, k * amp_dim)
        amps = amps.view(batch, seq, self.active_k, self.amp_dim)
        
        # Weight by softmax of scores
        weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # (batch, seq, k, 1)
        amps = amps * weights
        
        # Normalize
        amps = F.normalize(amps, p=2, dim=-1)
        
        return topk_indices, amps
    
    def to_dense(
        self,
        prime_indices: torch.Tensor,
        prime_amps: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert sparse representation to dense.
        
        Returns:
            dense: (batch, seq, num_primes, amp_dim)
        """
        batch, seq, k = prime_indices.shape
        device = prime_indices.device
        
        # Initialize dense tensor
        dense = torch.zeros(
            batch, seq, self.num_primes, self.amp_dim,
            device=device, dtype=prime_amps.dtype
        )
        
        # Scatter amplitudes to correct positions
        indices_expanded = prime_indices.unsqueeze(-1).expand(-1, -1, -1, self.amp_dim)
        dense.scatter_(2, indices_expanded, prime_amps)
        
        return dense
