"""
Kuramoto Synchronization for Attention Modulation.

Implements coupled oscillator dynamics to modulate attention patterns,
enabling emergent synchronization that can enhance coherence.

Mathematical Foundation (Kuramoto Model):
    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)
    
Order Parameter:
    r·e^(iψ) = (1/N) Σ_j e^(iθ_j)
    r ∈ [0,1] measures global synchronization
    
Attention Modulation:
    A' = A · f(r, θ)  where f depends on phase coherence
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Golden ratio for natural frequencies
PHI: float = (1 + math.sqrt(5)) / 2


def compute_order_parameter(phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Kuramoto order parameter from phases.
    
    r·e^(iψ) = (1/N) Σ_j e^(iθ_j)
    
    Args:
        phases: (..., N) oscillator phases in radians
        
    Returns:
        r: (...,) order parameter magnitude [0, 1]
        psi: (...,) mean phase
    """
    # Complex exponential
    exp_phases = torch.stack([
        torch.cos(phases),
        torch.sin(phases)
    ], dim=-1)  # (..., N, 2)
    
    # Mean
    mean_exp = exp_phases.mean(dim=-2)  # (..., 2)
    
    # Magnitude and phase
    r = torch.sqrt(mean_exp[..., 0]**2 + mean_exp[..., 1]**2)
    psi = torch.atan2(mean_exp[..., 1], mean_exp[..., 0])
    
    return r, psi


def kuramoto_step(
    phases: torch.Tensor,
    frequencies: torch.Tensor,
    coupling: float,
    dt: float = 0.01,
) -> torch.Tensor:
    """
    One integration step of Kuramoto dynamics.
    
    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)
    
    Args:
        phases: (..., N) current phases
        frequencies: (..., N) natural frequencies
        coupling: K, coupling strength
        dt: integration time step
        
    Returns:
        new_phases: (..., N) updated phases
    """
    N = phases.size(-1)
    
    # Phase differences: θ_j - θ_i for all pairs
    # phases: (..., N)
    # Expand to (..., N, 1) - (..., 1, N) = (..., N, N)
    phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # (..., N, N)
    
    # Interaction term: (K/N) Σ_j sin(θ_j - θ_i)
    interaction = (coupling / N) * torch.sin(phase_diff).sum(dim=-1)  # (..., N)
    
    # Euler step
    d_phases = frequencies + interaction
    new_phases = phases + dt * d_phases
    
    # Wrap to [-π, π]
    new_phases = torch.remainder(new_phases + math.pi, 2 * math.pi) - math.pi
    
    return new_phases


class KuramotoModule(nn.Module):
    """
    Learnable Kuramoto oscillator module.
    
    Maps input features to oscillator phases, runs Kuramoto dynamics,
    and outputs synchronized phase information.
    
    Args:
        input_dim: Dimension of input features
        num_oscillators: Number of coupled oscillators (default: same as input positions)
        coupling: Base coupling strength K
        num_steps: Number of integration steps
        dt: Time step
        use_learned_frequencies: Whether to learn natural frequencies
    """
    
    def __init__(
        self,
        input_dim: int,
        num_oscillators: Optional[int] = None,
        coupling: float = 1.0,
        num_steps: int = 5,
        dt: float = 0.01,
        use_learned_frequencies: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_oscillators = num_oscillators
        self.coupling = coupling
        self.num_steps = num_steps
        self.dt = dt
        
        # Project input to initial phases
        self.phase_proj = nn.Linear(input_dim, 1)
        
        # Learnable frequencies
        if use_learned_frequencies:
            self.freq_proj = nn.Linear(input_dim, 1)
        else:
            self.freq_proj = None
        
        # Learnable coupling modulation
        self.coupling_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Run Kuramoto dynamics on input states.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            return_trajectory: Whether to return full phase trajectory
            
        Returns:
            final_phases: (batch, seq) final oscillator phases
            order_param: (batch,) final order parameter r
            trajectory: (batch, num_steps, seq) if return_trajectory
        """
        batch, seq, _ = hidden_states.shape
        
        # Initialize phases from input
        phases = self.phase_proj(hidden_states).squeeze(-1)  # (batch, seq)
        phases = torch.tanh(phases) * math.pi  # Initial phases in [-π, π]
        
        # Compute frequencies
        if self.freq_proj is not None:
            frequencies = self.freq_proj(hidden_states).squeeze(-1)  # (batch, seq)
            # Scale to reasonable range and add golden ratio spacing
            frequencies = frequencies + torch.arange(seq, device=phases.device) / PHI
        else:
            # Natural frequencies proportional to position
            frequencies = torch.arange(seq, device=phases.device, dtype=phases.dtype)
            frequencies = frequencies.unsqueeze(0).expand(batch, -1) / PHI
        
        # Effective coupling
        K = self.coupling * torch.sigmoid(self.coupling_scale)
        
        # Trajectory storage
        if return_trajectory:
            trajectory = [phases.unsqueeze(1)]
        
        # Run Kuramoto dynamics
        for _ in range(self.num_steps):
            phases = kuramoto_step(phases, frequencies, K.item(), self.dt)
            if return_trajectory:
                trajectory.append(phases.unsqueeze(1))
        
        # Compute final order parameter
        r, _ = compute_order_parameter(phases)
        
        if return_trajectory:
            trajectory = torch.cat(trajectory, dim=1)  # (batch, num_steps+1, seq)
            return phases, r, trajectory
        
        return phases, r, None


class KuramotoAttentionModulator(nn.Module):
    """
    Modulate attention scores using Kuramoto synchronization.
    
    The modulation works by:
    1. Running Kuramoto dynamics on query/key positions
    2. Computing phase coherence between positions
    3. Scaling attention scores by coherence
    
    This encourages the model to attend to positions that are
    "in phase" with the query, promoting coherent information flow.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        coupling: Kuramoto coupling strength
        num_steps: Integration steps
        modulation_strength: How strongly to modulate attention (0-1)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        coupling: float = 1.0,
        num_steps: int = 5,
        modulation_strength: float = 0.3,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.modulation_strength = modulation_strength
        
        # Per-head Kuramoto modules
        self.kuramoto = KuramotoModule(
            input_dim=hidden_dim // num_heads,
            coupling=coupling,
            num_steps=num_steps,
        )
        
        # Learnable modulation strength per head
        self.head_modulation = nn.Parameter(
            torch.ones(num_heads) * modulation_strength
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modulate attention scores based on Kuramoto dynamics.
        
        Args:
            query: (batch, num_heads, seq_q, head_dim)
            key: (batch, num_heads, seq_k, head_dim)
            attention_scores: (batch, num_heads, seq_q, seq_k)
            
        Returns:
            modulated_scores: (batch, num_heads, seq_q, seq_k)
            order_param: (batch, num_heads) synchronization level
        """
        batch, num_heads, seq_q, head_dim = query.shape
        seq_k = key.size(2)
        
        # Process each head
        order_params = []
        modulated = []
        
        for h in range(num_heads):
            q_h = query[:, h]  # (batch, seq_q, head_dim)
            k_h = key[:, h]    # (batch, seq_k, head_dim)
            scores_h = attention_scores[:, h]  # (batch, seq_q, seq_k)
            
            # Run Kuramoto on query positions
            q_phases, q_r, _ = self.kuramoto(q_h)  # (batch, seq_q)
            
            # Run Kuramoto on key positions
            k_phases, k_r, _ = self.kuramoto(k_h)  # (batch, seq_k)
            
            order_params.append((q_r + k_r) / 2)  # Average order param
            
            # Phase coherence between query and key positions
            # phase_diff[i,j] = |θ_q[i] - θ_k[j]|
            phase_diff = q_phases.unsqueeze(-1) - k_phases.unsqueeze(-2)  # (batch, seq_q, seq_k)
            coherence = torch.cos(phase_diff)  # ∈ [-1, 1], high when in phase
            
            # Modulation factor: boost in-phase attention, suppress out-of-phase
            # modulation ∈ [1 - α, 1 + α] where α is head_modulation
            alpha = torch.sigmoid(self.head_modulation[h])
            modulation = 1.0 + alpha * coherence
            
            modulated_h = scores_h * modulation
            modulated.append(modulated_h.unsqueeze(1))
        
        modulated_scores = torch.cat(modulated, dim=1)  # (batch, heads, seq_q, seq_k)
        order_param = torch.stack(order_params, dim=1)  # (batch, heads)
        
        return modulated_scores, order_param


class GlobalSynchronizationLayer(nn.Module):
    """
    Global synchronization layer that provides sequence-level coherence.
    
    Instead of modulating attention, this layer adds a synchronized
    component to hidden states based on their phase relationships.
    
    Args:
        hidden_dim: Hidden dimension
        coupling: Kuramoto coupling strength
        num_steps: Number of integration steps
        blend_factor: How much synchronized signal to blend (0-1)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        coupling: float = 1.5,
        num_steps: int = 10,
        blend_factor: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.blend_factor = blend_factor
        
        # Kuramoto for global synchronization
        self.kuramoto = KuramotoModule(
            input_dim=hidden_dim,
            coupling=coupling,
            num_steps=num_steps,
            use_learned_frequencies=True,
        )
        
        # Project phases back to hidden space
        self.phase_to_hidden = nn.Linear(2, hidden_dim)  # cos(θ), sin(θ)
        
        # Learnable blend factor
        self.blend = nn.Parameter(torch.tensor(blend_factor))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add synchronized component to hidden states.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            
        Returns:
            output: (batch, seq, hidden_dim)
            order_param: (batch,) synchronization level
        """
        # Run Kuramoto
        phases, r, _ = self.kuramoto(hidden_states)  # phases: (batch, seq)
        
        # Convert phases to 2D representation
        phase_2d = torch.stack([
            torch.cos(phases),
            torch.sin(phases)
        ], dim=-1)  # (batch, seq, 2)
        
        # Project to hidden dimension
        phase_hidden = self.phase_to_hidden(phase_2d)  # (batch, seq, hidden_dim)
        
        # Blend with original
        # Blend factor scales with order parameter - more sync = more blend
        effective_blend = torch.sigmoid(self.blend) * r.unsqueeze(-1).unsqueeze(-1)
        output = hidden_states + effective_blend * phase_hidden
        
        return output, r


class AdaptiveCouplingLayer(nn.Module):
    """
    Adaptive coupling that adjusts Kuramoto dynamics based on content.
    
    The coupling strength K is modulated by the input, allowing the
    model to learn when strong vs weak synchronization is beneficial.
    
    Args:
        hidden_dim: Hidden dimension
        min_coupling: Minimum coupling strength
        max_coupling: Maximum coupling strength
        num_steps: Integration steps
    """
    
    def __init__(
        self,
        hidden_dim: int,
        min_coupling: float = 0.1,
        max_coupling: float = 3.0,
        num_steps: int = 5,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.min_coupling = min_coupling
        self.max_coupling = max_coupling
        self.num_steps = num_steps
        
        # Project to coupling strength
        self.coupling_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        
        # Phase projection
        self.phase_proj = nn.Linear(hidden_dim, 1)
        self.freq_proj = nn.Linear(hidden_dim, 1)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim + 3, hidden_dim)  # +3 for phase info
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply adaptive Kuramoto synchronization.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            
        Returns:
            output: (batch, seq, hidden_dim)
            order_param: (batch,) synchronization level
            coupling_used: (batch,) effective coupling strength
        """
        batch, seq, _ = hidden_states.shape
        
        # Compute adaptive coupling from global context
        global_context = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        coupling_scale = self.coupling_proj(global_context).squeeze(-1)  # (batch,)
        K = self.min_coupling + coupling_scale * (self.max_coupling - self.min_coupling)
        
        # Initialize phases
        phases = self.phase_proj(hidden_states).squeeze(-1)  # (batch, seq)
        phases = torch.tanh(phases) * math.pi
        
        # Frequencies
        frequencies = self.freq_proj(hidden_states).squeeze(-1)  # (batch, seq)
        
        # Run Kuramoto with batch-specific coupling
        for _ in range(self.num_steps):
            # Manual implementation to use per-batch coupling
            phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)
            interaction = (K.view(-1, 1) / seq) * torch.sin(phase_diff).sum(dim=-1)
            phases = phases + 0.01 * (frequencies + interaction)
            phases = torch.remainder(phases + math.pi, 2 * math.pi) - math.pi
        
        # Order parameter
        r, psi = compute_order_parameter(phases)
        
        # Augment hidden states with phase information
        phase_info = torch.stack([
            torch.cos(phases),
            torch.sin(phases),
            r.unsqueeze(-1).expand(-1, seq)
        ], dim=-1)  # (batch, seq, 3)
        
        augmented = torch.cat([hidden_states, phase_info], dim=-1)
        output = self.out_proj(augmented)
        
        return output, r, K
