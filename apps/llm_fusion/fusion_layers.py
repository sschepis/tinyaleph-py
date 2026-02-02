"""
Resonance Fusion Layers for LLM Grafting.

Core fusion layers that combine prime projection, quaternion operations,
Kuramoto synchronization, and coherence gating into grafting modules
for existing transformer architectures.
"""
import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig, COHERENCE_THRESHOLD
from .prime_embeddings import PrimeProjection, PrimeBackProjection
from .quaternion_layers import (
    quaternion_multiply,
    quaternion_normalize,
    QuaternionRotationLayer,
)
from .kuramoto_attention import (
    KuramotoModule,
    GlobalSynchronizationLayer,
    compute_order_parameter,
)


class CoherenceGatingLayer(nn.Module):
    """
    Gate information flow based on coherence (inverse entropy).
    
    High coherence = confident, stable → pass through
    Low coherence = uncertain, unstable → attenuate
    
    C = 1 - S/S_max where S is entropy
    
    Args:
        hidden_dim: Hidden dimension
        threshold: Coherence threshold for full pass-through
        soft_gate: Use soft gating vs hard threshold
    """
    
    def __init__(
        self,
        hidden_dim: int,
        threshold: float = COHERENCE_THRESHOLD,
        soft_gate: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.soft_gate = soft_gate
        
        # Compute coherence from hidden states
        # Simplified architecture for better gradient flow
        self.coherence_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),  # Tanh instead of GELU for bounded activations
            nn.Linear(hidden_dim // 4, 1),
        )
        
        # Initialize for positive coherence output
        with torch.no_grad():
            self.coherence_proj[-1].bias.fill_(1.0)  # Bias toward positive
        
        # Learnable threshold
        self.threshold_param = nn.Parameter(torch.tensor(threshold))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        external_coherence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply coherence gating.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            external_coherence: (batch, seq) optional external coherence values
            
        Returns:
            gated: (batch, seq, hidden_dim)
            coherence: (batch, seq) coherence values
        """
        if external_coherence is not None:
            coherence = external_coherence
        else:
            # Apply sigmoid to get coherence in [0, 1]
            raw_coherence = self.coherence_proj(hidden_states).squeeze(-1)  # (batch, seq)
            coherence = torch.sigmoid(raw_coherence)
        
        # Compute gate values
        if self.soft_gate:
            # Soft sigmoid gate centered at threshold
            threshold = torch.sigmoid(self.threshold_param)
            gate = torch.sigmoid(10 * (coherence - threshold))  # Steep sigmoid
        else:
            # Hard threshold
            gate = (coherence > self.threshold).float()
        
        # Apply gate
        gate = gate.unsqueeze(-1)  # (batch, seq, 1)
        gated = hidden_states * gate
        
        return gated, coherence.squeeze(-1) if coherence.dim() > 2 else coherence


class NeuralPRSC(nn.Module):
    """
    Neural Prime Resonance Semantic Coherence layer.
    
    Implements compositional semantics through prime number interference
    patterns, adapted for neural network use.
    
    Composition:
        |ψ_AB⟩ = |ψ_A⟩ ⊗ |ψ_B⟩  (tensor product of prime states)
        
    Binding:
        B(A, B) = prime(A) * prime(B)  (prime multiplication)
        
    Args:
        hidden_dim: Hidden dimension
        num_primes: Number of prime basis states
        composition_weight: Weight for composed semantics
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_primes: int = 25,
        composition_weight: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_primes = num_primes
        self.composition_weight = composition_weight
        
        # Prime projection
        self.prime_proj = PrimeProjection(
            input_dim=hidden_dim,
            num_primes=num_primes,
            use_quaternion=True,
        )
        
        # Back projection from composed space
        self.back_proj = PrimeBackProjection(
            num_primes=num_primes,
            output_dim=hidden_dim,
            use_quaternion=True,
        )
        
        # Learnable composition weight
        self.comp_weight = nn.Parameter(torch.tensor(composition_weight))
        
        # Binding key projection
        self.key_proj = nn.Linear(hidden_dim, num_primes)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        binding_keys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply PRSC composition.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            binding_keys: (batch, seq) optional semantic binding keys
            
        Returns:
            output: (batch, seq, hidden_dim)
            coherence: (batch, seq) compositional coherence
        """
        batch, seq, _ = hidden_states.shape
        
        # Project to prime space
        prime_states, entropy = self.prime_proj(hidden_states, return_entropy=True)
        # prime_states: (batch, seq, num_primes, 4)
        
        # Compute binding keys if not provided
        if binding_keys is None:
            key_logits = self.key_proj(hidden_states)  # (batch, seq, num_primes)
            binding_keys = torch.argmax(key_logits, dim=-1)  # (batch, seq)
        
        # Compose adjacent positions through quaternion multiplication
        # This creates interference patterns in prime space
        if seq > 1:
            # Shift for composition: compose position i with i+1
            prime_left = prime_states[:, :-1]  # (batch, seq-1, num_primes, 4)
            prime_right = prime_states[:, 1:]  # (batch, seq-1, num_primes, 4)
            
            # Quaternion composition
            composed = quaternion_multiply(prime_left, prime_right)
            composed = quaternion_normalize(composed)
            
            # Pad back to seq length (first position gets identity)
            identity = torch.zeros_like(prime_states[:, :1])
            identity[..., 0] = 1.0  # Unit quaternion
            composed = torch.cat([identity, composed], dim=1)
        else:
            composed = prime_states
        
        # Blend original with composed
        weight = torch.sigmoid(self.comp_weight)
        blended = (1 - weight) * prime_states + weight * composed
        
        # Back project
        output = self.back_proj(blended)
        
        # Coherence from entropy
        coherence = self.prime_proj.coherence(blended)
        
        return output, coherence


class SedenionMemoryCell(nn.Module):
    """
    Simplified Sedenion Memory Field for neural network use.
    
    Uses 16-dimensional hypercomplex representation for rich
    memory encoding with temporal decay.
    
    Args:
        hidden_dim: Hidden dimension
        memory_dim: Sedenion dimension (16)
        decay_rate: Memory decay rate
        max_memories: Maximum stored memories
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int = 16,
        decay_rate: float = 0.01,
        max_memories: int = 100,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.decay_rate = decay_rate
        self.max_memories = max_memories
        
        # Project to/from sedenion space
        self.to_sedenion = nn.Linear(hidden_dim, memory_dim)
        self.from_sedenion = nn.Linear(memory_dim, hidden_dim)
        
        # Memory attention
        self.memory_query = nn.Linear(hidden_dim, memory_dim)
        
        # Register memory buffer (not a parameter)
        self.register_buffer(
            "memory_bank",
            torch.zeros(max_memories, memory_dim)
        )
        self.register_buffer(
            "memory_ages",
            torch.zeros(max_memories)
        )
        self.register_buffer(
            "num_memories",
            torch.tensor(0)
        )
    
    def store(self, hidden_states: torch.Tensor, importance: float = 1.0):
        """
        Store hidden states in memory.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            importance: Importance weight for storage
        """
        # Only store during training
        if not self.training:
            return
        
        # Convert to sedenion
        sedenion = self.to_sedenion(hidden_states)  # (batch, seq, 16)
        
        # Average over batch and sequence for storage
        to_store = sedenion.mean(dim=(0, 1))  # (16,)
        to_store = to_store * importance
        
        # Age existing memories
        self.memory_ages += 1
        
        # Find slot (oldest or empty)
        if self.num_memories < self.max_memories:
            idx = self.num_memories.item()
            self.num_memories += 1
        else:
            idx = torch.argmax(self.memory_ages).item()
        
        # Store
        self.memory_bank[idx] = to_store
        self.memory_ages[idx] = 0
    
    def retrieve(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Retrieve relevant memories.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            
        Returns:
            memory_contribution: (batch, seq, hidden_dim)
        """
        if self.num_memories == 0:
            return torch.zeros_like(hidden_states)
        
        batch, seq, _ = hidden_states.shape
        
        # Query
        query = self.memory_query(hidden_states)  # (batch, seq, 16)
        
        # Active memories
        n = self.num_memories.item()
        memories = self.memory_bank[:n]  # (n, 16)
        ages = self.memory_ages[:n]  # (n,)
        
        # Temporal decay
        decay = torch.exp(-self.decay_rate * ages)  # (n,)
        memories = memories * decay.unsqueeze(-1)
        
        # Attention
        # query: (batch, seq, 16), memories: (n, 16)
        scores = torch.matmul(query, memories.T)  # (batch, seq, n)
        weights = F.softmax(scores, dim=-1)
        
        # Retrieve
        retrieved = torch.matmul(weights, memories)  # (batch, seq, 16)
        
        # Back to hidden
        return self.from_sedenion(retrieved)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        store: bool = True,
        importance: float = 0.8,
    ) -> torch.Tensor:
        """
        Retrieve memories and optionally store current states.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            store: Whether to store current states
            importance: Storage importance
            
        Returns:
            memory_contribution: (batch, seq, hidden_dim)
        """
        if store:
            self.store(hidden_states, importance)
        
        return self.retrieve(hidden_states)


class ResonanceFusionLayer(nn.Module):
    """
    Main fusion layer that grafts onto transformer hidden states.
    
    Combines:
    - Prime projection (semantic basis)
    - Quaternion rotation (geometric transformation)
    - Kuramoto synchronization (coherence dynamics)
    - Coherence gating (stability control)
    - Optional PRSC and SMF
    
    Fusion modes:
    - "parallel": Compute resonance in parallel, add to residual
    - "sequential": Process through resonance, then gate
    - "residual": Blend resonance output with original
    
    Args:
        hidden_dim: Hidden dimension of base model
        config: FusionConfig with component settings
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[FusionConfig] = None,
    ):
        super().__init__()
        
        if config is None:
            config = FusionConfig()
        
        self.hidden_dim = hidden_dim
        self.config = config
        self.fusion_mode = config.fusion_mode
        self.fusion_norm = config.fusion_norm
        
        # Fusion weight
        if config.learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(config.fusion_alpha))
        else:
            self.register_buffer("alpha", torch.tensor(config.fusion_alpha))
        
        # Build enabled components
        self.components = nn.ModuleDict()
        
        if "prime" in config.enabled_components:
            self.components["prime"] = PrimeProjection(
                input_dim=hidden_dim,
                num_primes=config.prime.num_primes,
                use_quaternion=config.prime.use_quaternion,
                normalize=config.prime.normalize,
                learnable_phases=config.prime.learnable_phases,
            )
            # Back projection
            self.components["prime_back"] = PrimeBackProjection(
                num_primes=config.prime.num_primes,
                output_dim=hidden_dim,
                use_quaternion=config.prime.use_quaternion,
            )
        
        if "quaternion" in config.enabled_components:
            self.components["quaternion"] = QuaternionRotationLayer(
                hidden_dim=hidden_dim,
                use_position_rotation=True,
            )
        
        if "kuramoto" in config.enabled_components and config.kuramoto.enabled:
            self.components["kuramoto"] = GlobalSynchronizationLayer(
                hidden_dim=hidden_dim,
                coupling=config.kuramoto.coupling,
                num_steps=5,
                blend_factor=0.1,
            )
        
        if "coherence_gate" in config.enabled_components:
            self.components["gate"] = CoherenceGatingLayer(
                hidden_dim=hidden_dim,
                threshold=config.prsc.coherence_threshold if config.prsc else COHERENCE_THRESHOLD,
                soft_gate=True,
            )
        
        if "prsc" in config.enabled_components and config.prsc.enabled:
            self.components["prsc"] = NeuralPRSC(
                hidden_dim=hidden_dim,
                num_primes=config.prime.num_primes,
                composition_weight=config.prsc.composition_weight,
            )
        
        if "smf" in config.enabled_components and config.smf.enabled:
            self.components["smf"] = SedenionMemoryCell(
                hidden_dim=hidden_dim,
                decay_rate=config.smf.decay_rate,
                max_memories=config.smf.max_moments,
            )
        
        # Final projection for parallel/residual modes
        if self.fusion_mode in ["parallel", "residual"]:
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.out_proj.bias)
            nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        
        # Layer norm for output
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Apply resonance fusion to hidden states.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            attention_mask: (batch, seq) optional mask
            position_ids: (batch, seq) optional position indices
            return_metrics: Whether to return diagnostic metrics
            
        Returns:
            output: (batch, seq, hidden_dim)
            metrics: Dict with coherence, entropy, etc. if return_metrics
        """
        residual = hidden_states
        metrics = {} if return_metrics else None
        
        # Current processing state
        x = hidden_states
        coherence = None
        
        # === Prime Projection ===
        if "prime" in self.components:
            prime_states, entropy = self.components["prime"](x, return_entropy=True)
            coherence = self.components["prime"].coherence(prime_states)
            
            if return_metrics:
                metrics["prime_entropy"] = entropy.mean().item()
                metrics["prime_coherence"] = coherence.mean().item()
            
            # Back project
            x = self.components["prime_back"](prime_states)
        
        # === Quaternion Rotation ===
        if "quaternion" in self.components:
            x = self.components["quaternion"](x, positions=position_ids)
        
        # === Kuramoto Synchronization ===
        if "kuramoto" in self.components:
            x, order_param = self.components["kuramoto"](x)
            
            if return_metrics:
                metrics["kuramoto_order"] = order_param.mean().item()
        
        # === PRSC Composition ===
        if "prsc" in self.components:
            prsc_out, prsc_coherence = self.components["prsc"](x)
            x = x + prsc_out * self.config.prsc.composition_weight
            
            if coherence is None:
                coherence = prsc_coherence
            else:
                coherence = (coherence + prsc_coherence) / 2
            
            if return_metrics:
                metrics["prsc_coherence"] = prsc_coherence.mean().item()
        
        # === SMF Memory ===
        if "smf" in self.components:
            memory = self.components["smf"](x, store=self.training)
            x = x + memory * self.config.smf.memory_weight
        
        # === Coherence Gating ===
        if "gate" in self.components:
            x, gate_coherence = self.components["gate"](x, external_coherence=coherence)
            
            if return_metrics:
                metrics["gate_coherence"] = gate_coherence.mean().item()
        
        # === Fusion Mode ===
        if self.fusion_norm == "fusion":
            x = self.layer_norm(x)

        if self.fusion_mode == "parallel":
            # Add to residual with learned weight
            alpha = torch.sigmoid(self.alpha) if isinstance(self.alpha, nn.Parameter) else self.alpha
            output = residual + alpha * self.out_proj(x)
            
        elif self.fusion_mode == "sequential":
            # Direct sequential processing
            output = x
            
        elif self.fusion_mode == "residual":
            # Blend with residual
            alpha = torch.sigmoid(self.alpha) if isinstance(self.alpha, nn.Parameter) else self.alpha
            output = (1 - alpha) * residual + alpha * self.out_proj(x)

        if self.fusion_norm == "output":
            output = self.layer_norm(output)
        
        if return_metrics:
            metrics["fusion_alpha"] = (
                torch.sigmoid(self.alpha).item() 
                if isinstance(self.alpha, nn.Parameter) 
                else self.alpha.item()
            )
        
        return output, metrics


class MultiLayerFusion(nn.Module):
    """
    Manages fusion layers across multiple transformer layers.
    
    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        config: FusionConfig specifying which layers get fusion
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        config: Optional[FusionConfig] = None,
    ):
        super().__init__()
        
        if config is None:
            config = FusionConfig()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.config = config
        
        # Create fusion layers for specified positions
        self.fusion_layers = nn.ModuleDict()
        
        for pos in config.fusion_positions:
            if 0 <= pos < num_layers:
                self.fusion_layers[str(pos)] = ResonanceFusionLayer(
                    hidden_dim=hidden_dim,
                    config=config,
                )
    
    def get_fusion_layer(self, layer_idx: int) -> Optional[ResonanceFusionLayer]:
        """Get fusion layer for specific transformer layer, or None."""
        key = str(layer_idx)
        if key in self.fusion_layers:
            return self.fusion_layers[key]
        return None
    
    def has_fusion(self, layer_idx: int) -> bool:
        """Check if layer has fusion."""
        return str(layer_idx) in self.fusion_layers
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Apply fusion for specific layer if configured.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            layer_idx: Which transformer layer
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            return_metrics: Whether to return metrics
            
        Returns:
            output: (batch, seq, hidden_dim) - unchanged if no fusion
            metrics: Metrics dict if fusion applied and return_metrics=True
        """
        fusion_layer = self.get_fusion_layer(layer_idx)
        
        if fusion_layer is None:
            return hidden_states, None
        
        return fusion_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_metrics=return_metrics,
        )
