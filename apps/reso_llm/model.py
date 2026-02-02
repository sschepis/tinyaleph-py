"""
ResoLLM Model Architecture.

Standard Mode (default):
    Matches jupyter2.py architecture exactly for compatibility with Colab-trained models.
    Core components:
    - ResoFormerBlock with RoPE positional encoding
    - Coherence gating for stable generation
    - Optional Kuramoto dynamics for attention synchronization

Extended Mode (optional):
    Additional physics-based features when config extensions are enabled:
    - AgencyLayer: Self-directed attention and goal formation
    - PRSC: Prime Resonance Semantic Coherence for compositional semantics
    - TemporalSMF: Holographic memory with episodic tagging
    - EntanglementNetwork: Multi-agent coordination
    - StabilityMonitor: Predictive Lyapunov analysis
    - StochasticResonance: Controlled noise injection
"""
import sys
import os
import math
import time
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any, Union

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Use PyTorch backend
import torch
import torch.nn as nn
from apps.resoformer.torch_backend import ResoFormerBlock, get_device

from apps.reso_llm.config import ResoLLMConfig


# =============================================================================
# Extended Components (Lazy Imports - Only loaded when extensions enabled)
# =============================================================================

def _import_extensions():
    """Lazy import of extension modules."""
    from tinyaleph.observer.smf import SedenionMemoryField
    from tinyaleph.observer.agency import AgencyLayer, Goal, Action, AttentionFocus
    from tinyaleph.observer.prsc import PRSC, SemanticBinding
    from tinyaleph.physics.kuramoto import KuramotoModel
    from tinyaleph.physics.entropy import EntropyTracker, StabilityClass
    from tinyaleph.physics.lyapunov import LyapunovTracker
    from tinyaleph.network.entanglement import EntanglementNetwork, EntanglementSource
    from tinyaleph.hilbert.state import PrimeState
    
    return {
        'SedenionMemoryField': SedenionMemoryField,
        'AgencyLayer': AgencyLayer,
        'Goal': Goal,
        'Action': Action,
        'AttentionFocus': AttentionFocus,
        'PRSC': PRSC,
        'SemanticBinding': SemanticBinding,
        'KuramotoModel': KuramotoModel,
        'EntropyTracker': EntropyTracker,
        'StabilityClass': StabilityClass,
        'LyapunovTracker': LyapunovTracker,
        'EntanglementNetwork': EntanglementNetwork,
        'EntanglementSource': EntanglementSource,
        'PrimeState': PrimeState,
    }


# =============================================================================
# Extended Component Classes (Only instantiated when extensions enabled)
# =============================================================================

class AgencyAttentionModulator(nn.Module):
    """
    Modulates attention weights based on Agency Layer's focus and goals.
    
    Maps agency attention foci to attention weight adjustments.
    """
    
    def __init__(self, dim: int, max_foci: int = 5):
        super().__init__()
        self.dim = dim
        self.max_foci = max_foci
        # Learned projection from focus features to attention bias
        self.focus_projection = nn.Linear(4, dim)  # novelty, relevance, intensity, duration
        
    def forward(
        self,
        attention_weights: torch.Tensor,
        foci: List,
        token_primes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Modulate attention weights based on agency foci.
        """
        if not foci:
            return attention_weights
            
        batch, heads, seq_len, _ = attention_weights.shape
        device = attention_weights.device
        
        # Compute focus-based bias
        focus_bias = torch.zeros(seq_len, device=device)
        
        for focus in foci[:self.max_foci]:
            # Create feature vector
            features = torch.tensor([
                focus.novelty,
                focus.relevance,
                focus.intensity,
                min(focus.duration / 10000, 1.0),
            ], device=device)
            
            # If we have prime mappings, boost positions with matching primes
            if token_primes and focus.primes:
                for i, prime in enumerate(token_primes):
                    if prime in focus.primes:
                        focus_bias[i] += focus.intensity
        
        # Apply bias
        if focus_bias.sum() > 0:
            focus_bias = focus_bias / (focus_bias.sum() + 1e-8)
            focus_bias = focus_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            attention_weights = attention_weights + 0.1 * focus_bias
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
        return attention_weights


class PRSCSemanticLayer(nn.Module):
    """
    Neural interface to PRSC for compositional semantics.
    """
    
    def __init__(self, vocab_size: int, dim: int, prsc):
        super().__init__()
        self.prsc = prsc
        self.vocab_size = vocab_size
        self.dim = dim
        
        self.to_prime_features = nn.Linear(dim, 64)
        self.from_prime_features = nn.Linear(64, dim)
        
    def compose_semantics(
        self,
        embeddings: torch.Tensor,
        concepts: List[str]
    ) -> torch.Tensor:
        """Compose multiple concept embeddings through prime interference."""
        if len(concepts) < 2:
            return embeddings
            
        composed = self.prsc.compose(concepts[:min(len(concepts), 10)])
        if composed is None:
            return embeddings
            
        coherence = self.prsc.global_coherence
        return embeddings * (1 - 0.1 * coherence) + 0.1 * coherence * embeddings.mean(dim=1, keepdim=True)


class StochasticResonanceLayer(nn.Module):
    """
    Implements stochastic resonance for escaping local minima.
    """
    
    def __init__(
        self,
        noise_amplitude: float = 0.1,
        signal_threshold: float = 0.3,
        optimal_noise_ratio: float = 0.5
    ):
        super().__init__()
        self.noise_amplitude = noise_amplitude
        self.signal_threshold = signal_threshold
        self.optimal_noise_ratio = optimal_noise_ratio
        
    def forward(
        self,
        x: torch.Tensor,
        repetition_score: float = 0.0,
        escape_threshold: float = 0.8
    ) -> torch.Tensor:
        """Apply stochastic resonance."""
        if not self.training and repetition_score > escape_threshold:
            noise = torch.randn_like(x) * self.noise_amplitude
            
            signal_strength = x.abs()
            weak_signal_mask = signal_strength < self.signal_threshold
            
            noise_scale = torch.where(
                weak_signal_mask,
                torch.ones_like(x) * self.optimal_noise_ratio,
                torch.ones_like(x) * 0.1
            )
            
            return x + noise * noise_scale
            
        return x


class PredictiveStabilityMonitor:
    """
    Monitors and predicts system stability using Lyapunov analysis.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 0.1,
        predictive_horizon: int = 5
    ):
        ext = _import_extensions()
        self.entropy_tracker = ext['EntropyTracker'](window_size=window_size)
        self.lyapunov_tracker = ext['LyapunovTracker']()
        self.StabilityClass = ext['StabilityClass']
        self.threshold = threshold
        self.predictive_horizon = predictive_horizon
        self.entropy_history = []
        self.lyapunov_history = []
        
    def record(self, entropy: float) -> None:
        """Record an entropy observation."""
        self.entropy_tracker.record(entropy)
        self.entropy_history.append(entropy)
        
        lyap = self.entropy_tracker.lyapunov_exponent()
        self.lyapunov_history.append(lyap)
        
    def current_stability(self):
        """Get current stability classification."""
        return self.entropy_tracker.stability()
        
    def predict_instability(self) -> Tuple[bool, float, int]:
        """Predict if instability is approaching."""
        if len(self.lyapunov_history) < 3:
            return False, 0.0, -1
            
        recent = self.lyapunov_history[-3:]
        trend = (recent[-1] - recent[0]) / 2
        
        current_lyap = recent[-1]
        
        if trend > 0 and current_lyap > 0:
            steps_to_threshold = max(1, int((self.threshold - current_lyap) / trend))
            
            if steps_to_threshold <= self.predictive_horizon:
                confidence = min(1.0, trend * 10)
                return True, confidence, steps_to_threshold
                
        return False, 0.0, -1
        
    def suggest_temperature_adjustment(self, current_temp: float) -> float:
        """Suggest temperature adjustment based on stability."""
        stability = self.current_stability()
        will_destabilize, confidence, _ = self.predict_instability()
        
        if stability == self.StabilityClass.DIVERGENT:
            return max(0.3, current_temp * 0.5)
        elif stability == self.StabilityClass.METASTABLE:
            return max(0.4, current_temp * 0.7)
        elif will_destabilize and confidence > 0.5:
            return max(0.5, current_temp * 0.85)
        elif stability in (self.StabilityClass.STABLE, self.StabilityClass.COLLAPSED):
            return min(1.5, current_temp * 1.05)
            
        return current_temp


# =============================================================================
# Main Model Class
# =============================================================================

class ResoLLMModel(nn.Module):
    """
    Resonant Large Language Model.
    
    Standard Mode (config.standard_mode=True):
        Matches jupyter2.py architecture exactly for compatibility with
        Colab-trained models. Uses:
        - Token embedding + ResoFormerBlocks + output head
        - RoPE positional encoding (built into ResoFormerBlock)
        - Coherence gating (optional, in ResoFormerBlock)
    
    Extended Mode (config.standard_mode=False and extensions enabled):
        Additional physics-based features:
        - Agency Layer: Self-directed attention and goal formation
        - PRSC: Compositional semantics through prime interference
        - Temporal SMF: Holographic memory with episodic tagging
        - Entanglement Network: Multi-agent coordination
        - Predictive Stability: Lyapunov-based hallucination prevention
        - Stochastic Resonance: Escaping repetitive patterns
    """
    
    def __init__(self, config: ResoLLMConfig):
        super().__init__()
        self.config = config
        
        # =====================================================================
        # Core Layers (Always present - matches jupyter2.py exactly)
        # =====================================================================
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            ResoFormerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                use_gating=config.use_coherence_gating,
                coherence_threshold=config.coherence_threshold
            ) for _ in range(config.num_layers)
        ])
        self.head = nn.Linear(config.dim, config.vocab_size)
        
        # =====================================================================
        # Extended Components (Only if extensions enabled)
        # =====================================================================
        if not config.standard_mode and config.has_extensions_enabled():
            self._init_extended_components()
        else:
            # Initialize placeholders for extended components
            self.agency = None
            self.agency_modulator = None
            self.prsc = None
            self.prsc_layer = None
            self.smf = None
            self.kuramoto = None
            self.entanglement = None
            self.stability_monitor = None
            self.stochastic_resonance = None
        
        # Tracking state
        self._current_coherence = 1.0
        self._current_entropy = 0.0
        self._repetition_score = 0.0
        self._recent_tokens: List[int] = []
        
    def _init_extended_components(self):
        """Initialize extended components based on config."""
        config = self.config
        ext = _import_extensions()
        
        # 1. Agency Layer
        self.agency = None
        self.agency_modulator = None
        if config.agency.enabled:
            self.agency = ext['AgencyLayer'](
                max_foci=config.agency.max_foci,
                max_goals=config.agency.max_goals,
                attention_decay_rate=config.agency.attention_decay_rate,
                novelty_weight=config.agency.novelty_weight,
                relevance_weight=config.agency.relevance_weight,
                intensity_weight=config.agency.intensity_weight,
            )
            self.agency_modulator = AgencyAttentionModulator(
                dim=config.dim,
                max_foci=config.agency.max_foci
            )
            
        # 2. PRSC Semantic Layer
        self.prsc = None
        self.prsc_layer = None
        if config.prsc.enabled:
            self.prsc = ext['PRSC'](coherence_threshold=config.prsc.coherence_threshold)
            self.prsc_layer = PRSCSemanticLayer(
                vocab_size=config.vocab_size,
                dim=config.dim,
                prsc=self.prsc
            )
            self._seed_prsc_from_landscape()
            
        # 3. Temporal SMF (Sedenion Memory Field)
        self.smf = None
        if config.temporal_smf.enabled:
            self.smf = ext['SedenionMemoryField'](
                decay_rate=config.temporal_smf.memory_decay_rate,
                max_moments=config.temporal_smf.max_moments
            )
            
        # 4. Kuramoto Dynamics (available in standard mode too, just optional)
        self.kuramoto = None
        if config.use_kuramoto_dynamics:
            self.kuramoto = ext['KuramotoModel'](
                n_oscillators=config.max_seq_len,
                coupling=config.kuramoto_coupling
            )
            
        # 5. Entanglement Network
        self.entanglement = None
        if config.entanglement.enabled:
            self.entanglement = ext['EntanglementNetwork']()
            self.entanglement.add_node(config.entanglement.node_id)
            self.entanglement.source = ext['EntanglementSource'](
                base_fidelity=config.entanglement.base_fidelity,
                success_probability=config.entanglement.success_probability,
                default_primes=config.entanglement.default_primes
            )
            
        # 6. Stability Monitor
        self.stability_monitor = None
        if config.stability.enabled:
            self.stability_monitor = PredictiveStabilityMonitor(
                window_size=config.stability.entropy_window,
                threshold=config.stability.lyapunov_threshold,
                predictive_horizon=config.stability.predictive_horizon
            )
            
        # 7. Stochastic Resonance
        self.stochastic_resonance = None
        if config.stochastic_resonance.enabled:
            self.stochastic_resonance = StochasticResonanceLayer(
                noise_amplitude=config.stochastic_resonance.noise_amplitude,
                signal_threshold=config.stochastic_resonance.signal_threshold,
                optimal_noise_ratio=config.stochastic_resonance.optimal_noise_ratio
            )

    def _seed_prsc_from_landscape(self) -> None:
        config = self.config
        if not config.prsc.landscape_path or self.prsc is None:
            return

        try:
            from apps.semantic_premodel.io import load_landscape
            from tinyaleph.hilbert.state import PrimeState
        except Exception:
            return

        try:
            landscape = load_landscape(config.prsc.landscape_path)
        except Exception:
            return

        entries = list(landscape.entries.values())
        if not entries:
            return

        entries.sort(key=lambda e: (-e.confidence, e.prime))
        meaning_counts = Counter(entry.meaning for entry in entries)
        primes = sorted(landscape.entries.keys())
        mirror_prime = None
        try:
            mirror_prime = landscape.metadata.get("mirror", {}).get("prime")
        except Exception:
            mirror_prime = None

        max_bindings = config.prsc.max_bindings or len(entries)
        min_conf = config.prsc.landscape_min_confidence
        bound = 0

        for entry in entries:
            if bound >= max_bindings:
                break
            if entry.confidence < min_conf:
                continue

            concept = entry.meaning
            if meaning_counts.get(entry.meaning, 0) > 1:
                concept = f"{entry.meaning}#{entry.prime}"

            try:
                state = PrimeState.basis(entry.prime, primes)
            except Exception:
                continue

            metadata = {
                "prime": entry.prime,
                "meaning": entry.meaning,
                "origin": entry.origin,
                "category": entry.category,
                "role": entry.role,
                "adjective": entry.adjective,
                "mirror": entry.prime == mirror_prime,
                "source": "semantic_premodel",
            }

            self.prsc.bind(concept, state, strength=entry.confidence, metadata=metadata)
            bound += 1

    def forward(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        training: bool = None,
        return_state: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass.
        
        In standard mode, this matches jupyter2.py exactly:
        1. Embed tokens
        2. Pass through ResoFormerBlocks (with RoPE + optional coherence gating)
        3. Output head to logits
        
        In extended mode, additional processing is applied.
        
        Args:
            token_ids: Tensor of token indices (batch, seq_len) or list of ints
            training: Optional override for training mode
            return_state: Whether to return internal state dict
            
        Returns:
            Logits tensor, or (logits, state_dict) if return_state=True
        """
        is_training = self.training if training is None else training
        
        # Handle list input (from ResonantGenerator)
        if isinstance(token_ids, list):
            device = self.embedding.weight.device
            x_input = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        else:
            x_input = token_ids

        batch_size, seq_len = x_input.shape
        
        # 1. Embed
        x = self.embedding(x_input)
        
        # Extended mode processing (before blocks)
        sync_factor = 1.0
        memory_context = None
        agency_state = None
        
        if not self.config.standard_mode and not is_training:
            # Apply PRSC Semantic Composition
            if self.prsc_layer:
                try:
                    pass  # Simplified for now
                except Exception:
                    pass
                
            # Apply Memory Context (SMF)
            if self.smf:
                try:
                    orientation = self.smf.get_orientation()
                    if orientation is not None:
                        coherence_weight = float(orientation[0]) if len(orientation) > 0 else 0.5
                        memory_context = {"coherence": coherence_weight, "orientation": orientation}
                except Exception:
                    pass
                
            # Apply Kuramoto Dynamics
            if self.kuramoto:
                try:
                    self.kuramoto.step()
                    sync_r = self.kuramoto.synchronization()
                    sync_factor = 0.5 + 0.5 * sync_r
                    x = x * sync_factor
                except Exception:
                    pass
                
            # Agency Layer Update
            if self.agency:
                try:
                    agency_state = self.agency.update({
                        "smf": self.smf,
                        "coherence": self._current_coherence,
                        "entropy": self._current_entropy,
                    })
                except Exception:
                    pass
            
        # 2. Transformer Layers (core - always runs)
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Agency attention modulation (extended mode only)
            if self.agency_modulator and agency_state and i == len(self.blocks) // 2:
                try:
                    pass  # Simplified for now
                except Exception:
                    pass
                
        # Extended mode processing (after blocks)
        if not self.config.standard_mode and not is_training:
            # Apply Stochastic Resonance
            if self.stochastic_resonance:
                try:
                    x = self.stochastic_resonance(x, self._repetition_score)
                except Exception:
                    pass
            
        # 3. Output Head
        logits = self.head(x)
        
        # Record entropy for stability monitoring (extended mode)
        if not self.config.standard_mode and self.stability_monitor and not is_training:
            try:
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
                self._current_entropy = entropy
                self.stability_monitor.record(entropy)
            except Exception:
                pass
            
        self._current_coherence = sync_factor
        
        # Flatten for generator compatibility if dealing with list input
        if isinstance(token_ids, list):
            output = logits.view(-1)
        else:
            output = logits
            
        if return_state:
            stability_value = "unknown"
            if self.stability_monitor:
                try:
                    stability_value = self.stability_monitor.current_stability().value
                except Exception:
                    pass
            state = {
                "coherence": self._current_coherence,
                "entropy": self._current_entropy,
                "sync_factor": sync_factor,
                "memory_context": memory_context,
                "agency_state": agency_state,
                "stability": stability_value,
            }
            return output, state
            
        return output

    # =========================================================================
    # Extended Mode Helper Methods
    # =========================================================================
    
    def update_memory(self, text: str, importance: float = 1.0):
        """Update the holographic memory with new text."""
        if self.smf:
            self.smf.encode(text, importance)
            
    def update_repetition_score(self, new_token: int):
        """Update repetition tracking for stochastic resonance."""
        self._recent_tokens.append(new_token)
        if len(self._recent_tokens) > 50:
            self._recent_tokens = self._recent_tokens[-50:]
            
        if len(self._recent_tokens) > 10:
            unique = len(set(self._recent_tokens[-20:]))
            self._repetition_score = 1.0 - (unique / 20.0)
        else:
            self._repetition_score = 0.0
            
    def get_stability(self) -> float:
        """Get current system stability."""
        if self.stability_monitor:
            try:
                stability = self.stability_monitor.current_stability()
                ext = _import_extensions()
                StabilityClass = ext['StabilityClass']
                stability_map = {
                    StabilityClass.COLLAPSED: 1.0,
                    StabilityClass.STABLE: 0.9,
                    StabilityClass.CRITICAL: 0.6,
                    StabilityClass.METASTABLE: 0.4,
                    StabilityClass.DIVERGENT: 0.0,
                }
                return stability_map.get(stability, 0.5)
            except Exception:
                pass
        if self.kuramoto:
            return self.kuramoto.synchronization()
        if self.smf:
            return self.smf.mean_coherence
        return 1.0
        
    def get_suggested_temperature(self, current_temp: float = 0.7) -> float:
        """Get suggested temperature based on stability analysis."""
        if self.stability_monitor and self.config.stability.auto_temperature_adjust:
            suggested = self.stability_monitor.suggest_temperature_adjustment(current_temp)
            return max(
                self.config.stability.min_temperature,
                min(self.config.stability.max_temperature, suggested)
            )
        return current_temp

    # =========================================================================
    # Save/Load Methods
    # =========================================================================
    
    def save(self, path: str, save_config: bool = True):
        """
        Save model weights and optionally config.
        
        Args:
            path: Path to save checkpoint
            save_config: Whether to save config alongside weights
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        from dataclasses import asdict
        
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': asdict(self.config) if save_config else None
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint_config(cls, path: str) -> dict:
        """
        Load config from a checkpoint file.
        
        For new-format checkpoints, reads the embedded config.
        For legacy checkpoints (jupyter2.py format), extracts config from state_dict shapes.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Config dict or None if not found
        """
        try:
            device = get_device()
            checkpoint = torch.load(path, weights_only=False, map_location=device)
            
            # New format: config is embedded
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                return checkpoint['config']
            
            # Legacy format (jupyter2.py style): extract config from state_dict shapes
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
            
            if 'embedding.weight' in state_dict:
                vocab_size, dim = state_dict['embedding.weight'].shape
                
                # Count layers by looking at block indices
                num_layers = 0
                for key in state_dict.keys():
                    if key.startswith('blocks.') and '.norm1.weight' in key:
                        layer_idx = int(key.split('.')[1])
                        num_layers = max(num_layers, layer_idx + 1)
                
                # Get num_heads from attention projection shape
                num_heads = 8  # default
                if 'blocks.0.attn.q_proj.weight' in state_dict:
                    q_shape = state_dict['blocks.0.attn.q_proj.weight'].shape
                    num_heads = q_shape[0] // 64
                    if num_heads == 0:
                        num_heads = max(1, dim // 64)
                
                # Get FFN dim
                ffn_dim = dim * 4
                if 'blocks.0.ffn.0.weight' in state_dict:
                    ffn_dim = state_dict['blocks.0.ffn.0.weight'].shape[0]
                
                return {
                    'vocab_size': vocab_size,
                    'dim': dim,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'ffn_dim': ffn_dim,
                    'max_seq_len': 512,  # default, can't determine from state_dict
                    'standard_mode': True,  # Legacy checkpoints are always standard mode
                    '_extracted_from_state_dict': True
                }
        except Exception as e:
            print(f"Warning: Could not extract config from checkpoint: {e}")
        return None
    
    @classmethod
    def config_matches(cls, path: str, config: ResoLLMConfig) -> bool:
        """
        Check if checkpoint config matches the provided config.
        
        Args:
            path: Path to checkpoint
            config: Config to compare against
            
        Returns:
            True if configs match (or checkpoint has no config), False otherwise
        """
        saved_config = cls.load_checkpoint_config(path)
        if saved_config is None:
            return True
            
        return (
            saved_config.get('dim') == config.dim and
            saved_config.get('num_layers') == config.num_layers and
            saved_config.get('num_heads') == config.num_heads and
            saved_config.get('vocab_size') == config.vocab_size
        )
            
    @classmethod
    def load(cls, path: str, config: ResoLLMConfig, strict: bool = True) -> 'ResoLLMModel':
        """
        Load model weights.
        
        Supports both:
        1. jupyter2.py format (state_dict only)
        2. New format (dict with config + state_dict)
        
        Args:
            path: Path to the checkpoint file
            config: Model configuration
            strict: If False, allows loading weights that don't fully match
                   (useful when checkpoint architecture differs from config)
                   
        Returns:
            Loaded model
        """
        device = get_device()
        print(f"Loading model on device: {device}")
        
        model = cls(config)
        
        # Handle both old (state_dict only) and new (dict with config) formats
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Legacy format: entire file is state_dict
            state_dict = checkpoint
        
        if strict:
            model.load_state_dict(state_dict)
        else:
            # Load with strict=False to handle architecture mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {len(unexpected_keys)} keys")
        
        model = model.to(device)
        return model
    
    @classmethod
    def from_checkpoint(cls, path: str, strict: bool = True) -> 'ResoLLMModel':
        """
        Load model directly from checkpoint, auto-detecting config.
        
        This is the recommended way to load models trained with jupyter2.py
        as it automatically detects the architecture from the checkpoint.
        
        Args:
            path: Path to checkpoint
            strict: Whether to strictly match state_dict keys
            
        Returns:
            Loaded model with correct configuration
        """
        saved_config = cls.load_checkpoint_config(path)
        
        if saved_config is None:
            raise ValueError(f"Could not extract config from checkpoint: {path}")
        
        # Create config from saved values
        config = ResoLLMConfig(
            vocab_size=saved_config.get('vocab_size', 50257),
            dim=saved_config.get('dim', 768),
            num_layers=saved_config.get('num_layers', 12),
            num_heads=saved_config.get('num_heads', 12),
            ffn_dim=saved_config.get('ffn_dim', saved_config.get('dim', 768) * 4),
            max_seq_len=saved_config.get('max_seq_len', 1024),
            standard_mode=saved_config.get('standard_mode', True)
        )
        
        # Ensure standard mode for legacy checkpoints
        if saved_config.get('_extracted_from_state_dict', False):
            config.disable_extensions()
        
        return cls.load(path, config, strict=strict)
