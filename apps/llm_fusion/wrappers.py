"""
Model Wrappers for LLM Fusion.

Provides wrappers that inject resonance fusion layers into pre-trained
HuggingFace transformer models using forward hooks.
"""
import sys
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from .config import FusionConfig
from .fusion_layers import ResonanceFusionLayer, MultiLayerFusion


class OutputAdapter(nn.Module):
    """
    Small adapter that modifies logits based on fusion signals.
    
    This allows the fusion layers to influence token predictions
    without modifying the frozen LM head.
    """
    def __init__(self, hidden_dim: int, vocab_size: int, rank: int = 64):
        super().__init__()
        self.rank = rank
        # Low-rank projection for efficiency
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        
        # Initialize near-zero for stable training start
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)
        
        # Learnable gate for controlling adapter strength
        self.gate = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logit adjustment from hidden states.
        
        Args:
            hidden_states: (batch, seq, hidden_dim)
            
        Returns:
            logit_delta: (batch, seq, vocab_size) - addition to base logits
        """
        x = self.down(hidden_states)  # (batch, seq, rank)
        x = F.gelu(x)
        logit_delta = self.up(x)  # (batch, seq, vocab_size)
        return torch.sigmoid(self.gate) * logit_delta


class ResonanceWrapper(nn.Module):
    """
    Wrapper that injects resonance fusion into a pre-trained transformer.
    
    Uses forward hooks to intercept hidden states at specified layers
    and apply resonance fusion without modifying the base model structure.
    
    Also includes an output adapter that allows fusion to influence
    token predictions without modifying the frozen LM head.
    
    Supports:
    - GPT-2 and GPT-Neo (decoder-only)
    - Llama and Mistral (decoder-only)  
    - BERT and RoBERTa (encoder-only)
    - Any model with numbered transformer layers
    
    Args:
        base_model: Pre-trained HuggingFace model
        config: FusionConfig specifying fusion positions and components
        freeze_base: Whether to freeze base model parameters
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> config = FusionConfig.for_gpt2()
        >>> wrapped = ResonanceWrapper(model, config)
        >>> # Train only fusion layers
        >>> wrapped.freeze_base_model()
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[FusionConfig] = None,
        freeze_base: bool = True,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.config = config or FusionConfig()
        
        # Detect model architecture
        self.model_type = self._detect_model_type()
        self.hidden_dim = self._get_hidden_dim()
        self.num_layers = self._get_num_layers()
        
        # Update config with inferred hidden dim if not set
        if self.config.hidden_dim is None:
            self.config.hidden_dim = self.hidden_dim
        
        # Validate fusion positions
        self.config.fusion_positions = [
            p for p in self.config.fusion_positions
            if 0 <= p < self.num_layers
        ]
        
        # Create multi-layer fusion module
        self.fusion = MultiLayerFusion(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            config=self.config,
        )
        
        # Create output adapter for knowledge injection
        # This allows fusion to influence token predictions without modifying the frozen LM head
        vocab_size = self._get_vocab_size()
        self.output_adapter = OutputAdapter(
            hidden_dim=self.hidden_dim,
            vocab_size=vocab_size,
            rank=self.config.adapter_rank,  # Use rank from config
        )
        print(f"Output adapter created: {sum(p.numel() for p in self.output_adapter.parameters()):,} params")
        
        # Hook handles for cleanup
        self._hook_handles: List[Any] = []
        
        # Metrics storage for inference
        self.last_metrics: Dict[int, Dict[str, Any]] = {}
        
        # Store last hidden states for output adapter
        self._last_hidden_states: Optional[torch.Tensor] = None
        
        # Freeze base model if requested
        if freeze_base:
            self.freeze_base_model()
        
        # Install hooks
        self._install_hooks()
    
    def _detect_model_type(self) -> str:
        """Detect the type of transformer model."""
        model_name = type(self.base_model).__name__.lower()
        
        if "gpt2" in model_name or "gpt2" in str(type(self.base_model)):
            return "gpt2"
        elif "gptneo" in model_name:
            return "gpt_neo"
        elif "llama" in model_name:
            return "llama"
        elif "mistral" in model_name:
            return "mistral"
        elif "bert" in model_name:
            return "bert"
        elif "roberta" in model_name:
            return "roberta"
        elif "opt" in model_name:
            return "opt"
        else:
            # Try to detect from config
            if hasattr(self.base_model, "config"):
                config = self.base_model.config
                if hasattr(config, "model_type"):
                    return config.model_type
            return "unknown"
    
    def _get_hidden_dim(self) -> int:
        """Get hidden dimension from model config."""
        if hasattr(self.base_model, "config"):
            config = self.base_model.config
            if hasattr(config, "hidden_size"):
                return config.hidden_size
            elif hasattr(config, "n_embd"):
                return config.n_embd
            elif hasattr(config, "d_model"):
                return config.d_model
        
        # Fallback: try to infer from embedding layer
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Embedding):
                return module.embedding_dim
        
        raise ValueError("Could not determine hidden dimension")
    
    def _get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if hasattr(self.base_model, "config"):
            config = self.base_model.config
            if hasattr(config, "num_hidden_layers"):
                return config.num_hidden_layers
            elif hasattr(config, "n_layer"):
                return config.n_layer
            elif hasattr(config, "num_layers"):
                return config.num_layers
        
        # Fallback: count layer modules
        layers = self._get_layer_modules()
        return len(layers)
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size from model config."""
        if hasattr(self.base_model, "config"):
            config = self.base_model.config
            if hasattr(config, "vocab_size"):
                return config.vocab_size
        # Fallback
        return 32000
    
    def _get_layer_modules(self) -> List[Tuple[str, nn.Module]]:
        """Get list of transformer layer modules."""
        # Common patterns for finding layers
        patterns = [
            "transformer.h",  # GPT-2
            "model.layers",   # Llama
            "model.decoder.layers",  # OPT
            "encoder.layer",  # BERT
            "gpt_neox.layers",  # GPT-NeoX
            "layers",  # Generic
        ]
        
        for pattern in patterns:
            parts = pattern.split(".")
            module = self.base_model
            try:
                for part in parts:
                    module = getattr(module, part)
                if isinstance(module, (nn.ModuleList, list)):
                    return [(f"{pattern}.{i}", m) for i, m in enumerate(module)]
            except AttributeError:
                continue
        
        # Fallback: find any ModuleList of layers
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                first = module[0]
                # Check if it looks like a transformer layer
                if hasattr(first, "forward") and any(
                    hasattr(first, attr) for attr in ["attention", "self_attn", "attn"]
                ):
                    return [(f"{name}.{i}", m) for i, m in enumerate(module)]
        
        return []
    
    def _install_hooks(self):
        """Install forward hooks on transformer layers."""
        layers = self._get_layer_modules()
        
        if not layers:
            print(f"Warning: Could not find transformer layers in {self.model_type} model")
            return
        
        for idx, (name, layer) in enumerate(layers):
            if self.fusion.has_fusion(idx):
                hook = layer.register_forward_hook(
                    partial(self._fusion_hook, layer_idx=idx)
                )
                self._hook_handles.append(hook)
    
    def _fusion_hook(
        self,
        module: nn.Module,
        args: Tuple,
        output: Any,
        layer_idx: int,
    ) -> Any:
        """
        Forward hook that applies fusion to layer output.
        
        Handles various output formats from different model types.
        """
        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        elif isinstance(output, dict):
            hidden_states = output.get("hidden_states", output.get("last_hidden_state"))
            rest = None
        else:
            hidden_states = output
            rest = None
        
        # Apply fusion
        fused, metrics = self.fusion(
            hidden_states,
            layer_idx=layer_idx,
            return_metrics=True,
        )
        
        # Store metrics
        if metrics:
            self.last_metrics[layer_idx] = metrics
        
        # Reconstruct output
        if isinstance(output, tuple):
            return (fused,) + rest
        elif isinstance(output, dict):
            output["hidden_states"] = fused
            return output
        else:
            return fused
    
    def freeze_base_model(self, unfreeze_lm_head: bool = False):
        """
        Freeze base model parameters.
        
        Args:
            unfreeze_lm_head: If True, keeps the LM head trainable.
                             Default False to prevent catastrophic forgetting.
                             Use the output_adapter instead for knowledge injection.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Optionally unfreeze LM head (not recommended - causes catastrophic forgetting)
        if unfreeze_lm_head:
            if hasattr(self.base_model, 'lm_head'):
                for param in self.base_model.lm_head.parameters():
                    param.requires_grad = True
                print("WARNING: LM head unfrozen - risk of catastrophic forgetting")
            elif hasattr(self.base_model, 'output'):
                for param in self.base_model.output.parameters():
                    param.requires_grad = True
                print("WARNING: Output layer unfrozen - risk of catastrophic forgetting")
    
    def unfreeze_base_model(self):
        """Unfreeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def get_fusion_parameters(self) -> List[nn.Parameter]:
        """Get all trainable fusion layer parameters (fusion + output adapter)."""
        params = list(self.fusion.parameters())
        params.extend(list(self.output_adapter.parameters()))
        return params

    def get_fusion_only_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters for fusion layers only (no adapter)."""
        return list(self.fusion.parameters())

    def get_adapter_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters for output adapter only."""
        return list(self.output_adapter.parameters())
    
    def num_fusion_parameters(self) -> int:
        """Count number of trainable fusion parameters."""
        fusion_params = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for p in self.output_adapter.parameters() if p.requires_grad)
        return fusion_params + adapter_params
    
    def num_base_parameters(self) -> int:
        """Count number of base model parameters."""
        return sum(p.numel() for p in self.base_model.parameters())
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        """
        Forward pass through wrapped model.
        
        All arguments are passed to the base model. Fusion is applied
        automatically via hooks. The output adapter modifies logits
        to inject knowledge from fusion layers.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments for base model
            
        Returns:
            Base model output with fusion-modified logits
        """
        # Clear previous metrics
        self.last_metrics.clear()
        
        # Request hidden states so we can use them for output adapter
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        
        # Get the final hidden states (after all fusion layers have been applied)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            final_hidden = outputs.hidden_states[-1]
        else:
            final_hidden = None
        
        # Apply output adapter to modify logits
        # We apply this during both training and inference to ensure consistency
        if hasattr(outputs, "logits") and final_hidden is not None:
            logit_delta = self.output_adapter(final_hidden)
            outputs.logits = outputs.logits + logit_delta
        
        return outputs
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text using the wrapped model.
        
        Fusion is applied during generation via hooks.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Generation arguments
            
        Returns:
            Generated token IDs
        """
        return self.base_model.generate(input_ids=input_ids, **kwargs)
    
    def get_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Get metrics from last forward pass."""
        return self.last_metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all fusion layers."""
        if not self.last_metrics:
            return {}
        
        # Collect all metric keys
        all_keys = set()
        for layer_metrics in self.last_metrics.values():
            all_keys.update(layer_metrics.keys())
        
        # Average each metric
        averages = {}
        for key in all_keys:
            values = [
                m[key] for m in self.last_metrics.values()
                if key in m
            ]
            if values:
                averages[key] = sum(values) / len(values)
        
        return averages
    
    def remove_hooks(self):
        """Remove all installed hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
    
    def save_fusion_weights(self, path: str):
        """Save fusion layer and output adapter weights."""
        combined_state = {
            "fusion": self.fusion.state_dict(),
            "output_adapter": self.output_adapter.state_dict(),
        }
        torch.save(combined_state, path)
    
    def load_fusion_weights(self, path: str, strict: bool = True):
        """Load fusion layer and output adapter weights."""
        state_dict = torch.load(path, map_location="cpu")
        
        # Handle both old format (fusion only) and new format (fusion + adapter)
        if "fusion" in state_dict:
            self.fusion.load_state_dict(state_dict["fusion"], strict=strict)
            if "output_adapter" in state_dict:
                self.output_adapter.load_state_dict(state_dict["output_adapter"], strict=strict)
        else:
            # Old format: just fusion weights
            self.fusion.load_state_dict(state_dict, strict=strict)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[FusionConfig] = None,
        **model_kwargs,
    ) -> "ResonanceWrapper":
        """
        Load pre-trained model and wrap with fusion layers.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            config: FusionConfig for fusion layers
            **model_kwargs: Arguments for model loading
            
        Returns:
            ResonanceWrapper with loaded model
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers library required for from_pretrained")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        
        return cls(base_model, config=config)


class LightweightWrapper(nn.Module):
    """
    Lightweight wrapper that adds fusion as a post-processing layer.
    
    Instead of hooks, this wrapper applies fusion after the base model's
    forward pass. Simpler but only modifies final hidden states.
    
    Args:
        base_model: Pre-trained model
        config: FusionConfig
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[FusionConfig] = None,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.config = config or FusionConfig.minimal()
        
        # Get hidden dim
        if hasattr(base_model, "config"):
            self.hidden_dim = getattr(
                base_model.config,
                "hidden_size",
                getattr(base_model.config, "n_embd", 768)
            )
        else:
            self.hidden_dim = 768
        
        # Single fusion layer for output
        self.fusion = ResonanceFusionLayer(
            hidden_dim=self.hidden_dim,
            config=self.config,
        )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> Any:
        """Forward pass with post-fusion."""
        # Get base model output with hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        
        # Apply fusion to last hidden state
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]
        else:
            # Assume first element is hidden states
            hidden_states = outputs[0]
        
        fused, _ = self.fusion(hidden_states)
        
        # Replace in outputs
        if hasattr(outputs, "last_hidden_state"):
            outputs.last_hidden_state = fused
        elif isinstance(outputs, tuple):
            outputs = (fused,) + outputs[1:]
        
        return outputs
