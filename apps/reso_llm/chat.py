"""
Interactive chat interface for Reso-LLM.

Supports both:
- Standard mode: Models trained with jupyter2.py (Colab notebooks)
- Extended mode: Models with Agency, PRSC, SMF, and other extensions

Features:
- Auto-detects model config from checkpoint
- Models named by size: tiny.pt, small.pt, medium.pt, large.pt, xl.pt
- GPT-2 style BPE tokenization
- Resonant attention with coherence gating
- Optional v2 extensions with graceful degradation
- Stability monitoring (extended mode)
"""
import sys
import os
import argparse

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from apps.reso_llm.config import ResoLLMConfig
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.tokenizer import create_default_tokenizer, create_bpe_tokenizer, create_char_tokenizer
from apps.reso_llm.inference import ResoLLMInference

# Checkpoint directory
CHECKPOINT_DIR = "apps/reso_llm/checkpoints"


def get_checkpoint_path(size: str) -> str:
    """Get checkpoint path for a given model size."""
    return os.path.join(CHECKPOINT_DIR, f"{size}.pt")


def find_available_checkpoint() -> tuple:
    """
    Find the first available checkpoint.
    
    Returns:
        (size, path) tuple or (None, None) if no checkpoint found
    """
    # Check in order of preference: medium, small, tiny, large, xl
    for size in ["medium", "small", "tiny", "large", "xl"]:
        path = get_checkpoint_path(size)
        if os.path.exists(path):
            return size, path
    return None, None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reso-LLM Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect available checkpoint
  python apps/reso_llm/chat.py

  # Use specific model size
  python apps/reso_llm/chat.py --config medium

  # Load custom checkpoint (e.g., from jupyter2.py training)
  python apps/reso_llm/chat.py --checkpoint checkpoints/model.pt

  # Enable extended features
  python apps/reso_llm/chat.py --extended

  # Custom generation settings
  python apps/reso_llm/chat.py --temperature 0.7 --max-length 150
"""
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--config", type=str, default=None,
        choices=["tiny", "small", "base", "medium", "large", "xl"],
        help="Model size to load (auto-detects if not specified)"
    )
    model_group.add_argument(
        "--checkpoint", type=str, default=None,
        help="Explicit checkpoint path (overrides --config)"
    )
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument(
        "--standard", action="store_true",
        help="Force standard mode (ignore extensions in checkpoint)"
    )
    mode_group.add_argument(
        "--extended", action="store_true",
        help="Force extended features (if available)"
    )
    
    # Generation settings
    gen_group = parser.add_argument_group('Generation Settings')
    gen_group.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    gen_group.add_argument(
        "--max-length", type=int, default=100,
        help="Maximum generation length (default: 100)"
    )
    gen_group.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling (default: 50)"
    )
    gen_group.add_argument(
        "--top-p", type=float, default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)"
    )
    
    # Debug
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    return args


def load_model_from_checkpoint(checkpoint_path: str, force_standard: bool = False, verbose: bool = False):
    """
    Load a model from checkpoint, auto-detecting configuration.
    
    Supports both jupyter2.py format and new format checkpoints.
    
    Args:
        checkpoint_path: Path to checkpoint file
        force_standard: If True, force standard mode ignoring checkpoint config
        verbose: Print detailed loading information
        
    Returns:
        (model, tokenizer, config) tuple
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Try to extract config from checkpoint
    saved_config = ResoLLMModel.load_checkpoint_config(checkpoint_path)
    
    if saved_config is not None:
        is_legacy = saved_config.get('_extracted_from_state_dict', False)
        ckpt_is_standard = saved_config.get('standard_mode', True)
        
        # Determine effective mode
        if force_standard:
            use_standard_mode = True
        else:
            use_standard_mode = ckpt_is_standard

        if verbose:
            if is_legacy:
                print("  Format: Legacy (jupyter2.py style)")
            else:
                print("  Format: New (embedded config)")
            print(f"  Checkpoint Mode: {'Standard' if ckpt_is_standard else 'Extended'}")
            print(f"  Effective Mode: {'Standard' if use_standard_mode else 'Extended'}")
        
        # Create config from checkpoint values
        dim = saved_config.get('dim', 768)
        config = ResoLLMConfig(
            vocab_size=saved_config.get('vocab_size', 50257),
            dim=dim,
            num_layers=saved_config.get('num_layers', 12),
            num_heads=saved_config.get('num_heads', 12),
            ffn_dim=saved_config.get('ffn_dim', dim * 4),
            max_seq_len=saved_config.get('max_seq_len', 1024),
            standard_mode=use_standard_mode
        )
        
        if use_standard_mode:
            config.disable_extensions()
        else:
            # Check if extension configs are missing (legacy save issue)
            # If so, enable default extensions to ensure model structure matches checkpoint
            has_extension_configs = any(k in saved_config for k in [
                'agency', 'prsc', 'temporal_smf', 'entanglement', 'stability', 'stochastic_resonance'
            ])
            
            if not has_extension_configs:
                if verbose:
                    print("  Warning: Extended mode detected but extension configs missing.")
                    print("  Enabling default extensions to match model structure.")
                config.enable_extensions()
            
            # Extended mode: Hydrate extension configs from saved_config
            from apps.reso_llm.config import (
                AgencyConfig, PRSCConfig, TemporalSMFConfig,
                EntanglementConfig, StabilityConfig, StochasticResonanceConfig
            )
            
            # Helper to hydrate dataclass from dict
            def hydrate(field_name, cls, cfg_obj):
                if field_name in saved_config:
                    data = saved_config[field_name]
                    if isinstance(data, dict):
                        setattr(cfg_obj, field_name, cls(**data))
            
            hydrate('agency', AgencyConfig, config)
            hydrate('prsc', PRSCConfig, config)
            hydrate('temporal_smf', TemporalSMFConfig, config)
            hydrate('entanglement', EntanglementConfig, config)
            hydrate('stability', StabilityConfig, config)
            hydrate('stochastic_resonance', StochasticResonanceConfig, config)

        print(f"  Model: dim={config.dim}, layers={config.num_layers}, heads={config.num_heads}")
        print(f"  Vocab: {config.vocab_size}")
        print(f"  Mode: {'Standard' if config.standard_mode else 'Extended'}")
        
        # Choose tokenizer based on vocab size
        # IMPORTANT: Models with vocab_size=50257 were trained WITHOUT special chat tokens
        # We must NOT add chat tokens or the tokenizer vocab will mismatch the model
        if config.vocab_size == 50257:
            print("  Tokenizer: GPT-2 BPE (no chat tokens - base vocab)")
            tokenizer = create_bpe_tokenizer(add_chat_tokens=False)
        elif config.vocab_size == 50264:
            print("  Tokenizer: GPT-2 BPE (with chat tokens)")
            tokenizer = create_bpe_tokenizer(add_chat_tokens=True)
        else:
            print(f"  Tokenizer: Character-level (vocab={config.vocab_size})")
            tokenizer = create_char_tokenizer(config.vocab_size)
        
        # Load model
        try:
            use_strict = not is_legacy
            model = ResoLLMModel.load(checkpoint_path, config, strict=use_strict)
            print(f"  Loaded successfully! (strict={use_strict})")
        except Exception as e:
            print(f"  Error loading with strict={use_strict}: {e}")
            print("  Retrying with strict=False...")
            try:
                model = ResoLLMModel.load(checkpoint_path, config, strict=False)
                print("  Loaded with strict=False")
            except Exception as e2:
                print(f"  Failed to load: {e2}")
                print("  Initializing untrained model...")
                model = ResoLLMModel(config)
    else:
        # Could not read config - use default medium config
        print("  Warning: Could not read config from checkpoint")
        print("  Using default medium configuration")
        
        tokenizer = create_default_tokenizer()
        config = ResoLLMConfig.medium(standard=use_standard)
        config.vocab_size = tokenizer.vocab_size
        
        try:
            model = ResoLLMModel.load(checkpoint_path, config, strict=False)
            print("  Loaded with strict=False")
        except Exception as e:
            print(f"  Error: {e}")
            print("  Initializing untrained model...")
            model = ResoLLMModel(config)
    
    return model, tokenizer, config


def create_fresh_model(size: str, use_standard: bool = True):
    """
    Create a fresh untrained model.
    
    Args:
        size: Model size name
        use_standard: If True, use standard mode
        
    Returns:
        (model, tokenizer, config) tuple
    """
    print(f"Creating untrained {size} model...")
    
    tokenizer = create_default_tokenizer()
    config = ResoLLMConfig.from_size(size, standard=use_standard)
    config.vocab_size = tokenizer.vocab_size
    
    if not use_standard:
        config.enable_extensions()
    
    print(f"  Model: dim={config.dim}, layers={config.num_layers}")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Mode: {'Standard' if config.standard_mode else 'Extended'}")
    print("  Tokenizer: GPT-2 BPE")
    
    model = ResoLLMModel(config)
    
    return model, tokenizer, config


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Reso-LLM Chat Interface")
    print("=" * 60)
    
    if args.extended:
        print("Mode: Extended (with Agency, PRSC, SMF, Stability)")
    else:
        print("Mode: Standard (jupyter2.py compatible)")
    print()
    
    # Determine checkpoint path and load model
    if args.checkpoint:
        # Explicit checkpoint path provided
        checkpoint_path = args.checkpoint
        if os.path.exists(checkpoint_path):
            model, tokenizer, config = load_model_from_checkpoint(
                checkpoint_path,
                force_standard=args.standard,
                verbose=args.verbose
            )
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return
            
    elif args.config:
        # Size specified - use that size's checkpoint
        checkpoint_path = get_checkpoint_path(args.config)
        if os.path.exists(checkpoint_path):
            model, tokenizer, config = load_model_from_checkpoint(
                checkpoint_path,
                force_standard=args.standard,
                verbose=args.verbose
            )
        else:
            print(f"No checkpoint found for size '{args.config}'")
            model, tokenizer, config = create_fresh_model(
                args.config,
                use_standard=not args.extended  # For fresh models, explicit extended flag matters
            )
    else:
        # Auto-detect: find first available checkpoint
        model_size, checkpoint_path = find_available_checkpoint()
        if checkpoint_path:
            model, tokenizer, config = load_model_from_checkpoint(
                checkpoint_path,
                force_standard=args.standard,
                verbose=args.verbose
            )
        else:
            print("No checkpoint found.")
            print("Use --config to specify model size or train a model first.")
            print("  Example: python apps/reso_llm/train.py --config medium")
            print()
            print("Starting with untrained medium model for demo purposes...")
            model, tokenizer, config = create_fresh_model(
                "medium",
                use_standard=not args.extended
            )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,}")
    
    # Show enabled features
    if not config.standard_mode:
        enabled = []
        if config.agency.enabled:
            enabled.append("Agency")
        if config.prsc.enabled:
            enabled.append("PRSC")
        if config.temporal_smf.enabled:
            enabled.append("TemporalSMF")
        if config.stability.enabled:
            enabled.append("Stability")
        if config.entanglement.enabled:
            enabled.append("Entanglement")
        if config.stochastic_resonance.enabled:
            enabled.append("StochasticResonance")
        if enabled:
            print(f"Extensions: {', '.join(enabled)}")
    
    print()
    print("-" * 60)
    
    # Create inference engine
    engine = ResoLLMInference(
        model,
        tokenizer,
        temperature=args.temperature,
        max_length=args.max_length
    )
    
    # Start interactive session
    engine.interactive_session()


if __name__ == "__main__":
    main()
