"""
Training loop for Reso-LLM v2.

Implements a standard training loop with:
- Cross-entropy loss
- PyTorch autograd for gradient computation
- Periodic validation
- Model checkpointing by size (tiny.pt, small.pt, medium.pt, etc.)
- Automatic checkpoint continuation
- Stability monitoring during training
- Multi-dataset support for training on combined datasets
- **Template-based data formatting** for consistent training

Default: medium model, timdettmers/openassistant-guanaco dataset

Template Types:
    - completion: Raw text (no structure)
    - chat: Multi-turn conversations (matches inference.py format)
    - instruction: Instruction-following (Alpaca, Dolly, etc.)

Multi-Dataset Training:
    # Train on recommended datasets (auto-detects templates)
    python train.py --recommended-datasets
    
    # Train on specific multiple datasets
    python train.py --datasets timdettmers/openassistant-guanaco databricks/databricks-dolly-15k tatsu-lab/alpaca
    
    # Show template report for datasets
    python train.py --datasets dataset1 dataset2 --show-template-report
    
    # Control tokens per dataset
    python train.py --datasets dataset1 dataset2 --max-tokens-per-dataset 2000000
"""
import sys
import os
import math
import time
import argparse
from typing import Optional

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# PyTorch backend
import torch
import torch.nn as nn
import torch.optim as optim

from apps.reso_llm.config import ResoLLMConfig, small_config, medium_config
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.dataset import (
    TextDataset,
    HuggingFaceDataset,
    GuanacoDataset,
    MultiDataset,
    create_guanaco_dataset,
    create_recommended_multi_dataset,
    verify_dataset_fusion,
    DEFAULT_DATASET
)
from apps.reso_llm.training_templates import (
    TrainingTemplateType,
    get_template_for_dataset,
    print_template_report,
    RECOMMENDED_DATASETS,
)
from tinyaleph.ml.resoformer import Tensor

from apps.resoformer.torch_backend import get_device

# Checkpoint directory
CHECKPOINT_DIR = "apps/reso_llm/checkpoints"


def get_checkpoint_path(size: str) -> str:
    """Get checkpoint path for a given model size."""
    return os.path.join(CHECKPOINT_DIR, f"{size}.pt")


class ResoLLMTrainer:
    """
    Trainer for Reso-LLM v2 (PyTorch version).
    
    Features:
    - AdamW optimizer with warmup and cosine decay
    - Gradient clipping
    - Stability monitoring
    - Automatic checkpoint saving (per epoch only)
    """
    
    def __init__(self, 
                 model: ResoLLMModel, 
                 dataset: TextDataset,
                 val_dataset: Optional[TextDataset] = None,
                 learning_rate: float = 3e-4,
                 warmup_steps: int = 100,
                 max_steps: int = 10000):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler with warmup + cosine decay
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Reduce LR on Plateau (for validation loss)
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.losses = []
        self.best_val_loss = float('inf')
        
    def train_step(self) -> tuple:
        """
        Perform a single training step using PyTorch autograd.
        
        Returns:
            (loss, grad_norm) tuple
        """
        self.model.train()
        
        # 1. Get batch
        x_custom, y_custom = self.dataset.get_batch()
        
        # Convert to torch tensors and move to device
        x = torch.tensor(x_custom.data, dtype=torch.long).view(
            self.dataset.batch_size, self.dataset.seq_len
        ).to(self.device)
        y = torch.tensor(y_custom.data, dtype=torch.long).view(
            self.dataset.batch_size, self.dataset.seq_len
        ).to(self.device)
        
        # 2. Forward pass
        self.optimizer.zero_grad()
        logits = self.model(x)
        
        # 3. Compute loss
        # Flatten for CrossEntropyLoss: (batch * seq, vocab) vs (batch * seq)
        loss = self.criterion(
            logits.view(-1, self.model.config.vocab_size), 
            y.view(-1)
        )
        
        # 4. Backward & Update
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item(), grad_norm.item()
        
    def train(
        self, 
        epochs: int = 1, 
        save_path: str = "checkpoints/model.pt",
        log_interval: int = 10
    ):
        """
        Run training loop.
        
        Saves checkpoint at the end of each epoch (no intermediate step saves).
        
        Args:
            epochs: Number of epochs to train
            save_path: Path to save checkpoints
            log_interval: Steps between logging
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Dataset: {len(self.dataset.data):,} tokens")
        print(f"Batch size: {self.dataset.batch_size}")
        print(f"Sequence length: {self.dataset.seq_len}")
        print(f"Steps per epoch: {len(self.dataset)}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        steps_per_epoch = len(self.dataset)
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            for i in range(steps_per_epoch):
                step_start = time.time()
                loss, grad_norm = self.train_step()
                step_duration = time.time() - step_start
                
                epoch_loss += loss
                epoch_grad_norm += grad_norm
                self.losses.append(loss)
                self.step += 1
                
                # Log progress
                if i % log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    stability = self.model.get_stability()
                    print(
                        f"  Step {i:4d}/{steps_per_epoch} | "
                        f"Loss: {loss:.4f} | "
                        f"Grad: {grad_norm:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Stability: {stability:.3f} | "
                        f"Time: {step_duration:.3f}s"
                    )
            
            avg_loss = epoch_loss / steps_per_epoch
            avg_grad_norm = epoch_grad_norm / steps_per_epoch
            duration = time.time() - start_time
            
            print(f"\nEpoch {epoch+1} completed in {duration:.2f}s")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Avg Grad Norm: {avg_grad_norm:.4f}")
            
            # Validation
            if self.val_dataset:
                val_loss = self.evaluate()
                print(f"  Validation Loss: {val_loss:.4f}")
                
                # Step plateau scheduler on validation loss
                self.plateau_scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = save_path.replace(".pt", "_best.pt")
                    self.model.save(best_path)
                    print(f"  New best model saved: {best_path}")
            else:
                # Step plateau scheduler on training loss if no validation set
                self.plateau_scheduler.step(avg_loss)
                
            # Save epoch checkpoint (only save at end of each epoch)
            self.model.save(save_path)
            print(f"  Saved checkpoint: {save_path}")
            
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if not self.val_dataset:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for x_custom, y_custom in self.val_dataset.iterate_batches():
                x = torch.tensor(x_custom.data, dtype=torch.long).view(
                    self.val_dataset.batch_size, self.val_dataset.seq_len
                ).to(self.device)
                y = torch.tensor(y_custom.data, dtype=torch.long).view(
                    self.val_dataset.batch_size, self.val_dataset.seq_len
                ).to(self.device)
                
                logits = self.model(x)
                loss = self.criterion(
                    logits.view(-1, self.model.config.vocab_size), 
                    y.view(-1)
                )
                
                total_loss += loss.item()
                count += 1
                
                # Limit validation steps
                if count >= 100:
                    break
            
        return total_loss / max(1, count)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Reso-LLM v2")
    
    # Configuration file (overrides other args if provided)
    parser.add_argument("--from-config", type=str, default=None,
                       help="Load training configuration from JSON file")
    
    # Model configuration - default is medium
    parser.add_argument("--config", type=str, default="medium",
                       choices=["tiny", "small", "medium", "large", "xl", "xxl"],
                       help="Model configuration preset (default: medium)")
    
    # Dataset - single or multiple
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                       help="HuggingFace dataset name (single dataset)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                       help="Multiple HuggingFace dataset names (space-separated)")
    parser.add_argument("--recommended-datasets", action="store_true",
                       help="Use recommended multi-dataset: guanaco, databricks-dolly-15k, alpaca")
    parser.add_argument("--seq-len", type=int, default=256,
                       help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--max-tokens-per-dataset", type=int, default=5_000_000,
                       help="Max tokens per dataset when using --datasets")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--warmup", type=int, default=100,
                       help="Warmup steps")
    
    # Control flags
    parser.add_argument("--reset", action="store_true",
                       help="Reset training - ignore existing checkpoint and train fresh")
    parser.add_argument("--show-template-report", action="store_true",
                       help="Show template mapping report for datasets and exit")
    
    # Optional extensions
    parser.add_argument("--use-agency", action="store_true", help="Enable Agency Layer")
    parser.add_argument("--use-prsc", action="store_true", help="Enable PRSC Semantic Layer")
    parser.add_argument("--use-temporal-smf", action="store_true", help="Enable Temporal SMF")
    parser.add_argument("--use-entanglement", action="store_true", help="Enable Entanglement Network")
    parser.add_argument("--use-stability", action="store_true", help="Enable Stability Monitor")
    parser.add_argument("--use-stochastic-resonance", action="store_true", help="Enable Stochastic Resonance")
    parser.add_argument("--landscape", type=str, default=None,
                       help="Path to semantic landscape JSON (enables PRSC)")
    parser.add_argument("--landscape-min-confidence", type=float, default=None,
                       help="Minimum confidence required to bind landscape entries")

    args = parser.parse_args()
    
    # If --from-config is provided, load and override args
    if args.from_config:
        args = load_config_file(args)
    
    return args


def load_config_file(args):
    """Load training configuration from JSON file and update args."""
    import json
    
    config_path = args.from_config
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Map config fields to args
    args.config = config.get('model_size', args.config)
    args.epochs = config.get('epochs', args.epochs)
    args.batch_size = config.get('batch_size', args.batch_size)
    args.seq_len = config.get('seq_len', args.seq_len)
    args.lr = config.get('learning_rate', args.lr)
    args.warmup = config.get('warmup_steps', args.warmup)
    args.reset = config.get('reset_checkpoint', args.reset)
    args.max_tokens_per_dataset = config.get('max_tokens_per_dataset', args.max_tokens_per_dataset)
    
    # Extensions
    args.use_agency = config.get('use_agency', False)
    args.use_prsc = config.get('use_prsc', False)
    args.use_temporal_smf = config.get('use_temporal_smf', False)
    args.use_entanglement = config.get('use_entanglement', False)
    args.use_stability = config.get('use_stability', False)
    args.use_stochastic_resonance = config.get('use_stochastic_resonance', False)
    args.landscape = config.get('landscape_path', args.landscape)
    args.landscape_min_confidence = config.get('landscape_min_confidence', args.landscape_min_confidence)

    # Handle datasets
    datasets = config.get('datasets', [])
    if len(datasets) > 1:
        args.datasets = datasets
        args.dataset = DEFAULT_DATASET  # Not used when datasets is set
    elif len(datasets) == 1:
        args.dataset = datasets[0]
        args.datasets = None
    
    # Print loaded config
    print(f"  Model: {args.config}")
    print(f"  Datasets: {datasets}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    
    return args


def main():
    """Main training function."""
    args = parse_args()
    
    # Handle --show-template-report flag
    if args.show_template_report:
        if args.datasets:
            dataset_list = args.datasets
        elif args.recommended_datasets:
            dataset_list = RECOMMENDED_DATASETS
        else:
            dataset_list = [args.dataset]
        
        print(print_template_report(dataset_list))
        return
    
    print("=" * 60)
    print("Reso-LLM v2 Training")
    print("=" * 60)
    
    # Tokenizer first (to get vocab size)
    print("\nInitializing tokenizer...")
    tokenizer = create_default_tokenizer()
    
    # Configuration - use preset size
    config = ResoLLMConfig.from_size(args.config)
    config.vocab_size = tokenizer.vocab_size  # Match tokenizer vocab
    
    # Apply extensions from args (loaded from config file)
    if hasattr(args, 'use_agency') and (
        args.use_agency or args.use_prsc or args.use_temporal_smf or
        args.use_entanglement or args.use_stability or args.use_stochastic_resonance or
        args.landscape
    ):
        print("\nEnabling extended features...")
        config.standard_mode = False
        
        if args.use_agency:
            print("  - Agency Layer: Enabled")
            config.agency.enabled = True
            
        if args.use_prsc or args.landscape:
            print("  - PRSC Semantic Layer: Enabled")
            config.prsc.enabled = True
            if args.landscape:
                print(f"    Landscape: {args.landscape}")
                config.prsc.landscape_path = args.landscape
            if args.landscape_min_confidence is not None:
                config.prsc.landscape_min_confidence = args.landscape_min_confidence
            
        if args.use_temporal_smf:
            print("  - Temporal SMF: Enabled")
            config.temporal_smf.enabled = True
            
        if args.use_entanglement:
            print("  - Entanglement Network: Enabled")
            config.entanglement.enabled = True
            
        if args.use_stability:
            print("  - Stability Monitor: Enabled")
            config.stability.enabled = True
            
        if args.use_stochastic_resonance:
            print("  - Stochastic Resonance: Enabled")
            config.stochastic_resonance.enabled = True

    print(f"\nModel size: {args.config}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Dimension: {config.dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  FFN dim: {config.ffn_dim}")
    print(f"  Max seq len: {config.max_seq_len}")
    
    # Dataset - support single, multiple, or recommended datasets
    if args.recommended_datasets:
        print("\nUsing recommended multi-dataset (guanaco, databricks-dolly-15k, alpaca)")
        # Show template mapping
        print("\nTemplate Mapping:")
        for ds_name in RECOMMENDED_DATASETS:
            template_type = get_template_for_dataset(ds_name)
            print(f"  {ds_name}: {template_type.value}")
        
        dataset = create_recommended_multi_dataset(
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_tokens_per_dataset=args.max_tokens_per_dataset
        )
    elif args.datasets:
        print(f"\nLoading multiple datasets: {', '.join(args.datasets)}")
        # Show template mapping
        print("\nTemplate Mapping:")
        for ds_name in args.datasets:
            template_type = get_template_for_dataset(ds_name)
            print(f"  {ds_name}: {template_type.value}")
        
        dataset = MultiDataset(
            dataset_names=args.datasets,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_tokens_per_dataset=args.max_tokens_per_dataset,
            shuffle=True,
            validate_output=True
        )
        
        # Run comprehensive validation for multi-dataset
        print("\nRunning multi-dataset validation...")
        validation_report = dataset.validate(num_samples=5, verbose=True)
        
        if not validation_report["valid"]:
            print("\nERROR: Multi-dataset validation failed!")
            for issue in validation_report["issues"]:
                print(f"  - {issue}")
            print("Exiting. Please check your dataset configuration.")
            return
    elif args.dataset == DEFAULT_DATASET:
        print(f"\nLoading dataset: {args.dataset}")
        # Use the optimized Guanaco loader
        dataset = create_guanaco_dataset(
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
    else:
        print(f"\nLoading dataset: {args.dataset}")
        # Use generic HuggingFace loader
        dataset = HuggingFaceDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
    
    if len(dataset.data) == 0:
        print("Error: No data loaded. Exiting.")
        return
    
    # Verify dataset fusion quality
    print("\n" + "=" * 60)
    print("Verifying Dataset Fusion Quality")
    print("=" * 60)
    verification = verify_dataset_fusion(dataset, num_samples=3, verbose=True)
    
    if not verification["valid"]:
        print("\nERROR: Dataset fusion verification failed!")
        for issue in verification["issues"]:
            print(f"  - {issue}")
        print("Exiting. Please check your dataset configuration.")
        return
    
    # Checkpoint path based on model size
    checkpoint_path = get_checkpoint_path(args.config)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Model loading logic:
    # - Always try to load existing checkpoint for the requested size
    # - Only skip if --reset flag is set
    if args.reset:
        print(f"\n--reset flag set. Creating fresh {args.config} model...")
        model = ResoLLMModel(config)
    elif os.path.exists(checkpoint_path):
        print(f"\nFound existing checkpoint: {checkpoint_path}")
        
        # Check if checkpoint config matches
        saved_config = ResoLLMModel.load_checkpoint_config(checkpoint_path)
        
        if saved_config is not None:
            config_matches = ResoLLMModel.config_matches(checkpoint_path, config)
            is_legacy = saved_config.get('_extracted_from_state_dict', False)
            
            if config_matches:
                if is_legacy:
                    print("  (Legacy format - config extracted from weights)")
                print("  Config matches - loading checkpoint...")
                try:
                    model = ResoLLMModel.load(checkpoint_path, config)
                    print("  Checkpoint loaded - continuing training")
                except Exception as e:
                    print(f"  Error loading checkpoint: {e}")
                    print("  Creating fresh model...")
                    model = ResoLLMModel(config)
            else:
                # Config mismatch - this shouldn't happen if checkpoint is named by size
                # But handle it gracefully
                print(f"  WARNING: Checkpoint config doesn't match {args.config} preset!")
                print(f"    Saved: dim={saved_config.get('dim')}, layers={saved_config.get('num_layers')}")
                print(f"    Expected: dim={config.dim}, layers={config.num_layers}")
                print("  Creating fresh model (use --reset to overwrite checkpoint)...")
                model = ResoLLMModel(config)
        else:
            # Could not read config - try loading anyway
            print("  Could not read config from checkpoint, attempting load...")
            try:
                model = ResoLLMModel.load(checkpoint_path, config, strict=False)
                print("  Checkpoint loaded (strict=False)")
            except Exception as e:
                print(f"  Error: {e}")
                print("  Creating fresh model...")
                model = ResoLLMModel(config)
    else:
        print(f"\nNo checkpoint found for {args.config}. Creating fresh model...")
        model = ResoLLMModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Trainer
    trainer = ResoLLMTrainer(
        model=model, 
        dataset=dataset, 
        learning_rate=args.lr,
        warmup_steps=args.warmup
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train(
        epochs=args.epochs, 
        save_path=checkpoint_path
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
