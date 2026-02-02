"""
Training Utilities for LLM Fusion.

Provides specialized training for resonance fusion layers while keeping
the base model frozen. Includes multi-objective loss functions.
"""
import math
import os
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import FusionConfig


@dataclass
class TrainingConfig:
    """Configuration for fusion layer training."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adapter_lr_mult: float = 1.0
    fusion_lr_mult: float = 0.3
    adapter_warmup_steps: int = 0
    warmup_steps: int = 100
    max_steps: int = 10000
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    batch_size: int = 1
    
    # Scheduling
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    
    # Loss weights
    lm_loss_weight: float = 1.0
    coherence_loss_weight: float = 0.1
    entropy_loss_weight: float = 0.05
    kuramoto_loss_weight: float = 0.05
    
    # Regularization
    fusion_alpha_penalty: float = 0.01  # Penalize large fusion weights
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Output
    output_dir: str = "fusion_training"
    save_total_limit: int = 3
    
    # Hardware
    fp16: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ResonanceLoss(nn.Module):
    """
    Multi-objective loss for resonance fusion training.
    
    Combines:
    - Language modeling loss (cross-entropy)
    - Coherence maximization (encourage high coherence)
    - Entropy regularization (prevent collapse)
    - Kuramoto synchronization (encourage stable phase dynamics)
    
    L = L_lm + λ_c * L_coherence + λ_e * L_entropy + λ_k * L_kuramoto
    
    Args:
        config: TrainingConfig with loss weights
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        super().__init__()
        
        if config is None:
            config = TrainingConfig()
        
        self.lm_weight = config.lm_loss_weight
        self.coherence_weight = config.coherence_loss_weight
        self.entropy_weight = config.entropy_loss_weight
        self.kuramoto_weight = config.kuramoto_loss_weight
        self.alpha_penalty = config.fusion_alpha_penalty
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        metrics: Optional[Dict[str, float]] = None,
        fusion_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            logits: (batch, seq, vocab) model output logits
            labels: (batch, seq) target token IDs
            metrics: Dict with coherence, entropy, kuramoto_order from fusion
            fusion_alpha: Fusion weight parameter for regularization
            
        Returns:
            total_loss: Combined scalar loss
            loss_components: Dict with individual loss values
        """
        components = {}
        
        # Language modeling loss
        vocab_size = logits.size(-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        components["lm_loss"] = lm_loss.item()
        
        total_loss = self.lm_weight * lm_loss
        
        # Coherence loss (minimize negative coherence = maximize coherence)
        if metrics and "gate_coherence" in metrics:
            coherence = metrics["gate_coherence"]
            coherence_loss = 1.0 - coherence  # Loss = 0 when coherence = 1
            components["coherence_loss"] = coherence_loss
            total_loss = total_loss + self.coherence_weight * coherence_loss
        
        # Entropy regularization (prevent collapse to single prime)
        if metrics and "prime_entropy" in metrics:
            entropy = metrics["prime_entropy"]
            # We want moderate entropy - not too low (collapse) or too high (random)
            # Optimal entropy ~= log2(num_primes) / 2
            target_entropy = 2.0  # Roughly log2(25)/2
            entropy_loss = (entropy - target_entropy) ** 2
            components["entropy_loss"] = entropy_loss
            total_loss = total_loss + self.entropy_weight * entropy_loss
        
        # Kuramoto synchronization loss
        if metrics and "kuramoto_order" in metrics:
            order = metrics["kuramoto_order"]
            # Encourage synchronization but not too strong
            target_order = 0.7
            kuramoto_loss = (order - target_order) ** 2
            components["kuramoto_loss"] = kuramoto_loss
            total_loss = total_loss + self.kuramoto_weight * kuramoto_loss
        
        # Alpha regularization
        if fusion_alpha is not None:
            alpha_val = torch.sigmoid(fusion_alpha).mean()
            alpha_loss = self.alpha_penalty * alpha_val ** 2
            components["alpha_loss"] = alpha_loss.item()
            total_loss = total_loss + alpha_loss
        
        components["total_loss"] = total_loss.item()
        
        return total_loss, components


class FusionTrainer:
    """
    Trainer for resonance fusion layers.
    
    Handles:
    - Training loop with gradient accumulation
    - Learning rate scheduling
    - Logging and checkpointing
    - Evaluation during training
    
    Args:
        model: ResonanceWrapper with fusion layers
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        tokenizer: Tokenizer for data processing
        config: TrainingConfig
        fusion_config: FusionConfig for the model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Any = None,
        config: Optional[TrainingConfig] = None,
        fusion_config: Optional[FusionConfig] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.fusion_config = fusion_config or FusionConfig()
        
        # Setup device
        self.device = self._get_device()
        self.model.to(self.device)
        
        # Create loss function
        self.loss_fn = ResonanceLoss(self.config)
        
        # Setup optimizer (only fusion parameters)
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self._fusion_warmup_active = False
        
        # Logging
        self.log_history: List[Dict[str, Any]] = []
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Optional adapter-only warmup
        if self.config.adapter_warmup_steps > 0:
            self._set_fusion_requires_grad(False)
            self._fusion_warmup_active = True
    
    def _get_device(self) -> torch.device:
        """Determine device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for fusion parameters (including output adapter)."""
        # Prefer explicit adapter/fusion separation for controlled deltas
        if hasattr(self.model, "get_adapter_parameters") and hasattr(self.model, "get_fusion_only_parameters"):
            adapter_params = [p for p in self.model.get_adapter_parameters() if p.requires_grad]
            fusion_params = [p for p in self.model.get_fusion_only_parameters() if p.requires_grad]
            param_groups = []
            if fusion_params:
                param_groups.append({
                    "params": fusion_params,
                    "lr": self.config.learning_rate * self.config.fusion_lr_mult,
                })
            if adapter_params:
                param_groups.append({
                    "params": adapter_params,
                    "lr": self.config.learning_rate * self.config.adapter_lr_mult,
                })
            if param_groups:
                return torch.optim.AdamW(
                    param_groups,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
        
        # Fallback: single param group (fusion + adapter or all params)
        if hasattr(self.model, "get_fusion_parameters"):
            params = self.model.get_fusion_parameters()
        elif hasattr(self.model, "fusion"):
            params = list(self.model.fusion.parameters())
        else:
            params = self.model.parameters()
        
        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
            )
        elif self.config.scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.max_steps,
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
            )

    def _set_fusion_requires_grad(self, enabled: bool):
        """Enable/disable gradients for fusion layers only."""
        if hasattr(self.model, "get_fusion_only_parameters"):
            for param in self.model.get_fusion_only_parameters():
                param.requires_grad = enabled
        elif hasattr(self.model, "fusion"):
            for param in self.model.fusion.parameters():
                param.requires_grad = enabled
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Training summary with metrics
        """
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        self.model.train()
        
        # Training loop
        accumulation_loss = 0.0
        accumulation_steps = 0
        
        while self.global_step < self.config.max_steps:
            for batch in train_loader:
                if self.global_step >= self.config.max_steps:
                    break

                if self._fusion_warmup_active and self.global_step >= self.config.adapter_warmup_steps:
                    self._set_fusion_requires_grad(True)
                    self._fusion_warmup_active = False
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                # Get metrics from fusion layers
                metrics = None
                if hasattr(self.model, "get_average_metrics"):
                    metrics = self.model.get_average_metrics()
                
                # Get fusion alpha for regularization
                fusion_alpha = None
                if hasattr(self.model, "fusion") and hasattr(self.model.fusion, "fusion_layers"):
                    alphas = []
                    for layer in self.model.fusion.fusion_layers.values():
                        if hasattr(layer, "alpha") and isinstance(layer.alpha, nn.Parameter):
                            alphas.append(layer.alpha)
                    if alphas:
                        fusion_alpha = torch.stack(alphas)
                
                # Compute loss
                loss, loss_components = self.loss_fn(logits, labels, metrics, fusion_alpha)
                
                # Scale for accumulation
                loss = loss / self.config.gradient_accumulation
                
                # Backward
                loss.backward()
                
                accumulation_loss += loss.item()
                accumulation_steps += 1
                
                # Update weights
                if accumulation_steps >= self.config.gradient_accumulation:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        log_entry = {
                            "step": self.global_step,
                            "loss": accumulation_loss,
                            "lr": self.scheduler.get_last_lr()[0],
                            **loss_components,
                        }
                        if metrics:
                            log_entry.update(metrics)
                        
                        self.log_history.append(log_entry)
                        self._print_log(log_entry)
                    
                    # Evaluation
                    if self.eval_dataset and self.global_step % self.config.eval_interval == 0:
                        eval_loss = self.evaluate()
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint("best")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                    
                    accumulation_loss = 0.0
                    accumulation_steps = 0
            
            self.epoch += 1
        
        # Final save
        self.save_checkpoint("final")
        
        # Save training log
        self._save_log()
        
        return {
            "final_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "log_history": self.log_history,
        }
    
    def evaluate(self) -> float:
        """
        Evaluate on evaluation dataset.
        
        Returns:
            Average evaluation loss
        """
        if self.eval_dataset is None:
            return float('inf')
        
        self.model.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                
                outputs = self.model(input_ids=input_ids)
                
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                loss, _ = self.loss_fn(logits, labels)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Eval loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        path = Path(self.config.output_dir) / f"checkpoint_{name}"
        path.mkdir(parents=True, exist_ok=True)
        
        # Save fusion weights
        if hasattr(self.model, "save_fusion_weights"):
            self.model.save_fusion_weights(str(path / "fusion_weights.pt"))
        else:
            torch.save(
                self.model.state_dict(),
                str(path / "model_state.pt")
            )
        
        # Save optimizer and scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }, str(path / "trainer_state.pt"))
        
        # Save configs
        with open(path / "training_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"Saved checkpoint: {path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if self.config.save_total_limit <= 0:
            return
        
        output_dir = Path(self.config.output_dir)
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_step_")],
            key=lambda x: int(x.name.split("_")[-1])
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)
    
    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        path = Path(path)
        
        # Load fusion weights
        if hasattr(self.model, "load_fusion_weights"):
            self.model.load_fusion_weights(str(path / "fusion_weights.pt"))
        else:
            state_dict = torch.load(str(path / "model_state.pt"))
            self.model.load_state_dict(state_dict)
        
        # Load trainer state
        trainer_state = torch.load(str(path / "trainer_state.pt"))
        self.optimizer.load_state_dict(trainer_state["optimizer"])
        self.scheduler.load_state_dict(trainer_state["scheduler"])
        self.global_step = trainer_state["global_step"]
        self.epoch = trainer_state["epoch"]
        self.best_eval_loss = trainer_state["best_eval_loss"]
        
        print(f"Loaded checkpoint from {path} at step {self.global_step}")
    
    def _print_log(self, entry: Dict[str, Any]):
        """Print log entry."""
        parts = [f"Step {entry['step']}"]
        parts.append(f"loss={entry['loss']:.4f}")
        parts.append(f"lr={entry['lr']:.2e}")
        
        if "gate_coherence" in entry:
            parts.append(f"coherence={entry['gate_coherence']:.3f}")
        if "kuramoto_order" in entry:
            parts.append(f"kuramoto={entry['kuramoto_order']:.3f}")
        
        print(" | ".join(parts))
    
    def _save_log(self):
        """Save training log to file."""
        log_path = Path(self.config.output_dir) / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.log_history, f, indent=2)


def create_fusion_dataset(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 512,
    assistant_start_token_ids: Optional[List[int]] = None,
    assistant_end_token_ids: Optional[List[int]] = None,
    assistant_start: Optional[str] = None,
    assistant_end: Optional[str] = None,
) -> Dataset:
    """
    Create a simple dataset for fusion training.
    
    Args:
        texts: List of text samples
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        PyTorch Dataset
    """
    class SimpleDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings["input_ids"])
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["labels"][idx],
            }
    
    use_offsets = bool(assistant_start and assistant_end and getattr(tokenizer, "is_fast", False))
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=use_offsets,
    )

    labels = encodings["input_ids"].clone()
    attention_mask = encodings.get("attention_mask")
    if use_offsets and "offset_mapping" in encodings:
        offset_mapping = encodings.pop("offset_mapping")
        matched = 0
        fallback = 0
        for i in range(labels.size(0)):
            offsets = offset_mapping[i]
            if isinstance(offsets, torch.Tensor):
                offsets = offsets.tolist()
            mask = _assistant_only_mask_from_offsets(
                texts[i],
                offsets,
                assistant_start,
                assistant_end,
            )
            if not any(mask):
                mask = [1] * len(mask)
                fallback += 1
            else:
                matched += 1
            masked_labels = [
                token_id if keep else -100
                for token_id, keep in zip(encodings["input_ids"][i].tolist(), mask)
            ]
            labels[i] = torch.tensor(masked_labels, dtype=labels.dtype)
        total = labels.size(0)
        print(f"  Assistant-only loss masking: {matched}/{total} samples matched, {fallback} fallback")
    elif assistant_start_token_ids and assistant_end_token_ids:
        matched = 0
        fallback = 0
        for i in range(labels.size(0)):
            input_ids = encodings["input_ids"][i].tolist()
            mask = _assistant_only_mask(input_ids, assistant_start_token_ids, assistant_end_token_ids)
            if not any(mask):
                # Fallback to full loss if no assistant spans found
                mask = [1] * len(input_ids)
                fallback += 1
            else:
                matched += 1
            masked_labels = [
                token_id if keep else -100
                for token_id, keep in zip(input_ids, mask)
            ]
            labels[i] = torch.tensor(masked_labels, dtype=labels.dtype)
        total = labels.size(0)
        print(f"  Assistant-only loss masking: {matched}/{total} samples matched, {fallback} fallback")

    if attention_mask is not None:
        labels[attention_mask == 0] = -100

    encodings["labels"] = labels
    
    return SimpleDataset(encodings)


def _assistant_only_mask(
    input_ids: List[int],
    assistant_start_ids: List[int],
    assistant_end_ids: List[int],
) -> List[int]:
    """Build a mask that keeps only assistant spans for loss computation."""
    mask = [0] * len(input_ids)

    start_positions = _find_subsequence_positions(input_ids, assistant_start_ids)
    if not start_positions:
        return mask
    end_positions = _find_subsequence_positions(input_ids, assistant_end_ids)

    for start_pos in start_positions:
        content_start = start_pos + len(assistant_start_ids)
        end_pos = len(input_ids)
        for candidate in end_positions:
            if candidate >= content_start:
                end_pos = candidate
                break
        for i in range(content_start, end_pos):
            mask[i] = 1

    return mask


def _find_subsequence_positions(seq: List[int], pattern: List[int]) -> List[int]:
    """Find all starting indices of a subsequence in a sequence."""
    if not pattern:
        return []
    positions = []
    last_start = len(seq) - len(pattern)
    for i in range(last_start + 1):
        if seq[i:i + len(pattern)] == pattern:
            positions.append(i)
    return positions


def _assistant_only_mask_from_offsets(
    text: str,
    offsets: List[List[int]],
    assistant_start: str,
    assistant_end: str,
) -> List[int]:
    """Build a mask that keeps only assistant spans using character offsets."""
    if not assistant_start or not assistant_end:
        return [0] * len(offsets)

    spans = []
    search_from = 0
    while True:
        start_idx = text.find(assistant_start, search_from)
        if start_idx == -1:
            break
        content_start = start_idx + len(assistant_start)
        end_idx = text.find(assistant_end, content_start)
        if end_idx == -1:
            end_idx = len(text)
        spans.append((content_start, end_idx))
        search_from = end_idx + len(assistant_end)

    if not spans:
        return [0] * len(offsets)

    mask = [0] * len(offsets)
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for span_start, span_end in spans:
            if start >= span_start and end <= span_end:
                mask[i] = 1
                break
    return mask


def quick_train(
    model: nn.Module,
    texts: List[str],
    tokenizer: Any,
    steps: int = 1000,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """
    Quick training helper for fusion layers.
    
    Args:
        model: ResonanceWrapper
        texts: Training texts
        tokenizer: Tokenizer
        steps: Number of training steps
        learning_rate: Learning rate
        
    Returns:
        Training results
    """
    dataset = create_fusion_dataset(texts, tokenizer)
    
    config = TrainingConfig(
        learning_rate=learning_rate,
        max_steps=steps,
        log_interval=max(1, steps // 20),
        eval_interval=steps + 1,  # No eval
        save_interval=steps,
    )
    
    trainer = FusionTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        config=config,
    )
    
    return trainer.train()
