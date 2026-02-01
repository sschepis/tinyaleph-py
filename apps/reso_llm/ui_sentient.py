from __future__ import annotations

"""
Sentient LLM Training Interface
================================
A comprehensive training dashboard for the Sentient LLM system with:
- Real-time metrics visualization
- Multiple chart types (loss, gradients, learning rate)
- Configuration presets
- Export/import settings
- Live training logs
- Chat interface for testing
"""

import argparse
import json
import os
import sys
import threading
import time
import queue
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from collections import deque

import pandas as pd
import torch

import gradio as gr

# Ensure project root is importable when launched directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tinyaleph.training.sentient_llm import LoopConfig, run_sentient_llm_loop
from tinyaleph.training.sentient_llm.teacher_local import LocalTeacher
from tinyaleph.training.sentient_llm.observer_runtime import SentientObserverWrapper
from tinyaleph.training.sentient_llm.adapters.reso_llm_student import CallableStudentTrainer
from tinyaleph.training.sentient_llm.schema import TrainingShard

from apps.reso_llm.inference import ResoLLMInference, build_chat_prompt, extract_assistant_response
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.config import ResoLLMConfig
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.train import ResoLLMTrainer, get_checkpoint_path, CHECKPOINT_DIR
from apps.reso_llm.data_sources import (
    LMStudioConfig, LMStudioTeacher, create_teacher_generator,
    load_conversation_dataset, Conversation
)
from apps.resoformer.torch_backend import get_device
from tinyaleph.ml.resoformer import Tensor


# -------------------------- Teacher / Student helpers -------------------------


def _load_or_init_model(model_path: str, tokenizer_vocab: int, size: str = "tiny") -> ResoLLMModel:
    config = ResoLLMConfig.from_size(size)
    config.vocab_size = tokenizer_vocab
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    if model_path and os.path.exists(model_path):
        try:
            return ResoLLMModel.load(model_path, config, strict=False)
        except Exception:
            pass
    return ResoLLMModel(config)


def make_teacher_generator(model_path: str):
    # Deterministic teacher: returns clean JSON shards so training data is not gibberish.
    # If you prefer live LLM generation, replace this with an actual ResoLLMInference call
    # and ensure the model can follow the JSON-instruction in teacher_local._prompt_for_symbol.
    def gen(prompt: str) -> str:
        return json.dumps(
            [
                {
                    "kind": "label",
                    "input_text": "Name the concept symbol.",
                    "target_text": "A canonical symbol in the lexicon.",
                },
                {
                    "kind": "definition",
                    "input_text": "Define the symbol and its role.",
                    "target_text": "A stable semantic concept used for continual learning.",
                },
                {
                    "kind": "example",
                    "input_text": "Use the symbol in context.",
                    "target_text": "In practice, this symbol anchors meaning across episodes.",
                },
            ]
        )

    return gen


# Simple shard-backed trainer
def make_student_trainer(
    seq_len: int = 256,
    batch_size: int = 4,
    lr: float = 3e-4,
    model_size: str = "tiny",
    warmup_steps: int = 20,
    max_train_steps: int = 16,
    use_extensions: bool = False,
    use_kuramoto: bool = True,
    use_smf: bool = True,
    use_coherence: bool = True,
):
    tokenizer = create_default_tokenizer()
    ckpt = get_checkpoint_path(model_size)
    
    # Create model config with optional extensions
    config = ResoLLMConfig.from_size(model_size, standard=not use_extensions)
    config.vocab_size = tokenizer.vocab_size
    config.use_kuramoto_dynamics = use_kuramoto
    config.use_smf_memory = use_smf
    config.use_coherence_gating = use_coherence
    
    # Load or initialize model
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    if ckpt and os.path.exists(ckpt):
        try:
            model = ResoLLMModel.load(ckpt, config, strict=False)
        except Exception:
            model = ResoLLMModel(config)
    else:
        model = ResoLLMModel(config)
    
    device = get_device()
    model = model.to(device)

    dummy_tokens = list(range(seq_len * batch_size + 2))
    dummy_ds = _ShardDataset(tokens=dummy_tokens, seq_len=seq_len, batch_size=batch_size)
    trainer = ResoLLMTrainer(
        model=model,
        dataset=dummy_ds,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        max_steps=10_000,
    )
    
    # Store max_train_steps for use in train_fn
    trainer_max_steps = max_train_steps

    last_dataset: Optional[_ShardDataset] = None

    def train_fn(shards: List[TrainingShard]) -> Dict[str, Any]:
        nonlocal last_dataset
        if not shards:
            return {"trained_shards": 0, "steps": 0}

        text_tokens: List[int] = []
        for sh in shards:
            chat = _format_shard(sh)
            text_tokens.extend(tokenizer.encode(chat))

        ds = _ShardDataset(tokens=text_tokens, seq_len=seq_len, batch_size=batch_size)
        trainer.dataset = ds
        last_dataset = ds

        steps = min(max(1, len(ds)), trainer_max_steps)
        losses: List[float] = []
        grads: List[float] = []
        for _ in range(steps):
            loss, grad = trainer.train_step()
            losses.append(loss)
            grads.append(grad)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        trainer.model.save(ckpt)

        return {
            "trained_shards": len(shards),
            "steps": steps,
            "loss_mean": float(sum(losses) / max(1, len(losses))),
            "grad_norm_mean": float(sum(grads) / max(1, len(grads))),
        }

    def eval_fn() -> Dict[str, Any]:
        if not last_dataset:
            return {"ok": True, "eval_loss": None}
        trainer.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for _ in range(min(len(last_dataset), 4)):
                x, y = last_dataset.get_batch()
                x_t = torch.tensor(x.data, dtype=torch.long, device=device).view(batch_size, seq_len)
                y_t = torch.tensor(y.data, dtype=torch.long, device=device).view(batch_size, seq_len)
                logits = trainer.model(x_t)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, trainer.model.config.vocab_size),
                    y_t.view(-1),
                )
                total += loss.item()
                count += 1
        trainer.model.train()
        return {"ok": True, "eval_loss": total / max(1, count)}

    return CallableStudentTrainer(train_fn=train_fn, eval_fn=eval_fn)


def _format_shard(shard: TrainingShard) -> str:
    chat = (
        f"<|user|>\n{shard.input_text}\n<|endofuser|>\n"
        f"<|assistant|>\n{shard.target_text}\n<|endofassistant|>"
    )
    return chat


@dataclass
class _ShardDataset:
    tokens: List[int]
    seq_len: int
    batch_size: int

    def __len__(self) -> int:
        n_sequences = max(1, (len(self.tokens) - 1) // self.seq_len)
        return max(1, n_sequences // self.batch_size)

    def get_batch(self) -> (Tensor, Tensor):
        import torch as _torch

        max_idx = len(self.tokens) - self.seq_len - 1
        batch_x: List[int] = []
        batch_y: List[int] = []
        for _ in range(self.batch_size):
            idx = 0 if max_idx <= 0 else _torch.randint(0, max_idx, ()).item()
            x_seq = self.tokens[idx : idx + self.seq_len]
            y_seq = self.tokens[idx + 1 : idx + self.seq_len + 1]
            batch_x.extend(x_seq)
            batch_y.extend(y_seq)
        return Tensor(batch_x, (self.batch_size, self.seq_len)), Tensor(batch_y, (self.batch_size, self.seq_len))


# Global cache for loaded dataset to avoid reloading
_PRETRAIN_CACHE: Dict[str, List[Any]] = {}


def _run_pretraining(
    training_params: "TrainingParams", 
    student, 
    out_dir: str,
    progress_callback: Optional[Callable[[float], None]] = None
):
    """Pretrain on HuggingFace dataset before starting the sentient loop."""
    import json
    from apps.reso_llm.data_sources import load_conversation_dataset
    from tinyaleph.training.sentient_llm.schema import TrainingShard
    
    cache_key = f"{training_params.hf_dataset_name}:{training_params.hf_max_samples}"
    
    # Check cache first
    if cache_key in _PRETRAIN_CACHE:
        _add_log(f"Using cached dataset: {training_params.hf_dataset_name}")
        print(f"[Pretrain] Using cached dataset: {training_params.hf_dataset_name}")
        conversations = _PRETRAIN_CACHE[cache_key]
    else:
        _add_log(f"Loading dataset: {training_params.hf_dataset_name}")
        _add_log(f"Max samples: {training_params.hf_max_samples}")
        print(f"[Pretrain] Loading dataset: {training_params.hf_dataset_name}")
        print(f"[Pretrain] Max samples: {training_params.hf_max_samples}")
        
        try:
            # load_conversation_dataset is a generator, collect it
            conversations = list(load_conversation_dataset(
                dataset_name=training_params.hf_dataset_name,
                max_samples=training_params.hf_max_samples,
            ))
            # Cache for future runs
            _PRETRAIN_CACHE[cache_key] = conversations
        except Exception as e:
            print(f"[Pretrain] Failed to load dataset: {e}")
            return
    
    print(f"[Pretrain] Loaded {len(conversations)} conversations", flush=True)
    
    # Debug: print first conversation structure
    if conversations:
        conv0 = conversations[0]
        print(f"[Pretrain] First conversation type: {type(conv0)}", flush=True)
        if hasattr(conv0, 'turns'):
            print(f"[Pretrain] Has {len(conv0.turns)} turns", flush=True)
            if conv0.turns:
                turn0 = conv0.turns[0]
                print(f"[Pretrain] First turn type: {type(turn0)}", flush=True)
                print(f"[Pretrain] First turn: {turn0}", flush=True)
    
    print(f"[Pretrain] Converting conversations to training shards...", flush=True)
    
    # Convert conversations to training shards
    shards = []
    try:
        for idx, conv in enumerate(conversations):
            if idx > 0 and idx % 100 == 0:
                print(f"[Pretrain] Processing conversation {idx}/{len(conversations)}... ({len(shards)} shards so far)", flush=True)
            
            # Get turns - handle both Conversation objects and dicts
            if hasattr(conv, 'turns'):
                turns = conv.turns
            elif isinstance(conv, dict) and 'turns' in conv:
                turns = conv['turns']
            else:
                turns = []
            
            # Process consecutive pairs (user, assistant)
            for i in range(len(turns)):
                turn = turns[i]
                
                # Get role and content
                if hasattr(turn, 'role'):
                    role = turn.role
                    content = turn.content
                elif isinstance(turn, dict):
                    role = turn.get('role', '')
                    content = turn.get('content', '')
                else:
                    continue
                
                # Look for user turn followed by assistant turn
                if role == "user" and i + 1 < len(turns):
                    next_turn = turns[i + 1]
                    if hasattr(next_turn, 'role'):
                        next_role = next_turn.role
                        next_content = next_turn.content
                    elif isinstance(next_turn, dict):
                        next_role = next_turn.get('role', '')
                        next_content = next_turn.get('content', '')
                    else:
                        continue
                    
                    if next_role == "assistant" and content and next_content:
                        import uuid
                        # Truncate long texts to avoid sequence length issues
                        # Model seq_len is typically 256-1024, leave room for special tokens
                        seq_len = getattr(training_params, 'seq_len', 256)
                        max_chars = seq_len * 4  # ~4 chars per token
                        truncated_input = content[:max_chars]
                        truncated_target = next_content[:max_chars]
                        
                        shard = TrainingShard(
                            shard_id=f"pretrain_{uuid.uuid4().hex[:8]}",
                            created_at=time.time(),
                            symbol_ids=["SYM:pretrain"],
                            kind="qa",  # Must be one of: label, definition, example, qa, tool, contrastive
                            input_text=truncated_input,
                            target_text=truncated_target,
                        )
                        shards.append(shard)
    except Exception as e:
        import traceback
        print(f"[Pretrain] Error during shard conversion: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return
    
    print(f"[Pretrain] Created {len(shards)} training shards", flush=True)
    
    if not shards:
        print("[Pretrain] No shards created, skipping pretraining")
        return
    
    # Run pretraining epochs
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    
    for epoch in range(training_params.pretrain_epochs):
        print(f"[Pretrain] Epoch {epoch + 1}/{training_params.pretrain_epochs}", flush=True)
        
        # Shuffle shards for each epoch
        import random
        epoch_shards = shards.copy()
        random.shuffle(epoch_shards)
        
        # Train in batches
        batch_size = training_params.batch_size * 8  # Larger batches for pretraining
        total_batches = (len(epoch_shards) + batch_size - 1) // batch_size
        print(f"[Pretrain] Training {total_batches} batches of size {batch_size}...", flush=True)
        
        for batch_idx, i in enumerate(range(0, len(epoch_shards), batch_size)):
            batch = epoch_shards[i:i + batch_size]
            
            try:
                result = student.train_on_shards(batch)
            except Exception as train_err:
                import traceback
                _add_log(f"Training error: {train_err}", "ERROR")
                print(f"[Pretrain] Training error: {train_err}", flush=True)
                print(traceback.format_exc(), flush=True)
                continue
            
            # Log metrics
            metric = {
                "cycle": -(training_params.pretrain_epochs - epoch),  # Negative cycle for pretrain
                "kind": "pretrain",
                "metrics": result,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metric) + "\n")
            
            loss = result.get("loss_mean", 0)
            grad = result.get("grad_norm_mean", 0)
            
            # Update progress
            total_steps = training_params.pretrain_epochs * total_batches
            current_step = epoch * total_batches + batch_idx + 1
            progress = current_step / total_steps
            if progress_callback:
                progress_callback(progress)
            
            # Log every few batches to avoid spam
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                _add_log(f"Epoch {epoch + 1}/{training_params.pretrain_epochs}, batch {batch_idx + 1}/{total_batches}: loss={loss:.4f}, grad={grad:.4f}")
            
            print(f"[Pretrain] Epoch {epoch + 1}, batch {batch_idx + 1}/{total_batches}, loss: {loss:.4f}", flush=True)
    
    _add_log("Pretraining complete")
    print("[Pretrain] Pretraining complete", flush=True)


# -------------------------- Gradio App -------------------------


@dataclass
class TrainingParams:
    """Container for all training parameters"""
    seq_len: int = 256
    batch_size: int = 4
    learning_rate: float = 3e-4
    model_size: str = "tiny"
    warmup_steps: int = 20
    max_train_steps: int = 16
    use_extensions: bool = False
    use_kuramoto: bool = True
    use_smf: bool = True
    use_coherence: bool = True
    # Data/Teacher options
    use_hf_dataset: bool = False
    hf_dataset_name: str = "timdettmers/openassistant-guanaco"
    hf_max_samples: int = 5000
    pretrain_epochs: int = 1
    use_lmstudio: bool = False
    lmstudio_url: str = "http://localhost:1234"
    lmstudio_model: str = "llama-3.1-8b-lexi-uncensored-v2"
    use_rlhf: bool = False
    rlhf_batch_size: int = 16
    dpo_beta: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingParams":
        """Import configuration from dictionary"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Configuration presets for common use cases
PRESETS: Dict[str, Dict[str, Any]] = {
    "quick_test": {
        "name": "Quick Test",
        "description": "Fast iteration for testing (tiny model, minimal epochs)",
        "model_size": "tiny",
        "seq_len": 128,
        "batch_size": 2,
        "learning_rate": 5e-4,
        "warmup_steps": 10,
        "max_train_steps": 8,
        "use_hf_dataset": False,
        "hf_max_samples": 100,
        "pretrain_epochs": 1,
    },
    "balanced": {
        "name": "Balanced Training",
        "description": "Good balance of speed and quality",
        "model_size": "small",
        "seq_len": 256,
        "batch_size": 4,
        "learning_rate": 3e-4,
        "warmup_steps": 50,
        "max_train_steps": 16,
        "use_hf_dataset": True,
        "hf_max_samples": 2000,
        "pretrain_epochs": 2,
    },
    "quality": {
        "name": "Quality Focus",
        "description": "Longer training for better results",
        "model_size": "medium",
        "seq_len": 512,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "warmup_steps": 100,
        "max_train_steps": 32,
        "use_hf_dataset": True,
        "hf_max_samples": 10000,
        "pretrain_epochs": 5,
    },
    "production": {
        "name": "Production Training",
        "description": "Full training run for deployment",
        "model_size": "large",
        "seq_len": 1024,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "warmup_steps": 200,
        "max_train_steps": 64,
        "use_hf_dataset": True,
        "hf_max_samples": 50000,
        "pretrain_epochs": 10,
    },
}


# Model size specifications
MODEL_SPECS = {
    "tiny": {"params": "~10M", "d_model": 256, "n_layers": 4, "n_heads": 4},
    "small": {"params": "~125M", "d_model": 768, "n_layers": 12, "n_heads": 12},
    "medium": {"params": "~350M", "d_model": 1024, "n_layers": 24, "n_heads": 16},
    "large": {"params": "~760M", "d_model": 1280, "n_layers": 36, "n_heads": 20},
    "xl": {"params": "~1.3B", "d_model": 1600, "n_layers": 48, "n_heads": 25},
}


# Global log buffer for UI
_LOG_BUFFER: deque = deque(maxlen=500)
_LOG_LOCK = threading.Lock()


def _add_log(message: str, level: str = "INFO"):
    """Add a log message to the global buffer"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    with _LOG_LOCK:
        _LOG_BUFFER.append(f"[{timestamp}] [{level}] {message}")


def _get_logs(n: int = 50) -> str:
    """Get recent log messages"""
    with _LOG_LOCK:
        logs = list(_LOG_BUFFER)[-n:]
    return "\n".join(logs) if logs else "No logs yet..."


def _format_model_info(model_size: str, use_extensions: bool = False) -> str:
    """Format model information for display"""
    spec = MODEL_SPECS.get(model_size, MODEL_SPECS["tiny"])
    info = f"""**Model: {model_size.upper()}**
- Parameters: {spec['params']}
- Hidden Size: {spec['d_model']}
- Layers: {spec['n_layers']}
- Attention Heads: {spec['n_heads']}
- Extensions: {'Enabled' if use_extensions else 'Standard'}
"""
    return info


class LoopThread:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.running = False
        self.last_error: Optional[str] = None
        self.out_dir: Optional[str] = None
        self.active_config: Optional[Dict[str, Any]] = None
        self._start_lock = threading.Lock()
        self.start_time: Optional[float] = None
        self.current_phase: str = "idle"
        self.progress: float = 0.0

    def start(self, cfg: LoopConfig, teacher_model: str, training_params: TrainingParams):
        with self._start_lock:
            if self.running:
                _add_log("Already running, ignoring start request", "WARN")
                return False
            # Check if thread is still alive
            if self.thread is not None and self.thread.is_alive():
                _add_log("Previous thread still alive, ignoring start request", "WARN")
                return False
        self.stop_event.clear()
        self.last_error = None
        self.out_dir = cfg.out_dir
        self.start_time = time.time()
        self.current_phase = "initializing"
        self.progress = 0.0
        
        # Store active config for display
        self.active_config = {
            "loop": {
                "run_id": cfg.run_id,
                "max_cycles": cfg.max_cycles,
                "train_every": cfg.train_every_cycles,
                "eval_every": cfg.eval_every_cycles,
                "mint_stability": cfg.mint_stability_threshold,
                "mint_novelty": cfg.mint_min_novelty,
            },
            "training": {
                "model_size": training_params.model_size,
                "batch_size": training_params.batch_size,
                "seq_len": training_params.seq_len,
                "learning_rate": training_params.learning_rate,
                "warmup_steps": training_params.warmup_steps,
            },
            "model_features": {
                "extensions": training_params.use_extensions,
                "kuramoto": training_params.use_kuramoto,
                "smf": training_params.use_smf,
                "coherence": training_params.use_coherence,
            },
            "data_teacher": {
                "hf_dataset": training_params.use_hf_dataset,
                "hf_dataset_name": training_params.hf_dataset_name if training_params.use_hf_dataset else None,
                "lmstudio": training_params.use_lmstudio,
                "lmstudio_model": training_params.lmstudio_model if training_params.use_lmstudio else None,
                "rlhf": training_params.use_rlhf,
            }
        }

        def _run():
            try:
                _add_log(f"Starting training run: {cfg.run_id}")
                _add_log(f"Model: {training_params.model_size}, Device: {get_device()}")
                
                self.current_phase = "loading_model"
                observer = SentientObserverWrapper()
                _add_log("Observer initialized")
                
                # Create teacher generator based on settings
                if training_params.use_lmstudio:
                    _add_log(f"Connecting to LMStudio: {training_params.lmstudio_url}")
                    teacher_generator = create_teacher_generator(
                        use_lmstudio=True,
                        lmstudio_url=training_params.lmstudio_url,
                        lmstudio_model=training_params.lmstudio_model,
                    )
                    _add_log(f"LMStudio connected, model: {training_params.lmstudio_model}")
                else:
                    teacher_generator = make_teacher_generator(teacher_model)
                    _add_log("Using static teacher generator")
                
                teacher = LocalTeacher(generator=teacher_generator)
                
                _add_log(f"Creating student trainer ({training_params.model_size})...")
                student = make_student_trainer(
                    seq_len=training_params.seq_len,
                    batch_size=training_params.batch_size,
                    lr=training_params.learning_rate,
                    model_size=training_params.model_size,
                    warmup_steps=training_params.warmup_steps,
                    max_train_steps=training_params.max_train_steps,
                    use_extensions=training_params.use_extensions,
                    use_kuramoto=training_params.use_kuramoto,
                    use_smf=training_params.use_smf,
                    use_coherence=training_params.use_coherence,
                )
                _add_log("Student trainer ready")
                
                # Optionally pretrain on HuggingFace dataset before symbol learning
                if training_params.use_hf_dataset:
                    self.current_phase = "pretraining"
                    _add_log(f"Starting pretraining on {training_params.hf_dataset_name}...")
                    _run_pretraining(
                        training_params, 
                        student,
                        cfg.out_dir,
                        progress_callback=lambda p: setattr(self, 'progress', p * 0.5)  # 0-50% for pretrain
                    )
                    _add_log("Pretraining complete")
                
                # The loop itself does not check an external stop_event; we bound by max_cycles
                self.current_phase = "sentient_loop"
                _add_log(f"Starting sentient loop (max {cfg.max_cycles} cycles)...")
                run_sentient_llm_loop(cfg, observer, teacher, student)
                _add_log("Training complete!")
                self.current_phase = "complete"
                
            except Exception as e:
                import traceback
                self.last_error = f"{e}\n{traceback.format_exc()}"
                _add_log(f"Error: {e}", "ERROR")
                self.current_phase = "error"
            finally:
                self.running = False

        self.thread = threading.Thread(target=_run, daemon=True)
        self.running = True
        self.thread.start()
        return True

    def stop(self):
        # Currently loop stops via max_cycles; to hard stop, restart process or add stop flag in loop.
        self.stop_event.set()
        self.running = False

    def status(self) -> str:
        if self.running:
            return "Loop running"
        if self.last_error:
            return f"Loop stopped with error: {self.last_error}"
        return "Loop idle"


def load_metrics(out_dir: str, limit: int = 200) -> List[Dict[str, Any]]:
    path = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines[-limit:]


def load_shard_count(out_dir: str) -> int:
    path = os.path.join(out_dir, "shards.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def clear_metrics(out_dir: str) -> None:
    """Clear existing metrics and shards files for a fresh start"""
    for fname in ["metrics.jsonl", "shards.jsonl", "learning_state.json"]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            os.remove(path)


def chat_reply(chat_history, model_path: str, model_size: str, user_msg: str):
    tokenizer = create_default_tokenizer()
    size = model_size or "tiny"
    ckpt = model_path or get_checkpoint_path(size)
    model = _load_or_init_model(ckpt, tokenizer_vocab=tokenizer.vocab_size, size=size)
    device = get_device()
    model = model.to(device)
    infer = ResoLLMInference(model=model, tokenizer=tokenizer)

    # Build conversation from history (Gradio 6.x uses messages format by default)
    messages = []
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, dict):
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                # ChatMessage object
                messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_msg})

    prompt = build_chat_prompt(messages, add_assistant_prompt=True)
    result = infer.generate(prompt, max_length=200, temperature=0.8)
    reply = extract_assistant_response(result.text)
    
    # Return in Gradio 6.x Chatbot format: list of dicts with 'role' and 'content'
    if chat_history is None:
        chat_history = []
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": reply})
    return chat_history, ""


def build_interface():
    loop_thread = LoopThread()
    
    # Custom CSS for improved styling
    custom_css = """
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header-title {
        color: white;
        font-size: 2em;
        margin: 0;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1em;
    }
    .status-running {
        color: #22c55e;
        font-weight: bold;
    }
    .status-idle {
        color: #6b7280;
    }
    .status-error {
        color: #ef4444;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 8px;
        padding: 15px;
    }
    .log-panel {
        font-family: 'Fira Code', 'Monaco', monospace;
        font-size: 12px;
        background: #0d1117;
        color: #c9d1d9;
        border-radius: 6px;
    }
    """

    with gr.Blocks(title="Sentient LLM Training", css=custom_css, theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
    )) as demo:
        
        # Header Section
        gr.Markdown("""
        # 🧠 Sentient LLM Training Dashboard
        *Continual learning with semantic symbol emergence and neural-symbolic integration*
        """)
        
        # Status bar
        with gr.Row():
            with gr.Column(scale=3):
                status_display = gr.Markdown("**Status:** 🔵 Idle", elem_classes=["status-idle"])
            with gr.Column(scale=1):
                elapsed_time = gr.Textbox(label="Elapsed", value="00:00:00", interactive=False)
            with gr.Column(scale=1):
                phase_display = gr.Textbox(label="Phase", value="idle", interactive=False)

        # Main content area
        with gr.Row():
            # Left panel: Configuration
            with gr.Column(scale=2):
                # Quick actions
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="🎯 Quick Presets",
                        choices=["(custom)"] + list(PRESETS.keys()),
                        value="(custom)",
                        scale=2
                    )
                    apply_preset_btn = gr.Button("Apply", size="sm", scale=1)
                
                with gr.Tabs() as config_tabs:
                    # ===== BASIC TAB =====
                    with gr.Tab("📋 Basic", id="tab_basic"):
                        run_id = gr.Textbox(
                            label="Run ID", 
                            value="sentient_run",
                            info="Unique identifier for this training run"
                        )
                        out_dir = gr.Textbox(
                            label="Output Directory", 
                            value="runs/sentient",
                            info="Directory for checkpoints and logs"
                        )
                        max_cycles = gr.Number(
                            label="Max Cycles", 
                            value=1000, 
                            precision=0,
                            info="Maximum training cycles (0 = unlimited)"
                        )
                        
                    # ===== MODEL TAB =====
                    with gr.Tab("🏗️ Model", id="tab_model"):
                        model_size = gr.Dropdown(
                            label="Model Size",
                            choices=["tiny", "small", "medium", "large", "xl"],
                            value="tiny"
                        )
                        model_info_display = gr.Markdown(
                            value=_format_model_info("tiny"),
                            elem_classes=["metric-card"]
                        )
                        
                        gr.Markdown("**Architecture Features**")
                        use_extensions = gr.Checkbox(
                            label="Extended Features (Agency, PRSC, Temporal)",
                            value=False
                        )
                        with gr.Row():
                            use_kuramoto = gr.Checkbox(label="Kuramoto Dynamics", value=True)
                            use_smf = gr.Checkbox(label="SMF Memory", value=True)
                            use_coherence = gr.Checkbox(label="Coherence Gating", value=True)
                        
                    # ===== TRAINING TAB =====
                    with gr.Tab("⚡ Training", id="tab_training"):
                        with gr.Row():
                            learning_rate = gr.Number(
                                label="Learning Rate",
                                value=3e-4
                            )
                            batch_size = gr.Slider(
                                label="Batch Size",
                                minimum=1, maximum=32, value=4, step=1
                            )
                        with gr.Row():
                            seq_len = gr.Slider(
                                label="Sequence Length",
                                minimum=64, maximum=1024, value=256, step=64
                            )
                            warmup_steps = gr.Slider(
                                label="Warmup Steps",
                                minimum=0, maximum=500, value=20, step=10
                            )
                        max_train_steps = gr.Slider(
                            label="Max Train Steps/Batch",
                            minimum=1, maximum=128, value=16, step=1,
                            info="Gradient updates per shard batch"
                        )
                        
                    # ===== LOOP TAB =====
                    with gr.Tab("🔄 Loop", id="tab_loop"):
                        with gr.Row():
                            train_every = gr.Slider(
                                label="Train Every N Cycles",
                                minimum=1, maximum=20, value=1, step=1
                            )
                            eval_every = gr.Slider(
                                label="Eval Every N Cycles",
                                minimum=1, maximum=50, value=10, step=1
                            )
                        with gr.Row():
                            shards_per_new = gr.Slider(
                                label="Shards/New Symbol",
                                minimum=1, maximum=50, value=24, step=1
                            )
                            shards_per_update = gr.Slider(
                                label="Shards/Updated Symbol",
                                minimum=1, maximum=20, value=8, step=1
                            )
                        with gr.Row():
                            max_train_shards = gr.Slider(
                                label="Max Train Shards/Step",
                                minimum=16, maximum=512, value=128, step=16
                            )
                            replay_sample = gr.Slider(
                                label="Replay Sample/Step",
                                minimum=0, maximum=256, value=96, step=16
                            )
                        
                    # ===== MINTING TAB =====
                    with gr.Tab("💎 Minting", id="tab_minting"):
                        gr.Markdown("**Symbol Emergence Thresholds**")
                        mint_stability = gr.Slider(
                            label="Min Stability",
                            value=0.92, minimum=0.5, maximum=1.0, step=0.01,
                            info="Higher = more stable symbols only"
                        )
                        mint_novelty = gr.Slider(
                            label="Min Novelty",
                            value=0.20, minimum=0.0, maximum=1.0, step=0.01,
                            info="Higher = more unique concepts only"
                        )
                        teacher_model = gr.Textbox(
                            label="Teacher Checkpoint",
                            value="",
                            info="Optional: Path to teacher model"
                        )
                    
                    # ===== DATA/TEACHER TAB =====
                    with gr.Tab("📚 Data/Teacher", id="tab_data"):
                        gr.Markdown("### 📊 Pretraining Dataset")
                        use_hf_dataset = gr.Checkbox(
                            label="Enable HuggingFace Pretraining",
                            value=False
                        )
                        hf_dataset_name = gr.Textbox(
                            label="Dataset",
                            value="timdettmers/openassistant-guanaco"
                        )
                        with gr.Row():
                            hf_max_samples = gr.Slider(
                                label="Max Samples",
                                minimum=100, maximum=50000, value=5000, step=100
                            )
                            pretrain_epochs = gr.Slider(
                                label="Epochs",
                                minimum=1, maximum=20, value=3, step=1
                            )
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🤖 LMStudio Teacher")
                        use_lmstudio = gr.Checkbox(
                            label="Use LMStudio as Teacher",
                            value=False
                        )
                        with gr.Row():
                            lmstudio_url = gr.Textbox(
                                label="URL",
                                value="http://localhost:1234",
                                scale=2
                            )
                            lmstudio_model = gr.Textbox(
                                label="Model",
                                value="llama-3.1-8b-lexi-uncensored-v2",
                                scale=2
                            )
                        with gr.Row():
                            lmstudio_test_btn = gr.Button("🔌 Test Connection", size="sm")
                            lmstudio_status = gr.Markdown("Not tested")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🎯 RLHF/DPO")
                        use_rlhf = gr.Checkbox(
                            label="Enable RLHF/DPO Training",
                            value=False
                        )
                        with gr.Row():
                            rlhf_batch_size = gr.Slider(
                                label="Preference Batch Size",
                                minimum=4, maximum=64, value=16, step=4
                            )
                            dpo_beta = gr.Slider(
                                label="DPO Beta",
                                minimum=0.01, maximum=1.0, value=0.1, step=0.01
                            )
                    
                    # ===== IMPORT/EXPORT TAB =====
                    with gr.Tab("💾 Config", id="tab_config"):
                        gr.Markdown("### Export/Import Configuration")
                        config_json = gr.Code(
                            label="Configuration JSON",
                            language="json",
                            lines=10
                        )
                        with gr.Row():
                            export_btn = gr.Button("📤 Export Config")
                            import_btn = gr.Button("📥 Import Config")
                        config_status = gr.Markdown("")

                # Control buttons
                gr.Markdown("---")
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Training", variant="primary", size="lg", scale=2)
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg", scale=1)
                    clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1)

            # Right panel: Monitoring
            with gr.Column(scale=3):
                with gr.Tabs():
                    # ===== METRICS TAB =====
                    with gr.Tab("📈 Metrics"):
                        # Stats row
                        with gr.Row():
                            cycle_count = gr.Number(label="Cycle", value=0, interactive=False)
                            shard_count = gr.Number(label="Shards", value=0, interactive=False)
                            current_loss = gr.Number(label="Loss", value=0.0, interactive=False)
                            tokens_per_sec = gr.Number(label="Tok/s", value=0, interactive=False)
                        
                        # Loss chart
                        loss_plot = gr.LinePlot(
                            x="cycle", y="value", color="kind",
                            height=250,
                            title="Training Loss"
                        )
                        
                        # Gradient norm chart
                        with gr.Accordion("Gradient Norm", open=False):
                            grad_plot = gr.LinePlot(
                                x="cycle", y="value", color="kind",
                                height=200,
                                title="Gradient Norm"
                            )
                    
                    # ===== LOGS TAB =====
                    with gr.Tab("📝 Logs"):
                        log_display = gr.Textbox(
                            label="Training Logs",
                            value="Waiting for training to start...",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            elem_classes=["log-panel"]
                        )
                        with gr.Row():
                            refresh_logs_btn = gr.Button("🔄 Refresh", size="sm")
                            clear_logs_btn = gr.Button("🗑️ Clear Logs", size="sm")
                    
                    # ===== CONFIG DISPLAY TAB =====
                    with gr.Tab("⚙️ Active Config"):
                        model_info = gr.JSON(label="Current Configuration")

        # Chat section
        gr.Markdown("---")
        gr.Markdown("## 💬 Chat with Trained Model")
        with gr.Row():
            with gr.Column(scale=3):
                chatbox = gr.Chatbot(label="Conversation", height=300)
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type a message to test the model...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Column(scale=1):
                gr.Markdown("**Chat Settings**")
                chat_temp = gr.Slider(
                    label="Temperature",
                    minimum=0.1, maximum=2.0, value=0.8, step=0.1
                )
                chat_max_len = gr.Slider(
                    label="Max Length",
                    minimum=50, maximum=500, value=200, step=10
                )
                clear_chat_btn = gr.Button("Clear Chat")

        # ===== EVENT HANDLERS =====
        
        # Update model info when size changes
        def update_model_info(size, extensions):
            return _format_model_info(size, extensions)
        
        model_size.change(
            update_model_info, 
            inputs=[model_size, use_extensions], 
            outputs=[model_info_display]
        )
        use_extensions.change(
            update_model_info, 
            inputs=[model_size, use_extensions], 
            outputs=[model_info_display]
        )
        
        # Apply preset
        def apply_preset(preset_name):
            if preset_name == "(custom)" or preset_name not in PRESETS:
                return [gr.update()] * 10  # No changes
            
            preset = PRESETS[preset_name]
            return [
                gr.update(value=preset.get("model_size", "tiny")),
                gr.update(value=preset.get("seq_len", 256)),
                gr.update(value=preset.get("batch_size", 4)),
                gr.update(value=preset.get("learning_rate", 3e-4)),
                gr.update(value=preset.get("warmup_steps", 20)),
                gr.update(value=preset.get("max_train_steps", 16)),
                gr.update(value=preset.get("use_hf_dataset", False)),
                gr.update(value=preset.get("hf_max_samples", 5000)),
                gr.update(value=preset.get("pretrain_epochs", 3)),
                f"Applied preset: **{preset.get('name', preset_name)}**\n\n{preset.get('description', '')}"
            ]
        
        apply_preset_btn.click(
            apply_preset,
            inputs=[preset_dropdown],
            outputs=[
                model_size, seq_len, batch_size, learning_rate, warmup_steps, 
                max_train_steps, use_hf_dataset, hf_max_samples, pretrain_epochs,
                config_status
            ]
        )
        
        # Export config
        def export_config(
            model_size_val, seq_len_val, batch_size_val, learning_rate_val, warmup_steps_val,
            max_train_steps_val, use_hf_dataset_val, hf_dataset_name_val, hf_max_samples_val,
            pretrain_epochs_val, use_lmstudio_val, lmstudio_url_val, lmstudio_model_val
        ):
            config = {
                "model_size": model_size_val,
                "seq_len": seq_len_val,
                "batch_size": batch_size_val,
                "learning_rate": learning_rate_val,
                "warmup_steps": warmup_steps_val,
                "max_train_steps": max_train_steps_val,
                "use_hf_dataset": use_hf_dataset_val,
                "hf_dataset_name": hf_dataset_name_val,
                "hf_max_samples": hf_max_samples_val,
                "pretrain_epochs": pretrain_epochs_val,
                "use_lmstudio": use_lmstudio_val,
                "lmstudio_url": lmstudio_url_val,
                "lmstudio_model": lmstudio_model_val,
            }
            return json.dumps(config, indent=2), "✅ Configuration exported"
        
        export_btn.click(
            export_config,
            inputs=[
                model_size, seq_len, batch_size, learning_rate, warmup_steps,
                max_train_steps, use_hf_dataset, hf_dataset_name, hf_max_samples,
                pretrain_epochs, use_lmstudio, lmstudio_url, lmstudio_model
            ],
            outputs=[config_json, config_status]
        )
        
        # Import config
        def import_config(config_str):
            try:
                config = json.loads(config_str)
                return [
                    gr.update(value=config.get("model_size", "tiny")),
                    gr.update(value=config.get("seq_len", 256)),
                    gr.update(value=config.get("batch_size", 4)),
                    gr.update(value=config.get("learning_rate", 3e-4)),
                    gr.update(value=config.get("warmup_steps", 20)),
                    gr.update(value=config.get("max_train_steps", 16)),
                    gr.update(value=config.get("use_hf_dataset", False)),
                    gr.update(value=config.get("hf_dataset_name", "timdettmers/openassistant-guanaco")),
                    gr.update(value=config.get("hf_max_samples", 5000)),
                    gr.update(value=config.get("pretrain_epochs", 3)),
                    gr.update(value=config.get("use_lmstudio", False)),
                    gr.update(value=config.get("lmstudio_url", "http://localhost:1234")),
                    gr.update(value=config.get("lmstudio_model", "llama-3.1-8b-lexi-uncensored-v2")),
                    "✅ Configuration imported"
                ]
            except Exception as e:
                return [gr.update()] * 13 + [f"❌ Import failed: {e}"]
        
        import_btn.click(
            import_config,
            inputs=[config_json],
            outputs=[
                model_size, seq_len, batch_size, learning_rate, warmup_steps,
                max_train_steps, use_hf_dataset, hf_dataset_name, hf_max_samples,
                pretrain_epochs, use_lmstudio, lmstudio_url, lmstudio_model,
                config_status
            ]
        )

        def do_start(
            run_id_val, out_dir_val, max_cycles_val,
            model_size_val, use_extensions_val, use_kuramoto_val, use_smf_val, use_coherence_val,
            learning_rate_val, batch_size_val, seq_len_val, warmup_steps_val, max_train_steps_val,
            train_every_val, eval_every_val, shards_per_new_val, shards_per_update_val, max_train_shards_val, replay_sample_val,
            mint_stability_val, mint_novelty_val, teacher_model_val,
            # Data/Teacher inputs
            use_hf_dataset_val, hf_dataset_name_val, hf_max_samples_val, pretrain_epochs_val,
            use_lmstudio_val, lmstudio_url_val, lmstudio_model_val,
            use_rlhf_val, rlhf_batch_size_val, dpo_beta_val
        ):
            # Build loop config with log callback for UI
            cfg = LoopConfig(
                run_id=run_id_val,
                out_dir=out_dir_val,
                max_cycles=int(max_cycles_val),
                mint_stability_threshold=float(mint_stability_val),
                mint_min_novelty=float(mint_novelty_val),
                train_every_cycles=int(train_every_val),
                eval_every_cycles=int(eval_every_val),
                shards_per_new_symbol=int(shards_per_new_val),
                shards_per_updated_symbol=int(shards_per_update_val),
                max_train_shards_per_step=int(max_train_shards_val),
                replay_sample_per_step=int(replay_sample_val),
                log_fn=_add_log,  # Pipe loop logs to UI
            )
            
            # Build training params with Data/Teacher options
            training_params = TrainingParams(
                seq_len=int(seq_len_val),
                batch_size=int(batch_size_val),
                learning_rate=float(learning_rate_val),
                model_size=model_size_val,
                warmup_steps=int(warmup_steps_val),
                max_train_steps=int(max_train_steps_val),
                use_extensions=use_extensions_val,
                use_kuramoto=use_kuramoto_val,
                use_smf=use_smf_val,
                use_coherence=use_coherence_val,
                # Data/Teacher options
                use_hf_dataset=use_hf_dataset_val,
                hf_dataset_name=hf_dataset_name_val,
                hf_max_samples=int(hf_max_samples_val),
                pretrain_epochs=int(pretrain_epochs_val),
                use_lmstudio=use_lmstudio_val,
                lmstudio_url=lmstudio_url_val,
                lmstudio_model=lmstudio_model_val,
                use_rlhf=use_rlhf_val,
                rlhf_batch_size=int(rlhf_batch_size_val),
                dpo_beta=float(dpo_beta_val),
            )
            
            _add_log(f"Starting training: {run_id_val}")
            ok = loop_thread.start(cfg, teacher_model_val, training_params)
            config_display = loop_thread.active_config if ok else {}
            status_text = "**Status:** 🟢 Running" if ok else "**Status:** 🔴 Failed to start"
            return status_text, gr.update(), config_display

        start_btn.click(
            do_start,
            inputs=[
                run_id, out_dir, max_cycles,
                model_size, use_extensions, use_kuramoto, use_smf, use_coherence,
                learning_rate, batch_size, seq_len, warmup_steps, max_train_steps,
                train_every, eval_every, shards_per_new, shards_per_update, max_train_shards, replay_sample,
                mint_stability, mint_novelty, teacher_model,
                # Data/Teacher inputs
                use_hf_dataset, hf_dataset_name, hf_max_samples, pretrain_epochs,
                use_lmstudio, lmstudio_url, lmstudio_model,
                use_rlhf, rlhf_batch_size, dpo_beta
            ],
            outputs=[status_display, loss_plot, model_info],
        )

        def do_stop():
            loop_thread.stop()
            _add_log("Training stopped by user")
            return "**Status:** 🟡 Stopping..."

        stop_btn.click(do_stop, outputs=status_display)
        
        def do_clear(out_dir_val):
            clear_metrics(out_dir_val)
            _add_log("Metrics cleared")
            empty_df = pd.DataFrame(columns=["cycle", "value", "kind"])
            return "**Status:** 🔵 Metrics cleared", empty_df, 0, 0.0
        
        clear_btn.click(do_clear, inputs=[out_dir], outputs=[status_display, loss_plot, shard_count, current_loss])
        
        # Log panel handlers
        def refresh_logs():
            return _get_logs(50)
        
        def clear_logs():
            with _LOG_LOCK:
                _LOG_BUFFER.clear()
            return "Logs cleared"
        
        refresh_logs_btn.click(refresh_logs, outputs=[log_display])
        clear_logs_btn.click(clear_logs, outputs=[log_display])
        
        # Clear chat
        def do_clear_chat():
            return [], ""
        
        clear_chat_btn.click(do_clear_chat, outputs=[chatbox, chat_input])
        
        def do_test_lmstudio(url, model):
            """Test LMStudio connection with diagnostics using native API"""
            try:
                import requests
                
                # Normalize URL: remove /v1 suffix if present
                base_url = url
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
                if base_url.endswith("/"):
                    base_url = base_url[:-1]
                
                # First check if server is reachable using native API
                try:
                    models_response = requests.get(f"{base_url}/api/v1/models", timeout=5)
                    if models_response.status_code != 200:
                        # Try legacy endpoint
                        models_response = requests.get(f"{base_url}/v1/models", timeout=5)
                        if models_response.status_code != 200:
                            return f"❌ Server reachable but models endpoint returned {models_response.status_code}"
                    
                    models_data = models_response.json()
                    available = models_data.get("data", [])
                    model_ids = [m.get("id", "") for m in available]
                    models_str = ", ".join(model_ids[:3]) if model_ids else "none loaded"
                    
                except requests.exceptions.ConnectionError:
                    return f"❌ Cannot connect to {base_url}. Is LMStudio running?"
                except requests.exceptions.Timeout:
                    return f"❌ Connection to {base_url} timed out"
                
                # Try a simple generation using native /api/v1/chat endpoint
                from apps.reso_llm.data_sources import LMStudioConfig, LMStudioTeacher
                config = LMStudioConfig(base_url=base_url, model=model, timeout=30)
                teacher = LMStudioTeacher(config)
                response = teacher.generate(
                    "Reply with exactly: OK",
                    max_tokens=10,
                    temperature=0.1,
                    retry_count=0  # Don't retry for test
                )
                _add_log(f"LMStudio connected: {model}")
                return f"✅ Connected! Models: {models_str}. Response: {response[:30]}"
            except RuntimeError as e:
                error_str = str(e)
                if "crashed" in error_str.lower():
                    return f"⚠️ Model crashed. Try reloading it in LMStudio, or use a smaller model."
                if "timeout" in error_str.lower():
                    return f"⚠️ Request timed out. Model may be loading or too slow."
                return f"❌ API error: {error_str[:100]}"
            except Exception as e:
                return f"❌ Connection failed: {type(e).__name__}: {e}"
        
        lmstudio_test_btn.click(do_test_lmstudio, inputs=[lmstudio_url, lmstudio_model], outputs=[lmstudio_status])

        def do_refresh(out_dir_val):
            """Refresh all metrics and status displays"""
            metrics = load_metrics(out_dir_val)
            loss_rows = []
            grad_rows = []
            current_cycle = 0
            last_loss = 0.0
            
            for m in metrics:
                metrics_dict = m.get("metrics", {}) or {}
                kind = m.get("kind", "train")
                cycle = m.get("cycle", 0)
                current_cycle = max(current_cycle, cycle)
                
                # Extract loss
                loss_val = None
                if kind in ("train", "pretrain"):
                    loss_val = metrics_dict.get("loss_mean") or metrics_dict.get("loss")
                elif kind == "eval":
                    loss_val = metrics_dict.get("eval_loss")
                
                if loss_val is None:
                    for k, v in metrics_dict.items():
                        if isinstance(v, (int, float)) and "loss" in k.lower():
                            loss_val = v
                            break
                
                if loss_val is not None and isinstance(loss_val, (int, float)):
                    loss_rows.append({"cycle": cycle, "value": float(loss_val), "kind": kind})
                    last_loss = float(loss_val)
                
                # Extract gradient norm
                grad_val = metrics_dict.get("grad_norm_mean")
                if grad_val is not None and isinstance(grad_val, (int, float)):
                    grad_rows.append({"cycle": cycle, "value": float(grad_val), "kind": kind})
            
            loss_df = pd.DataFrame(loss_rows) if loss_rows else pd.DataFrame(columns=["cycle", "value", "kind"])
            grad_df = pd.DataFrame(grad_rows) if grad_rows else pd.DataFrame(columns=["cycle", "value", "kind"])
            
            # Update status based on loop state
            if loop_thread.running:
                elapsed = time.time() - (loop_thread.start_time or time.time())
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                status = "**Status:** 🟢 Running"
                phase = loop_thread.current_phase
            else:
                elapsed_str = "00:00:00"
                if loop_thread.last_error:
                    status = "**Status:** 🔴 Error"
                    phase = "error"
                elif loop_thread.current_phase == "complete":
                    status = "**Status:** ✅ Complete"
                    phase = "complete"
                else:
                    status = "**Status:** 🔵 Idle"
                    phase = "idle"
            
            # Get logs
            logs = _get_logs(50)
            
            return (
                loss_df, 
                grad_df,
                load_shard_count(out_dir_val), 
                current_cycle,
                last_loss,
                status,
                elapsed_str,
                phase,
                logs
            )

        # Auto-refresh timer
        refresh_timer = gr.Timer(1.0, active=True)
        refresh_timer.tick(
            fn=do_refresh, 
            inputs=[out_dir], 
            outputs=[
                loss_plot, 
                grad_plot,
                shard_count, 
                cycle_count,
                current_loss,
                status_display,
                elapsed_time,
                phase_display,
                log_display
            ]
        )

        # Chat handler with temperature and max length
        def chat_with_settings(chat_history, model_path, model_sz, user_msg, temp, max_len):
            return chat_reply(chat_history, model_path, model_sz, user_msg)
        
        send_btn.click(
            chat_with_settings, 
            inputs=[chatbox, teacher_model, model_size, chat_input, chat_temp, chat_max_len], 
            outputs=[chatbox, chat_input]
        )
        
        # Submit on Enter in chat input
        chat_input.submit(
            chat_with_settings,
            inputs=[chatbox, teacher_model, model_size, chat_input, chat_temp, chat_max_len],
            outputs=[chatbox, chat_input]
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_interface()
    demo.queue().launch(server_name=args.listen, server_port=args.port)


if __name__ == "__main__":
    main()
