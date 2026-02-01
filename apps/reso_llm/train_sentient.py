from __future__ import annotations
import argparse
from typing import Dict, Any, List

import os
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch

from tinyaleph.training.sentient_llm import LoopConfig, run_sentient_llm_loop
from tinyaleph.training.sentient_llm.teacher_local import LocalTeacher
from tinyaleph.training.sentient_llm.adapters.observer_adapter import CallableObserverAdapter
from tinyaleph.training.sentient_llm.adapters.reso_llm_student import CallableStudentTrainer
from tinyaleph.training.sentient_llm.schema import TrainingShard, ObserverEpisode, ObserverSymbol, SymbolId, now_ts
from tinyaleph.training.sentient_llm.observer_runtime import SentientObserverWrapper

from apps.reso_llm.inference import ResoLLMInference
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.config import ResoLLMConfig
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.train import ResoLLMTrainer, get_checkpoint_path, CHECKPOINT_DIR
from apps.reso_llm.dataset import validate_format
from apps.resoformer.torch_backend import get_device
from tinyaleph.ml.resoformer import Tensor


# ---- YOU WIRE THESE TO YOUR REAL OBJECTS ----

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
    """
    Use local Reso-LLM inference as the teacher generator.
    Returns a callable: generator(prompt: str) -> str
    """

    tokenizer = create_default_tokenizer()
    ckpt = model_path or get_checkpoint_path("tiny")
    model = _load_or_init_model(ckpt, tokenizer_vocab=tokenizer.vocab_size)
    device = get_device()
    model = model.to(device)
    infer = ResoLLMInference(
        model=model,
        tokenizer=tokenizer,
        temperature=0.8,
        max_length=160,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
    )

    def gen(prompt: str) -> str:
        res = infer.generate(prompt, max_length=160, temperature=0.8)
        return res.text

    return gen


def make_observer():
    """
    Compose the real sentient observer primitives (BoundaryLayer + AgencyLayer + SMF)
    via SentientObserverWrapper to satisfy the protocol.
    """

    wrapper = SentientObserverWrapper()
    return wrapper


@dataclass
class _ShardDataset:
    tokens: List[int]
    tokenizer_vocab: int
    seq_len: int
    batch_size: int

    def __post_init__(self):
        # Ensure enough data to sample
        min_len = self.seq_len * self.batch_size + 2
        if len(self.tokens) < min_len and self.tokens:
            reps = math.ceil(min_len / len(self.tokens))
            self.tokens = (self.tokens * reps)[: min_len + 1]

    def __len__(self) -> int:
        n_sequences = max(1, (len(self.tokens) - 1) // self.seq_len)
        return max(1, n_sequences // self.batch_size)

    def get_batch(self) -> (Tensor, Tensor):
        max_idx = len(self.tokens) - self.seq_len - 1
        batch_x: List[int] = []
        batch_y: List[int] = []
        for _ in range(self.batch_size):
            idx = 0 if max_idx <= 0 else torch.randint(0, max_idx, ()).item()
            x_seq = self.tokens[idx : idx + self.seq_len]
            y_seq = self.tokens[idx + 1 : idx + self.seq_len + 1]
            batch_x.extend(x_seq)
            batch_y.extend(y_seq)
        return Tensor(batch_x, (self.batch_size, self.seq_len)), Tensor(batch_y, (self.batch_size, self.seq_len))


def _format_shard(shard: TrainingShard) -> str:
    chat = (
        f"<|user|>\n{shard.input_text}\n<|endofuser|>\n"
        f"<|assistant|>\n{shard.target_text}\n<|endofassistant|>"
    )
    ok, _ = validate_format(chat)
    if not ok:
        # ensure template markers are present
        chat = (
            "<|user|>\n" + shard.input_text.strip() + "\n<|endofuser|>\n"
            "<|assistant|>\n" + shard.target_text.strip() + "\n<|endofassistant|>"
        )
    return chat


def make_student_trainer(seq_len: int = 256, batch_size: int = 4, lr: float = 3e-4):
    """
    Adapts TrainingShard batches into the existing Reso-LLM trainer.
    """

    tokenizer = create_default_tokenizer()
    ckpt = get_checkpoint_path("tiny")
    model = _load_or_init_model(ckpt, tokenizer_vocab=tokenizer.vocab_size)
    device = get_device()
    model = model.to(device)

    # Lightweight trainer; dataset will be swapped per call
    dummy_ds = _ShardDataset(tokens=list(range(seq_len * batch_size + 2)), tokenizer_vocab=tokenizer.vocab_size, seq_len=seq_len, batch_size=batch_size)
    trainer = ResoLLMTrainer(
        model=model,
        dataset=dummy_ds,  # will be replaced in train_fn
        learning_rate=lr,
        warmup_steps=20,
        max_steps=10_000,
    )

    last_dataset: Optional[_ShardDataset] = None

    def train_fn(shards: List[TrainingShard]) -> Dict[str, Any]:
        nonlocal last_dataset
        if not shards:
            return {"trained_shards": 0, "steps": 0}

        text_tokens: List[int] = []
        for sh in shards:
            chat = _format_shard(sh)
            text_tokens.extend(tokenizer.encode(chat))

        ds = _ShardDataset(tokens=text_tokens, tokenizer_vocab=tokenizer.vocab_size, seq_len=seq_len, batch_size=batch_size)
        trainer.dataset = ds
        last_dataset = ds

        steps = min( max(1, len(ds)), 16)
        losses: List[float] = []
        grads: List[float] = []
        for _ in range(steps):
            loss, grad = trainer.train_step()
            losses.append(loss)
            grads.append(grad)

        # Save incremental checkpoint
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


# -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="sentient_run")
    ap.add_argument("--out-dir", default="runs/sentient")
    ap.add_argument("--max-cycles", type=int, default=10_000)

    ap.add_argument("--teacher-model", default="")
    ap.add_argument("--mint-stability", type=float, default=0.92)
    ap.add_argument("--mint-novelty", type=float, default=0.20)

    args = ap.parse_args()

    cfg = LoopConfig(
        run_id=args.run_id,
        out_dir=args.out_dir,
        max_cycles=args.max_cycles,
        mint_stability_threshold=args.mint_stability,
        mint_min_novelty=args.mint_novelty,
    )

    observer = make_observer()
    teacher = LocalTeacher(generator=make_teacher_generator(args.teacher_model))
    student = make_student_trainer()

    run_sentient_llm_loop(cfg, observer, teacher, student)


if __name__ == "__main__":
    main()
