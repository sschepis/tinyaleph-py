from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import os
import json

from .interfaces import LoopConfig, SentientObserver, TeacherModel, StudentTrainer
from .schema import ObserverEpisode, ObserverSymbol, SymbolId, now_ts
from .replay import ReplayBuffer, append_jsonl


@dataclass
class LoopState:
    cycle: int = 0
    known_symbols: Dict[SymbolId, float] = None  # id -> last_seen_stability

    @classmethod
    def create(cls) -> "LoopState":
        return cls(cycle=0, known_symbols={})


def _should_mint(sym: ObserverSymbol, cfg: LoopConfig) -> bool:
    return (sym.stability >= cfg.mint_stability_threshold) and (sym.novelty >= cfg.mint_min_novelty)


def _append_metrics(out_dir: str, cycle: int, kind: str, metrics: Dict) -> None:
    path = os.path.join(out_dir, "metrics.jsonl")
    rec = {"t": now_ts(), "cycle": cycle, "kind": kind, "metrics": metrics}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_sentient_llm_loop(
    cfg: LoopConfig,
    observer: SentientObserver,
    teacher: TeacherModel,
    student: StudentTrainer,
) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    replay = ReplayBuffer.create(cfg.replay_max_shards)
    state = LoopState.create()

    episodes_recent: List[ObserverEpisode] = []
    
    # Helper to log to both console and callback
    def _log(msg: str, level: str = "INFO"):
        print(f"[Loop] {msg}", flush=True)
        if cfg.log_fn:
            try:
                cfg.log_fn(msg, level)
            except Exception:
                pass
    
    _log(f"Starting sentient loop with max_cycles={cfg.max_cycles}")
    _log(f"Mint thresholds: stability >= {cfg.mint_stability_threshold}, novelty >= {cfg.mint_min_novelty}")

    for cycle in range(cfg.max_cycles):
        state.cycle = cycle

        ep = observer.step()
        episodes_recent.append(ep)
        if len(episodes_recent) > 64:
            episodes_recent.pop(0)

        symbols = observer.get_symbols(None)

        new_syms: List[ObserverSymbol] = []
        upd_syms: List[ObserverSymbol] = []
        
        # Log symbol status periodically
        if cycle % 10 == 0 and symbols:
            sample_sym = symbols[0]
            _log(f"Cycle {cycle}: {len(symbols)} symbols, sample: id={sample_sym.id}, stability={sample_sym.stability:.3f}, novelty={sample_sym.novelty:.3f}")

        for s in symbols:
            prev = state.known_symbols.get(s.id)
            if prev is None:
                if _should_mint(s, cfg):
                    new_syms.append(s)
            else:
                # "updated" means stability shifted a bit; adjust if you prefer novelty-based
                if abs(s.stability - prev) > 0.02 and s.stability >= cfg.mint_stability_threshold:
                    upd_syms.append(s)

        shards = []
        # Limit symbols processed per cycle to avoid long blocking
        MAX_SYMBOLS_PER_CYCLE = 5  # Process at most 5 symbols per cycle
        processed_syms: List[ObserverSymbol] = []  # Track which symbols we actually processed
        
        if new_syms:
            batch_syms = new_syms[:MAX_SYMBOLS_PER_CYCLE]
            remaining = len(new_syms) - len(batch_syms)
            _log(f"Cycle {cycle}: Generating shards for {len(batch_syms)} NEW symbols" + (f" ({remaining} queued)" if remaining > 0 else ""))
            shards += teacher.generate_shards(batch_syms, episodes_recent, cfg.shards_per_new_symbol)
            processed_syms.extend(batch_syms)  # Only mark batch as processed
            
        if upd_syms:
            batch_syms = upd_syms[:MAX_SYMBOLS_PER_CYCLE]
            remaining = len(upd_syms) - len(batch_syms)
            _log(f"Cycle {cycle}: Generating shards for {len(batch_syms)} UPDATED symbols" + (f" ({remaining} queued)" if remaining > 0 else ""))
            shards += teacher.generate_shards(batch_syms, episodes_recent, cfg.shards_per_updated_symbol)
            processed_syms.extend(batch_syms)  # Only mark batch as processed

        if shards:
            _log(f"Cycle {cycle}: Generated {len(shards)} shards total")
            append_jsonl(os.path.join(cfg.out_dir, "shards.jsonl"), shards)
            replay.add_many(shards)

            # Only mark PROCESSED symbols as known
            for s in processed_syms:
                state.known_symbols[s.id] = s.stability

            # minimal lexicon update: use label shards
            lex_updates = {}
            for sh in shards:
                if sh.kind == "label" and sh.symbol_ids:
                    lex_updates[sh.symbol_ids[0]] = {"label": sh.target_text.strip()}
            if lex_updates:
                observer.upsert_lexicon(lex_updates)

        if cycle % cfg.train_every_cycles == 0:
            batch: List = []
            batch.extend(shards[: cfg.max_train_shards_per_step])
            need = max(0, cfg.max_train_shards_per_step - len(batch))
            if need > 0:
                replay_shards = replay.sample(min(cfg.replay_sample_per_step, need))
                batch.extend(replay_shards)
                if cycle % 10 == 0:
                    _log(f"Cycle {cycle}: Batch has {len(shards)} new + {len(replay_shards)} replay = {len(batch)} shards")

            if batch:
                metrics = student.train_on_shards(batch)
                _append_metrics(cfg.out_dir, cycle, "train", metrics)
                if cycle % 10 == 0:
                    loss = metrics.get("loss_mean", 0)
                    _log(f"Cycle {cycle}: Training complete, loss={loss:.4f}")
            elif cycle % 50 == 0:
                _log(f"Cycle {cycle}: No batch to train (replay empty, no new shards)", "WARN")

        if cycle % cfg.eval_every_cycles == 0:
            em = student.evaluate()
            _append_metrics(cfg.out_dir, cycle, "eval", em)
            eval_loss = em.get("eval_loss", "N/A")
            _log(f"Cycle {cycle}: Eval loss={eval_loss}")
    
    # Save final state
    observer.save_state(os.path.join(cfg.out_dir, "learning_state.json"))

