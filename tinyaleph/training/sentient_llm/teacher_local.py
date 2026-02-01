from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
import json

from .schema import ObserverEpisode, ObserverSymbol, TrainingShard, now_ts


@dataclass
class LocalTeacher:
    """
    generator(prompt: str) -> str
    You provide generator: MLX, llama.cpp, reso_llm inference, etc.
    """
    generator: Callable[[str], str]

    def generate_shards(
        self,
        symbols: List[ObserverSymbol],
        episodes: List[ObserverEpisode],
        shards_per_symbol: int,
    ) -> List[TrainingShard]:
        out: List[TrainingShard] = []
        print(f"[Teacher] Generating shards for {len(symbols)} symbols, {shards_per_symbol} each", flush=True)
        for sym in symbols:
            prompt = self._prompt_for_symbol(sym, episodes)
            try:
                raw = self.generator(prompt)
                print(f"[Teacher] Got response for {sym.id}: {len(raw)} chars", flush=True)
            except Exception as e:
                print(f"[Teacher] ERROR generating for {sym.id}: {e}", flush=True)
                continue
            parsed = self._parse(sym, raw, shards_per_symbol)
            print(f"[Teacher] Parsed {len(parsed)} shards for {sym.id}", flush=True)
            out.extend(parsed)
        print(f"[Teacher] Total shards generated: {len(out)}", flush=True)
        return out

    def _prompt_for_symbol(self, sym: ObserverSymbol, episodes: List[ObserverEpisode]) -> str:
        return (
            "Generate training shards for a student model learning a new concept.\n"
            f"Symbol token: ⟦{sym.id}⟧\n"
            f"stability={sym.stability:.4f} novelty={sym.novelty:.4f}\n"
            f"prime_basis={sym.prime_basis}\n"
            "Return JSON array. Each item: {kind,input_text,target_text}.\n"
            "kind ∈ [label,definition,example,qa].\n"
            "Use ⟦SYM:xxxxxx⟧ tokens inside text where relevant.\n"
        )

    def _parse(self, sym: ObserverSymbol, raw: str, limit: int) -> List[TrainingShard]:
        shards: List[TrainingShard] = []
        
        # Try to extract JSON from the response (may be wrapped in text)
        json_str = raw
        if "```json" in raw:
            # Extract JSON from markdown code block
            start = raw.find("```json") + 7
            end = raw.find("```", start)
            if end > start:
                json_str = raw[start:end].strip()
        elif "[" in raw and "]" in raw:
            # Try to extract array
            start = raw.find("[")
            end = raw.rfind("]") + 1
            json_str = raw[start:end]
        
        try:
            arr = json.loads(json_str)
            if isinstance(arr, list):
                for i, item in enumerate(arr[:limit]):
                    kind = item.get("kind", "example")
                    # Validate kind is in allowed set
                    if kind not in ("label", "definition", "example", "qa", "tool", "contrastive"):
                        kind = "example"
                    shards.append(
                        TrainingShard(
                            shard_id=f"{sym.id}:{i}",
                            created_at=now_ts(),
                            symbol_ids=[sym.id],
                            kind=kind,
                            input_text=item.get("input_text", ""),
                            target_text=item.get("target_text", ""),
                            symbol_embedding=sym.embedding,
                        )
                    )
                print(f"[Teacher] Parsed {len(shards)} JSON shards", flush=True)
                return shards
            else:
                print(f"[Teacher] JSON parsed but not a list: {type(arr)}", flush=True)
        except json.JSONDecodeError as e:
            print(f"[Teacher] JSON parse error: {e}", flush=True)
            print(f"[Teacher] Raw response preview: {raw[:200]}...", flush=True)
        except Exception as e:
            print(f"[Teacher] Parse exception: {e}", flush=True)

        # fallback
        shards.append(
            TrainingShard(
                shard_id=f"{sym.id}:definition",
                created_at=now_ts(),
                symbol_ids=[sym.id],
                kind="definition",
                input_text=f"Define ⟦{sym.id}⟧.",
                target_text=raw.strip(),
                symbol_embedding=sym.embedding,
            )
        )
        return shards

