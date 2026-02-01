"""Dataset generation and replay utilities for key-derivation evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os
import random

from apps.ecdsa_ai.curve import CurvePoint, generate_keypair
from apps.ecdsa_ai.hnp import WeakNonceSignature, generate_weak_nonce_signatures, DEFAULT_BITS


@dataclass
class ReplayDataset:
    seed: int
    ecdsa_samples: List[Dict[str, Any]]
    hnp: Dict[str, Any]

    @property
    def ecdsa_pairs(self) -> List[tuple[int, CurvePoint]]:
        pairs = []
        for item in self.ecdsa_samples:
            priv = int(item["private_key"], 16)
            pub = item["public_key"]
            pairs.append((priv, CurvePoint(int(pub["x"], 16), int(pub["y"], 16))))
        return pairs

    @property
    def hnp_bits(self) -> int:
        return int(self.hnp.get("bits", DEFAULT_BITS))

    @property
    def hnp_mode(self) -> str:
        return str(self.hnp.get("mode", "low"))

    @property
    def hnp_private_key(self) -> int:
        return int(self.hnp["private_key"], 16)

    @property
    def hnp_public_key(self) -> CurvePoint:
        pub = self.hnp["public_key"]
        return CurvePoint(int(pub["x"], 16), int(pub["y"], 16))

    @property
    def hnp_signatures(self) -> List[WeakNonceSignature]:
        sigs = []
        for sig in self.hnp["signatures"]:
            k_full = sig.get("k_full")
            sigs.append(
                WeakNonceSignature(
                    r=int(sig["r"], 16),
                    s=int(sig["s"], 16),
                    z=int(sig["z"], 16),
                    k_low=int(sig["k_low"], 16),
                    k_full=int(k_full, 16) if k_full else None,
                )
            )
        return sigs


def _point_to_dict(point: CurvePoint) -> Dict[str, str]:
    return {"x": hex(point.x), "y": hex(point.y)}


def create_replay_dataset(
    seed: int,
    ecdsa_samples: int = 16,
    hnp_signatures: int = 20,
    hnp_bits: int = DEFAULT_BITS,
    hnp_mode: str = "low",
) -> ReplayDataset:
    rng = random.Random(seed)

    ecdsa_data = []
    for _ in range(ecdsa_samples):
        priv, pub = generate_keypair(rng)
        ecdsa_data.append({"private_key": hex(priv), "public_key": _point_to_dict(pub)})

    # Use a dedicated key for HNP evaluation.
    hnp_priv, hnp_pub = generate_keypair(rng)
    sigs = generate_weak_nonce_signatures(
        hnp_priv,
        hnp_signatures,
        seed=rng.randint(0, 2**31 - 1),
        bits=hnp_bits,
    )
    sig_data = []
    for sig in sigs:
        sig_data.append({
            "r": hex(sig.r),
            "s": hex(sig.s),
            "z": hex(sig.z),
            "k_low": hex(sig.k_low),
            "k_full": hex(sig.k_full) if sig.k_full is not None else None,
        })

    hnp_block = {
        "private_key": hex(hnp_priv),
        "public_key": _point_to_dict(hnp_pub),
        "signatures": sig_data,
        "bits": hnp_bits,
        "mode": hnp_mode,
    }

    return ReplayDataset(seed=seed, ecdsa_samples=ecdsa_data, hnp=hnp_block)


def save_dataset(path: str, dataset: ReplayDataset) -> None:
    payload = {
        "seed": dataset.seed,
        "ecdsa_samples": dataset.ecdsa_samples,
        "hnp": dataset.hnp,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_dataset(path: str) -> ReplayDataset:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ReplayDataset(
        seed=payload["seed"],
        ecdsa_samples=payload["ecdsa_samples"],
        hnp=payload["hnp"],
    )


def ensure_dataset(
    path: str | None,
    seed: int,
    ecdsa_samples: int,
    hnp_signatures: int,
    hnp_bits: int,
    hnp_mode: str,
) -> ReplayDataset:
    if path and os.path.exists(path):
        return load_dataset(path)

    dataset = create_replay_dataset(
        seed=seed,
        ecdsa_samples=ecdsa_samples,
        hnp_signatures=hnp_signatures,
        hnp_bits=hnp_bits,
        hnp_mode=hnp_mode,
    )
    if path:
        save_dataset(path, dataset)
    return dataset
