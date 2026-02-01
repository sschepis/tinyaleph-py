"""Weak-nonce ECDSA signature generation and HNP consistency evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import secrets

from apps.ecdsa_ai.curve import CurvePoint, Gx, Gy, N, scalar_mult, inv_mod

DEFAULT_BITS = 53


@dataclass(frozen=True)
class WeakNonceSignature:
    r: int
    s: int
    z: int
    k_low: int
    k_full: Optional[int] = None


def sign_with_nonce(private_key: int, z: int, k: int) -> Tuple[int, int]:
    point = scalar_mult(k, CurvePoint(Gx, Gy))
    if point is None:
        raise ValueError("Invalid nonce; resulted in point at infinity")
    r = point.x % N
    if r == 0:
        raise ValueError("Invalid signature; r == 0")

    k_inv = inv_mod(k % N, N)
    s = (k_inv * (z + r * private_key)) % N
    if s == 0:
        raise ValueError("Invalid signature; s == 0")
    return r, s


def _draw_nonce(randbits, bits: int) -> int:
    if bits <= 0:
        return 1
    if bits >= N.bit_length():
        return (randbits(N.bit_length()) % (N - 1)) + 1
    mask = (1 << bits) - 1
    return (randbits(bits) & mask) or 1


def generate_weak_nonce_signatures(
    private_key: int,
    count: int,
    seed: int | None = None,
    bits: int = DEFAULT_BITS,
) -> List[WeakNonceSignature]:
    if seed is not None:
        rng = random.Random(seed)
        randbits = rng.getrandbits
    else:
        randbits = secrets.randbits

    signatures: List[WeakNonceSignature] = []
    while len(signatures) < count:
        z = randbits(256) % N
        k = _draw_nonce(randbits, bits)
        try:
            r, s = sign_with_nonce(private_key, z, k)
        except ValueError:
            continue
        mask = (1 << bits) - 1 if bits > 0 else 0
        signatures.append(WeakNonceSignature(r=r, s=s, z=z, k_low=k & mask, k_full=k))
    return signatures


def hnp_consistency(
    private_key: int,
    signatures: List[WeakNonceSignature],
    bits: int = DEFAULT_BITS,
    mode: str = "low",
) -> Tuple[float, float]:
    if not signatures:
        return 0.0, 0.0

    matches = 0
    errors = []
    for sig in signatures:
        s_inv = inv_mod(sig.s, N)
        k_pred = (s_inv * (sig.z + sig.r * private_key)) % N

        if mode == "full":
            if sig.k_full is None:
                continue
            if k_pred == sig.k_full:
                matches += 1
            diff = abs(k_pred - sig.k_full)
            diff = min(diff, N - diff)
            errors.append(diff / N)
        else:
            mask = (1 << bits) - 1 if bits > 0 else 0
            k_low = k_pred & mask
            if k_low == sig.k_low:
                matches += 1
            diff = abs(k_low - sig.k_low)
            modulus = 1 << bits if bits > 0 else 1
            diff = min(diff, modulus - diff)
            errors.append(diff / modulus)

    total = len(errors)
    match_ratio = matches / total if total > 0 else 0.0
    avg_error = sum(errors) / total if total > 0 else 0.0
    return match_ratio, avg_error
