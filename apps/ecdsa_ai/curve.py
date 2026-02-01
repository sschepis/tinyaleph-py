"""Minimal secp256k1 operations for research-grade experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import random
import secrets

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


@dataclass(frozen=True)
class CurvePoint:
    x: int
    y: int


def inv_mod(a: int, m: int) -> int:
    if a == 0:
        raise ZeroDivisionError("Inverse of 0 does not exist")
    return pow(a, -1, m)


def is_on_curve(point: CurvePoint) -> bool:
    return (point.y * point.y - (point.x * point.x * point.x + 7)) % P == 0


def point_add(p1: Optional[CurvePoint], p2: Optional[CurvePoint]) -> Optional[CurvePoint]:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    if p1.x == p2.x and (p1.y + p2.y) % P == 0:
        return None

    if p1 == p2:
        slope = (3 * p1.x * p1.x) * inv_mod(2 * p1.y % P, P)
    else:
        slope = (p2.y - p1.y) * inv_mod((p2.x - p1.x) % P, P)

    slope %= P
    x_r = (slope * slope - p1.x - p2.x) % P
    y_r = (slope * (p1.x - x_r) - p1.y) % P
    return CurvePoint(x_r, y_r)


def scalar_mult(k: int, point: Optional[CurvePoint]) -> Optional[CurvePoint]:
    if k % N == 0 or point is None:
        return None
    if k < 0:
        return scalar_mult(-k, CurvePoint(point.x, (-point.y) % P))

    result = None
    addend = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return result


def generate_keypair(rng: Optional[random.Random] = None) -> Tuple[int, CurvePoint]:
    if rng is None:
        private_key = secrets.randbelow(N - 1) + 1
    else:
        private_key = rng.randrange(1, N)
    public_key = scalar_mult(private_key, CurvePoint(Gx, Gy))
    if public_key is None:
        raise RuntimeError("Failed to generate public key")
    return private_key, public_key
