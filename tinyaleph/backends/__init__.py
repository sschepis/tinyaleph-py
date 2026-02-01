"""
Domain-specific backends for TinyAleph.

Provides:
- cryptographic: Prime-based hashing, encryption, key derivation
- semantic: Semantic embedding and similarity
- scientific: Scientific computing primitives
- bioinformatics: Sequence analysis
"""

from tinyaleph.backends.cryptographic import (
    CryptographicBackend,
    PrimeStateKeyGenerator,
    EntropySensitiveEncryptor,
    HolographicKeyDistributor,
)

__all__ = [
    # Cryptographic
    "CryptographicBackend",
    "PrimeStateKeyGenerator",
    "EntropySensitiveEncryptor",
    "HolographicKeyDistributor",
]