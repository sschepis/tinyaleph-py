#!/usr/bin/env python3
"""
Example 13: Cryptographic Backend

Demonstrates the prime-based cryptographic capabilities:
- PrimeStateKeyGenerator for key derivation
- EntropySensitiveEncryptor for adaptive encryption
- HolographicKeyDistributor for distributed key sharing
- CryptographicBackend unified interface
"""

import sys
sys.path.insert(0, '..')

from tinyaleph.backends import (
    CryptographicBackend,
    PrimeStateKeyGenerator,
    EntropySensitiveEncryptor,
    HolographicKeyDistributor,
)
from tinyaleph.hilbert import PrimeState
from tinyaleph.core import Complex


def demonstrate_key_generation():
    """Demonstrate prime-based key generation."""
    print("=" * 60)
    print("PRIME STATE KEY GENERATOR")
    print("=" * 60)
    
    # Create key generator
    generator = PrimeStateKeyGenerator(key_bits=256)
    
    print("\nGenerating keys from prime states:")
    
    # Generate key from seed phrase
    seed = "TinyAleph secure key derivation"
    key1 = generator.generate_from_seed(seed)
    print(f"\n  Seed: '{seed}'")
    print(f"  Key (hex): {key1[:32]}...")
    print(f"  Key length: {len(key1) * 4} bits")
    
    # Generate key from prime state
    amplitudes = [Complex(0.4), Complex(0.2), Complex(0.15), Complex(0.1), Complex(0.1), Complex(0.05)]
    state = PrimeState(amplitudes=amplitudes, primes=[2, 3, 5, 7, 11, 13])
    key2 = generator.generate_from_state(state)
    print(f"\n  From PrimeState with primes [2, 3, 5, 7, 11, 13]:")
    print(f"  Key (hex): {key2[:32]}...")
    
    # Generate deterministic key pair
    print("\nDeterministic key pair generation:")
    public, private = generator.generate_keypair(seed)
    print(f"  Public key:  {public[:24]}...")
    print(f"  Private key: {private[:24]}...")
    
    # Key derivation function
    print("\nKey derivation chain:")
    derived = key1
    for i in range(3):
        derived = generator.derive_key(derived, f"chain-{i}")
        print(f"  Derived[{i}]: {derived[:24]}...")
    
    return generator


def demonstrate_encryption():
    """Demonstrate entropy-sensitive encryption."""
    print("\n" + "=" * 60)
    print("ENTROPY-SENSITIVE ENCRYPTOR")
    print("=" * 60)
    
    # Create encryptor
    encryptor = EntropySensitiveEncryptor()
    
    # Test messages with different entropy characteristics
    messages = [
        "Hello, World!",  # Low entropy (simple)
        "The quick brown fox jumps over the lazy dog",  # Medium entropy
        "7hE_qu1ck_br0wn_f0x!@#$%",  # Higher entropy
    ]
    
    key = "my_secret_key_256_bits_long_here"
    
    print("\nEncryption with entropy adaptation:")
    for msg in messages:
        # Analyze message entropy
        entropy = encryptor.message_entropy(msg)
        
        # Encrypt
        ciphertext = encryptor.encrypt(msg, key)
        
        # Decrypt to verify
        decrypted = encryptor.decrypt(ciphertext, key)
        
        print(f"\n  Original: '{msg}'")
        print(f"  Entropy: {entropy:.3f} bits/char")
        print(f"  Ciphertext length: {len(ciphertext)} bytes")
        print(f"  Decrypted: '{decrypted}'")
        print(f"  Verified: {decrypted == msg}")
    
    # Show entropy-based strength selection
    print("\nEntropy-based encryption strength:")
    for level in ['low', 'medium', 'high']:
        rounds = encryptor.rounds_for_strength(level)
        print(f"  {level:8s} entropy → {rounds} encryption rounds")
    
    return encryptor


def demonstrate_key_distribution():
    """Demonstrate holographic key distribution."""
    print("\n" + "=" * 60)
    print("HOLOGRAPHIC KEY DISTRIBUTOR")
    print("=" * 60)
    
    # Create distributor
    distributor = HolographicKeyDistributor(
        num_shares=5,
        threshold=3  # Need 3 of 5 shares to reconstruct
    )
    
    # Master key
    master_key = "MASTER_SECRET_KEY_FOR_DISTRIBUTION"
    print(f"\nMaster key: '{master_key}'")
    print(f"Scheme: {distributor.threshold}-of-{distributor.num_shares} threshold")
    
    # Split into shares
    shares = distributor.split(master_key)
    print("\nGenerated shares:")
    for i, share in enumerate(shares):
        print(f"  Share {i+1}: {share[:20]}...")
    
    # Reconstruct from different subsets
    print("\nReconstruction tests:")
    
    # Test with exactly threshold shares
    subset1 = shares[:3]  # First 3 shares
    recovered1 = distributor.reconstruct(subset1)
    print(f"  From shares [1,2,3]: '{recovered1}' (success: {recovered1 == master_key})")
    
    # Test with different subset
    subset2 = [shares[0], shares[2], shares[4]]  # Shares 1,3,5
    recovered2 = distributor.reconstruct(subset2)
    print(f"  From shares [1,3,5]: '{recovered2}' (success: {recovered2 == master_key})")
    
    # Test with more than threshold
    subset3 = shares[:4]  # 4 shares
    recovered3 = distributor.reconstruct(subset3)
    print(f"  From shares [1,2,3,4]: '{recovered3}' (success: {recovered3 == master_key})")
    
    # Test with fewer than threshold (should fail)
    subset4 = shares[:2]  # Only 2 shares
    try:
        recovered4 = distributor.reconstruct(subset4)
        print(f"  From shares [1,2]: Unexpected success")
    except Exception as e:
        print(f"  From shares [1,2]: Correctly failed - {type(e).__name__}")
    
    return distributor


def demonstrate_cryptographic_backend():
    """Demonstrate the unified CryptographicBackend."""
    print("\n" + "=" * 60)
    print("UNIFIED CRYPTOGRAPHIC BACKEND")
    print("=" * 60)
    
    backend = CryptographicBackend()
    
    print("\nBackend capabilities:")
    print("  - Prime-based key generation")
    print("  - Entropy-sensitive encryption")
    print("  - Threshold secret sharing")
    print("  - Hash computation")
    
    # Hash computation
    print("\nHash computation:")
    message = "Hello, TinyAleph!"
    hash_val = backend.hash(message)
    print(f"  Message: '{message}'")
    print(f"  Hash: {hash_val}")
    
    # HMAC
    print("\nHMAC (keyed hash):")
    key = "secret_key"
    hmac_val = backend.hmac(message, key)
    print(f"  Message: '{message}'")
    print(f"  Key: '{key}'")
    print(f"  HMAC: {hmac_val}")
    
    # Key generation with entropy
    print("\nKey generation with target entropy:")
    for entropy_bits in [64, 128, 256]:
        key = backend.generate_key(entropy_bits)
        print(f"  {entropy_bits}-bit: {key[:min(32, len(key))]}...")
    
    return backend


def demonstrate_prime_crypto_properties():
    """Demonstrate prime-based cryptographic properties."""
    print("\n" + "=" * 60)
    print("PRIME-BASED CRYPTOGRAPHIC PROPERTIES")
    print("=" * 60)
    
    generator = PrimeStateKeyGenerator()
    
    # Show how different prime states produce different keys
    print("\nPrime state → Key mapping:")
    
    states = [
        ([Complex(0.7), Complex(0.3)], [2, 3]),
        ([Complex(0.7), Complex(0.3)], [2, 5]),  # Different prime
        ([Complex(0.3), Complex(0.7)], [2, 3]),  # Same primes, different weights
        ([Complex(0.5), Complex(0.3), Complex(0.2)], [2, 3, 5]),  # More primes
    ]
    
    for i, (amps, primes) in enumerate(states):
        state = PrimeState(amplitudes=amps, primes=primes)
        key = generator.generate_from_state(state)
        print(f"\n  State {i+1}:")
        print(f"    Primes: {primes}")
        print(f"    Amplitudes: {[a.real for a in amps]}")
        print(f"    Key: {key[:32]}...")
    
    # Demonstrate avalanche effect
    print("\nAvalanche effect demonstration:")
    seed1 = "test_seed_1"
    seed2 = "test_seed_2"  # One character different
    
    key1 = generator.generate_from_seed(seed1)
    key2 = generator.generate_from_seed(seed2)
    
    # Count differing characters
    diff_chars = sum(1 for a, b in zip(key1, key2) if a != b)
    
    print(f"  Seed 1: '{seed1}'")
    print(f"  Seed 2: '{seed2}'")
    print(f"  Key 1: {key1[:32]}...")
    print(f"  Key 2: {key2[:32]}...")
    print(f"  Differing chars: {diff_chars}/{min(len(key1), len(key2))} (~50% expected for good avalanche)")


def main():
    """Run all cryptographic demonstrations."""
    print("ALEPH PRIME - CRYPTOGRAPHIC BACKEND EXAMPLES")
    print("=" * 60)
    
    demonstrate_key_generation()
    demonstrate_encryption()
    demonstrate_key_distribution()
    demonstrate_cryptographic_backend()
    demonstrate_prime_crypto_properties()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Cryptographic backend capabilities:
- Prime-state based key generation
- Entropy-sensitive adaptive encryption
- Threshold secret sharing (Shamir-like)
- Hash and HMAC computation
- Avalanche effect verification

Key features:
- Keys derived from prime number structure
- Entropy-adaptive cipher selection
- Holographic key distribution
- Deterministic key derivation

Security properties:
- Strong entropy from prime factorization
- Threshold reconstruction prevents single-point failure
- Entropy analysis for adaptive security
""")


if __name__ == "__main__":
    main()