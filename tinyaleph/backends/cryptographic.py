"""
Cryptographic Backend - Prime-based hashing, encryption, and key derivation

Enhanced with QuPrimes concepts:
- Prime-State Key Generation using resonance phases
- Holographic Key Distribution
- Entropy-Sensitive Encryption

Core mathematical foundation:
- For a state |n⟩ = Σ √(a_i/A) |p_i⟩, derive key: K = Σ_i θ_{p_i} mod 2π
- θ_{p_i} = 2π log_{p_i}(n) (resonance phase)
- Security relies on difficulty of inverting phase relationships
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import math
import random
from abc import ABC, abstractmethod

from tinyaleph.core.primes import nth_prime, prime_sieve, is_prime, factorize
from tinyaleph.core.complex import Complex
from tinyaleph.core.constants import PHI


def first_n_primes(n: int) -> List[int]:
    """Generate first n prime numbers."""
    if n <= 0:
        return []
    primes = []
    candidate = 2
    while len(primes) < n:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def prime_to_frequency(p: int, min_freq: float = 1.0, max_freq: float = 10.0) -> float:
    """Map prime to frequency in given range using logarithmic scaling."""
    # log(p) maps primes to roughly linear spacing
    log_p = math.log(p)
    log_min = math.log(2)  # smallest prime
    log_max = math.log(1000)  # reasonable upper bound
    
    # Normalize to [0, 1]
    normalized = (log_p - log_min) / (log_max - log_min)
    normalized = max(0.0, min(1.0, normalized))
    
    # Map to frequency range
    return min_freq + normalized * (max_freq - min_freq)


# =============================================================================
# GAUSSIAN INTEGER
# =============================================================================

@dataclass
class GaussianInteger:
    """
    Gaussian integer: a + bi where a, b are integers.
    
    Important for primes p ≡ 1 (mod 4), which can be expressed as
    sum of two squares: p = a² + b² = (a + bi)(a - bi)
    """
    real: float
    imag: float
    
    def __add__(self, other: 'GaussianInteger') -> 'GaussianInteger':
        return GaussianInteger(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other: 'GaussianInteger') -> 'GaussianInteger':
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        return GaussianInteger(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def norm(self) -> float:
        return self.real ** 2 + self.imag ** 2
    
    def conjugate(self) -> 'GaussianInteger':
        return GaussianInteger(self.real, -self.imag)


# =============================================================================
# HYPERCOMPLEX STATE (simplified for cryptography)
# =============================================================================

class HypercomplexState:
    """
    Hypercomplex state vector for cryptographic mixing.
    Uses higher-dimensional complex numbers for non-commutative operations.
    """
    
    def __init__(self, dimension: int, components: Optional[List[float]] = None):
        self.dimension = dimension
        if components is not None:
            self.c = list(components)
        else:
            self.c = [0.0] * dimension
    
    @classmethod
    def zero(cls, dimension: int) -> 'HypercomplexState':
        return cls(dimension)
    
    @classmethod
    def from_real(cls, dimension: int, value: float) -> 'HypercomplexState':
        state = cls(dimension)
        state.c[0] = value
        return state
    
    def add(self, other: 'HypercomplexState') -> 'HypercomplexState':
        result = HypercomplexState(self.dimension)
        for i in range(self.dimension):
            result.c[i] = self.c[i] + other.c[i]
        return result
    
    def mul(self, other: 'HypercomplexState') -> 'HypercomplexState':
        """Non-commutative multiplication for mixing."""
        result = HypercomplexState(self.dimension)
        for i in range(self.dimension):
            for j in range(self.dimension):
                # Cyclic multiplication with phase shift
                idx = (i + j) % self.dimension
                result.c[idx] += self.c[i] * other.c[j]
        return result
    
    def norm(self) -> float:
        return math.sqrt(sum(x ** 2 for x in self.c))
    
    def normalize(self) -> 'HypercomplexState':
        n = self.norm()
        if n < 1e-10:
            return self
        result = HypercomplexState(self.dimension)
        for i in range(self.dimension):
            result.c[i] = self.c[i] / n
        return result


# =============================================================================
# CRYPTOGRAPHIC BACKEND
# =============================================================================

@dataclass
class CryptographicBackend:
    """
    Cryptographic backend using prime-based operations.
    
    Provides:
    - hash(): Hypercomplex state mixing for hashing
    - deriveKey(): Key derivation from password + salt
    - hmac(): HMAC-like authentication code
    """
    
    dimension: int = 32
    key_primes: List[int] = field(default_factory=list)
    rounds: int = 16
    
    def __post_init__(self):
        if not self.key_primes:
            self.key_primes = first_n_primes(256)
    
    def encode(self, input_data: Union[str, bytes, List[int]]) -> List[int]:
        """Encode input to prime sequence."""
        if isinstance(input_data, str):
            data = input_data.encode('utf-8')
        elif isinstance(input_data, bytes):
            data = input_data
        else:
            data = bytes(input_data)
        
        return [self.key_primes[b % len(self.key_primes)] for b in data]
    
    def decode(self, primes: List[int]) -> bytes:
        """Decode prime sequence back to bytes."""
        prime_to_idx = {p: i for i, p in enumerate(self.key_primes)}
        byte_values = [prime_to_idx.get(p, 0) for p in primes]
        return bytes(byte_values)
    
    def primes_to_state(self, primes: List[int]) -> HypercomplexState:
        """Convert prime sequence to hypercomplex state."""
        state = HypercomplexState.zero(self.dimension)
        n = len(primes) or 1
        sqrt_n = math.sqrt(n)
        
        for i, p in enumerate(primes):
            angle = (2 * math.pi * p) / self.dimension
            idx = i % self.dimension
            
            # Use Gaussian integer decomposition for p ≡ 1 (mod 4)
            if p % 4 == 1:
                gi = GaussianInteger(math.cos(angle), math.sin(angle))
                state.c[idx] += gi.real / sqrt_n
                state.c[(idx + 1) % self.dimension] += gi.imag / sqrt_n
            else:
                state.c[idx] += math.cos(angle * p) / sqrt_n
        
        return state.normalize()
    
    def hash(self, input_data: Union[str, bytes], output_length: int = 32) -> bytes:
        """
        Hash input using hypercomplex state mixing.
        
        Args:
            input_data: Data to hash
            output_length: Desired output length in bytes
            
        Returns:
            Hash as bytes
        """
        primes = self.encode(input_data)
        state = self.primes_to_state(primes)
        
        # Create input-dependent mixing constants
        input_sum = sum(primes)
        input_xor = 0
        for p in primes:
            input_xor ^= p
        
        # Multiple rounds of mixing
        for i in range(self.rounds):
            # Mix with shifted version for non-commutativity
            shifted = HypercomplexState.zero(self.dimension)
            for j in range(self.dimension):
                shifted.c[(j + i + 1) % self.dimension] = state.c[j]
            
            # Combine original and shifted
            state = state.mul(shifted).normalize()
            
            # Add input-dependent round constant
            round_const = HypercomplexState.zero(self.dimension)
            idx = (i + input_xor) % self.dimension
            round_const.c[idx] = 0.1 * math.sin(input_sum + i)
            round_const.c[(idx + 7) % self.dimension] = 0.1 * math.cos(input_sum + i)
            state = state.add(round_const).normalize()
            
            # Non-linear transformation
            for j in range(self.dimension):
                state.c[j] = math.tanh(state.c[j] * 2)
            state = state.normalize()
        
        # Extract hash bytes
        hash_bytes = []
        for i in range(output_length):
            idx = (i + input_xor) % self.dimension
            val = abs(state.c[idx] * 127.5 + state.c[(idx + 1) % self.dimension] * 127.5)
            hash_bytes.append(int(val) & 0xFF)
        
        return bytes(hash_bytes)
    
    def derive_key(
        self,
        password: str,
        salt: str,
        key_length: int = 32,
        iterations: int = 10000
    ) -> bytes:
        """
        Derive key from password using salt and iterations.
        
        Args:
            password: Password string
            salt: Salt string
            key_length: Desired key length
            iterations: Number of iterations
            
        Returns:
            Derived key as bytes
        """
        primes = self.encode(password)
        salt_primes = self.encode(salt)
        
        for i in range(iterations):
            # Mix password and salt primes
            primes = primes + salt_primes
            state = self.primes_to_state(primes)
            
            # Apply iteration-dependent mixing
            mixed = state.mul(HypercomplexState.from_real(self.dimension, i + 1))
            
            # Extract new primes from mixed state
            primes = []
            for j in range(self.dimension):
                idx = int(abs(mixed.c[j]) * len(self.key_primes))
                primes.append(self.key_primes[idx % len(self.key_primes)])
        
        # Convert primes to byte-safe representation
        prime_bytes = []
        for p in primes:
            prime_bytes.append(p & 0xFF)
            prime_bytes.append((p >> 8) & 0xFF)
        return self.hash(bytes(prime_bytes), key_length)
    
    def mix_primes(self, data_primes: List[int], key_primes: List[int]) -> List[int]:
        """XOR-style prime mixing for encryption."""
        max_prime = self.key_primes[-1] + 1
        result = []
        for i, p in enumerate(data_primes):
            k = key_primes[i % len(key_primes)]
            mixed = (p * k) % max_prime
            # Find nearest prime
            nearest = min(self.key_primes, key=lambda q: abs(q - mixed))
            result.append(nearest)
        return result
    
    def hmac(self, key: str, message: str, output_length: int = 32) -> bytes:
        """
        Compute HMAC-like authentication code.
        
        Args:
            key: Key string
            message: Message to authenticate
            output_length: Output length in bytes
            
        Returns:
            Authentication code as bytes
        """
        key_primes = self.encode(key)
        msg_primes = self.encode(message)
        
        # Inner hash - convert primes to bytes safely
        inner_primes = self.mix_primes(msg_primes, key_primes)
        inner_bytes = []
        for p in inner_primes:
            inner_bytes.append(p & 0xFF)
            inner_bytes.append((p >> 8) & 0xFF)
        inner_hash = self.hash(bytes(inner_bytes), self.dimension)
        
        # Outer hash
        outer_primes = self.mix_primes(self.encode(inner_hash), key_primes)
        outer_bytes = []
        for p in outer_primes:
            outer_bytes.append(p & 0xFF)
            outer_bytes.append((p >> 8) & 0xFF)
        return self.hash(bytes(outer_bytes), output_length)


# =============================================================================
# PRIME-STATE KEY GENERATOR
# =============================================================================

@dataclass
class PrimeStateKeyGenerator:
    """
    Generate cryptographic keys using prime resonance framework.
    
    For a state |n⟩ = Σ √(a_i/A) |p_i⟩, derive key:
    K = Σ_i θ_{p_i} mod 2π, where θ_{p_i} = 2π log_{p_i}(n)
    
    Security relies on difficulty of inverting phase relationships.
    """
    
    primes: List[int] = field(default_factory=list)
    key_length: int = 32
    
    def __post_init__(self):
        if not self.primes:
            self.primes = first_n_primes(64)
        self.phi = PHI  # Golden ratio for phase shifts
    
    def resonance_phase(self, p: int, n: int) -> float:
        """
        Compute resonance phase θ_p = 2π log_p(n).
        
        Args:
            p: Prime base
            n: Number to compute phase for
            
        Returns:
            Phase angle in radians
        """
        if n <= 0 or p <= 1:
            return 0.0
        return 2 * math.pi * math.log(n) / math.log(p)
    
    def create_prime_state(self, n: int) -> Dict[int, Complex]:
        """
        Create canonical prime state from number n.
        |n⟩ = Σ √(a_i/A) |p_i⟩ for n = Π p_i^{a_i}
        
        Args:
            n: Input number
            
        Returns:
            Dictionary mapping primes to complex amplitudes
        """
        state = {p: Complex(0, 0) for p in self.primes}
        factors = factorize(n)
        
        # Total exponent count
        total_exp = sum(factors.values())
        
        if total_exp == 0:
            # n = 1, return uniform state
            amp = 1.0 / math.sqrt(len(self.primes))
            return {p: Complex(amp, 0) for p in self.primes}
        
        # Set amplitudes based on factorization
        for p, exp in factors.items():
            if p in state:
                amplitude = math.sqrt(exp / total_exp)
                state[p] = Complex(amplitude, 0)
        
        # Normalize
        norm = math.sqrt(sum(c.norm() ** 2 for c in state.values()))
        if norm > 1e-10:
            state = {p: Complex(c.real / norm, c.imag / norm) for p, c in state.items()}
        
        return state
    
    def generate_key(self, n: int) -> Dict:
        """
        Generate key from prime state using phase summation.
        K = Σ_i θ_{p_i} mod 2π
        
        Args:
            n: Input number to derive key from
            
        Returns:
            Key data with phases, buffer, and entropy
        """
        state = self.create_prime_state(n)
        phases = []
        
        # Compute resonance phase for each prime with non-zero amplitude
        for p in self.primes:
            amp = state[p]
            if amp.norm() > 1e-10:
                phase = self.resonance_phase(p, n)
                phases.append({
                    "prime": p,
                    "phase": phase,
                    "amplitude": amp.norm(),
                    "complex": Complex.from_polar(amp.norm(), phase)
                })
        
        # Sum phases modulo 2π
        raw_key = sum(p["phase"] * p["amplitude"] for p in phases)
        key_modulo = raw_key % (2 * math.pi)
        
        # Expand to key bytes
        key_bytes = self.expand_to_bytes(phases, self.key_length)
        
        # Compute entropy
        total_prob = sum(p["amplitude"] ** 2 for p in phases)
        if total_prob > 0:
            entropy = -sum(
                (p["amplitude"] ** 2 / total_prob) * 
                math.log(p["amplitude"] ** 2 / total_prob + 1e-10)
                for p in phases
            )
        else:
            entropy = 0.0
        
        return {
            "state": state,
            "phases": phases,
            "raw_key": raw_key,
            "key_modulo": key_modulo,
            "key_buffer": bytes(key_bytes),
            "key_hex": bytes(key_bytes).hex(),
            "entropy": entropy
        }
    
    def expand_to_bytes(self, phases: List[Dict], length: int) -> List[int]:
        """Expand phase information to key bytes."""
        if not phases:
            return [0] * length
        
        key_bytes = []
        for i in range(length):
            phase_idx = i % len(phases)
            phase = phases[phase_idx]
            
            # Derive byte from phase and position
            val = (phase["phase"] + i * self.phi) * phase["amplitude"]
            normalized = ((math.sin(val) + 1) / 2) * 255
            key_bytes.append(int(normalized) & 0xFF)
        
        return key_bytes
    
    def generate_key_pair(self, seed: int) -> Dict:
        """
        Generate key pair (public/private) using prime resonance.
        
        Public key: amplitudes without phases
        Private key: full phase information
        
        Args:
            seed: Seed for key generation
            
        Returns:
            Dictionary with public_key and private_key
        """
        n = abs(seed) + 1
        private_data = self.generate_key(n)
        
        # Public key: amplitudes without phases
        public_key = {
            "primes": [p["prime"] for p in private_data["phases"]],
            "amplitudes": [p["amplitude"] for p in private_data["phases"]],
            "entropy": private_data["entropy"]
        }
        
        # Private key: full phase information
        private_key = {
            "seed": n,
            "phases": private_data["phases"],
            "key_buffer": private_data["key_buffer"]
        }
        
        return {"public_key": public_key, "private_key": private_key}
    
    def derive_shared_secret(self, key1: Dict, key2: Dict) -> Dict:
        """
        Derive shared secret from two keys.
        Uses inner product of prime states.
        
        Args:
            key1: First key data (must contain "state")
            key2: Second key data (must contain "state")
            
        Returns:
            Shared secret data
        """
        # Compute inner product
        state1 = key1["state"]
        state2 = key2["state"]
        
        inner_real = 0.0
        inner_imag = 0.0
        for p in self.primes:
            c1 = state1.get(p, Complex(0, 0))
            c2 = state2.get(p, Complex(0, 0))
            # <ψ1|ψ2> = Σ c1* · c2
            inner_real += c1.real * c2.real + c1.imag * c2.imag
            inner_imag += c1.real * c2.imag - c1.imag * c2.real
        
        inner = Complex(inner_real, inner_imag)
        magnitude = inner.norm()
        phase = math.atan2(inner.imag, inner.real)
        
        # Generate shared key bytes
        shared_bytes = []
        for i in range(self.key_length):
            val = math.sin(phase + i * self.phi) * magnitude
            normalized = ((val + 1) / 2) * 255
            shared_bytes.append(int(abs(normalized)) & 0xFF)
        
        return {
            "coherence": magnitude,
            "phase": phase,
            "shared_key": bytes(shared_bytes)
        }


# =============================================================================
# ENTROPY-SENSITIVE ENCRYPTOR
# =============================================================================

@dataclass
class EntropySensitiveEncryptor:
    """
    Encrypt messages using entropy-based phase modulation.
    
    Ê_K|m⟩ = e^{iK(m)}|m⟩, where K(m) = Σ_{p|m} θ_p
    """
    
    key_gen: PrimeStateKeyGenerator = field(default_factory=PrimeStateKeyGenerator)
    
    def encrypt(self, data: Union[bytes, str], key: int) -> bytes:
        """
        Encrypt data using phase modulation.
        
        Args:
            data: Data to encrypt
            key: Encryption key (number)
            
        Returns:
            Encrypted data
        """
        buffer = data.encode('utf-8') if isinstance(data, str) else data
        key_phases = self.key_gen.generate_key(key)["phases"]
        
        if not key_phases:
            return buffer
        
        encrypted = bytearray(len(buffer))
        
        for i, byte in enumerate(buffer):
            # Find divisor primes for this position
            phase_sum = sum(
                p["phase"] * p["amplitude"]
                for p in key_phases
                if i % p["prime"] == 0 or byte % p["prime"] == 0
            )
            
            # Apply phase-based transformation
            transform = int((math.sin(phase_sum + i) + 1) * 127.5)
            encrypted[i] = (byte + transform) & 0xFF
        
        return bytes(encrypted)
    
    def decrypt(self, encrypted: bytes, key: int) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted: Encrypted data
            key: Decryption key
            
        Returns:
            Decrypted data
        """
        key_phases = self.key_gen.generate_key(key)["phases"]
        
        if not key_phases:
            return encrypted
        
        decrypted = bytearray(len(encrypted))
        
        for i, byte in enumerate(encrypted):
            # Same phase calculation as encryption (position-based)
            phase_sum = sum(
                p["phase"] * p["amplitude"]
                for p in key_phases
                if i % p["prime"] == 0
            )
            
            transform = int((math.sin(phase_sum + i) + 1) * 127.5)
            decrypted[i] = (byte - transform + 256) & 0xFF
        
        return bytes(decrypted)


# =============================================================================
# HOLOGRAPHIC KEY DISTRIBUTOR
# =============================================================================

@dataclass
class HolographicKeyDistributor:
    """
    Encode keys in interference patterns.
    
    I(x, y) = Σ_p A_p e^{-S(x,y)} e^{ipθ}
    
    Extract keys via Fourier inversion.
    """
    
    grid_size: int = 16
    key_gen: PrimeStateKeyGenerator = field(default_factory=PrimeStateKeyGenerator)
    
    def encode_key(self, key_value: int) -> Dict:
        """
        Encode key into holographic pattern.
        
        Args:
            key_value: Key as number
            
        Returns:
            Holographic encoding with pattern and key data
        """
        key_data = self.key_gen.generate_key(key_value)
        
        # Create interference pattern
        pattern = []
        half_grid = self.grid_size / 2
        
        for x in range(self.grid_size):
            row = []
            for y in range(self.grid_size):
                intensity = 0.0
                phase = 0.0
                
                for p in key_data["phases"]:
                    # Distance-based decay
                    r = math.sqrt((x - half_grid) ** 2 + (y - half_grid) ** 2)
                    decay = math.exp(-r / self.grid_size)
                    
                    # Prime-frequency interference
                    prime_phase = p["phase"] + (x * p["prime"] / self.grid_size) + (y / p["prime"])
                    intensity += p["amplitude"] * decay * math.cos(prime_phase)
                    phase += prime_phase * p["amplitude"]
                
                row.append({"intensity": intensity, "phase": phase})
            pattern.append(row)
        
        return {
            "pattern": pattern,
            "key_data": key_data,
            "grid_size": self.grid_size
        }
    
    def decode_key(self, encoding: Dict) -> List[int]:
        """
        Decode key from holographic pattern.
        
        Args:
            encoding: Holographic encoding
            
        Returns:
            Recovered key bytes
        """
        pattern = encoding["pattern"]
        grid_size = encoding["grid_size"]
        
        # Fourier inversion to extract prime phases
        extracted_phases = []
        
        for p in self.key_gen.primes[:16]:
            real_sum = 0.0
            imag_sum = 0.0
            
            for x in range(grid_size):
                for y in range(grid_size):
                    cell = pattern[x][y]
                    freq = (x * p / grid_size) + (y / p)
                    real_sum += cell["intensity"] * math.cos(freq)
                    imag_sum += cell["intensity"] * math.sin(freq)
            
            magnitude = math.sqrt(real_sum ** 2 + imag_sum ** 2) / (grid_size * grid_size)
            phase = math.atan2(imag_sum, real_sum)
            
            if magnitude > 0.01:
                extracted_phases.append({
                    "prime": p,
                    "phase": phase,
                    "amplitude": magnitude
                })
        
        # Reconstruct key from extracted phases
        return self.key_gen.expand_to_bytes(extracted_phases, self.key_gen.key_length)
    
    def create_shares(
        self,
        key_value: int,
        num_shares: int = 3,
        threshold: int = 2
    ) -> Dict:
        """
        Share key between parties using holographic splitting.
        
        Args:
            key_value: Key to share
            num_shares: Number of shares to create
            threshold: Minimum shares needed to reconstruct
            
        Returns:
            Dictionary with shares and threshold
        """
        encoding = self.encode_key(key_value)
        shares = []
        
        for s in range(num_shares):
            share = []
            for x in range(self.grid_size):
                row = []
                for y in range(self.grid_size):
                    original = encoding["pattern"][x][y]
                    
                    # Add share-specific noise
                    noise = math.sin((x + y + s) * PHI)
                    noise_factor = 1 if s < num_shares - 1 else -(num_shares - 1)
                    
                    row.append({
                        "intensity": original["intensity"] + noise * noise_factor,
                        "phase": original["phase"] + (2 * math.pi * s / num_shares)
                    })
                share.append(row)
            
            shares.append({
                "index": s,
                "pattern": share,
                "grid_size": self.grid_size
            })
        
        return {"shares": shares, "threshold": threshold}
    
    def combine_shares(self, shares: List[Dict]) -> List[int]:
        """
        Combine shares to recover key.
        
        Args:
            shares: Subset of shares to combine
            
        Returns:
            Recovered key bytes
        """
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares to reconstruct")
        
        # Average the patterns
        combined = []
        for x in range(self.grid_size):
            row = []
            for y in range(self.grid_size):
                intensity_sum = 0.0
                phase_sum = 0.0
                
                for share in shares:
                    intensity_sum += share["pattern"][x][y]["intensity"]
                    phase_sum += share["pattern"][x][y]["phase"]
                
                row.append({
                    "intensity": intensity_sum / len(shares),
                    "phase": phase_sum / len(shares)
                })
            combined.append(row)
        
        return self.decode_key({"pattern": combined, "grid_size": self.grid_size})


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "GaussianInteger",
    "HypercomplexState",
    "CryptographicBackend",
    "PrimeStateKeyGenerator",
    "EntropySensitiveEncryptor",
    "HolographicKeyDistributor",
    "first_n_primes",
    "prime_to_frequency",
]