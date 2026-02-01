"""
Prime number utilities and generators.

Provides efficient prime number operations used throughout TinyAleph
for prime Hilbert space construction and resonance computing.

Uses pure Python by default. Numpy is optional and used for optimizations
when available.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Dict, Iterator, Tuple


@lru_cache(maxsize=1024)
def is_prime(n: int) -> bool:
    """
    Check if n is prime using trial division.
    
    Args:
        n: Integer to check
        
    Returns:
        True if n is prime
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def prime_sieve(limit: int) -> List[int]:
    """
    Generate all primes up to limit using Sieve of Eratosthenes.
    
    Args:
        limit: Upper bound (inclusive)
        
    Returns:
        List of primes up to limit
    """
    if limit < 2:
        return []
    
    # Pure Python sieve
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    
    return [i for i, is_p in enumerate(sieve) if is_p]


def first_n_primes(n: int) -> List[int]:
    """
    Generate the first n prime numbers.
    
    Args:
        n: Number of primes to generate
        
    Returns:
        List of first n primes
    """
    if n <= 0:
        return []
    
    # Estimate upper bound using prime number theorem
    if n < 6:
        limit = 15
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n)) + 2))
    
    primes = prime_sieve(limit)
    while len(primes) < n:
        limit *= 2
        primes = prime_sieve(limit)
    
    return primes[:n]


def prime_generator() -> Iterator[int]:
    """
    Infinite generator of prime numbers.
    
    Yields:
        Prime numbers in ascending order
    """
    yield 2
    n = 3
    while True:
        if is_prime(n):
            yield n
        n += 2


@lru_cache(maxsize=1024)
def factorize(n: int) -> Dict[int, int]:
    """
    Compute the prime factorization of n.
    
    Args:
        n: Positive integer to factorize
        
    Returns:
        Dictionary mapping prime factors to their exponents
        
    Examples:
        >>> factorize(60)
        {2: 2, 3: 1, 5: 1}
    """
    if n <= 1:
        return {}
    
    factors: Dict[int, int] = {}
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors


# Alias for backward compatibility
prime_factorization = factorize


def euler_totient(n: int) -> int:
    """
    Compute Euler's totient function φ(n).
    
    φ(n) = number of integers in [1, n] coprime to n
    
    Uses the formula: φ(n) = n * Π(1 - 1/p) for all prime p dividing n
    """
    if n <= 1:
        return n
    
    result = n
    factors = factorize(n)
    
    for p in factors:
        result -= result // p
    
    return result


def mobius(n: int) -> int:
    """
    Compute the Möbius function μ(n).
    
    μ(n) = 0 if n has a squared prime factor
    μ(n) = (-1)^k if n is product of k distinct primes
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    factors = factorize(n)
    
    for exp in factors.values():
        if exp > 1:
            return 0
    
    return (-1) ** len(factors)


# Pre-computed primes for fast lookup
_PRIME_CACHE: List[int] = []


def _ensure_primes(n: int) -> None:
    """Ensure at least n primes are cached."""
    global _PRIME_CACHE
    if len(_PRIME_CACHE) < n:
        _PRIME_CACHE = first_n_primes(max(n, 100))


def nth_prime(n: int) -> int:
    """
    Return the n-th prime number (1-indexed).
    
    Args:
        n: Index of prime to return (1 = 2, 2 = 3, etc.)
        
    Returns:
        The n-th prime number
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    _ensure_primes(n)
    return _PRIME_CACHE[n - 1]


@lru_cache(maxsize=1024)
def prime_index(p: int) -> int:
    """
    Return the index of prime p (1-indexed).
    
    Args:
        p: A prime number
        
    Returns:
        Index of p in sequence of primes (2 -> 1, 3 -> 2, 5 -> 3, ...)
        
    Raises:
        ValueError: If p is not prime
    """
    if not is_prime(p):
        raise ValueError(f"{p} is not prime")
    
    # For small primes, count directly
    if p <= 2:
        return 1
    
    count = 1  # Count 2
    n = 3
    while n < p:
        if is_prime(n):
            count += 1
        n += 2
    
    return count + 1


def next_prime(n: int) -> int:
    """
    Return the smallest prime greater than n.
    
    Args:
        n: Starting point
        
    Returns:
        Next prime after n
    """
    if n < 2:
        return 2
    
    candidate = n + 1 if n % 2 == 0 else n + 2
    if candidate == 3:
        return 3
    
    while not is_prime(candidate):
        candidate += 2
    
    return candidate


def prev_prime(n: int) -> int:
    """
    Return the largest prime less than n.
    
    Args:
        n: Starting point
        
    Returns:
        Previous prime before n
        
    Raises:
        ValueError: If no prime exists less than n
    """
    if n <= 2:
        raise ValueError("No prime less than 2")
    if n == 3:
        return 2
    
    candidate = n - 1 if n % 2 == 0 else n - 2
    
    while candidate > 1 and not is_prime(candidate):
        candidate -= 2
    
    if candidate < 2:
        return 2
    
    return candidate


def prime_pi(n: int) -> int:
    """
    Count primes up to n (prime counting function π(n)).
    
    Args:
        n: Upper bound
        
    Returns:
        Number of primes ≤ n
    """
    if n < 2:
        return 0
    return len(prime_sieve(n))


def is_prime_power(n: int) -> Tuple[bool, int, int]:
    """
    Check if n is a prime power p^k.
    
    Returns:
        (is_prime_power, prime, exponent)
        If not a prime power, returns (False, 0, 0)
    """
    if n <= 1:
        return (False, 0, 0)
    
    factors = factorize(n)
    
    if len(factors) == 1:
        p, k = next(iter(factors.items()))
        return (True, p, k)
    
    return (False, 0, 0)


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)


def coprime(a: int, b: int) -> bool:
    """Check if a and b are coprime (gcd = 1)."""
    return gcd(a, b) == 1


def prime_weights(primes: List[int]) -> List[float]:
    """
    Compute normalized weights for a list of primes.
    
    Uses log(p) normalization for frequency-based weighting.
    
    Args:
        primes: List of prime numbers
        
    Returns:
        Normalized weight list
    """
    if not primes:
        return []
    
    logs = [math.log(p) for p in primes]
    total = sum(logs)
    
    if total < 1e-10:
        return [1.0 / len(primes)] * len(primes)
    
    return [log / total for log in logs]


# Initialize prime cache at import
_ensure_primes(100)


def get_prime(index: int) -> int:
    """
    Get prime by index (0-indexed) with caching.
    
    Uses pre-computed list for fast lookup.
    """
    _ensure_primes(index + 1)
    return _PRIME_CACHE[index]


# Aliases for compatibility
def primes_up_to(limit: int) -> List[int]:
    """
    Alias for prime_sieve - get all primes up to limit.
    
    Args:
        limit: Upper bound (inclusive)
        
    Returns:
        List of primes up to limit
    """
    return prime_sieve(limit)


def prime_to_frequency(p: int, base_freq: float = 1.0) -> float:
    """
    Convert a prime to a frequency using logarithmic spacing.
    
    Uses the formula: f = base_freq * log(p) / log(2)
    
    This creates frequencies that are spaced by musical intervals,
    with the ratio between adjacent primes being roughly constant
    on a logarithmic scale.
    
    Args:
        p: Prime number
        base_freq: Base frequency (default 1.0 Hz)
        
    Returns:
        Frequency corresponding to the prime
    """
    if p < 2:
        return base_freq
    return base_freq * math.log(p) / math.log(2)