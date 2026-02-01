"""
Generic Cayley-Dickson construction for 2^n dimensional algebras.

The Cayley-Dickson construction generates a sequence of algebras:
- 2^0 = 1: Real numbers ℝ
- 2^1 = 2: Complex numbers ℂ
- 2^2 = 4: Quaternions ℍ
- 2^3 = 8: Octonions 𝕆
- 2^4 = 16: Sedenions 𝕊
- 2^n: General hypercomplex numbers

Each step loses some algebraic property:
- Reals: Ordered field
- Complex: Not ordered, but commutative
- Quaternions: Not commutative, but associative
- Octonions: Not associative, but alternative
- Sedenions: Not alternative, but power-associative
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from functools import lru_cache
from typing import Union


class Hypercomplex:
    """
    Cayley-Dickson algebra of dimension 2^n.
    
    This class provides a generic implementation of hypercomplex numbers
    using the Cayley-Dickson construction. It supports arbitrary power-of-2
    dimensions.
    
    Attributes:
        dim: Dimension of the algebra (must be power of 2)
        c: Component array of length dim
    
    Examples:
        >>> # Create a sedenion (16D)
        >>> s = Hypercomplex(16, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        >>> s.norm()
        1.0
        
        >>> # Create an octonion (8D)
        >>> o = Hypercomplex(8, np.ones(8) / np.sqrt(8))
        >>> o.entropy()  # Maximum entropy for uniform distribution
    """
    
    __slots__ = ('dim', 'c')
    
    def __init__(self, dim: int, components: NDArray | None = None):
        """
        Initialize a hypercomplex number.
        
        Args:
            dim: Dimension (must be power of 2: 1, 2, 4, 8, 16, ...)
            components: Array of components (defaults to zero vector)
            
        Raises:
            ValueError: If dim is not a power of 2
        """
        if not (dim > 0 and (dim & (dim - 1)) == 0):
            raise ValueError(f"Dimension must be power of 2, got {dim}")
        
        self.dim = dim
        if components is not None:
            if len(components) != dim:
                raise ValueError(f"Components length {len(components)} != dimension {dim}")
            self.c = np.asarray(components, dtype=np.float64)
        else:
            self.c = np.zeros(dim, dtype=np.float64)
    
    @classmethod
    def basis(cls, dim: int, index: int) -> Hypercomplex:
        """
        Create basis element e_i.
        
        Args:
            dim: Dimension of the algebra
            index: Basis element index (0 to dim-1)
            
        Returns:
            Hypercomplex number with 1 in position index, 0 elsewhere
        """
        if not 0 <= index < dim:
            raise ValueError(f"Index {index} out of range [0, {dim})")
        c = np.zeros(dim)
        c[index] = 1.0
        return cls(dim, c)
    
    @classmethod
    def real(cls, dim: int, value: float = 1.0) -> Hypercomplex:
        """Create a real number (only e_0 component)."""
        c = np.zeros(dim)
        c[0] = value
        return cls(dim, c)
    
    @classmethod
    def random(cls, dim: int, normalize: bool = True) -> Hypercomplex:
        """
        Create random hypercomplex number.
        
        Args:
            dim: Dimension of the algebra
            normalize: If True, return unit hypercomplex
        """
        c = np.random.randn(dim)
        if normalize:
            c = c / np.linalg.norm(c)
        return cls(dim, c)
    
    @staticmethod
    @lru_cache(maxsize=2048)
    def _multiply_indices(dim: int, i: int, j: int) -> tuple[int, int]:
        """
        Get (result_index, sign) for e_i * e_j.
        
        This implements the Cayley-Dickson multiplication table recursively.
        Uses caching for efficiency since the table is accessed repeatedly.
        
        Returns:
            (k, s) where e_i * e_j = s * e_k, with s ∈ {-1, +1}
        """
        # Base cases
        if i == 0:
            return (j, 1)
        if j == 0:
            return (i, 1)
        if i == j:
            return (0, -1)
        
        # Recursive Cayley-Dickson construction
        half = dim // 2
        
        if i < half and j < half:
            # Both in lower half: use smaller algebra multiplication
            return Hypercomplex._multiply_indices(half, i, j)
        elif i < half:
            # i in lower, j in upper
            k, s = Hypercomplex._multiply_indices(half, i, j - half)
            return (k + half, s)
        elif j < half:
            # i in upper, j in lower
            k, s = Hypercomplex._multiply_indices(half, i - half, j)
            return (k + half, s)
        else:
            # Both in upper half
            k, s = Hypercomplex._multiply_indices(half, j - half, i - half)
            return (k, -s)
    
    def __add__(self, other: Union[Hypercomplex, float, int]) -> Hypercomplex:
        """Component-wise addition."""
        if isinstance(other, (int, float)):
            result = self.c.copy()
            result[0] += other
            return Hypercomplex(self.dim, result)
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} != {other.dim}")
        return Hypercomplex(self.dim, self.c + other.c)
    
    def __radd__(self, other: Union[float, int]) -> Hypercomplex:
        return self.__add__(other)
    
    def __sub__(self, other: Union[Hypercomplex, float, int]) -> Hypercomplex:
        """Component-wise subtraction."""
        if isinstance(other, (int, float)):
            result = self.c.copy()
            result[0] -= other
            return Hypercomplex(self.dim, result)
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} != {other.dim}")
        return Hypercomplex(self.dim, self.c - other.c)
    
    def __mul__(self, other: Union[Hypercomplex, float, int]) -> Hypercomplex:
        """
        Cayley-Dickson multiplication.
        
        Note: For dim >= 8 (octonions), multiplication is NOT associative.
        For dim >= 16 (sedenions), there are zero divisors.
        """
        if isinstance(other, (int, float)):
            return Hypercomplex(self.dim, self.c * other)
        
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} != {other.dim}")
        
        result = np.zeros(self.dim)
        for i in range(self.dim):
            if abs(self.c[i]) < 1e-15:
                continue
            for j in range(self.dim):
                if abs(other.c[j]) < 1e-15:
                    continue
                k, s = self._multiply_indices(self.dim, i, j)
                result[k] += s * self.c[i] * other.c[j]
        
        return Hypercomplex(self.dim, result)
    
    def __rmul__(self, other: Union[float, int]) -> Hypercomplex:
        """Right multiplication by scalar."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[Hypercomplex, float, int]) -> Hypercomplex:
        """Division by scalar or hypercomplex number."""
        if isinstance(other, (int, float)):
            return Hypercomplex(self.dim, self.c / other)
        # h1 / h2 = h1 * h2^(-1)
        return self * other.inverse()
    
    def __neg__(self) -> Hypercomplex:
        """Negate all components."""
        return Hypercomplex(self.dim, -self.c)
    
    def __abs__(self) -> float:
        """Return the norm."""
        return self.norm()
    
    def __eq__(self, other: object) -> bool:
        """Check equality with tolerance."""
        if isinstance(other, Hypercomplex):
            if self.dim != other.dim:
                return False
            return bool(np.allclose(self.c, other.c, atol=1e-10))
        return False
    
    def __getitem__(self, index: int) -> float:
        """Get component by index."""
        return float(self.c[index])
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set component by index."""
        self.c[index] = value
    
    def conj(self) -> Hypercomplex:
        """
        Return the conjugate.
        
        conj(a + Σ b_i e_i) = a - Σ b_i e_i
        """
        result = np.zeros(self.dim)
        result[0] = self.c[0]
        result[1:] = -self.c[1:]
        return Hypercomplex(self.dim, result)
    
    def norm2(self) -> float:
        """Return the squared norm: |h|² = h * conj(h)."""
        return float(np.dot(self.c, self.c))
    
    def norm(self) -> float:
        """Return the norm: |h| = √(Σ c_i²)."""
        return float(np.sqrt(self.norm2()))
    
    def normalize(self) -> Hypercomplex:
        """Return unit hypercomplex (norm = 1)."""
        n = self.norm()
        if n < 1e-10:
            return Hypercomplex.real(self.dim, 1.0)
        return Hypercomplex(self.dim, self.c / n)
    
    def inverse(self) -> Hypercomplex:
        """
        Return the multiplicative inverse.
        
        h^(-1) = conj(h) / |h|²
        
        Warning: For dim >= 16 (sedenions), there exist zero divisors
        that have no inverse.
        """
        n2 = self.norm2()
        if n2 < 1e-15:
            raise ZeroDivisionError("Cannot invert zero hypercomplex")
        return Hypercomplex(self.dim, self.conj().c / n2)
    
    def entropy(self) -> float:
        """
        Calculate Shannon entropy of the component distribution.
        
        H = -Σ p_i log₂(p_i) where p_i = c_i² / |h|²
        
        Returns:
            Entropy in bits, range [0, log₂(dim)]
        """
        n = self.norm()
        if n < 1e-10:
            return 0.0
        
        probs = (self.c / n) ** 2
        probs = probs[probs > 1e-10]
        return float(-np.sum(probs * np.log2(probs)))
    
    def coherence(self) -> float:
        """
        Calculate coherence as inverse normalized entropy.
        
        C = 1 - H / log₂(dim)
        
        Returns:
            Coherence in [0, 1], where 1 = pure state, 0 = maximum entropy
        """
        max_entropy = np.log2(self.dim)
        if max_entropy < 1e-10:
            return 1.0
        return 1.0 - self.entropy() / max_entropy
    
    @property
    def real_part(self) -> float:
        """Return the real (scalar) part."""
        return float(self.c[0])
    
    @property
    def imag_parts(self) -> NDArray:
        """Return the imaginary parts as array."""
        return self.c[1:].copy()
    
    def is_pure(self, tol: float = 1e-10) -> bool:
        """Check if hypercomplex is pure (real part ≈ 0)."""
        return abs(self.c[0]) < tol
    
    def is_unit(self, tol: float = 1e-10) -> bool:
        """Check if hypercomplex is unit (|h| ≈ 1)."""
        return abs(self.norm() - 1.0) < tol
    
    def as_quaternion(self):
        """Convert to Quaternion if dim == 4."""
        if self.dim != 4:
            raise ValueError(f"Cannot convert dim={self.dim} to Quaternion")
        from tinyaleph.core.quaternion import Quaternion
        return Quaternion(w=self.c[0], x=self.c[1], y=self.c[2], z=self.c[3])
    
    def as_complex(self):
        """Convert to Complex if dim == 2."""
        if self.dim != 2:
            raise ValueError(f"Cannot convert dim={self.dim} to Complex")
        from tinyaleph.core.complex import Complex
        return Complex(re=self.c[0], im=self.c[1])
    
    @staticmethod
    def dim_name(dim: int) -> str:
        """Get the mathematical name for a given dimension."""
        names = {
            1: "Real",
            2: "Complex",
            4: "Quaternion",
            8: "Octonion",
            16: "Sedenion",
            32: "Pathion",
            64: "Chingon",
            128: "Routon",
            256: "Voudon",
        }
        return names.get(dim, f"{dim}D Hypercomplex")
    
    def __repr__(self) -> str:
        """String representation."""
        name = self.dim_name(self.dim)
        return f"{name}({self.c})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        parts = []
        if abs(self.c[0]) > 1e-10:
            parts.append(f"{self.c[0]:.4g}")
        
        for i in range(1, min(self.dim, 8)):  # Show first 8 components
            if abs(self.c[i]) > 1e-10:
                sign = "+" if self.c[i] > 0 and parts else ""
                parts.append(f"{sign}{self.c[i]:.4g}e_{i}")
        
        if self.dim > 8:
            nonzero = np.sum(np.abs(self.c[8:]) > 1e-10)
            if nonzero > 0:
                parts.append(f"... (+{nonzero} more)")
        
        return "".join(parts) if parts else "0"