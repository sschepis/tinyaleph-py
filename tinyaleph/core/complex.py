"""
Complex number operations with full operator overloading.

This module provides a Complex class that integrates with Pydantic
for validation and serialization, while providing full arithmetic
operations and utility methods.

Uses pure Python math by default. Numpy is optional and only needed
for numpy interoperability methods.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union


@dataclass
class Complex:
    """
    Complex number with full arithmetic operations.
    
    Represents z = real + imag*i where i² = -1.
    
    Attributes:
        real: Real part
        imag: Imaginary part
    
    Examples:
        >>> z = Complex(3.0, 4.0)
        >>> z.norm()
        5.0
        >>> z.phase()
        0.9272952180016122
    """
    
    real: float = 0.0
    imag: float = 0.0
    
    # Aliases for compatibility
    @property
    def re(self) -> float:
        return self.real
    
    @property
    def im(self) -> float:
        return self.imag
    
    @classmethod
    def from_polar(cls, r: float, theta: float) -> Complex:
        """
        Create complex number from polar form.
        
        Args:
            r: Magnitude (radius)
            theta: Phase angle in radians
            
        Returns:
            Complex number r * e^(i*theta)
        """
        return cls(r * math.cos(theta), r * math.sin(theta))
    
    @classmethod
    def zero(cls) -> Complex:
        """Return the additive identity (0 + 0i)."""
        return cls(0.0, 0.0)
    
    @classmethod
    def one(cls) -> Complex:
        """Return the multiplicative identity (1 + 0i)."""
        return cls(1.0, 0.0)
    
    @classmethod
    def i(cls) -> Complex:
        """Return the imaginary unit (0 + 1i)."""
        return cls(0.0, 1.0)
    
    def __add__(self, other: Union[Complex, float, int]) -> Complex:
        """Add two complex numbers or complex + real."""
        if isinstance(other, (int, float)):
            return Complex(self.real + other, self.imag)
        return Complex(self.real + other.real, self.imag + other.imag)
    
    def __radd__(self, other: Union[float, int]) -> Complex:
        """Right addition for real + complex."""
        return self.__add__(other)
    
    def __sub__(self, other: Union[Complex, float, int]) -> Complex:
        """Subtract two complex numbers or complex - real."""
        if isinstance(other, (int, float)):
            return Complex(self.real - other, self.imag)
        return Complex(self.real - other.real, self.imag - other.imag)
    
    def __rsub__(self, other: Union[float, int]) -> Complex:
        """Right subtraction for real - complex."""
        return Complex(other - self.real, -self.imag)
    
    def __mul__(self, other: Union[Complex, float, int]) -> Complex:
        """
        Multiply two complex numbers or complex * scalar.
        
        For complex numbers: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        """
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.imag * other)
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __rmul__(self, other: Union[float, int]) -> Complex:
        """Right multiplication for scalar * complex."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[Complex, float, int]) -> Complex:
        """
        Divide two complex numbers or complex / scalar.
        
        For complex division: z1/z2 = z1 * conj(z2) / |z2|²
        """
        if isinstance(other, (int, float)):
            return Complex(self.real / other, self.imag / other)
        denom = other.magnitude_squared()
        if denom < 1e-15:
            raise ZeroDivisionError("Division by zero complex number")
        return Complex(
            (self.real * other.real + self.imag * other.imag) / denom,
            (self.imag * other.real - self.real * other.imag) / denom
        )
    
    def __neg__(self) -> Complex:
        """Negate the complex number."""
        return Complex(-self.real, -self.imag)
    
    def __abs__(self) -> float:
        """Return the magnitude (norm)."""
        return self.magnitude()
    
    def __eq__(self, other: object) -> bool:
        """Check equality with tolerance."""
        if isinstance(other, Complex):
            return abs(self.real - other.real) < 1e-10 and abs(self.imag - other.imag) < 1e-10
        if isinstance(other, (int, float)):
            return abs(self.real - other) < 1e-10 and abs(self.imag) < 1e-10
        return False
    
    def conjugate(self) -> Complex:
        """
        Return the complex conjugate.
        
        conj(a + bi) = a - bi
        """
        return Complex(self.real, -self.imag)
    
    def conj(self) -> Complex:
        """Alias for conjugate()."""
        return self.conjugate()
    
    def magnitude_squared(self) -> float:
        """
        Return the squared magnitude.
        
        |z|² = real² + imag²
        """
        return self.real ** 2 + self.imag ** 2
    
    def norm2(self) -> float:
        """Alias for magnitude_squared()."""
        return self.magnitude_squared()
    
    def magnitude(self) -> float:
        """
        Return the magnitude (modulus).
        
        |z| = √(real² + imag²)
        """
        return math.sqrt(self.magnitude_squared())
    
    def norm(self) -> float:
        """Alias for magnitude()."""
        return self.magnitude()
    
    def phase(self) -> float:
        """
        Return the phase angle in radians.
        
        arg(z) = atan2(imag, real)
        """
        return math.atan2(self.imag, self.real)
    
    def exp(self) -> Complex:
        """
        Return e^z using Euler's formula.
        
        e^(a + bi) = e^a * (cos(b) + i*sin(b))
        """
        ea = math.exp(self.real)
        return Complex(ea * math.cos(self.imag), ea * math.sin(self.imag))
    
    def log(self) -> Complex:
        """
        Return the principal value of the natural logarithm.
        
        log(z) = log|z| + i*arg(z)
        """
        return Complex(math.log(self.magnitude()), self.phase())
    
    def sqrt(self) -> Complex:
        """Return the principal square root."""
        r = self.magnitude()
        theta = self.phase()
        return Complex.from_polar(math.sqrt(r), theta / 2)
    
    def pow(self, n: Union[int, float, Complex]) -> Complex:
        """
        Raise to a power.
        
        z^n = e^(n * log(z))
        """
        if isinstance(n, (int, float)):
            if n == 0:
                return Complex.one()
            if n == 1:
                return Complex(self.real, self.imag)
            if isinstance(n, int) and n > 0:
                result = Complex.one()
                for _ in range(n):
                    result = result * self
                return result
            # Use log for non-integer powers
            log_z = self.log()
            return (log_z * n).exp()
        else:
            # Complex exponent
            log_z = self.log()
            return (n * log_z).exp()
    
    def to_builtin(self) -> complex:
        """Convert to Python's built-in complex type."""
        return complex(self.real, self.imag)
    
    @classmethod
    def from_builtin(cls, z: complex) -> Complex:
        """Create from Python's built-in complex type."""
        return cls(z.real, z.imag)
    
    def to_numpy(self):
        """Convert to numpy complex number. Requires numpy."""
        try:
            import numpy as np
            return np.complex128(self.real + 1j * self.imag)
        except ImportError:
            raise ImportError("numpy required for to_numpy()")
    
    @classmethod
    def from_numpy(cls, z) -> Complex:
        """Create from numpy complex number."""
        return cls(float(z.real), float(z.imag))
    
    def __repr__(self) -> str:
        """String representation."""
        sign = "+" if self.imag >= 0 else "-"
        return f"Complex({self.real:.6g} {sign} {abs(self.imag):.6g}i)"
    
    def __str__(self) -> str:
        """Human-readable string."""
        if abs(self.imag) < 1e-10:
            return f"{self.real:.6g}"
        if abs(self.real) < 1e-10:
            return f"{self.imag:.6g}i"
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.6g} {sign} {abs(self.imag):.6g}i"
    
    def __hash__(self) -> int:
        """Hash based on real and imag parts."""
        return hash((round(self.real, 10), round(self.imag, 10)))
    
    # Method aliases for compatibility with JavaScript-style APIs
    def add(self, other: Union[Complex, float, int]) -> Complex:
        """Method alias for __add__."""
        return self.__add__(other)
    
    def sub(self, other: Union[Complex, float, int]) -> Complex:
        """Method alias for __sub__."""
        return self.__sub__(other)
    
    def mul(self, other: Union[Complex, float, int]) -> Complex:
        """Method alias for __mul__."""
        return self.__mul__(other)
    
    def div(self, other: Union[Complex, float, int]) -> Complex:
        """Method alias for __truediv__."""
        return self.__truediv__(other)
    
    def norm_sq(self) -> float:
        """Alias for magnitude_squared()."""
        return self.magnitude_squared()