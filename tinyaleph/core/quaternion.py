"""
Hamilton quaternion algebra with full operations.

Quaternions extend complex numbers to 4 dimensions with three imaginary units
i, j, k satisfying: i² = j² = k² = ijk = -1

Properties:
- Non-commutative multiplication
- Division algebra (every non-zero element has inverse)
- Unit quaternions represent 3D rotations

Uses pure Python math by default. Numpy is optional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union, List


@dataclass
class Quaternion:
    """
    Hamilton quaternion: q = w + xi + yj + zk
    
    Quaternions form a 4-dimensional associative normed division algebra
    over the real numbers.
    
    Attributes:
        w: Scalar (real) part
        i: i component (also accessible as x)
        j: j component (also accessible as y)
        k: k component (also accessible as z)
    
    Examples:
        >>> q = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity
        >>> q.norm()
        1.0
        >>> q1 = Quaternion(0, 1, 0, 0)  # Pure i
        >>> q2 = Quaternion(0, 0, 1, 0)  # Pure j
        >>> (q1 * q2).k  # i * j = k
        1.0
    """
    
    w: float = 1.0
    i: float = 0.0
    j: float = 0.0
    k: float = 0.0
    
    # Aliases for compatibility
    @property
    def x(self) -> float:
        return self.i
    
    @property
    def y(self) -> float:
        return self.j
    
    @property
    def z(self) -> float:
        return self.k
    
    @classmethod
    def identity(cls) -> Quaternion:
        """Return the multiplicative identity (1, 0, 0, 0)."""
        return cls(1.0, 0.0, 0.0, 0.0)
    
    @classmethod
    def zero(cls) -> Quaternion:
        """Return the additive identity (0, 0, 0, 0)."""
        return cls(0.0, 0.0, 0.0, 0.0)
    
    @classmethod
    def I(cls) -> Quaternion:
        """Return the i basis element."""
        return cls(0.0, 1.0, 0.0, 0.0)
    
    @classmethod
    def J(cls) -> Quaternion:
        """Return the j basis element."""
        return cls(0.0, 0.0, 1.0, 0.0)
    
    @classmethod
    def K(cls) -> Quaternion:
        """Return the k basis element."""
        return cls(0.0, 0.0, 0.0, 1.0)
    
    @classmethod
    def from_axis_angle(cls, axis: Tuple[float, float, float], angle: float) -> Quaternion:
        """
        Create rotation quaternion from axis-angle representation.
        
        Args:
            axis: Unit rotation axis (ax, ay, az)
            angle: Rotation angle in radians
            
        Returns:
            Unit quaternion representing the rotation
        """
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-10:
            return cls.identity()
        
        half_angle = angle / 2
        s = math.sin(half_angle) / norm
        return cls(
            w=math.cos(half_angle),
            i=ax * s,
            j=ay * s,
            k=az * s
        )
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        """
        Create rotation quaternion from Euler angles (ZYX convention).
        
        Args:
            roll: Rotation around X axis (radians)
            pitch: Rotation around Y axis (radians)
            yaw: Rotation around Z axis (radians)
            
        Returns:
            Unit quaternion representing the rotation
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        return cls(
            w=cr * cp * cy + sr * sp * sy,
            i=sr * cp * cy - cr * sp * sy,
            j=cr * sp * cy + sr * cp * sy,
            k=cr * cp * sy - sr * sp * cy
        )
    
    @classmethod
    def from_vector(cls, v: Tuple[float, float, float]) -> Quaternion:
        """Create pure quaternion from 3D vector (w=0)."""
        return cls(0.0, v[0], v[1], v[2])
    
    @classmethod
    def from_components(cls, components: List[float]) -> Quaternion:
        """Create from list [w, i, j, k]."""
        if len(components) != 4:
            raise ValueError("Need exactly 4 components")
        return cls(components[0], components[1], components[2], components[3])
    
    def to_list(self) -> List[float]:
        """Convert to list [w, i, j, k]."""
        return [self.w, self.i, self.j, self.k]
    
    def __add__(self, other: Union[Quaternion, float, int]) -> Quaternion:
        """Add two quaternions or quaternion + scalar."""
        if isinstance(other, (int, float)):
            return Quaternion(self.w + other, self.i, self.j, self.k)
        return Quaternion(
            self.w + other.w,
            self.i + other.i,
            self.j + other.j,
            self.k + other.k
        )
    
    def __radd__(self, other: Union[float, int]) -> Quaternion:
        """Right addition."""
        return self.__add__(other)
    
    def __sub__(self, other: Union[Quaternion, float, int]) -> Quaternion:
        """Subtract two quaternions or quaternion - scalar."""
        if isinstance(other, (int, float)):
            return Quaternion(self.w - other, self.i, self.j, self.k)
        return Quaternion(
            self.w - other.w,
            self.i - other.i,
            self.j - other.j,
            self.k - other.k
        )
    
    def __rsub__(self, other: Union[float, int]) -> Quaternion:
        """Right subtraction."""
        return Quaternion(other - self.w, -self.i, -self.j, -self.k)
    
    def __mul__(self, other: Union[Quaternion, float, int]) -> Quaternion:
        """
        Hamilton product (quaternion multiplication).
        
        For quaternions q1 = (w1, x1, y1, z1) and q2 = (w2, x2, y2, z2):
        q1 * q2 follows the rules i² = j² = k² = ijk = -1
        
        Note: Quaternion multiplication is NOT commutative.
        """
        if isinstance(other, (int, float)):
            return Quaternion(
                self.w * other,
                self.i * other,
                self.j * other,
                self.k * other
            )
        
        # Hamilton product
        return Quaternion(
            self.w * other.w - self.i * other.i - self.j * other.j - self.k * other.k,
            self.w * other.i + self.i * other.w + self.j * other.k - self.k * other.j,
            self.w * other.j - self.i * other.k + self.j * other.w + self.k * other.i,
            self.w * other.k + self.i * other.j - self.j * other.i + self.k * other.w
        )
    
    def __rmul__(self, other: Union[float, int]) -> Quaternion:
        """Right multiplication by scalar."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[Quaternion, float, int]) -> Quaternion:
        """Divide quaternion by scalar or another quaternion."""
        if isinstance(other, (int, float)):
            return Quaternion(
                self.w / other,
                self.i / other,
                self.j / other,
                self.k / other
            )
        # q1 / q2 = q1 * q2^(-1)
        return self * other.inverse()
    
    def __neg__(self) -> Quaternion:
        """Negate all components."""
        return Quaternion(-self.w, -self.i, -self.j, -self.k)
    
    def __abs__(self) -> float:
        """Return the norm."""
        return self.norm()
    
    def __eq__(self, other: object) -> bool:
        """Check equality with tolerance."""
        if isinstance(other, Quaternion):
            return (
                abs(self.w - other.w) < 1e-10 and
                abs(self.i - other.i) < 1e-10 and
                abs(self.j - other.j) < 1e-10 and
                abs(self.k - other.k) < 1e-10
            )
        return False
    
    def __hash__(self) -> int:
        """Hash based on components."""
        return hash((
            round(self.w, 10),
            round(self.i, 10),
            round(self.j, 10),
            round(self.k, 10)
        ))
    
    def conjugate(self) -> Quaternion:
        """
        Return the quaternion conjugate.
        
        conj(w + xi + yj + zk) = w - xi - yj - zk
        
        For unit quaternions, conj(q) = q^(-1)
        """
        return Quaternion(self.w, -self.i, -self.j, -self.k)
    
    def conj(self) -> Quaternion:
        """Alias for conjugate()."""
        return self.conjugate()
    
    def norm_squared(self) -> float:
        """Return the squared norm (norm²)."""
        return self.w ** 2 + self.i ** 2 + self.j ** 2 + self.k ** 2
    
    def norm2(self) -> float:
        """Alias for norm_squared()."""
        return self.norm_squared()
    
    def norm(self) -> float:
        """Return the quaternion norm (magnitude)."""
        return math.sqrt(self.norm_squared())
    
    def normalize(self) -> Quaternion:
        """
        Return a unit quaternion (norm = 1).
        
        Returns identity if norm is too small.
        """
        n = self.norm()
        if n < 1e-10:
            return Quaternion.identity()
        return self / n
    
    def normalized(self) -> Quaternion:
        """Alias for normalize()."""
        return self.normalize()
    
    def inverse(self) -> Quaternion:
        """
        Return the multiplicative inverse.
        
        q^(-1) = conj(q) / |q|²
        """
        n2 = self.norm_squared()
        if n2 < 1e-15:
            raise ZeroDivisionError("Cannot invert zero quaternion")
        return Quaternion(
            self.w / n2,
            -self.i / n2,
            -self.j / n2,
            -self.k / n2
        )
    
    def commutator(self, other: Quaternion) -> Quaternion:
        """
        Compute the commutator [q1, q2] = q1*q2 - q2*q1.
        
        The commutator measures non-commutativity:
        - Zero if q1 and q2 commute
        - Non-zero otherwise
        """
        return self * other - other * self
    
    def dot(self, other: Quaternion) -> float:
        """
        Compute the dot product (as 4D vectors).
        
        Useful for measuring similarity between quaternions.
        """
        return self.w * other.w + self.i * other.i + self.j * other.j + self.k * other.k
    
    def slerp(self, other: Quaternion, t: float) -> Quaternion:
        """
        Spherical linear interpolation between unit quaternions.
        
        Args:
            other: Target quaternion
            t: Interpolation parameter in [0, 1]
            
        Returns:
            Interpolated unit quaternion
        """
        dot = self.dot(other)
        
        # Handle opposite quaternions (same rotation, different sign)
        q2 = other
        if dot < 0:
            q2 = -other
            dot = -dot
        
        # For very similar quaternions, use linear interpolation
        if dot > 0.9995:
            result = Quaternion(
                self.w + t * (q2.w - self.w),
                self.i + t * (q2.i - self.i),
                self.j + t * (q2.j - self.j),
                self.k + t * (q2.k - self.k)
            )
            return result.normalize()
        
        # Spherical interpolation
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        s1 = math.sin((1 - t) * theta) / sin_theta
        s2 = math.sin(t * theta) / sin_theta
        
        return Quaternion(
            s1 * self.w + s2 * q2.w,
            s1 * self.i + s2 * q2.i,
            s1 * self.j + s2 * q2.j,
            s1 * self.k + s2 * q2.k
        )
    
    def rotate_vector(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Rotate a 3D vector using this quaternion.
        
        v' = q * v * q^(-1) where v is embedded as pure quaternion
        
        Args:
            v: 3D vector (x, y, z)
            
        Returns:
            Rotated vector
        """
        q_v = Quaternion.from_vector(v)
        q_rotated = self * q_v * self.inverse()
        return (q_rotated.i, q_rotated.j, q_rotated.k)
    
    def to_euler(self) -> Tuple[float, float, float]:
        """
        Convert to Euler angles (ZYX convention).
        
        Returns:
            (roll, pitch, yaw) in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.i + self.j * self.k)
        cosr_cosp = 1 - 2 * (self.i * self.i + self.j * self.j)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.j - self.k * self.i)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.k + self.i * self.j)
        cosy_cosp = 1 - 2 * (self.j * self.j + self.k * self.k)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)
    
    def to_rotation_matrix(self) -> List[List[float]]:
        """
        Convert unit quaternion to 3x3 rotation matrix.
        
        Returns:
            3x3 list of lists representing the rotation
        """
        n = self.normalize()
        
        return [
            [1 - 2*(n.j**2 + n.k**2), 2*(n.i*n.j - n.k*n.w), 2*(n.i*n.k + n.j*n.w)],
            [2*(n.i*n.j + n.k*n.w), 1 - 2*(n.i**2 + n.k**2), 2*(n.j*n.k - n.i*n.w)],
            [2*(n.i*n.k - n.j*n.w), 2*(n.j*n.k + n.i*n.w), 1 - 2*(n.i**2 + n.j**2)]
        ]
    
    def to_numpy_matrix(self):
        """Convert to numpy 3x3 rotation matrix. Requires numpy."""
        try:
            import numpy as np
            return np.array(self.to_rotation_matrix())
        except ImportError:
            raise ImportError("numpy required for to_numpy_matrix()")
    
    @property
    def vector_part(self) -> Tuple[float, float, float]:
        """Return the vector (imaginary) part as a tuple."""
        return (self.i, self.j, self.k)
    
    @property
    def scalar_part(self) -> float:
        """Return the scalar (real) part."""
        return self.w
    
    def is_pure(self, tol: float = 1e-10) -> bool:
        """Check if quaternion is pure (w ≈ 0)."""
        return abs(self.w) < tol
    
    def is_unit(self, tol: float = 1e-10) -> bool:
        """Check if quaternion is unit (|q| ≈ 1)."""
        return abs(self.norm() - 1.0) < tol
    
    def exp(self) -> Quaternion:
        """
        Compute quaternion exponential.
        
        exp(q) = exp(w) * (cos|v| + v/|v| * sin|v|)
        where v = (i, j, k) is the vector part
        """
        v_norm = math.sqrt(self.i**2 + self.j**2 + self.k**2)
        exp_w = math.exp(self.w)
        
        if v_norm < 1e-10:
            return Quaternion(exp_w, 0, 0, 0)
        
        cos_v = math.cos(v_norm)
        sin_v = math.sin(v_norm)
        scale = exp_w * sin_v / v_norm
        
        return Quaternion(
            exp_w * cos_v,
            self.i * scale,
            self.j * scale,
            self.k * scale
        )
    
    def log(self) -> Quaternion:
        """
        Compute quaternion logarithm.
        
        log(q) = log|q| + v/|v| * arccos(w/|q|)
        """
        q_norm = self.norm()
        v_norm = math.sqrt(self.i**2 + self.j**2 + self.k**2)
        
        if q_norm < 1e-10:
            raise ValueError("Cannot compute log of zero quaternion")
        
        if v_norm < 1e-10:
            return Quaternion(math.log(q_norm), 0, 0, 0)
        
        theta = math.acos(max(-1, min(1, self.w / q_norm)))
        scale = theta / v_norm
        
        return Quaternion(
            math.log(q_norm),
            self.i * scale,
            self.j * scale,
            self.k * scale
        )
    
    def pow(self, t: float) -> Quaternion:
        """Raise quaternion to power t: q^t = exp(t * log(q))."""
        if self.norm() < 1e-10:
            return Quaternion.zero()
        return (self.log() * t).exp()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Quaternion(w={self.w:.6g}, i={self.i:.6g}, j={self.j:.6g}, k={self.k:.6g})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        parts = []
        if abs(self.w) > 1e-10:
            parts.append(f"{self.w:.4g}")
        if abs(self.i) > 1e-10:
            sign = "+" if self.i > 0 and parts else ""
            parts.append(f"{sign}{self.i:.4g}i")
        if abs(self.j) > 1e-10:
            sign = "+" if self.j > 0 and parts else ""
            parts.append(f"{sign}{self.j:.4g}j")
        if abs(self.k) > 1e-10:
            sign = "+" if self.k > 0 and parts else ""
            parts.append(f"{sign}{self.k:.4g}k")
        return "".join(parts) if parts else "0"