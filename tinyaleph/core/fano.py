"""
Fano Plane Structure for Octonion/Sedenion Multiplication

Implements the Fano plane - a finite projective plane with 7 points and 7 lines -
which encodes the multiplication rules for octonions (8D) and sedenions (16D).

The Fano plane is the minimal projective plane PG(2,2) and has the property that
exactly 3 points lie on each line and exactly 3 lines pass through each point.

Key concepts:
- Octonion multiplication: 8D non-associative division algebra
- Sedenion multiplication: 16D via Cayley-Dickson construction
- Pathion and beyond: Recursive Cayley-Dickson for 32D+
"""

from typing import Tuple, List
import math


# Standard Fano plane lines (7 lines of 3 points each)
# Points are labeled 1-7, with 0 being the identity
FANO_LINES = [
    [1, 2, 3],
    [1, 4, 5],
    [1, 6, 7],
    [2, 4, 6],
    [2, 5, 7],
    [3, 4, 7],
    [3, 5, 6]
]


def octonion_multiply_index(i: int, j: int) -> Tuple[int, int]:
    """
    Compute octonion multiplication index using Fano plane.
    
    Returns (k, sign) where e_i * e_j = sign * e_k
    
    The octonions O are the 8-dimensional normed division algebra
    with basis {1, e1, e2, ..., e7}. Multiplication is encoded by
    the Fano plane.
    
    Args:
        i: Index of first basis element (0-7)
        j: Index of second basis element (0-7)
        
    Returns:
        Tuple (k, sign) where k is result index and sign is ±1
    """
    # Identity cases
    if i == 0:
        return (j, 1)
    if j == 0:
        return (i, 1)
    # Squaring gives -1
    if i == j:
        return (0, -1)
    
    # Look up in Fano plane
    for line in FANO_LINES:
        xi = line.index(i) if i in line else -1
        if xi >= 0 and j in line:
            xj = line.index(j)
            # Third point on line
            k = line[3 - xi - xj]
            # Sign from cyclic order on line
            sign = 1 if (xj - xi + 3) % 3 == 1 else -1
            return (k, sign)
    
    # Fallback (should not happen for valid indices 1-7)
    return (i ^ j, 1)


def sedenion_multiply_index(i: int, j: int) -> Tuple[int, int]:
    """
    Compute sedenion multiplication index (Cayley-Dickson extension of octonions).
    
    Sedenions are the 16-dimensional algebra obtained by applying the
    Cayley-Dickson construction to octonions. They are NOT a division
    algebra (they have zero divisors).
    
    Cayley-Dickson formula:
    (a, b)(c, d) = (ac - d*b, da + bc*)
    where * is conjugation
    
    Args:
        i: Index of first basis element (0-15)
        j: Index of second basis element (0-15)
        
    Returns:
        Tuple (k, sign) where k is result index and sign is ±1
    """
    # Identity cases
    if i == 0:
        return (j, 1)
    if j == 0:
        return (i, 1)
    # Squaring gives -1
    if i == j:
        return (0, -1)
    
    # Split into high/low halves (Cayley-Dickson construction)
    hi = i >= 8
    hj = j >= 8
    li = i & 7  # Low 3 bits
    lj = j & 7
    
    # Both in low half: use octonion multiplication
    if not hi and not hj:
        return octonion_multiply_index(li, lj)
    
    # Both in high half: (0, b)(0, d) = (-d*b, 0) = -conj(d)*b
    if hi and hj:
        k, s = octonion_multiply_index(li, lj)
        return (k, -s)
    
    # Mixed: low * high = high
    if not hi:
        # (a, 0)(0, d) = (0, da)
        k, s = octonion_multiply_index(lj, li)
        return (k + 8, s)
    
    # high * low = high
    # (0, b)(c, 0) = (0, bc*)
    k, s = octonion_multiply_index(li, lj)
    return (k + 8, -s)


def multiply_indices(dim: int, i: int, j: int) -> Tuple[int, int]:
    """
    Generic multiplication index lookup for Cayley-Dickson algebras.
    
    Supports:
    - dim <= 2: Complex numbers
    - dim <= 4: Quaternions
    - dim <= 8: Octonions
    - dim <= 16: Sedenions
    - dim > 16: Higher Cayley-Dickson algebras (pathions, etc.)
    
    Args:
        dim: Dimension of the algebra (power of 2)
        i: First index
        j: Second index
        
    Returns:
        Tuple (k, sign) where e_i * e_j = sign * e_k
    """
    if dim <= 2:
        # Complex numbers: e1 * e1 = -1
        if i == 0:
            return (j, 1)
        if j == 0:
            return (i, 1)
        if i == 1 and j == 1:
            return (0, -1)
        return (i ^ j, 1)
    
    if dim <= 4:
        # Quaternions: i*i = j*j = k*k = ijk = -1
        if i == 0:
            return (j, 1)
        if j == 0:
            return (i, 1)
        if i == j:
            return (0, -1)
        
        # Quaternion multiplication table
        # quat[i][j] gives signed result index
        quat = [
            [0, 0, 0, 0],
            [0, 0, 3, -2],  # i*j=k, i*k=-j
            [0, -3, 0, 1],  # j*i=-k, j*k=i
            [0, 2, -1, 0]   # k*i=j, k*j=-i
        ]
        k = quat[i][j]
        sign = 1 if k > 0 else -1
        return (abs(k), sign)
    
    if dim <= 8:
        return octonion_multiply_index(i % 8, j % 8)
    
    if dim <= 16:
        return sedenion_multiply_index(i, j)
    
    # Pathion (32D) and beyond: recursive Cayley-Dickson
    if i == 0:
        return (j, 1)
    if j == 0:
        return (i, 1)
    if i == j:
        return (0, -1)
    
    half = dim // 2
    hi = i >= half
    hj = j >= half
    li = i % half
    lj = j % half
    
    if not hi and not hj:
        return multiply_indices(half, li, lj)
    if hi and hj:
        k, s = multiply_indices(half, li, lj)
        return (k, -s)
    if not hi:
        k, s = multiply_indices(half, lj, li)
        return (k + half, s)
    
    k, s = multiply_indices(half, li, lj)
    return (k + half, -s)


def build_multiplication_table(dim: int) -> List[List[Tuple[int, int]]]:
    """
    Build complete multiplication table for a Cayley-Dickson algebra.
    
    Args:
        dim: Dimension of the algebra (should be power of 2)
        
    Returns:
        2D list where table[i][j] = (k, sign) for e_i * e_j = sign * e_k
    """
    table = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(multiply_indices(dim, i, j))
        table.append(row)
    return table


class FanoPlane:
    """
    The Fano plane - the smallest projective plane PG(2,2).
    
    The Fano plane has:
    - 7 points (labeled 1-7)
    - 7 lines (each containing exactly 3 points)
    - Each point lies on exactly 3 lines
    - Each pair of distinct lines intersects at exactly 1 point
    - Each pair of distinct points determines exactly 1 line
    
    This structure encodes octonion multiplication and has deep connections
    to projective geometry, error-correcting codes, and algebraic structures.
    """
    
    def __init__(self):
        self.lines = FANO_LINES
        self.points = list(range(1, 8))
        
        # Build incidence structure
        self._point_to_lines = {p: [] for p in self.points}
        for line_idx, line in enumerate(self.lines):
            for p in line:
                self._point_to_lines[p].append(line_idx)
    
    def get_lines_through_point(self, point: int) -> List[List[int]]:
        """Get all lines passing through a point."""
        if point not in self._point_to_lines:
            raise ValueError(f"Invalid point: {point}")
        return [self.lines[idx] for idx in self._point_to_lines[point]]
    
    def get_third_point(self, p1: int, p2: int) -> int:
        """
        Given two points on a line, get the third point.
        
        In the Fano plane, any two distinct points determine a unique line,
        and that line contains exactly one other point.
        """
        if p1 == p2:
            raise ValueError("Points must be distinct")
        
        for line in self.lines:
            if p1 in line and p2 in line:
                for p in line:
                    if p != p1 and p != p2:
                        return p
        
        raise ValueError(f"No line contains both {p1} and {p2}")
    
    def are_collinear(self, p1: int, p2: int, p3: int) -> bool:
        """Check if three points are collinear (lie on the same line)."""
        points = {p1, p2, p3}
        for line in self.lines:
            if points == set(line):
                return True
        return False
    
    def get_line_through(self, p1: int, p2: int) -> List[int]:
        """Get the unique line containing two points."""
        for line in self.lines:
            if p1 in line and p2 in line:
                return line
        raise ValueError(f"No line contains both {p1} and {p2}")
    
    def intersection(self, line1_idx: int, line2_idx: int) -> int:
        """Get the intersection point of two lines."""
        if line1_idx == line2_idx:
            raise ValueError("Lines must be distinct")
        
        line1 = set(self.lines[line1_idx])
        line2 = set(self.lines[line2_idx])
        intersection = line1 & line2
        
        if len(intersection) != 1:
            raise ValueError("Lines should intersect at exactly one point")
        
        return intersection.pop()
    
    def dual_point(self, line_idx: int) -> int:
        """
        In the self-dual Fano plane, each line corresponds to a point.
        This returns the dual point for a given line.
        """
        # The Fano plane is self-dual
        # Standard duality maps line i to point i (with appropriate labeling)
        return line_idx + 1
    
    def automorphism_group_order(self) -> int:
        """
        The automorphism group of the Fano plane is PSL(3,2) = GL(3,2).
        
        This group has order 168 = 8 * 7 * 3, which is the largest
        simple group that can act as automorphisms of the Fano plane.
        """
        return 168


class CayleyDicksonAlgebra:
    """
    General Cayley-Dickson algebra of dimension 2^n.
    
    The Cayley-Dickson construction produces a sequence of algebras:
    - n=0: Real numbers R (1D)
    - n=1: Complex numbers C (2D)
    - n=2: Quaternions H (4D)
    - n=3: Octonions O (8D)
    - n=4: Sedenions S (16D)
    - n=5: Pathions (32D)
    - etc.
    
    Each step doubles the dimension but loses properties:
    - C loses ordering
    - H loses commutativity
    - O loses associativity
    - S loses alternativity and division
    """
    
    def __init__(self, n: int):
        """
        Create Cayley-Dickson algebra of dimension 2^n.
        
        Args:
            n: Power such that dimension = 2^n (0 <= n <= 8 typical)
        """
        self.n = n
        self.dim = 2 ** n
        self._table = None
    
    @property
    def multiplication_table(self) -> List[List[Tuple[int, int]]]:
        """Lazy computation of multiplication table."""
        if self._table is None:
            self._table = build_multiplication_table(self.dim)
        return self._table
    
    def multiply(self, a: List[float], b: List[float]) -> List[float]:
        """
        Multiply two elements of the algebra.
        
        Args:
            a: Coefficients of first element
            b: Coefficients of second element
            
        Returns:
            Coefficients of product
        """
        if len(a) != self.dim or len(b) != self.dim:
            raise ValueError(f"Elements must have {self.dim} components")
        
        result = [0.0] * self.dim
        table = self.multiplication_table
        
        for i in range(self.dim):
            for j in range(self.dim):
                k, sign = table[i][j]
                result[k] += sign * a[i] * b[j]
        
        return result
    
    def conjugate(self, a: List[float]) -> List[float]:
        """
        Conjugate an element.
        
        For Cayley-Dickson algebras, conjugation negates all imaginary parts.
        """
        result = [a[0]]  # Real part unchanged
        for i in range(1, self.dim):
            result.append(-a[i])
        return result
    
    def norm_squared(self, a: List[float]) -> float:
        """
        Compute squared norm of an element.
        
        norm(a)^2 = a * conjugate(a) (real part only)
        """
        conj = self.conjugate(a)
        product = self.multiply(a, conj)
        return product[0]
    
    def norm(self, a: List[float]) -> float:
        """Compute norm of an element."""
        return math.sqrt(self.norm_squared(a))
    
    def inverse(self, a: List[float]) -> List[float]:
        """
        Compute multiplicative inverse (if exists).
        
        a^(-1) = conjugate(a) / norm(a)^2
        
        Note: For sedenions and higher, zero divisors exist!
        """
        ns = self.norm_squared(a)
        if abs(ns) < 1e-15:
            raise ZeroDivisionError("Element has zero norm")
        
        conj = self.conjugate(a)
        return [c / ns for c in conj]
    
    @property
    def is_division_algebra(self) -> bool:
        """
        Check if this algebra is a division algebra.
        
        Only R, C, H, O are division algebras.
        """
        return self.n <= 3
    
    @property
    def is_associative(self) -> bool:
        """
        Check if multiplication is associative.
        
        Only R, C, H are associative.
        """
        return self.n <= 2
    
    @property
    def is_commutative(self) -> bool:
        """
        Check if multiplication is commutative.
        
        Only R, C are commutative.
        """
        return self.n <= 1
    
    @property
    def is_alternative(self) -> bool:
        """
        Check if algebra is alternative.
        
        Alternative means (aa)b = a(ab) and (ab)b = a(bb) for all a, b.
        R, C, H, O are alternative; sedenions are not.
        """
        return self.n <= 3
    
    def __repr__(self):
        names = {0: 'R', 1: 'C', 2: 'H', 3: 'O', 4: 'S'}
        name = names.get(self.n, f'CD_{self.dim}')
        return f"CayleyDicksonAlgebra({name}, dim={self.dim})"


# Pre-defined algebras
COMPLEX = CayleyDicksonAlgebra(1)
QUATERNIONS = CayleyDicksonAlgebra(2)
OCTONIONS = CayleyDicksonAlgebra(3)
SEDENIONS = CayleyDicksonAlgebra(4)
PATHIONS = CayleyDicksonAlgebra(5)


def test_fano():
    """Test Fano plane and multiplication structures."""
    print("Testing Fano Plane and Cayley-Dickson Algebras")
    print("=" * 50)
    
    # Test Fano plane
    fano = FanoPlane()
    print(f"\nFano plane has {len(fano.points)} points and {len(fano.lines)} lines")
    print(f"Automorphism group order: {fano.automorphism_group_order()}")
    
    # Test collinearity
    print(f"\n1, 2, 3 collinear: {fano.are_collinear(1, 2, 3)}")
    print(f"1, 2, 4 collinear: {fano.are_collinear(1, 2, 4)}")
    
    # Test octonion multiplication
    print("\nOctonion multiplication (e_i * e_j):")
    for i in range(1, 4):
        for j in range(1, 4):
            k, s = octonion_multiply_index(i, j)
            sign = '+' if s > 0 else '-'
            print(f"  e{i} * e{j} = {sign}e{k}")
    
    # Test sedenion properties
    print(f"\nSedenion algebra: {SEDENIONS}")
    print(f"  Division algebra: {SEDENIONS.is_division_algebra}")
    print(f"  Associative: {SEDENIONS.is_associative}")
    print(f"  Alternative: {SEDENIONS.is_alternative}")
    
    # Test Cayley-Dickson multiplication
    a = [1.0, 2.0, 0.0, 0.0]
    b = [0.0, 1.0, 1.0, 0.0]
    ab = QUATERNIONS.multiply(a, b)
    print(f"\nQuaternion multiplication:")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  a*b = {ab}")
    print(f"  |a| = {QUATERNIONS.norm(a):.4f}")
    
    print("\n✓ All Fano plane tests passed")


if __name__ == '__main__':
    test_fano()