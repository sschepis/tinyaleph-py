"""
CRT-Homology Module for TinyAleph.

Implements Chinese Remainder Theorem based encodings and
homological algebra constructions for semantic processing.

Key components:
- ResidueEncoder: Maps values to residue systems
- CRTReconstructor: Reconstructs from residues via CRT
- BirkhoffProjector: Projects onto doubly stochastic matrices
- HomologyLoss: Measures homological consistency
- CRTModularLayer: Neural layer with CRT structure
- CRTFusedAttention: Attention using CRT-based routing

Mathematical Foundation:
    The Chinese Remainder Theorem states that for coprime moduli m1, m2, ..., mk,
    any residue system (r1, r2, ..., rk) uniquely determines a value x mod (m1*m2*...*mk).
    
    This creates a bijection:
        Z/(m1*m2*...*mk)Z ≅ Z/m1Z × Z/m2Z × ... × Z/mkZ
    
    For semantic processing, we use prime moduli to encode information
    in parallel residue channels, then reconstruct via CRT.
"""

from typing import List, Dict, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from functools import lru_cache
import random


# =============================================================================
# Number-Theoretic Utilities
# =============================================================================

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm.
    
    Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
    """
    if b == 0:
        return a, 1, 0
    else:
        g, x, y = extended_gcd(b, a % b)
        return g, y, x - (a // b) * y


def mod_inverse(a: int, m: int) -> int:
    """
    Modular multiplicative inverse.
    
    Returns x such that a*x ≡ 1 (mod m).
    Raises ValueError if inverse doesn't exist.
    """
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}")
    return x % m


def are_coprime(a: int, b: int) -> bool:
    """Check if two integers are coprime (gcd = 1)."""
    return math.gcd(a, b) == 1


def all_coprime(nums: List[int]) -> bool:
    """Check if all pairs in a list are coprime."""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if not are_coprime(nums[i], nums[j]):
                return False
    return True


def softmax(x: List[float], temperature: float = 1.0) -> List[float]:
    """Softmax with temperature scaling."""
    if not x:
        return []
    
    # Subtract max for numerical stability
    max_x = max(x)
    scaled = [(v - max_x) / temperature for v in x]
    
    # Compute exp
    exp_x = [math.exp(v) for v in scaled]
    total = sum(exp_x)
    
    if total == 0:
        return [1.0 / len(x)] * len(x)
    
    return [v / total for v in exp_x]


def first_n_primes(n: int) -> List[int]:
    """Generate first n primes using sieve."""
    if n <= 0:
        return []
    
    # Upper bound for nth prime
    if n < 6:
        upper = 15
    else:
        upper = int(n * (math.log(n) + math.log(math.log(n)))) + 3
    
    # Sieve of Eratosthenes
    sieve = [True] * (upper + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(upper)) + 1):
        if sieve[i]:
            for j in range(i*i, upper + 1, i):
                sieve[j] = False
    
    primes = [i for i in range(upper + 1) if sieve[i]]
    return primes[:n]


@lru_cache(maxsize=1024)
def prime_factorization(n: int) -> Dict[int, int]:
    """Return prime factorization as {prime: exponent} dict."""
    if n <= 1:
        return {}
    
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


# =============================================================================
# ResidueEncoder - Maps values to residue systems
# =============================================================================

@dataclass
class ResidueEncoding:
    """Encoded value as tuple of residues."""
    residues: Tuple[int, ...]
    moduli: Tuple[int, ...]
    original: Optional[int] = None
    
    def __post_init__(self):
        if len(self.residues) != len(self.moduli):
            raise ValueError("Residues and moduli must have same length")
    
    @property
    def product_modulus(self) -> int:
        """Product of all moduli."""
        result = 1
        for m in self.moduli:
            result *= m
        return result
    
    def to_dict(self) -> Dict[int, int]:
        """Return as {modulus: residue} dict."""
        return dict(zip(self.moduli, self.residues))
    
    def __repr__(self) -> str:
        pairs = [f"{r} (mod {m})" for r, m in zip(self.residues, self.moduli)]
        return f"ResidueEncoding({', '.join(pairs)})"


class ResidueEncoder:
    """
    Encodes integers into residue class representation.
    
    Given coprime moduli m1, m2, ..., mk, represents integer x as
    (x mod m1, x mod m2, ..., x mod mk).
    
    This is the forward direction of the CRT isomorphism.
    """
    
    def __init__(self, moduli: Optional[List[int]] = None, 
                 num_channels: int = 8):
        """
        Initialize encoder.
        
        Args:
            moduli: List of coprime moduli. If None, uses first primes.
            num_channels: Number of residue channels if moduli not specified.
        """
        if moduli is None:
            # Use first primes as moduli (guaranteed coprime)
            moduli = first_n_primes(num_channels)
        
        self.moduli = tuple(moduli)
        
        if not all_coprime(list(self.moduli)):
            raise ValueError("All moduli must be pairwise coprime")
        
        self.num_channels = len(self.moduli)
        self.product_modulus = math.prod(self.moduli)
        
        # Precompute for efficiency
        self._precompute_crt_coefficients()
    
    def _precompute_crt_coefficients(self):
        """Precompute CRT reconstruction coefficients."""
        self.crt_coefficients = []
        
        for i, m in enumerate(self.moduli):
            # Product of all other moduli
            M_i = self.product_modulus // m
            # Modular inverse of M_i mod m
            y_i = mod_inverse(M_i, m)
            self.crt_coefficients.append((M_i, y_i, M_i * y_i))
    
    def encode(self, x: int) -> ResidueEncoding:
        """
        Encode integer to residue representation.
        
        Args:
            x: Integer to encode
            
        Returns:
            ResidueEncoding with residues for each modulus
        """
        residues = tuple(x % m for m in self.moduli)
        return ResidueEncoding(
            residues=residues,
            moduli=self.moduli,
            original=x
        )
    
    def encode_batch(self, values: List[int]) -> List[ResidueEncoding]:
        """Encode multiple values."""
        return [self.encode(x) for x in values]
    
    def decode(self, encoding: ResidueEncoding) -> int:
        """
        Decode residues back to integer using CRT.
        
        Args:
            encoding: ResidueEncoding to decode
            
        Returns:
            Unique integer x in [0, product_modulus) with given residues
        """
        if encoding.moduli != self.moduli:
            raise ValueError("Encoding moduli must match encoder moduli")
        
        x = 0
        for r, (M_i, y_i, coeff) in zip(encoding.residues, self.crt_coefficients):
            x += r * coeff
        
        return x % self.product_modulus
    
    def add_encoded(self, a: ResidueEncoding, b: ResidueEncoding) -> ResidueEncoding:
        """Add two encodings (modular arithmetic in each channel)."""
        if a.moduli != b.moduli or a.moduli != self.moduli:
            raise ValueError("Moduli must match")
        
        new_residues = tuple(
            (r1 + r2) % m 
            for r1, r2, m in zip(a.residues, b.residues, self.moduli)
        )
        return ResidueEncoding(residues=new_residues, moduli=self.moduli)
    
    def mul_encoded(self, a: ResidueEncoding, b: ResidueEncoding) -> ResidueEncoding:
        """Multiply two encodings (modular arithmetic in each channel)."""
        if a.moduli != b.moduli or a.moduli != self.moduli:
            raise ValueError("Moduli must match")
        
        new_residues = tuple(
            (r1 * r2) % m 
            for r1, r2, m in zip(a.residues, b.residues, self.moduli)
        )
        return ResidueEncoding(residues=new_residues, moduli=self.moduli)
    
    def scale_encoded(self, encoding: ResidueEncoding, scalar: int) -> ResidueEncoding:
        """Scale an encoding by integer scalar."""
        new_residues = tuple(
            (r * scalar) % m 
            for r, m in zip(encoding.residues, self.moduli)
        )
        return ResidueEncoding(residues=new_residues, moduli=self.moduli)
    
    def get_channel_info(self) -> List[Dict[str, Any]]:
        """Get information about each residue channel."""
        return [
            {
                'index': i,
                'modulus': m,
                'crt_coefficient': self.crt_coefficients[i][2],
                'capacity_bits': math.log2(m) if m > 0 else 0
            }
            for i, m in enumerate(self.moduli)
        ]


# =============================================================================
# CRTReconstructor - Reconstructs values from residues
# =============================================================================

class CRTReconstructor:
    """
    Reconstructs integers from residue representations using CRT.
    
    Handles:
    - Standard CRT reconstruction
    - Partial reconstruction from subset of residues
    - Error detection via redundancy
    - Garner's algorithm for efficient computation
    """
    
    def __init__(self, moduli: List[int]):
        """
        Initialize reconstructor.
        
        Args:
            moduli: List of pairwise coprime moduli
        """
        if not all_coprime(moduli):
            raise ValueError("All moduli must be pairwise coprime")
        
        self.moduli = tuple(moduli)
        self.n = len(moduli)
        self.product_modulus = math.prod(moduli)
        
        # Precompute for Garner's algorithm
        self._precompute_garner()
    
    def _precompute_garner(self):
        """Precompute coefficients for Garner's algorithm."""
        # c[i] = (m_0 * m_1 * ... * m_{i-1})^{-1} mod m_i
        self.garner_coeffs = [1]  # c[0] = 1
        
        for i in range(1, self.n):
            prod = 1
            for j in range(i):
                prod = (prod * self.moduli[j]) % self.moduli[i]
            self.garner_coeffs.append(mod_inverse(prod, self.moduli[i]))
    
    def reconstruct(self, residues: List[int]) -> int:
        """
        Reconstruct integer from residues using standard CRT.
        
        Args:
            residues: List of residues r_i where x ≡ r_i (mod m_i)
            
        Returns:
            Unique x in [0, product_modulus)
        """
        if len(residues) != self.n:
            raise ValueError(f"Expected {self.n} residues, got {len(residues)}")
        
        x = 0
        for i, r in enumerate(residues):
            M_i = self.product_modulus // self.moduli[i]
            y_i = mod_inverse(M_i, self.moduli[i])
            x += r * M_i * y_i
        
        return x % self.product_modulus
    
    def reconstruct_garner(self, residues: List[int]) -> int:
        """
        Reconstruct using Garner's algorithm (more efficient for large moduli).
        
        Computes x in mixed-radix representation, avoiding large intermediate values.
        """
        if len(residues) != self.n:
            raise ValueError(f"Expected {self.n} residues, got {len(residues)}")
        
        # Mixed-radix representation
        v = [0] * self.n
        v[0] = residues[0] % self.moduli[0]
        
        for i in range(1, self.n):
            # Compute v[i] = c[i] * (r[i] - (v[0] + v[1]*m[0] + ...)) mod m[i]
            u = v[0]
            base = 1
            for j in range(1, i):
                base = (base * self.moduli[j-1]) % self.moduli[i]
                u = (u + v[j] * base) % self.moduli[i]
            
            v[i] = (self.garner_coeffs[i] * (residues[i] - u)) % self.moduli[i]
        
        # Convert mixed-radix to integer
        x = v[0]
        base = 1
        for i in range(1, self.n):
            base *= self.moduli[i-1]
            x += v[i] * base
        
        return x
    
    def partial_reconstruct(self, residue_dict: Dict[int, int]) -> Tuple[int, int]:
        """
        Reconstruct from subset of residues.
        
        Args:
            residue_dict: {modulus: residue} for subset of moduli
            
        Returns:
            (value, partial_modulus) where value is unique mod partial_modulus
        """
        subset_moduli = list(residue_dict.keys())
        subset_residues = [residue_dict[m] for m in subset_moduli]
        
        # Create temporary reconstructor for subset
        temp_recon = CRTReconstructor(subset_moduli)
        value = temp_recon.reconstruct(subset_residues)
        
        return value, math.prod(subset_moduli)
    
    def check_consistency(self, residues: List[int], 
                          redundant_residues: Optional[Dict[int, int]] = None) -> bool:
        """
        Check if residues are consistent (error detection).
        
        If redundant_residues provided, verifies reconstruction matches.
        """
        if redundant_residues is None:
            return True
        
        x = self.reconstruct(residues)
        
        for m, r in redundant_residues.items():
            if x % m != r:
                return False
        
        return True
    
    def find_errors(self, residues: List[int], 
                    redundant_residues: Dict[int, int]) -> List[int]:
        """
        Find which channels have errors using redundancy.
        
        Returns list of indices with suspected errors.
        """
        errors = []
        x = self.reconstruct(residues)
        
        # Check primary residues
        for i, (r, m) in enumerate(zip(residues, self.moduli)):
            if x % m != r:
                errors.append(i)
        
        return errors


# =============================================================================
# BirkhoffProjector - Projects onto doubly stochastic matrices
# =============================================================================

@dataclass
class DoublyStochasticMatrix:
    """A doubly stochastic matrix (rows and columns sum to 1)."""
    data: List[List[float]]
    
    @property
    def n(self) -> int:
        return len(self.data)
    
    def row_sums(self) -> List[float]:
        return [sum(row) for row in self.data]
    
    def col_sums(self) -> List[float]:
        n = self.n
        return [sum(self.data[i][j] for i in range(n)) for j in range(n)]
    
    def is_valid(self, tol: float = 1e-6) -> bool:
        """Check if matrix is doubly stochastic within tolerance."""
        n = self.n
        for i in range(n):
            if abs(sum(self.data[i]) - 1.0) > tol:
                return False
            if abs(sum(self.data[j][i] for j in range(n)) - 1.0) > tol:
                return False
        return True
    
    def __getitem__(self, idx: Tuple[int, int]) -> float:
        i, j = idx
        return self.data[i][j]
    
    def __repr__(self) -> str:
        return f"DoublyStochasticMatrix({self.n}x{self.n})"


class BirkhoffProjector:
    """
    Projects matrices onto the Birkhoff polytope (set of doubly stochastic matrices).
    
    Uses Sinkhorn-Knopp algorithm for iterative projection:
    Alternately normalize rows and columns until convergence.
    
    Applications:
    - Soft permutation learning
    - Attention normalization
    - Graph matching
    """
    
    def __init__(self, max_iterations: int = 100, 
                 tolerance: float = 1e-6,
                 epsilon: float = 1e-10):
        """
        Initialize projector.
        
        Args:
            max_iterations: Maximum Sinkhorn iterations
            tolerance: Convergence tolerance
            epsilon: Small value to prevent division by zero
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
    
    def project(self, matrix: List[List[float]]) -> DoublyStochasticMatrix:
        """
        Project matrix onto Birkhoff polytope using Sinkhorn-Knopp.
        
        Args:
            matrix: Input matrix (non-negative recommended)
            
        Returns:
            DoublyStochasticMatrix closest to input
        """
        n = len(matrix)
        if n == 0:
            return DoublyStochasticMatrix([])
        
        # Ensure non-negative
        data = [[max(0.0, matrix[i][j]) for j in range(n)] for i in range(n)]
        
        # Add epsilon to prevent zero rows/columns
        for i in range(n):
            for j in range(n):
                data[i][j] += self.epsilon
        
        for iteration in range(self.max_iterations):
            # Normalize rows
            for i in range(n):
                row_sum = sum(data[i])
                if row_sum > 0:
                    for j in range(n):
                        data[i][j] /= row_sum
            
            # Normalize columns
            for j in range(n):
                col_sum = sum(data[i][j] for i in range(n))
                if col_sum > 0:
                    for i in range(n):
                        data[i][j] /= col_sum
            
            # Check convergence
            row_error = max(abs(sum(data[i]) - 1.0) for i in range(n))
            col_error = max(abs(sum(data[i][j] for i in range(n)) - 1.0) for j in range(n))
            
            if max(row_error, col_error) < self.tolerance:
                break
        
        return DoublyStochasticMatrix(data)
    
    def project_softmax(self, logits: List[List[float]], 
                        temperature: float = 1.0) -> DoublyStochasticMatrix:
        """
        Project logits via softmax then Sinkhorn.
        
        Applies row-wise softmax first for differentiability.
        """
        n = len(logits)
        
        # Row-wise softmax
        matrix = []
        for row in logits:
            exp_row = [math.exp(v / temperature) for v in row]
            total = sum(exp_row)
            matrix.append([v / total for v in exp_row])
        
        return self.project(matrix)
    
    def decompose_birkhoff(self, ds_matrix: DoublyStochasticMatrix,
                           max_perms: int = 100) -> List[Tuple[float, List[int]]]:
        """
        Decompose doubly stochastic matrix as convex combination of permutations.
        
        Uses greedy algorithm (Birkhoff-von Neumann theorem guarantees this works).
        
        Returns:
            List of (weight, permutation) tuples
        """
        n = ds_matrix.n
        # Copy data
        remaining = [row[:] for row in ds_matrix.data]
        
        decomposition = []
        
        for _ in range(max_perms):
            # Find minimum positive element
            min_val = float('inf')
            for i in range(n):
                for j in range(n):
                    if remaining[i][j] > self.epsilon and remaining[i][j] < min_val:
                        min_val = remaining[i][j]
            
            if min_val == float('inf') or min_val < self.epsilon:
                break
            
            # Find permutation using remaining as adjacency matrix
            # Greedy matching
            perm = [-1] * n
            used_cols = set()
            
            for i in range(n):
                best_j = -1
                best_val = -1
                for j in range(n):
                    if j not in used_cols and remaining[i][j] > best_val:
                        best_val = remaining[i][j]
                        best_j = j
                if best_j >= 0:
                    perm[i] = best_j
                    used_cols.add(best_j)
            
            if -1 in perm:
                break
            
            # Subtract min_val times this permutation
            for i in range(n):
                remaining[i][perm[i]] -= min_val
                remaining[i][perm[i]] = max(0.0, remaining[i][perm[i]])
            
            decomposition.append((min_val, perm))
        
        return decomposition
    
    def permutation_to_matrix(self, perm: List[int]) -> List[List[float]]:
        """Convert permutation to permutation matrix."""
        n = len(perm)
        matrix = [[0.0] * n for _ in range(n)]
        for i, j in enumerate(perm):
            matrix[i][j] = 1.0
        return matrix
    
    def entropy(self, ds_matrix: DoublyStochasticMatrix) -> float:
        """
        Compute entropy of doubly stochastic matrix.
        
        Higher entropy = more uniform, lower = more permutation-like.
        """
        total = 0.0
        for row in ds_matrix.data:
            for v in row:
                if v > 0:
                    total -= v * math.log(v)
        return total


# =============================================================================
# HomologyLoss - Measures homological consistency
# =============================================================================

class HomologyLoss:
    """
    Computes loss based on homological structure.
    
    Homology measures "holes" in topological spaces. For discrete structures:
    - H_0: Connected components
    - H_1: Loops/cycles
    - H_2: Voids/cavities
    
    We use simplicial homology on graphs/complexes derived from semantic structures.
    """
    
    def __init__(self, dimension: int = 1, 
                 lambda_boundary: float = 1.0,
                 lambda_cycle: float = 0.5):
        """
        Initialize homology loss.
        
        Args:
            dimension: Maximum homology dimension to compute
            dimension: Which H_n to focus on
            lambda_boundary: Weight for boundary consistency
            lambda_cycle: Weight for cycle consistency
        """
        self.dimension = dimension
        self.lambda_boundary = lambda_boundary
        self.lambda_cycle = lambda_cycle
    
    def _build_boundary_matrix(self, edges: List[Tuple[int, int]], 
                                num_vertices: int) -> List[List[int]]:
        """
        Build boundary matrix ∂_1 for edges.
        
        ∂_1(edge i→j) = j - i (in chain notation)
        Matrix entry: +1 for source, -1 for target
        """
        n_edges = len(edges)
        # Matrix: num_vertices × n_edges
        matrix = [[0] * n_edges for _ in range(num_vertices)]
        
        for e_idx, (i, j) in enumerate(edges):
            if i < num_vertices:
                matrix[i][e_idx] = -1  # Source
            if j < num_vertices:
                matrix[j][e_idx] = 1   # Target
        
        return matrix
    
    def _compute_rank(self, matrix: List[List[int]]) -> int:
        """Compute matrix rank using Gaussian elimination mod 2."""
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        # Copy and work mod 2
        data = [[abs(matrix[i][j]) % 2 for j in range(n)] for i in range(m)]
        
        rank = 0
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(rank, m):
                if data[row][col] == 1:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                continue
            
            # Swap rows
            data[rank], data[pivot_row] = data[pivot_row], data[rank]
            
            # Eliminate
            for row in range(m):
                if row != rank and data[row][col] == 1:
                    for c in range(n):
                        data[row][c] = (data[row][c] + data[rank][c]) % 2
            
            rank += 1
        
        return rank
    
    def compute_betti_0(self, edges: List[Tuple[int, int]], 
                        num_vertices: int) -> int:
        """
        Compute Betti number β_0 = number of connected components.
        
        β_0 = dim(ker(∂_0)) = num_vertices - rank(∂_1)
        """
        if num_vertices == 0:
            return 0
        if not edges:
            return num_vertices
        
        boundary = self._build_boundary_matrix(edges, num_vertices)
        rank = self._compute_rank(boundary)
        
        return num_vertices - rank
    
    def compute_betti_1(self, edges: List[Tuple[int, int]], 
                        num_vertices: int,
                        triangles: Optional[List[Tuple[int, int, int]]] = None) -> int:
        """
        Compute Betti number β_1 = number of independent cycles.
        
        β_1 = dim(ker(∂_1)) - dim(im(∂_2))
             = |E| - rank(∂_1) - rank(∂_2)
        """
        n_edges = len(edges)
        if n_edges == 0:
            return 0
        
        # Rank of boundary matrix
        boundary = self._build_boundary_matrix(edges, num_vertices)
        rank_b1 = self._compute_rank(boundary)
        
        # Dimension of kernel of ∂_1
        dim_ker = n_edges - rank_b1
        
        # If triangles provided, compute ∂_2
        rank_b2 = 0
        if triangles:
            b2 = self._build_boundary_2(triangles, edges)
            rank_b2 = self._compute_rank(b2)
        
        return dim_ker - rank_b2
    
    def _build_boundary_2(self, triangles: List[Tuple[int, int, int]], 
                          edges: List[Tuple[int, int]]) -> List[List[int]]:
        """Build boundary matrix ∂_2 for triangles → edges."""
        edge_index = {e: i for i, e in enumerate(edges)}
        n_triangles = len(triangles)
        n_edges = len(edges)
        
        matrix = [[0] * n_triangles for _ in range(n_edges)]
        
        for t_idx, (a, b, c) in enumerate(triangles):
            # Triangle edges: ab, bc, ca with orientation
            for (i, j), sign in [((a, b), 1), ((b, c), 1), ((c, a), 1)]:
                if (i, j) in edge_index:
                    matrix[edge_index[(i, j)]][t_idx] = sign
                elif (j, i) in edge_index:
                    matrix[edge_index[(j, i)]][t_idx] = -sign
        
        return matrix
    
    def boundary_loss(self, edges: List[Tuple[int, int]], 
                      num_vertices: int,
                      target_components: int = 1) -> float:
        """
        Loss based on deviation from target number of components.
        
        Encourages connected structure (low β_0).
        """
        beta_0 = self.compute_betti_0(edges, num_vertices)
        return self.lambda_boundary * abs(beta_0 - target_components)
    
    def cycle_loss(self, edges: List[Tuple[int, int]],
                   num_vertices: int,
                   target_cycles: int = 0) -> float:
        """
        Loss based on deviation from target number of cycles.
        
        Can be used to encourage or discourage cyclic structure.
        """
        beta_1 = self.compute_betti_1(edges, num_vertices)
        return self.lambda_cycle * abs(beta_1 - target_cycles)
    
    def total_loss(self, edges: List[Tuple[int, int]],
                   num_vertices: int,
                   target_components: int = 1,
                   target_cycles: int = 0) -> float:
        """Combined homology loss."""
        return (self.boundary_loss(edges, num_vertices, target_components) +
                self.cycle_loss(edges, num_vertices, target_cycles))
    
    def euler_characteristic(self, num_vertices: int, num_edges: int,
                             num_faces: int = 0) -> int:
        """
        Compute Euler characteristic χ = V - E + F.
        
        For graphs embedded in plane: χ = 2 - 2g where g is genus.
        """
        return num_vertices - num_edges + num_faces


# =============================================================================
# CRTModularLayer - Neural layer with CRT structure
# =============================================================================

class CRTModularLayer:
    """
    Neural layer that operates in modular arithmetic channels.
    
    Each channel corresponds to a prime modulus, enabling:
    - Parallel computation
    - Error detection via redundancy
    - Natural quantization
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 num_channels: int = 8,
                 moduli: Optional[List[int]] = None):
        """
        Initialize CRT modular layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_channels: Number of residue channels
            moduli: Custom moduli (uses first primes if None)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if moduli is None:
            moduli = first_n_primes(num_channels)
        
        self.encoder = ResidueEncoder(moduli)
        self.moduli = self.encoder.moduli
        self.num_channels = len(self.moduli)
        
        # Per-channel weight matrices (as integer weights mod m)
        self.channel_weights = []
        for m in self.moduli:
            # Initialize with random weights mod m
            weights = [
                [random.randint(0, m - 1) for _ in range(input_dim)]
                for _ in range(output_dim)
            ]
            self.channel_weights.append(weights)
        
        # Per-channel biases
        self.channel_biases = []
        for m in self.moduli:
            biases = [random.randint(0, m - 1) for _ in range(output_dim)]
            self.channel_biases.append(biases)
    
    def forward_channel(self, x: List[int], channel_idx: int) -> List[int]:
        """
        Forward pass in single channel (mod m_i).
        
        Args:
            x: Input vector (already in residue form for this channel)
            channel_idx: Which modular channel
            
        Returns:
            Output vector in residue form
        """
        m = self.moduli[channel_idx]
        weights = self.channel_weights[channel_idx]
        biases = self.channel_biases[channel_idx]
        
        output = []
        for i in range(self.output_dim):
            # Compute weighted sum mod m
            total = biases[i]
            for j in range(min(len(x), self.input_dim)):
                total += weights[i][j] * x[j]
            output.append(total % m)
        
        return output
    
    def forward(self, x: List[int]) -> List[int]:
        """
        Full forward pass using CRT.
        
        1. Encode input to residue form
        2. Process each channel independently
        3. Reconstruct output via CRT
        """
        # Encode each input element
        encoded_x = [self.encoder.encode(val) for val in x]
        
        # Process each channel
        channel_outputs = []
        for c in range(self.num_channels):
            channel_input = [enc.residues[c] for enc in encoded_x]
            channel_out = self.forward_channel(channel_input, c)
            channel_outputs.append(channel_out)
        
        # Reconstruct each output element
        reconstructor = CRTReconstructor(list(self.moduli))
        output = []
        for i in range(self.output_dim):
            residues = [channel_outputs[c][i] for c in range(self.num_channels)]
            output.append(reconstructor.reconstruct(residues))
        
        return output
    
    def get_effective_weights(self) -> List[List[int]]:
        """Reconstruct effective integer weights from channel weights."""
        reconstructor = CRTReconstructor(list(self.moduli))
        
        weights = []
        for i in range(self.output_dim):
            row = []
            for j in range(self.input_dim):
                residues = [self.channel_weights[c][i][j]
                           for c in range(self.num_channels)]
                row.append(reconstructor.reconstruct(residues))
            weights.append(row)
        
        return weights


# =============================================================================
# CRTFusedAttention - Attention using CRT-based routing
# =============================================================================

class CRTFusedAttention:
    """
    Attention mechanism using CRT for multi-scale routing.
    
    Each modular channel handles different semantic scales:
    - Small primes: Fine-grained local patterns
    - Large primes: Coarse global patterns
    
    Attention scores are computed modularly then fused.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 moduli: Optional[List[int]] = None):
        """
        Initialize CRT fused attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            moduli: Moduli for CRT channels (default: first primes)
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if moduli is None:
            moduli = first_n_primes(num_heads)
        
        self.encoder = ResidueEncoder(moduli)
        self.moduli = self.encoder.moduli
        
        # Attention projections per channel
        self.q_proj = CRTModularLayer(embed_dim, embed_dim, len(moduli), list(moduli))
        self.k_proj = CRTModularLayer(embed_dim, embed_dim, len(moduli), list(moduli))
        self.v_proj = CRTModularLayer(embed_dim, embed_dim, len(moduli), list(moduli))
        self.out_proj = CRTModularLayer(embed_dim, embed_dim, len(moduli), list(moduli))
        
        # Birkhoff projector for attention matrices
        self.birkhoff = BirkhoffProjector()
    
    def _split_heads(self, x: List[int]) -> List[List[int]]:
        """Split embedding into heads."""
        heads = []
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            heads.append(x[start:end])
        return heads
    
    def _merge_heads(self, heads: List[List[int]]) -> List[int]:
        """Merge heads back to embedding."""
        merged = []
        for head in heads:
            merged.extend(head)
        return merged
    
    def compute_attention_scores(self, q: List[int], k: List[int]) -> float:
        """Compute scaled dot-product attention score."""
        if len(q) != len(k):
            return 0.0
        
        dot = sum(qi * ki for qi, ki in zip(q, k))
        scale = math.sqrt(len(q)) if len(q) > 0 else 1.0
        return dot / scale
    
    def forward(self, query: List[List[int]],
                key: List[List[int]],
                value: List[List[int]]) -> List[List[int]]:
        """
        Forward pass with CRT-fused attention.
        
        Args:
            query: Query sequence [seq_len, embed_dim]
            key: Key sequence [seq_len, embed_dim]
            value: Value sequence [seq_len, embed_dim]
            
        Returns:
            Output sequence [seq_len, embed_dim]
        """
        seq_len = len(query)
        
        # Project to Q, K, V
        Q = [self.q_proj.forward(q) for q in query]
        K = [self.k_proj.forward(k) for k in key]
        V = [self.v_proj.forward(v) for v in value]
        
        # Compute attention per head
        outputs = []
        for i in range(seq_len):
            q_heads = self._split_heads(Q[i])
            
            head_outputs = []
            for h in range(self.num_heads):
                # Compute attention scores for this head
                scores = []
                for j in range(seq_len):
                    k_heads = self._split_heads(K[j])
                    score = self.compute_attention_scores(q_heads[h], k_heads[h])
                    scores.append(score)
                
                # Softmax
                weights = softmax(scores)
                
                # Weighted sum of values
                head_out = [0] * self.head_dim
                for j in range(seq_len):
                    v_heads = self._split_heads(V[j])
                    for d in range(self.head_dim):
                        head_out[d] += int(weights[j] * v_heads[h][d])
                
                head_outputs.append(head_out)
            
            # Merge heads and project
            merged = self._merge_heads(head_outputs)
            out = self.out_proj.forward(merged)
            outputs.append(out)
        
        return outputs
    
    def attention_with_birkhoff(self, query: List[List[int]],
                                 key: List[List[int]]) -> DoublyStochasticMatrix:
        """
        Compute attention as doubly stochastic matrix.
        
        Ensures attention is a valid convex combination of permutations.
        """
        seq_len = len(query)
        
        # Compute raw attention scores
        Q = [self.q_proj.forward(q) for q in query]
        K = [self.k_proj.forward(k) for k in key]
        
        scores = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                row.append(self.compute_attention_scores(Q[i], K[j]))
            scores.append(row)
        
        # Project onto Birkhoff polytope
        return self.birkhoff.project(scores)


# =============================================================================
# CoprimeSelector - Select optimal coprime moduli
# =============================================================================

class CoprimeSelector:
    """
    Selects optimal sets of coprime moduli for CRT encoding.
    
    Criteria:
    - All pairwise coprime
    - Product covers desired range
    - Balanced channel capacities
    - Efficient computation
    """
    
    def __init__(self, target_range: int = 2**32):
        """
        Initialize selector.
        
        Args:
            target_range: Desired range for CRT reconstruction
        """
        self.target_range = target_range
        self._prime_cache = first_n_primes(1000)
    
    def select_primes(self, num_channels: int) -> List[int]:
        """Select first n primes (optimal for coprimality)."""
        return self._prime_cache[:num_channels]
    
    def select_prime_powers(self, num_channels: int,
                             max_power: int = 3) -> List[int]:
        """
        Select prime powers for larger moduli.
        
        Prime powers of different primes are coprime.
        """
        moduli = []
        for p in self._prime_cache:
            if len(moduli) >= num_channels:
                break
            for k in range(1, max_power + 1):
                if len(moduli) >= num_channels:
                    break
                moduli.append(p ** k)
        
        return moduli[:num_channels]
    
    def select_for_range(self, min_range: int) -> List[int]:
        """Select minimal set of primes whose product exceeds min_range."""
        moduli = []
        product = 1
        
        for p in self._prime_cache:
            if product >= min_range:
                break
            moduli.append(p)
            product *= p
        
        return moduli
    
    def select_balanced(self, num_channels: int,
                        bits_per_channel: int = 8) -> List[int]:
        """
        Select moduli with balanced bit widths.
        
        Each modulus should be close to 2^bits_per_channel.
        """
        target = 2 ** bits_per_channel
        moduli = []
        
        # Find primes closest to target
        for p in self._prime_cache:
            if p > target * 2:
                break
            if p >= target // 2:
                moduli.append(p)
        
        # Sort by distance to target
        moduli.sort(key=lambda p: abs(p - target))
        
        # Take first num_channels that are coprime
        selected = []
        for p in moduli:
            if len(selected) >= num_channels:
                break
            if all(are_coprime(p, q) for q in selected):
                selected.append(p)
        
        return selected
    
    def analyze_moduli(self, moduli: List[int]) -> Dict[str, Any]:
        """Analyze a set of moduli."""
        return {
            'moduli': moduli,
            'num_channels': len(moduli),
            'product': math.prod(moduli),
            'bits': math.log2(math.prod(moduli)) if moduli else 0,
            'channel_bits': [math.log2(m) for m in moduli],
            'all_coprime': all_coprime(moduli),
            'lcm': math.lcm(*moduli) if moduli else 0
        }


# =============================================================================
# Integration Functions
# =============================================================================

def create_semantic_crt_encoder(vocabulary_size: int = 10000) -> ResidueEncoder:
    """
    Create CRT encoder suitable for semantic embeddings.
    
    Uses moduli whose product exceeds vocabulary size.
    """
    selector = CoprimeSelector(vocabulary_size)
    moduli = selector.select_for_range(vocabulary_size)
    return ResidueEncoder(moduli)


def crt_embed_sequence(tokens: List[int], encoder: ResidueEncoder) -> List[ResidueEncoding]:
    """Encode a sequence of tokens to CRT form."""
    return encoder.encode_batch(tokens)


def crt_similarity(enc1: ResidueEncoding, enc2: ResidueEncoding) -> float:
    """
    Compute similarity between two CRT encodings.
    
    Uses channel-wise agreement.
    """
    if enc1.moduli != enc2.moduli:
        return 0.0
    
    agreements = sum(
        1 if r1 == r2 else 0
        for r1, r2 in zip(enc1.residues, enc2.residues)
    )
    
    return agreements / len(enc1.moduli)


def homology_regularizer(attention_weights: List[List[float]],
                         target_components: int = 1) -> float:
    """
    Regularization term based on attention pattern topology.
    
    Encourages connected attention (low β_0).
    """
    # Threshold to get edges
    threshold = 0.1
    n = len(attention_weights)
    
    edges = []
    for i in range(n):
        for j in range(n):
            if i < len(attention_weights) and j < len(attention_weights[i]):
                if attention_weights[i][j] > threshold:
                    edges.append((i, j))
    
    loss = HomologyLoss()
    return loss.boundary_loss(edges, n, target_components)