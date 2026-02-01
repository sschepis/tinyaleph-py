"""
Tests for core mathematical primitives.

All tests use pure Python - numpy is optional.
"""
import pytest
import math
from tinyaleph.core.complex import Complex
from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.primes import (
    is_prime, first_n_primes, factorize, nth_prime,
    prime_sieve, prime_index, next_prime, prev_prime,
    euler_totient, mobius, gcd, coprime, prime_pi
)
from tinyaleph.core.constants import PHI, PHI_CONJUGATE, DELTA_S, FIBONACCI


class TestComplex:
    """Tests for Complex class."""
    
    def test_creation(self):
        z = Complex(real=3.0, imag=4.0)
        assert z.real == 3.0
        assert z.imag == 4.0
        # Test aliases
        assert z.re == 3.0
        assert z.im == 4.0
    
    def test_polar(self):
        z = Complex.from_polar(1.0, math.pi / 2)
        assert abs(z.real) < 1e-10
        assert abs(z.imag - 1.0) < 1e-10
    
    def test_polar_zero_angle(self):
        z = Complex.from_polar(5.0, 0.0)
        assert abs(z.real - 5.0) < 1e-10
        assert abs(z.imag) < 1e-10
    
    def test_norm(self):
        z = Complex(real=3.0, imag=4.0)
        assert z.norm() == 5.0
        assert z.magnitude() == 5.0
    
    def test_norm_squared(self):
        z = Complex(real=3.0, imag=4.0)
        assert z.norm2() == 25.0
        assert z.magnitude_squared() == 25.0
    
    def test_conjugate(self):
        z = Complex(real=3.0, imag=4.0)
        conj = z.conj()
        assert conj.real == 3.0
        assert conj.imag == -4.0
    
    def test_multiplication(self):
        z1 = Complex(real=1.0, imag=2.0)
        z2 = Complex(real=3.0, imag=4.0)
        result = z1 * z2
        # (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        assert abs(result.real - (-5.0)) < 1e-10
        assert abs(result.imag - 10.0) < 1e-10
    
    def test_scalar_multiplication(self):
        z = Complex(real=2.0, imag=3.0)
        result = z * 2.0
        assert result.real == 4.0
        assert result.imag == 6.0
    
    def test_addition(self):
        z1 = Complex(real=1.0, imag=2.0)
        z2 = Complex(real=3.0, imag=4.0)
        result = z1 + z2
        assert result.real == 4.0
        assert result.imag == 6.0
    
    def test_subtraction(self):
        z1 = Complex(real=5.0, imag=7.0)
        z2 = Complex(real=3.0, imag=4.0)
        result = z1 - z2
        assert result.real == 2.0
        assert result.imag == 3.0
    
    def test_division(self):
        z1 = Complex(real=1.0, imag=0.0)
        z2 = Complex(real=0.0, imag=1.0)  # i
        result = z1 / z2
        # 1/i = -i
        assert abs(result.real) < 1e-10
        assert abs(result.imag + 1.0) < 1e-10
    
    def test_negation(self):
        z = Complex(real=3.0, imag=4.0)
        neg = -z
        assert neg.real == -3.0
        assert neg.imag == -4.0
    
    def test_phase(self):
        z = Complex(real=1.0, imag=1.0)
        assert abs(z.phase() - math.pi / 4) < 1e-10
    
    def test_exp(self):
        z = Complex(real=0.0, imag=math.pi)
        result = z.exp()
        # e^(iπ) = -1
        assert abs(result.real + 1.0) < 1e-10
        assert abs(result.imag) < 1e-10
    
    def test_log(self):
        z = Complex(real=math.e, imag=0.0)
        result = z.log()
        assert abs(result.real - 1.0) < 1e-10
        assert abs(result.imag) < 1e-10
    
    def test_sqrt(self):
        z = Complex(real=-1.0, imag=0.0)
        result = z.sqrt()
        # sqrt(-1) = i
        assert abs(result.real) < 1e-10
        assert abs(result.imag - 1.0) < 1e-10
    
    def test_pow_integer(self):
        z = Complex(real=0.0, imag=1.0)  # i
        result = z.pow(2)
        # i^2 = -1
        assert abs(result.real + 1.0) < 1e-10
        assert abs(result.imag) < 1e-10
    
    def test_class_methods(self):
        assert Complex.zero() == Complex(0.0, 0.0)
        assert Complex.one() == Complex(1.0, 0.0)
        assert Complex.i() == Complex(0.0, 1.0)
    
    def test_to_builtin(self):
        z = Complex(real=3.0, imag=4.0)
        builtin = z.to_builtin()
        assert builtin == complex(3.0, 4.0)
    
    def test_from_builtin(self):
        z = Complex.from_builtin(complex(3.0, 4.0))
        assert z.real == 3.0
        assert z.imag == 4.0


class TestQuaternion:
    """Tests for Quaternion class."""
    
    def test_identity(self):
        q = Quaternion.identity()
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
    
    def test_zero(self):
        q = Quaternion.zero()
        assert q.w == 0.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
    
    def test_basis_elements(self):
        i = Quaternion.I()
        j = Quaternion.J()
        k = Quaternion.K()
        assert i.i == 1.0
        assert j.j == 1.0
        assert k.k == 1.0
    
    def test_norm(self):
        q = Quaternion(w=1.0, i=2.0, j=3.0, k=4.0)
        expected = math.sqrt(1 + 4 + 9 + 16)
        assert abs(q.norm() - expected) < 1e-10
    
    def test_normalize(self):
        q = Quaternion(w=1.0, i=2.0, j=3.0, k=4.0)
        n = q.normalize()
        assert abs(n.norm() - 1.0) < 1e-10
    
    def test_conjugate(self):
        q = Quaternion(w=1.0, i=2.0, j=3.0, k=4.0)
        conj = q.conj()
        assert conj.w == 1.0
        assert conj.i == -2.0
        assert conj.j == -3.0
        assert conj.k == -4.0
    
    def test_hamilton_product_ij_equals_k(self):
        i = Quaternion.I()
        j = Quaternion.J()
        result = i * j
        # i * j = k
        assert abs(result.w) < 1e-10
        assert abs(result.i) < 1e-10
        assert abs(result.j) < 1e-10
        assert abs(result.k - 1.0) < 1e-10
    
    def test_hamilton_product_jk_equals_i(self):
        j = Quaternion.J()
        k = Quaternion.K()
        result = j * k
        # j * k = i
        assert abs(result.i - 1.0) < 1e-10
    
    def test_hamilton_product_ki_equals_j(self):
        k = Quaternion.K()
        i = Quaternion.I()
        result = k * i
        # k * i = j
        assert abs(result.j - 1.0) < 1e-10
    
    def test_hamilton_product_non_commutative(self):
        i = Quaternion.I()
        j = Quaternion.J()
        ij = i * j
        ji = j * i
        # i*j = k but j*i = -k
        assert abs(ij.k - 1.0) < 1e-10
        assert abs(ji.k - (-1.0)) < 1e-10
    
    def test_i_squared(self):
        i = Quaternion.I()
        result = i * i
        # i² = -1
        assert abs(result.w + 1.0) < 1e-10
    
    def test_axis_angle(self):
        q = Quaternion.from_axis_angle((0, 0, 1), math.pi / 2)
        assert q.is_unit()
    
    def test_inverse(self):
        q = Quaternion(w=1.0, i=2.0, j=3.0, k=4.0)
        inv = q.inverse()
        result = q * inv
        # q * q^-1 = 1
        assert abs(result.w - 1.0) < 1e-6
        assert abs(result.i) < 1e-6
        assert abs(result.j) < 1e-6
        assert abs(result.k) < 1e-6
    
    def test_scalar_multiplication(self):
        q = Quaternion(w=1.0, i=2.0, j=3.0, k=4.0)
        result = q * 2.0
        assert result.w == 2.0
        assert result.i == 4.0
        assert result.j == 6.0
        assert result.k == 8.0
    
    def test_addition(self):
        q1 = Quaternion(w=1.0, i=2.0, j=3.0, k=4.0)
        q2 = Quaternion(w=5.0, i=6.0, j=7.0, k=8.0)
        result = q1 + q2
        assert result.w == 6.0
        assert result.i == 8.0
        assert result.j == 10.0
        assert result.k == 12.0
    
    def test_dot_product(self):
        q1 = Quaternion(w=1.0, i=0.0, j=0.0, k=0.0)
        q2 = Quaternion(w=1.0, i=0.0, j=0.0, k=0.0)
        assert q1.dot(q2) == 1.0
    
    def test_slerp_endpoints(self):
        q1 = Quaternion.identity()
        q2 = Quaternion.from_axis_angle((0, 0, 1), math.pi / 2)
        
        # t=0 should give q1
        result0 = q1.slerp(q2, 0.0)
        assert abs(result0.w - q1.w) < 1e-6
        
        # t=1 should give q2
        result1 = q1.slerp(q2, 1.0)
        assert abs(result1.w - q2.w) < 1e-6
    
    def test_rotate_vector(self):
        # 90 degree rotation around z-axis
        q = Quaternion.from_axis_angle((0, 0, 1), math.pi / 2)
        v = (1, 0, 0)
        rotated = q.rotate_vector(v)
        # Should rotate to (0, 1, 0)
        assert abs(rotated[0]) < 1e-6
        assert abs(rotated[1] - 1.0) < 1e-6
        assert abs(rotated[2]) < 1e-6
    
    def test_euler_roundtrip(self):
        roll, pitch, yaw = 0.1, 0.2, 0.3
        q = Quaternion.from_euler(roll, pitch, yaw)
        r2, p2, y2 = q.to_euler()
        assert abs(roll - r2) < 1e-6
        assert abs(pitch - p2) < 1e-6
        assert abs(yaw - y2) < 1e-6
    
    def test_exp_log(self):
        q = Quaternion(w=0.0, i=0.1, j=0.2, k=0.3)
        exp_q = q.exp()
        log_exp_q = exp_q.log()
        # Should get back approximately the original
        assert abs(log_exp_q.i - q.i) < 1e-6


class TestPrimes:
    """Tests for prime utilities."""
    
    def test_is_prime_small(self):
        assert is_prime(2)
        assert is_prime(3)
        assert is_prime(5)
        assert is_prime(7)
        assert is_prime(11)
        assert is_prime(13)
    
    def test_is_not_prime(self):
        assert not is_prime(0)
        assert not is_prime(1)
        assert not is_prime(4)
        assert not is_prime(6)
        assert not is_prime(8)
        assert not is_prime(9)
        assert not is_prime(10)
    
    def test_is_prime_large(self):
        assert is_prime(104729)  # 10000th prime
        assert not is_prime(104728)
    
    def test_first_n_primes(self):
        primes = first_n_primes(10)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    def test_first_n_primes_empty(self):
        assert first_n_primes(0) == []
    
    def test_factorize(self):
        factors = factorize(60)
        assert factors == {2: 2, 3: 1, 5: 1}
    
    def test_factorize_prime(self):
        factors = factorize(17)
        assert factors == {17: 1}
    
    def test_factorize_power_of_prime(self):
        factors = factorize(32)
        assert factors == {2: 5}
    
    def test_nth_prime(self):
        assert nth_prime(1) == 2
        assert nth_prime(2) == 3
        assert nth_prime(3) == 5
        assert nth_prime(10) == 29
    
    def test_prime_sieve(self):
        primes = prime_sieve(30)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    def test_prime_index(self):
        assert prime_index(2) == 1
        assert prime_index(3) == 2
        assert prime_index(5) == 3
        assert prime_index(29) == 10
    
    def test_next_prime(self):
        assert next_prime(2) == 3
        assert next_prime(3) == 5
        assert next_prime(10) == 11
        assert next_prime(1) == 2
    
    def test_prev_prime(self):
        assert prev_prime(3) == 2
        assert prev_prime(5) == 3
        assert prev_prime(12) == 11
    
    def test_euler_totient(self):
        assert euler_totient(1) == 1
        assert euler_totient(2) == 1
        assert euler_totient(6) == 2  # 1, 5 coprime to 6
        assert euler_totient(10) == 4  # 1, 3, 7, 9
    
    def test_mobius(self):
        assert mobius(1) == 1
        assert mobius(2) == -1
        assert mobius(6) == 1  # 2*3, two distinct primes
        assert mobius(4) == 0  # 2^2, squared factor
        assert mobius(30) == -1  # 2*3*5, three distinct primes
    
    def test_gcd(self):
        assert gcd(12, 8) == 4
        assert gcd(17, 13) == 1
        assert gcd(100, 25) == 25
    
    def test_coprime(self):
        assert coprime(7, 11)
        assert not coprime(6, 9)
    
    def test_prime_pi(self):
        assert prime_pi(10) == 4  # 2, 3, 5, 7
        assert prime_pi(30) == 10


class TestConstants:
    """Tests for mathematical constants."""
    
    def test_golden_ratio(self):
        assert abs(PHI - 1.618033988749895) < 1e-10
    
    def test_golden_ratio_property(self):
        # φ² = φ + 1
        assert abs(PHI * PHI - PHI - 1) < 1e-10
    
    def test_conjugate_golden_ratio(self):
        # ψ = 1/φ = φ - 1
        assert abs(PHI_CONJUGATE - (PHI - 1)) < 1e-10
        assert abs(PHI_CONJUGATE - 1/PHI) < 1e-10
    
    def test_delta_s(self):
        assert DELTA_S == 0.01
    
    def test_fibonacci(self):
        # F(n) + F(n+1) = F(n+2)
        for i in range(len(FIBONACCI) - 2):
            assert FIBONACCI[i] + FIBONACCI[i+1] == FIBONACCI[i+2]


# Optional tests that require numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestHypercomplexWithNumpy:
    """Tests for Hypercomplex class (requires numpy)."""
    
    def test_dimension_validation(self):
        from tinyaleph.core.hypercomplex import Hypercomplex
        with pytest.raises(ValueError):
            Hypercomplex(3)  # Not power of 2
    
    def test_real_creation(self):
        from tinyaleph.core.hypercomplex import Hypercomplex
        h = Hypercomplex.real(8, 5.0)
        assert h.c[0] == 5.0
        assert all(h.c[1:] == 0)
    
    def test_norm(self):
        from tinyaleph.core.hypercomplex import Hypercomplex
        components = np.array([1.0, 2.0, 3.0, 4.0])
        h = Hypercomplex(4, components)
        expected = np.sqrt(sum(components**2))
        assert abs(h.norm() - expected) < 1e-10
    
    def test_conjugate(self):
        from tinyaleph.core.hypercomplex import Hypercomplex
        components = np.array([1.0, 2.0, 3.0, 4.0])
        h = Hypercomplex(4, components)
        conj = h.conj()
        assert conj.c[0] == 1.0
        assert conj.c[1] == -2.0
        assert conj.c[2] == -3.0
        assert conj.c[3] == -4.0
    
    def test_scalar_multiplication(self):
        from tinyaleph.core.hypercomplex import Hypercomplex
        h = Hypercomplex(4, np.array([1.0, 2.0, 3.0, 4.0]))
        result = h * 2.0
        assert result.c[0] == 2.0
        assert result.c[1] == 4.0
    
    def test_entropy(self):
        from tinyaleph.core.hypercomplex import Hypercomplex
        # Uniform distribution has maximum entropy
        uniform = Hypercomplex(4, np.ones(4) / 2)
        # Single component has zero entropy
        pure = Hypercomplex.real(4, 1.0)
        assert pure.entropy() == 0.0
        assert uniform.entropy() > pure.entropy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])