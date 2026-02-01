"""
Tests for resonance module - ResonantFragment.
"""
import pytest
import math

# These modules require numpy
numpy_available = True
try:
    import numpy as np
except ImportError:
    numpy_available = False

pytestmark = pytest.mark.skipif(not numpy_available, reason="numpy required for resonance module")


class TestResonantFragment:
    """Tests for ResonantFragment class."""
    
    @pytest.fixture
    def fragment(self):
        """Create sample fragment for testing."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        return ResonantFragment.encode("Hello world")
    
    @pytest.fixture
    def empty_fragment(self):
        """Create empty fragment."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        return ResonantFragment()
    
    def test_empty_fragment(self, empty_fragment):
        """Test empty fragment creation."""
        assert len(empty_fragment.coeffs) == 0
        assert empty_fragment.entropy == 0.0
    
    def test_encode_string(self, fragment):
        """Test encoding string into fragment."""
        assert len(fragment.coeffs) > 0
        assert fragment.entropy >= 0
    
    def test_encode_empty_string(self):
        """Test encoding empty string."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        frag = ResonantFragment.encode("")
        assert len(frag.coeffs) == 0
    
    def test_encode_with_entropy(self):
        """Test encoding with spatial entropy parameter."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        frag = ResonantFragment.encode("Test", spatial_entropy=0.8)
        assert len(frag.coeffs) > 0
    
    def test_from_primes(self):
        """Test creating fragment from prime-amplitude pairs."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        frag = ResonantFragment.from_primes([(2, 0.5), (3, 0.5), (5, 0.5)])
        assert 2 in frag.coeffs
        assert 3 in frag.coeffs
        assert 5 in frag.coeffs
    
    def test_random_fragment(self):
        """Test creating random fragment."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        frag = ResonantFragment.random(n_primes=10)
        assert len(frag.coeffs) == 10
    
    def test_norm(self, fragment):
        """Test norm computation."""
        n = fragment.norm()
        assert n > 0
        # After encoding, should be normalized
        assert abs(n - 1.0) < 0.01
    
    def test_normalize(self, fragment):
        """Test normalization."""
        normalized = fragment.normalize()
        assert abs(normalized.norm() - 1.0) < 0.001
    
    def test_tensor_product(self, fragment):
        """Test tensor product (interference)."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        other = ResonantFragment.encode("World")
        result = fragment.tensor(other)
        assert len(result.coeffs) > 0
    
    def test_collapse(self, fragment):
        """Test collapse to single prime."""
        collapsed = fragment.collapse()
        assert len(collapsed.coeffs) == 1
        # The single coefficient should be 1.0
        for amp in collapsed.coeffs.values():
            assert abs(amp - 1.0) < 0.001
    
    def test_collapse_empty(self, empty_fragment):
        """Test collapse of empty fragment."""
        collapsed = empty_fragment.collapse()
        assert len(collapsed.coeffs) == 0
    
    def test_rotate_phase(self, fragment):
        """Test phase rotation."""
        rotated = fragment.rotate_phase(math.pi / 4)
        assert len(rotated.coeffs) == len(fragment.coeffs)
    
    def test_overlap(self, fragment):
        """Test overlap computation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        other = ResonantFragment.encode("Hello world")
        overlap = fragment.overlap(other)
        # Same string should have high overlap
        assert overlap > 0.5
    
    def test_overlap_different(self, fragment):
        """Test overlap with different fragment."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        other = ResonantFragment.encode("Completely different")
        overlap = fragment.overlap(other)
        # Different strings should have some overlap due to shared primes
        assert 0 <= overlap <= 1
    
    def test_distance(self, fragment):
        """Test distance computation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        other = ResonantFragment.encode("Hello world")
        dist = fragment.distance(other)
        # Same string should have zero distance
        assert dist < 0.1
    
    def test_distance_different(self, fragment):
        """Test distance to different fragment."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        other = ResonantFragment.encode("Completely different")
        dist = fragment.distance(other)
        assert dist > 0
    
    def test_dominant_prime(self, fragment):
        """Test finding dominant prime."""
        dominant = fragment.dominant_prime()
        assert dominant is not None
        assert dominant in fragment.coeffs
    
    def test_dominant_prime_empty(self, empty_fragment):
        """Test dominant prime of empty fragment."""
        dominant = empty_fragment.dominant_prime()
        assert dominant is None
    
    def test_primes(self, fragment):
        """Test getting sorted primes."""
        primes = fragment.primes()
        assert len(primes) > 0
        # Should be sorted
        assert primes == sorted(primes)
    
    def test_to_vector(self, fragment):
        """Test conversion to vector."""
        primes = fragment.primes()
        vec = fragment.to_vector(primes)
        assert len(vec) == len(primes)
    
    def test_from_vector(self):
        """Test creation from vector."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        primes = [2, 3, 5, 7]
        vec = np.array([0.5, 0.5, 0.5, 0.5])
        frag = ResonantFragment.from_vector(vec, primes)
        assert len(frag.coeffs) == 4
    
    def test_add_fragments(self, fragment):
        """Test fragment addition."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        other = ResonantFragment.encode("Other")
        result = fragment + other
        assert len(result.coeffs) > 0
    
    def test_scalar_multiply(self, fragment):
        """Test scalar multiplication."""
        scaled = fragment * 2.0
        # Check amplitudes are doubled
        for p in fragment.coeffs:
            assert abs(scaled.coeffs[p] - fragment.coeffs[p] * 2.0) < 0.001
    
    def test_right_scalar_multiply(self, fragment):
        """Test right scalar multiplication."""
        scaled = 2.0 * fragment
        for p in fragment.coeffs:
            assert abs(scaled.coeffs[p] - fragment.coeffs[p] * 2.0) < 0.001
    
    def test_repr(self, fragment):
        """Test string representation."""
        s = repr(fragment)
        assert "ResonantFragment" in s
    
    def test_str(self, fragment):
        """Test string conversion."""
        s = str(fragment)
        assert "ResonantFragment" in s
    
    def test_center_attribute(self, fragment):
        """Test center attribute."""
        center = fragment.center
        assert len(center) == 2
        assert isinstance(center[0], float)
        assert isinstance(center[1], float)


class TestFragmentOperations:
    """Test fragment operations."""
    
    def test_interference_pattern(self):
        """Test creating interference pattern from two fragments."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        
        f1 = ResonantFragment.encode("Alpha")
        f2 = ResonantFragment.encode("Beta")
        
        # Tensor creates interference
        interference = f1.tensor(f2)
        
        # Should have contributions from both
        assert len(interference.coeffs) > 0
    
    def test_round_trip_vector(self):
        """Test round-trip conversion to/from vector."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        
        original = ResonantFragment.encode("Test")
        primes = original.primes()
        vec = original.to_vector(primes)
        recovered = ResonantFragment.from_vector(vec, primes)
        
        # Should have same coefficients (approximately)
        for p in primes:
            assert abs(original.coeffs[p] - recovered.coeffs[p]) < 0.001
    
    def test_normalization_preserves_ratios(self):
        """Test that normalization preserves amplitude ratios."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        
        frag = ResonantFragment(coeffs={2: 1.0, 3: 2.0})
        normalized = frag.normalize()
        
        # Ratio should be preserved
        ratio_before = frag.coeffs[3] / frag.coeffs[2]
        ratio_after = normalized.coeffs[3] / normalized.coeffs[2]
        assert abs(ratio_before - ratio_after) < 0.001


class TestFragmentMeasurement:
    """Test measurement operations on fragments."""
    
    def test_collapse_probabilistic(self):
        """Test that collapse is probabilistic."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.resonance.fragment import ResonantFragment
        
        # Create uniform superposition
        frag = ResonantFragment.from_primes([
            (2, 1.0), (3, 1.0), (5, 1.0), (7, 1.0)
        ])
        
        # Collapse many times and check we get different results
        results = set()
        for _ in range(50):
            collapsed = frag.collapse()
            results.add(collapsed.dominant_prime())
        
        # Should get multiple different primes
        assert len(results) >= 2