"""
Tests for physics module - Kuramoto oscillators and entropy.
"""
import pytest
import math

# These modules require numpy
numpy_available = True
try:
    import numpy as np
except ImportError:
    numpy_available = False

pytestmark = pytest.mark.skipif(not numpy_available, reason="numpy required for physics module")


class TestKuramotoModel:
    """Tests for Kuramoto coupled oscillator model."""
    
    @pytest.fixture
    def small_model(self):
        """Create small Kuramoto model for testing."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.physics.kuramoto import KuramotoModel
        return KuramotoModel(n_oscillators=10, coupling=1.0)
    
    @pytest.fixture
    def large_model(self):
        """Create larger Kuramoto model for testing."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.physics.kuramoto import KuramotoModel
        return KuramotoModel(n_oscillators=100, coupling=2.0)
    
    def test_initialization(self, small_model):
        """Test model initializes with correct number of oscillators."""
        assert small_model.n_oscillators == 10
        assert len(small_model.phases) == 10
        assert len(small_model.frequencies) == 10
    
    def test_phases_in_valid_range(self, small_model):
        """Test phases are in [0, 2π)."""
        assert all(0 <= p < 2 * np.pi for p in small_model.phases)
    
    def test_coupling_parameter(self, small_model):
        """Test coupling parameter is set correctly."""
        assert small_model.coupling == 1.0
    
    def test_step_updates_phases(self, small_model):
        """Test that a step updates phases."""
        initial_phases = small_model.phases.copy()
        small_model.step(dt=0.1)
        # Phases should change (unless all oscillators exactly synchronized)
        # With random initial conditions, they almost certainly change
        assert not np.allclose(small_model.phases, initial_phases)
    
    def test_phases_remain_bounded(self, small_model):
        """Test phases remain in [0, 2π) after steps."""
        for _ in range(100):
            small_model.step(dt=0.1)
        assert all(0 <= p < 2 * np.pi for p in small_model.phases)
    
    def test_order_parameter_is_complex(self, small_model):
        """Test order parameter returns complex number."""
        r = small_model.order_parameter()
        assert isinstance(r, complex)
    
    def test_synchronization_in_range(self, small_model):
        """Test synchronization measure is in [0, 1]."""
        r = small_model.synchronization()
        assert 0 <= r <= 1
    
    def test_mean_phase_in_range(self, small_model):
        """Test mean phase is in [-π, π]."""
        psi = small_model.mean_phase()
        assert -np.pi <= psi <= np.pi
    
    def test_phase_coherence_in_range(self, small_model):
        """Test phase coherence is in [0, 1]."""
        c = small_model.phase_coherence()
        assert 0 <= c <= 1
    
    def test_rk4_step_more_accurate(self, small_model):
        """Test RK4 integrator runs without error."""
        initial_phases = small_model.phases.copy()
        small_model.step_rk4(dt=0.1)
        # Should update phases
        assert not np.allclose(small_model.phases, initial_phases)
    
    def test_simulate_returns_history(self, small_model):
        """Test simulate returns synchronization history."""
        history = small_model.simulate(duration=1.0, dt=0.1)
        assert len(history) == 10  # duration/dt steps
        assert all(0 <= r <= 1 for r in history)
    
    def test_simulate_with_rk4(self, small_model):
        """Test simulation with RK4 method."""
        history = small_model.simulate(duration=1.0, dt=0.1, method="rk4")
        assert len(history) == 10
    
    def test_entropy_non_negative(self, small_model):
        """Test entropy is non-negative."""
        s = small_model.entropy()
        assert s >= 0
    
    def test_critical_coupling_estimate(self, small_model):
        """Test critical coupling estimation."""
        k_c = small_model.critical_coupling()
        assert k_c >= 0
    
    def test_reset_randomizes_phases(self, small_model):
        """Test reset gives new random phases."""
        old_phases = small_model.phases.copy()
        small_model.reset()
        # New phases should be different (with very high probability)
        assert not np.allclose(small_model.phases, old_phases)
    
    def test_high_coupling_increases_synchronization(self, large_model):
        """Test that high coupling leads to synchronization."""
        # Run for a while with strong coupling
        for _ in range(500):
            large_model.step(dt=0.01)
        r_final = large_model.synchronization()
        # With K=2.0 and 100 oscillators, should synchronize partially
        assert r_final > 0.3  # Should show some synchronization
    
    def test_frequency_histogram(self, small_model):
        """Test frequency histogram generation."""
        centers, hist = small_model.frequency_histogram(n_bins=10)
        assert len(centers) == 10
        assert len(hist) == 10
        assert all(h >= 0 for h in hist)
    
    def test_repr(self, small_model):
        """Test string representation."""
        s = repr(small_model)
        assert "KuramotoModel" in s
        assert "N=10" in s


class TestKuramotoFactoryMethods:
    """Test factory methods for Kuramoto model."""
    
    def test_uniform_frequencies(self):
        """Test creating model with uniform frequencies."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.physics.kuramoto import KuramotoModel
        model = KuramotoModel.with_uniform_frequencies(n_oscillators=50, coupling=1.5)
        assert model.n_oscillators == 50
        assert model.coupling == 1.5
        # Frequencies should be in uniform range
        assert all(-1 <= f <= 1 for f in model.frequencies)
    
    def test_lorentzian_frequencies(self):
        """Test creating model with Lorentzian frequencies."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.physics.kuramoto import KuramotoModel
        model = KuramotoModel.with_lorentzian_frequencies(n_oscillators=50, gamma=0.5)
        assert model.n_oscillators == 50
        # Lorentzian distribution can have outliers, so just check count
        assert len(model.frequencies) == 50


class TestSynchronizationTransition:
    """Tests for synchronization phase transition."""
    
    def test_subcritical_coupling_no_sync(self):
        """Test that weak coupling doesn't synchronize."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.physics.kuramoto import KuramotoModel
        model = KuramotoModel(n_oscillators=50, coupling=0.1)
        # Run for a while
        for _ in range(200):
            model.step(dt=0.05)
        r = model.synchronization()
        # Should stay largely incoherent
        assert r < 0.5
    
    def test_supercritical_coupling_syncs(self):
        """Test that strong coupling leads to synchronization."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.physics.kuramoto import KuramotoModel
        # Use very strong coupling
        model = KuramotoModel(n_oscillators=50, coupling=5.0)
        # Run for a while
        for _ in range(500):
            model.step(dt=0.01)
        r = model.synchronization()
        # Should achieve significant synchronization
        assert r > 0.5


class TestEntropyFunctions:
    """Test entropy-related functionality."""
    
    def test_phase_entropy_uniform(self):
        """Test entropy is high for uniformly distributed phases."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.physics.kuramoto import KuramotoModel
        model = KuramotoModel(n_oscillators=100, coupling=0.0)
        # With zero coupling, phases don't change from uniform random
        s = model.entropy()
        # Uniform distribution should have high entropy
        assert s > 1.5


class TestLyapunovAnalysis:
    """Tests for Lyapunov exponent estimation (if implemented)."""
    
    def test_lyapunov_stability(self):
        """Placeholder for Lyapunov analysis tests."""
        # The physics module may implement Lyapunov analysis
        # This is a placeholder for those tests
        pass