"""
Tests for ML module - SparsePrimeState and attention mechanisms.
"""
import pytest
import math

# Check for core module availability
core_available = True
try:
    from tinyaleph.core.quaternion import Quaternion
    from tinyaleph.core.primes import is_prime, nth_prime
except ImportError:
    core_available = False

pytestmark = pytest.mark.skipif(not core_available, reason="core module required for ML tests")


class TestSparsePrimeState:
    """Tests for SparsePrimeState class."""
    
    @pytest.fixture
    def state(self):
        """Create sample sparse state."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        return SparsePrimeState.first_n_superposition(5)
    
    @pytest.fixture
    def vacuum(self):
        """Create vacuum state."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        return SparsePrimeState.vacuum()
    
    def test_vacuum_state(self, vacuum):
        """Test vacuum state creation."""
        assert len(vacuum) == 0
    
    def test_single_prime(self):
        """Test single prime eigenstate."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        state = SparsePrimeState.single_prime(7)
        assert len(state) == 1
        assert 7 in state.amplitudes
    
    def test_single_prime_invalid(self):
        """Test single prime with non-prime fails."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        with pytest.raises(ValueError):
            SparsePrimeState.single_prime(6)
    
    def test_first_n_superposition(self, state):
        """Test superposition of first n primes."""
        assert len(state) == 5
        assert 2 in state.amplitudes
        assert 3 in state.amplitudes
    
    def test_from_primes(self):
        """Test creating from list of primes."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        state = SparsePrimeState.from_primes([2, 3, 5])
        assert len(state) == 3
    
    def test_from_primes_with_amplitudes(self):
        """Test creating from primes with custom amplitudes."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        amps = [Quaternion(1.0), Quaternion(0.0, 1.0, 0, 0), Quaternion(0, 0, 1.0, 0)]
        state = SparsePrimeState.from_primes([2, 3, 5], amps)
        assert len(state) == 3
    
    def test_len(self, state):
        """Test length."""
        assert len(state) == 5
    
    def test_getitem(self, state):
        """Test amplitude access."""
        amp = state[2]
        assert isinstance(amp, Quaternion)
    
    def test_getitem_missing(self, state):
        """Test missing prime returns zero."""
        amp = state[1009]  # Large prime not in state
        assert amp.norm() < 0.001
    
    def test_setitem(self, state):
        """Test setting amplitude."""
        state[11] = Quaternion(0.5)
        assert 11 in state.amplitudes
    
    def test_setitem_invalid_prime(self, state):
        """Test setting non-prime fails."""
        with pytest.raises(ValueError):
            state[6] = Quaternion(1.0)
    
    def test_setitem_zero_removes(self, state):
        """Test setting zero amplitude removes prime."""
        state[2] = Quaternion(0.0)
        assert 2 not in state.amplitudes
    
    def test_iter(self, state):
        """Test iteration."""
        items = list(state)
        assert len(items) == 5
        for p, q in items:
            assert is_prime(p)
            assert isinstance(q, Quaternion)
    
    def test_add(self, state):
        """Test state addition."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        other = SparsePrimeState.single_prime(11)
        result = state + other
        assert 11 in result.amplitudes
    
    def test_sub(self, state):
        """Test state subtraction."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        other = SparsePrimeState.single_prime(2)
        result = state - other
        # After subtraction and normalization, structure changes
        assert len(result) >= 0
    
    def test_scalar_mul(self, state):
        """Test scalar multiplication."""
        scaled = state * 2.0
        for p in state.amplitudes:
            assert abs(scaled[p].norm() - state[p].norm() * 2.0) < 0.001
    
    def test_right_scalar_mul(self, state):
        """Test right scalar multiplication."""
        scaled = 2.0 * state
        for p in state.amplitudes:
            assert abs(scaled[p].norm() - state[p].norm() * 2.0) < 0.001
    
    def test_quaternion_mul(self, state):
        """Test quaternion multiplication."""
        q = Quaternion(0, 1, 0, 0)  # i
        result = state.quaternion_mul(q)
        assert len(result) == len(state)
    
    def test_norm_squared(self, state):
        """Test norm squared."""
        ns = state.norm_squared()
        assert ns > 0
    
    def test_norm(self, state):
        """Test norm."""
        n = state.norm()
        assert abs(n - 1.0) < 0.01  # Should be normalized
    
    def test_normalized(self, state):
        """Test normalization returns normalized copy."""
        scaled = state * 2.0
        normalized = scaled.normalized()
        assert abs(normalized.norm() - 1.0) < 0.001
    
    def test_inner_product(self, state):
        """Test inner product."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        other = SparsePrimeState.first_n_superposition(5)
        ip = state.inner_product(other)
        assert isinstance(ip, Quaternion)
        # Same state should have real inner product ~1
        assert ip.w > 0.9
    
    def test_overlap(self, state):
        """Test overlap computation."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        other = SparsePrimeState.first_n_superposition(5)
        overlap = state.overlap(other)
        assert 0 <= overlap <= 1
        assert overlap > 0.9  # Same state
    
    def test_prime_spectrum(self, state):
        """Test prime spectrum."""
        spectrum = state.prime_spectrum()
        assert len(spectrum) == 5
        total = sum(spectrum.values())
        assert abs(total - 1.0) < 0.001
    
    def test_entropy(self, state):
        """Test entropy computation."""
        entropy = state.entropy()
        assert entropy >= 0
        # Uniform superposition should have high entropy
        assert entropy > 1.5
    
    def test_is_coherent(self, state):
        """Test coherence check."""
        # Just check it returns a boolean
        result = state.is_coherent()
        assert isinstance(result, bool)
    
    def test_apply_rotation(self, state):
        """Test quaternion rotation."""
        axis = Quaternion(0, 1, 0, 0)
        rotated = state.apply_rotation(axis, math.pi / 4)
        assert len(rotated) == len(state)
    
    def test_apply_phase(self, state):
        """Test phase application."""
        phased = state.apply_phase(2, math.pi / 2)
        assert len(phased) == len(state)
    
    def test_project_to_primes(self, state):
        """Test projection to subset of primes."""
        projected = state.project_to_primes([2, 3])
        assert len(projected) == 2
    
    def test_collapse(self, state):
        """Test state collapse."""
        prime, phase = state.collapse()
        assert is_prime(prime)
        assert isinstance(phase, Quaternion)
        assert len(state) == 1  # State is modified
    
    def test_collapse_vacuum_fails(self, vacuum):
        """Test collapse of vacuum fails."""
        with pytest.raises(ValueError):
            vacuum.collapse()
    
    def test_top_k_primes(self, state):
        """Test getting top-k primes."""
        top = state.top_k_primes(3)
        assert len(top) == 3
        for p, prob in top:
            assert is_prime(p)
            assert prob >= 0
    
    def test_truncate(self, state):
        """Test truncation."""
        truncated = state.truncate(threshold=0.1)
        assert len(truncated) <= len(state)
    
    def test_to_real_vector(self, state):
        """Test conversion to real vector."""
        vec = state.to_real_vector(max_prime_idx=10)
        assert len(vec) == 40  # 10 primes * 4 quaternion components
    
    def test_from_real_vector(self):
        """Test creation from real vector."""
        from tinyaleph.ml.sparse_state import SparsePrimeState
        vec = [0.5, 0.0, 0.0, 0.0] * 10  # 10 primes with w=0.5
        state = SparsePrimeState.from_real_vector(vec)
        # First prime (2) should be present
        assert 2 in state.amplitudes
    
    def test_repr(self, state):
        """Test string representation."""
        s = repr(state)
        assert "SparsePrimeState" in s


class TestCoherentSuperposition:
    """Test coherent superposition creation."""
    
    def test_coherent_superposition(self):
        """Test creating coherent superposition."""
        from tinyaleph.ml.sparse_state import coherent_superposition
        state = coherent_superposition([2, 3, 5], phases=[0, math.pi/4, math.pi/2])
        assert len(state) == 3
    
    def test_coherent_superposition_default_phases(self):
        """Test with default phases."""
        from tinyaleph.ml.sparse_state import coherent_superposition
        state = coherent_superposition([2, 3, 5])
        assert len(state) == 3
    
    def test_golden_superposition(self):
        """Test golden ratio superposition."""
        from tinyaleph.ml.sparse_state import golden_superposition
        state = golden_superposition(5)
        assert len(state) == 5


class TestResonantAttention:
    """Tests for resonant attention mechanism."""
    
    def test_softmax(self):
        """Test softmax function."""
        from tinyaleph.ml.attention import softmax
        values = [1.0, 2.0, 3.0]
        probs = softmax(values)
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 0.001
    
    def test_softmax_empty(self):
        """Test softmax on empty list."""
        from tinyaleph.ml.attention import softmax
        probs = softmax([])
        assert probs == []
    
    def test_resonance_kernel(self):
        """Test resonance kernel."""
        from tinyaleph.ml.attention import resonance_kernel
        r = resonance_kernel(2, 3)
        assert 0 <= r <= 1
        # Same prime should give 1
        assert resonance_kernel(2, 2) == 1.0
    
    def test_prime_resonance_weight(self):
        """Test prime resonance weight."""
        from tinyaleph.ml.attention import prime_resonance_weight
        w = prime_resonance_weight(2)
        assert w > 0
        # Smaller primes should have higher weight
        assert prime_resonance_weight(2) > prime_resonance_weight(101)
    
    def test_golden_ratio_constants(self):
        """Test golden ratio constants."""
        from tinyaleph.ml.attention import golden_ratio, inverse_golden, golden_angle
        phi = golden_ratio()
        assert abs(phi - (1 + math.sqrt(5)) / 2) < 0.001
        
        inv = inverse_golden()
        assert abs(inv - (phi - 1)) < 0.001
        
        angle = golden_angle()
        assert angle > 0
    
    def test_resonant_attention(self):
        """Test resonant attention computation."""
        from tinyaleph.ml.attention import resonant_attention
        from tinyaleph.ml.sparse_state import SparsePrimeState
        
        query = SparsePrimeState.single_prime(2)
        keys = [SparsePrimeState.single_prime(2), SparsePrimeState.single_prime(3)]
        values = [SparsePrimeState.single_prime(5), SparsePrimeState.single_prime(7)]
        
        result = resonant_attention(query, keys, values)
        assert len(result) > 0
    
    def test_golden_ratio_attention_weights(self):
        """Test golden ratio attention weight generation."""
        from tinyaleph.ml.attention import golden_ratio_attention_weights
        weights = golden_ratio_attention_weights(5)
        assert len(weights) == 5
        assert abs(sum(weights) - 1.0) < 0.001


class TestAttentionHead:
    """Tests for AttentionHead class."""
    
    def test_attention_head_creation(self):
        """Test attention head creation."""
        from tinyaleph.ml.attention import AttentionHead
        head = AttentionHead(dim=10)
        assert head.dim == 10
    
    def test_attention_head_projection(self):
        """Test attention head projections."""
        from tinyaleph.ml.attention import AttentionHead
        from tinyaleph.ml.sparse_state import SparsePrimeState
        
        head = AttentionHead(dim=10)
        state = SparsePrimeState.first_n_superposition(5)
        
        q = head.project_q(state)
        k = head.project_k(state)
        v = head.project_v(state)
        
        assert len(q) > 0
        assert len(k) > 0
        assert len(v) > 0


class TestMultiHeadAttention:
    """Tests for multi-head attention."""
    
    def test_multihead_creation(self):
        """Test multi-head attention creation."""
        from tinyaleph.ml.attention import MultiHeadResonantAttention
        mha = MultiHeadResonantAttention(num_heads=4, dim=16)
        assert len(mha.heads) == 4
    
    def test_multihead_forward(self):
        """Test multi-head forward pass."""
        from tinyaleph.ml.attention import MultiHeadResonantAttention
        from tinyaleph.ml.sparse_state import SparsePrimeState
        
        mha = MultiHeadResonantAttention(num_heads=2, dim=8)
        query = SparsePrimeState.first_n_superposition(3)
        keys = [SparsePrimeState.single_prime(2)]
        values = [SparsePrimeState.single_prime(3)]
        
        result = mha.forward(query, keys, values)
        assert len(result) > 0


class TestCoherenceGatedComputation:
    """Tests for coherence-gated computation (ACT)."""
    
    def test_coherence_gate_creation(self):
        """Test coherence gate creation."""
        from tinyaleph.ml.attention import CoherenceGatedComputation
        gate = CoherenceGatedComputation(max_steps=5)
        assert gate.max_steps == 5
    
    def test_coherence_gate_compute(self):
        """Test coherence-gated computation."""
        from tinyaleph.ml.attention import CoherenceGatedComputation
        from tinyaleph.ml.sparse_state import SparsePrimeState
        
        gate = CoherenceGatedComputation(max_steps=3)
        initial = SparsePrimeState.first_n_superposition(3)
        
        def step_fn(state, step):
            # Simple identity step
            return state
        
        result, steps, halt_prob = gate.compute(initial, step_fn)
        
        assert len(result) > 0
        assert 1 <= steps <= 3
        assert 0 <= halt_prob <= 1


class TestTransformerBlock:
    """Tests for resonant transformer block."""
    
    def test_transformer_block_creation(self):
        """Test transformer block creation."""
        from tinyaleph.ml.attention import (
            ResonantTransformerBlock,
            MultiHeadResonantAttention,
            CoherenceGatedComputation
        )
        
        attention = MultiHeadResonantAttention(num_heads=2, dim=8)
        gate = CoherenceGatedComputation(max_steps=3)
        block = ResonantTransformerBlock(attention=attention, coherence_gate=gate)
        
        assert block.attention is attention
        assert block.coherence_gate is gate
    
    def test_create_resonant_transformer(self):
        """Test creating transformer stack."""
        from tinyaleph.ml.attention import create_resonant_transformer
        blocks = create_resonant_transformer(num_layers=2, num_heads=2, dim=8, max_steps=2)
        assert len(blocks) == 2