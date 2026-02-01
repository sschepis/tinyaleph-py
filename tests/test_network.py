"""
Tests for network module - identity and entanglement.
"""
import pytest
import math

# These modules require numpy
numpy_available = True
try:
    import numpy as np
except ImportError:
    numpy_available = False

pytestmark = pytest.mark.skipif(not numpy_available, reason="numpy required for network module")


class TestPrimeResonanceIdentity:
    """Tests for Prime Resonance Identity (PRI)."""
    
    @pytest.fixture
    def sample_pri(self):
        """Create sample PRI for testing."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import PrimeResonanceIdentity
        return PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
    
    def test_pri_creation(self, sample_pri):
        """Test PRI creation with prime components."""
        assert sample_pri.gaussian == 3
        assert sample_pri.eisenstein == 7
        assert sample_pri.quaternionic == 11
    
    def test_pri_signature(self, sample_pri):
        """Test signature property returns tuple."""
        sig = sample_pri.signature
        assert sig == (3, 7, 11)
    
    def test_pri_hash(self, sample_pri):
        """Test hash is computed from product."""
        h = sample_pri.hash
        expected = (3 * 7 * 11) % 1000000007
        assert h == expected
    
    def test_pri_product(self, sample_pri):
        """Test product property."""
        assert sample_pri.product == 3 * 7 * 11
    
    def test_pri_random(self):
        """Test random PRI creation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import PrimeResonanceIdentity
        pri = PrimeResonanceIdentity.random()
        assert isinstance(pri.gaussian, int)
        assert isinstance(pri.eisenstein, int)
        assert isinstance(pri.quaternionic, int)
    
    def test_pri_from_seed(self):
        """Test deterministic PRI creation from seed."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import PrimeResonanceIdentity
        pri1 = PrimeResonanceIdentity.from_seed(42)
        pri2 = PrimeResonanceIdentity.from_seed(42)
        assert pri1.signature == pri2.signature
    
    def test_pri_from_string(self):
        """Test PRI creation from string."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import PrimeResonanceIdentity
        pri = PrimeResonanceIdentity.from_string("test_node")
        assert isinstance(pri.gaussian, int)
    
    def test_entanglement_strength_identical(self):
        """Test entanglement strength between identical PRIs."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import PrimeResonanceIdentity
        pri1 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        pri2 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        strength = pri1.entanglement_strength(pri2)
        assert strength == 1.0
    
    def test_entanglement_strength_different(self):
        """Test entanglement strength between different PRIs."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import PrimeResonanceIdentity
        pri1 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        pri2 = PrimeResonanceIdentity(gaussian=5, eisenstein=13, quaternionic=17)
        strength = pri1.entanglement_strength(pri2)
        assert 0 <= strength < 1.0
    
    def test_resonance_frequency(self, sample_pri):
        """Test resonance frequency computation."""
        freq = sample_pri.resonance_frequency()
        expected = np.log(3) + np.log(7) + np.log(11)
        assert abs(freq - expected) < 0.001
    
    def test_phase(self, sample_pri):
        """Test phase computation."""
        phase = sample_pri.phase()
        assert 0 <= phase < 2 * np.pi
    
    def test_is_compatible(self, sample_pri):
        """Test compatibility check."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import PrimeResonanceIdentity
        pri2 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        assert sample_pri.is_compatible(pri2, threshold=0.5)
    
    def test_equality(self, sample_pri):
        """Test PRI equality."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import PrimeResonanceIdentity
        pri2 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        assert sample_pri == pri2
    
    def test_repr(self, sample_pri):
        """Test string representation."""
        s = repr(sample_pri)
        assert "PRI" in s
        assert "3" in s


class TestEntangledNode:
    """Tests for EntangledNode class."""
    
    @pytest.fixture
    def node(self):
        """Create sample entangled node."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import EntangledNode, PrimeResonanceIdentity
        pri = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        return EntangledNode(pri=pri)
    
    def test_node_creation(self, node):
        """Test node creation."""
        assert node.coherence == 1.0
        assert node.entropy == 0.0
        assert node.entangled_with == []
    
    def test_node_id(self, node):
        """Test node ID is derived from PRI."""
        assert node.id == node.pri.hash
    
    def test_node_random(self):
        """Test random node creation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import EntangledNode
        node = EntangledNode.random()
        assert node.coherence == 1.0
    
    def test_node_from_seed(self):
        """Test deterministic node creation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import EntangledNode
        node1 = EntangledNode.from_seed(42)
        node2 = EntangledNode.from_seed(42)
        assert node1.pri.signature == node2.pri.signature
    
    def test_can_entangle_high_coherence(self, node):
        """Test entanglement possible with high coherence."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import EntangledNode
        other = EntangledNode.random()
        # With random PRIs, compatibility varies
        # Just check the method runs
        result = node.can_entangle(other, threshold=0.0)
        assert isinstance(result, bool)
    
    def test_can_entangle_low_coherence(self, node):
        """Test entanglement blocked with low coherence."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import EntangledNode
        other = EntangledNode.random()
        other.coherence = 0.1  # Too low
        assert not node.can_entangle(other)
    
    def test_entangle_success(self, node):
        """Test successful entanglement."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import EntangledNode, PrimeResonanceIdentity
        # Create compatible node
        pri2 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        other = EntangledNode(pri=pri2)
        result = node.entangle(other)
        assert result
        assert other.id in node.entangled_with
    
    def test_disentangle(self, node):
        """Test disentanglement."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import EntangledNode, PrimeResonanceIdentity
        pri2 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        other = EntangledNode(pri=pri2)
        node.entangle(other)
        node.disentangle(other)
        assert other.id not in node.entangled_with
    
    def test_is_entangled_with(self, node):
        """Test entanglement check."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.identity import EntangledNode, PrimeResonanceIdentity
        pri2 = PrimeResonanceIdentity(gaussian=3, eisenstein=7, quaternionic=11)
        other = EntangledNode(pri=pri2)
        assert not node.is_entangled_with(other)
        node.entangle(other)
        assert node.is_entangled_with(other)
    
    def test_entanglement_count(self, node):
        """Test entanglement count."""
        assert node.entanglement_count() == 0
    
    def test_reset(self, node):
        """Test node reset."""
        node.entropy = 0.5
        node.coherence = 0.5
        node.reset()
        assert node.coherence == 1.0
        assert node.entropy == 0.0
    
    def test_decay(self, node):
        """Test decay function."""
        node.decay(rate=0.1)
        assert node.coherence < 1.0
        assert node.entropy > 0.0


class TestEntangledPair:
    """Tests for EntangledPair class."""
    
    @pytest.fixture
    def pair(self):
        """Create sample entangled pair."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import EntangledPair
        return EntangledPair(prime_a=2, prime_b=3)
    
    def test_pair_creation(self, pair):
        """Test pair creation."""
        assert pair.prime_a == 2
        assert pair.prime_b == 3
        assert pair.fidelity == 1.0
    
    def test_pair_invalid_prime(self):
        """Test pair creation with non-prime fails."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import EntangledPair
        with pytest.raises(ValueError):
            EntangledPair(prime_a=4, prime_b=3)
    
    def test_pair_same_prime(self):
        """Test pair creation with same prime fails."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import EntangledPair
        with pytest.raises(ValueError):
            EntangledPair(prime_a=3, prime_b=3)
    
    def test_measure_a(self, pair):
        """Test measurement on particle A."""
        outcome_a, predicted_b = pair.measure_a()
        assert outcome_a in [2, 3]
        assert predicted_b in [2, 3]
    
    def test_is_maximally_entangled(self, pair):
        """Test maximal entanglement check."""
        assert pair.is_maximally_entangled()
        pair.fidelity = 0.5
        assert not pair.is_maximally_entangled()


class TestEntanglementNetwork:
    """Tests for EntanglementNetwork class."""
    
    @pytest.fixture
    def network(self):
        """Create sample network."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import EntanglementNetwork
        return EntanglementNetwork()
    
    def test_add_node(self, network):
        """Test adding nodes."""
        network.add_node("A")
        network.add_node("B")
        assert "A" in network.nodes
        assert "B" in network.nodes
    
    def test_remove_node(self, network):
        """Test removing nodes."""
        network.add_node("A")
        network.remove_node("A")
        assert "A" not in network.nodes
    
    def test_establish_link(self, network):
        """Test establishing entanglement link."""
        network.add_node("A")
        network.add_node("B")
        # Link may or may not succeed (probabilistic)
        for _ in range(10):  # Try multiple times
            pair = network.establish_link("A", "B")
            if pair:
                assert pair.node_a == "A"
                assert pair.node_b == "B"
                break
    
    def test_are_entangled(self, network):
        """Test entanglement check."""
        network.add_node("A")
        network.add_node("B")
        assert not network.are_entangled("A", "B")
    
    def test_get_links(self, network):
        """Test getting links for a node."""
        network.add_node("A")
        links = network.get_links("A")
        assert links == []
    
    def test_total_entanglement(self, network):
        """Test total entanglement computation."""
        total = network.total_entanglement()
        assert total == 0.0  # Empty network
    
    def test_average_fidelity_empty(self, network):
        """Test average fidelity of empty network."""
        avg = network.average_fidelity()
        assert avg == 0.0


class TestEntanglementSource:
    """Tests for EntanglementSource class."""
    
    def test_generate_pair(self):
        """Test pair generation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import EntanglementSource
        source = EntanglementSource(success_probability=1.0)
        pair = source.generate()
        assert pair is not None
        assert pair.prime_a == 2
        assert pair.prime_b == 3
    
    def test_generate_n(self):
        """Test generating multiple pairs."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import EntanglementSource
        source = EntanglementSource(success_probability=1.0)
        pairs = source.generate_n(5)
        assert len(pairs) == 5


class TestBellStates:
    """Tests for Bell state functions."""
    
    def test_create_ghz_state(self):
        """Test GHZ state creation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import create_ghz_state
        state = create_ghz_state([2, 3, 5])
        assert len(state) == 2
        assert (2, 2, 2) in state
        assert (3, 3, 3) in state
    
    def test_create_w_state(self):
        """Test W state creation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import create_w_state
        state = create_w_state([2, 3, 5])
        assert len(state) == 3
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy computation."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.network.entanglement import entanglement_entropy
        from tinyaleph.core.complex import Complex
        # Bell state
        state = {
            (2, 2): Complex(1/math.sqrt(2), 0),
            (3, 3): Complex(1/math.sqrt(2), 0)
        }
        entropy = entanglement_entropy(state)
        assert entropy >= 0