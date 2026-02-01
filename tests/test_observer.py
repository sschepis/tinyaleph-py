"""
Tests for observer module - SedenionMemoryField and PRSC.
"""
import pytest
import math

# These modules require numpy
numpy_available = True
try:
    import numpy as np
except ImportError:
    numpy_available = False

pytestmark = pytest.mark.skipif(not numpy_available, reason="numpy required for observer module")


class TestSedenionMemoryField:
    """Tests for SedenionMemoryField class."""
    
    @pytest.fixture
    def smf(self):
        """Create sample SMF for testing."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.observer.smf import SedenionMemoryField
        return SedenionMemoryField(decay_rate=0.01, max_moments=100)
    
    def test_smf_creation(self, smf):
        """Test SMF creation."""
        assert smf.decay_rate == 0.01
        assert smf.max_moments == 100
        assert smf.current_time == 0.0
        assert smf.size == 0
    
    def test_encode_single(self, smf):
        """Test encoding a single memory."""
        moment = smf.encode("Hello world", importance=0.8)
        assert moment.content == "Hello world"
        assert moment.coherence > 0
        assert smf.size == 1
    
    def test_encode_multiple(self, smf):
        """Test encoding multiple memories."""
        smf.encode("First memory")
        smf.encode("Second memory")
        smf.encode("Third memory")
        assert smf.size == 3
    
    def test_recall_empty(self, smf):
        """Test recall from empty memory."""
        results = smf.recall("anything", top_k=5)
        assert results == []
    
    def test_recall_finds_relevant(self, smf):
        """Test recall returns relevant memories."""
        smf.encode("The quick brown fox")
        smf.encode("Jumps over the lazy dog")
        smf.encode("A completely different topic")
        results = smf.recall("quick fox", top_k=2)
        assert len(results) <= 2
    
    def test_recall_by_content(self, smf):
        """Test recall by content matching."""
        smf.encode("Important fact about primes")
        smf.encode("Another unrelated memory")
        results = smf.recall_by_content("primes")
        assert len(results) == 1
        assert "primes" in results[0].content
    
    def test_step_advances_time(self, smf):
        """Test step advances time."""
        smf.step(dt=1.0)
        assert smf.current_time == 1.0
        smf.step(dt=0.5)
        assert smf.current_time == 1.5
    
    def test_consolidate_empty(self, smf):
        """Test consolidate on empty memory."""
        result = smf.consolidate()
        assert result.dim == 16  # Sedenion dimension
    
    def test_consolidate_with_memories(self, smf):
        """Test consolidate with memories."""
        smf.encode("First memory")
        smf.encode("Second memory")
        result = smf.consolidate()
        assert result.dim == 16
    
    def test_superpose(self, smf):
        """Test superposition of memories."""
        m1 = smf.encode("First memory")
        m2 = smf.encode("Second memory")
        result = smf.superpose([m1, m2])
        assert result.dim == 16
    
    def test_superpose_with_weights(self, smf):
        """Test weighted superposition."""
        m1 = smf.encode("First memory")
        m2 = smf.encode("Second memory")
        result = smf.superpose([m1, m2], weights=[0.7, 0.3])
        assert result.dim == 16
    
    def test_decay_all(self, smf):
        """Test decay reduces coherence."""
        m = smf.encode("Test memory")
        initial_coherence = m.coherence
        smf.step(dt=100.0)  # Large time step
        smf.decay_all()
        # Coherence should decrease after decay
        if smf.size > 0:  # Memory might be pruned
            assert smf.moments[0].coherence < initial_coherence
    
    def test_clear(self, smf):
        """Test clearing all memories."""
        smf.encode("Memory 1")
        smf.encode("Memory 2")
        smf.clear()
        assert smf.size == 0
    
    def test_reset_time(self, smf):
        """Test resetting time."""
        smf.step(dt=10.0)
        smf.reset_time()
        assert smf.current_time == 0.0
    
    def test_total_entropy(self, smf):
        """Test total entropy computation."""
        smf.encode("Memory 1")
        smf.encode("Memory 2")
        entropy = smf.total_entropy
        assert entropy >= 0
    
    def test_mean_coherence(self, smf):
        """Test mean coherence computation."""
        smf.encode("Memory 1")
        smf.encode("Memory 2")
        coherence = smf.mean_coherence
        assert 0 <= coherence <= 1
    
    def test_mean_coherence_empty(self, smf):
        """Test mean coherence of empty memory."""
        coherence = smf.mean_coherence
        assert coherence == 1.0
    
    def test_interference(self, smf):
        """Test interference between memories."""
        m1 = smf.encode("First pattern")
        m2 = smf.encode("Second pattern")
        interference = smf.interference(m1, m2)
        assert interference.dim == 16
    
    def test_repr(self, smf):
        """Test string representation."""
        s = repr(smf)
        assert "SedenionMemoryField" in s
    
    def test_pruning(self):
        """Test automatic pruning when exceeding max_moments."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.observer.smf import SedenionMemoryField
        smf = SedenionMemoryField(max_moments=5)
        for i in range(10):
            smf.encode(f"Memory {i}")
        assert smf.size <= 5


class TestMemoryMoment:
    """Tests for MemoryMoment dataclass."""
    
    def test_moment_attributes(self):
        """Test moment has correct attributes."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.observer.smf import SedenionMemoryField
        smf = SedenionMemoryField()
        moment = smf.encode("Test content")
        
        assert hasattr(moment, 'sedenion')
        assert hasattr(moment, 'timestamp')
        assert hasattr(moment, 'entropy')
        assert hasattr(moment, 'coherence')
        assert hasattr(moment, 'content')
        
        assert moment.content == "Test content"
        assert moment.timestamp >= 0
        assert moment.entropy >= 0
        assert 0 <= moment.coherence <= 1


class TestPRSC:
    """Tests for Prime Resonance Semantic Coherence."""
    
    @pytest.fixture
    def prsc(self):
        """Create sample PRSC for testing."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.observer.prsc import PrimeResonanceSemanticCoherence
        return PrimeResonanceSemanticCoherence()
    
    def test_prsc_creation(self, prsc):
        """Test PRSC creation."""
        assert prsc is not None
    
    def test_bind_concept(self, prsc):
        """Test binding a concept to primes."""
        prsc.bind_concept("mathematics", [2, 3, 5, 7])
        binding = prsc.get_binding("mathematics")
        assert binding is not None
    
    def test_get_binding_unknown(self, prsc):
        """Test getting binding for unknown concept."""
        binding = prsc.get_binding("nonexistent")
        assert binding is None
    
    def test_unbind_concept(self, prsc):
        """Test unbinding a concept."""
        prsc.bind_concept("test", [2, 3])
        prsc.unbind_concept("test")
        assert prsc.get_binding("test") is None
    
    def test_compute_coherence(self, prsc):
        """Test computing coherence between concepts."""
        prsc.bind_concept("math", [2, 3, 5])
        prsc.bind_concept("physics", [2, 5, 7])
        coherence = prsc.compute_coherence("math", "physics")
        assert 0 <= coherence <= 1
    
    def test_compose_concepts(self, prsc):
        """Test composing multiple concepts."""
        prsc.bind_concept("math", [2, 3])
        prsc.bind_concept("physics", [5, 7])
        composed = prsc.compose_concepts(["math", "physics"])
        assert composed is not None
    
    def test_semantic_distance(self, prsc):
        """Test semantic distance computation."""
        prsc.bind_concept("math", [2, 3, 5])
        prsc.bind_concept("physics", [2, 5, 7])
        distance = prsc.semantic_distance("math", "physics")
        assert distance >= 0
    
    def test_resonance_strength(self, prsc):
        """Test resonance strength computation."""
        prsc.bind_concept("math", [2, 3])
        strength = prsc.resonance_strength("math")
        assert strength >= 0
    
    def test_list_concepts(self, prsc):
        """Test listing all concepts."""
        prsc.bind_concept("a", [2])
        prsc.bind_concept("b", [3])
        concepts = prsc.list_concepts()
        assert "a" in concepts
        assert "b" in concepts
    
    def test_clear_all(self, prsc):
        """Test clearing all bindings."""
        prsc.bind_concept("a", [2])
        prsc.bind_concept("b", [3])
        prsc.clear_all()
        assert len(prsc.list_concepts()) == 0


class TestPRSCWithState:
    """Tests for PRSC with PrimeState integration."""
    
    def test_bind_state(self):
        """Test binding PrimeState to concept."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.observer.prsc import PrimeResonanceSemanticCoherence
        from tinyaleph.hilbert.state import PrimeState
        
        prsc = PrimeResonanceSemanticCoherence()
        state = PrimeState.uniform()
        prsc.bind_state("uniform", state)
        
        binding = prsc.get_state_binding("uniform")
        assert binding is not None
    
    def test_state_coherence(self):
        """Test coherence between bound states."""
        if not numpy_available:
            pytest.skip("numpy required")
        from tinyaleph.observer.prsc import PrimeResonanceSemanticCoherence
        from tinyaleph.hilbert.state import PrimeState
        
        prsc = PrimeResonanceSemanticCoherence()
        prsc.bind_state("a", PrimeState.uniform())
        prsc.bind_state("b", PrimeState.uniform())
        
        coherence = prsc.state_coherence("a", "b")
        assert 0 <= coherence <= 1