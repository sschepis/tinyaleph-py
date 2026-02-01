"""
Tests for runtime module - AlephEngine and transport.
"""
import pytest
import asyncio
import math

# Check for core module availability
core_available = True
try:
    from tinyaleph.hilbert.state import PrimeState
except ImportError:
    core_available = False

pytestmark = pytest.mark.skipif(not core_available, reason="core module required for runtime tests")


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        from tinyaleph.runtime.engine import EngineConfig
        config = EngineConfig()
        assert config.max_iterations == 100
        assert config.enable_distributed == False
    
    def test_custom_config(self):
        """Test custom configuration."""
        from tinyaleph.runtime.engine import EngineConfig
        config = EngineConfig(
            coherence_threshold=0.5,
            entropy_threshold=2.0,
            max_iterations=50
        )
        assert config.coherence_threshold == 0.5
        assert config.max_iterations == 50


class TestEngineState:
    """Tests for EngineState dataclass."""
    
    def test_default_state(self):
        """Test default engine state."""
        from tinyaleph.runtime.engine import EngineState
        state = EngineState()
        assert state.iteration == 0
        assert state.coherence == 1.0
        assert state.halted == False
    
    def test_state_with_prime_state(self):
        """Test engine state with prime state."""
        from tinyaleph.runtime.engine import EngineState
        ps = PrimeState.uniform()
        state = EngineState(prime_state=ps)
        assert state.prime_state is ps


class TestAlephEngine:
    """Tests for AlephEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for testing."""
        from tinyaleph.runtime.engine import AlephEngine
        return AlephEngine()
    
    @pytest.fixture
    def engine_with_config(self):
        """Create engine with custom config."""
        from tinyaleph.runtime.engine import AlephEngine, EngineConfig
        config = EngineConfig(max_iterations=5)
        return AlephEngine(config=config)
    
    def test_engine_creation(self, engine):
        """Test engine creation."""
        assert engine is not None
        assert engine.config is not None
    
    def test_engine_initial_state(self, engine):
        """Test initial engine state."""
        assert engine.state.iteration == 0
        assert not engine.state.halted
    
    def test_bind_concept(self, engine):
        """Test binding concept to state."""
        state = PrimeState.uniform()
        engine.bind_concept("test", state)
        binding = engine.get_binding("test")
        assert binding is not None
    
    def test_unbind_concept(self, engine):
        """Test unbinding concept."""
        state = PrimeState.uniform()
        engine.bind_concept("test", state)
        engine.unbind_concept("test")
        assert engine.get_binding("test") is None
    
    def test_get_binding_unknown(self, engine):
        """Test getting unknown binding."""
        assert engine.get_binding("nonexistent") is None
    
    def test_compose_concepts(self, engine):
        """Test composing concepts."""
        engine.bind_concept("a", PrimeState.uniform())
        engine.bind_concept("b", PrimeState.uniform())
        composed = engine.compose_concepts(["a", "b"])
        assert composed is not None
    
    def test_compose_concepts_empty(self, engine):
        """Test composing empty list."""
        composed = engine.compose_concepts([])
        assert composed is not None
    
    def test_global_coherence(self, engine):
        """Test global coherence computation."""
        coherence = engine.global_coherence
        assert 0 <= coherence <= 1
    
    def test_register_hook(self, engine):
        """Test registering event hook."""
        called = []
        def callback(eng, **kwargs):
            called.append(True)
        
        engine.register_hook('pre_step', callback)
        assert callback in engine.hooks['pre_step']
    
    def test_unregister_hook(self, engine):
        """Test unregistering event hook."""
        def callback(eng, **kwargs):
            pass
        
        engine.register_hook('pre_step', callback)
        engine.unregister_hook('pre_step', callback)
        assert callback not in engine.hooks['pre_step']
    
    @pytest.mark.asyncio
    async def test_step(self, engine):
        """Test single step execution."""
        engine.state.prime_state = PrimeState.uniform()
        state = await engine.step()
        assert state.iteration == 1
    
    @pytest.mark.asyncio
    async def test_step_halts_on_max_iterations(self, engine_with_config):
        """Test step halts at max iterations."""
        engine_with_config.state.prime_state = PrimeState.uniform()
        for _ in range(10):
            state = await engine_with_config.step()
            if state.halted:
                break
        assert engine_with_config.state.halted
        assert engine_with_config.state.halt_reason == 'max_iterations'
    
    @pytest.mark.asyncio
    async def test_run(self, engine_with_config):
        """Test running until halt."""
        state = await engine_with_config.run()
        assert state.halted
    
    def test_run_sync(self, engine_with_config):
        """Test synchronous run."""
        state = engine_with_config.run_sync()
        assert state.halted
    
    def test_collapse(self, engine):
        """Test state collapse."""
        engine.state.prime_state = PrimeState.uniform()
        prime, prob = engine.collapse()
        assert prime >= 2
        assert prob > 0
    
    def test_collapse_no_state(self, engine):
        """Test collapse with no state."""
        prime, prob = engine.collapse()
        assert prime == 2
        assert prob == 1.0
    
    def test_reset(self, engine):
        """Test engine reset."""
        engine.state.iteration = 10
        engine.state.halted = True
        engine.reset()
        assert engine.state.iteration == 0
        assert not engine.state.halted
    
    def test_time_property(self, engine):
        """Test time property."""
        assert engine.time == 0.0
    
    def test_is_running(self, engine):
        """Test is_running property."""
        assert engine.is_running
        engine.state.halted = True
        assert not engine.is_running
    
    def test_repr(self, engine):
        """Test string representation."""
        s = repr(engine)
        assert "AlephEngine" in s


class TestEngineHooks:
    """Test engine hook system."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for testing."""
        from tinyaleph.runtime.engine import AlephEngine, EngineConfig
        config = EngineConfig(max_iterations=3)
        return AlephEngine(config=config)
    
    @pytest.mark.asyncio
    async def test_pre_step_hook(self, engine):
        """Test pre_step hook is called."""
        pre_step_count = [0]
        
        def on_pre_step(eng, **kwargs):
            pre_step_count[0] += 1
        
        engine.register_hook('pre_step', on_pre_step)
        engine.state.prime_state = PrimeState.uniform()
        
        await engine.step()
        assert pre_step_count[0] == 1
    
    @pytest.mark.asyncio
    async def test_post_step_hook(self, engine):
        """Test post_step hook is called."""
        post_step_count = [0]
        
        def on_post_step(eng, **kwargs):
            post_step_count[0] += 1
        
        engine.register_hook('post_step', on_post_step)
        engine.state.prime_state = PrimeState.uniform()
        
        await engine.step()
        assert post_step_count[0] == 1
    
    @pytest.mark.asyncio
    async def test_on_halt_hook(self, engine):
        """Test on_halt hook is called."""
        halt_reason = [None]
        
        def on_halt(eng, reason=None, **kwargs):
            halt_reason[0] = reason
        
        engine.register_hook('on_halt', on_halt)
        
        await engine.run()
        assert halt_reason[0] == 'max_iterations'
    
    def test_on_bind_hook(self, engine):
        """Test on_bind hook is called."""
        bound_concepts = []
        
        def on_bind(eng, concept=None, **kwargs):
            bound_concepts.append(concept)
        
        engine.register_hook('on_bind', on_bind)
        engine.bind_concept("test", PrimeState.uniform())
        
        assert "test" in bound_concepts


class TestTransportLayer:
    """Tests for transport layer (if implemented)."""
    
    def test_local_transport(self):
        """Test local transport creation."""
        try:
            from tinyaleph.runtime.transport import LocalTransport
            transport = LocalTransport()
            assert transport is not None
        except ImportError:
            pytest.skip("Transport module not implemented")
    
    def test_message_creation(self):
        """Test message creation."""
        try:
            from tinyaleph.runtime.transport import Message
            msg = Message(source="A", target="B", payload={"data": 1})
            assert msg.source == "A"
            assert msg.target == "B"
        except ImportError:
            pytest.skip("Transport module not implemented")


class TestDistributedEngine:
    """Tests for distributed engine features."""
    
    def test_distributed_mode_disabled(self):
        """Test distributed mode is disabled by default."""
        from tinyaleph.runtime.engine import AlephEngine
        engine = AlephEngine()
        assert not engine.config.enable_distributed


class TestCoherenceHalting:
    """Tests for coherence-based halting."""
    
    @pytest.mark.asyncio
    async def test_halts_on_low_coherence(self):
        """Test engine halts when coherence drops below threshold."""
        from tinyaleph.runtime.engine import AlephEngine, EngineConfig
        
        config = EngineConfig(
            coherence_threshold=0.9,
            max_iterations=100
        )
        engine = AlephEngine(config=config)
        
        # Manually set low coherence
        engine.state.coherence = 0.5
        engine.state.prime_state = PrimeState.uniform()
        
        await engine.step()
        
        assert engine.state.halted
        assert engine.state.halt_reason == 'coherence_threshold'
    
    @pytest.mark.asyncio
    async def test_halts_on_high_entropy(self):
        """Test engine halts when entropy exceeds threshold."""
        from tinyaleph.runtime.engine import AlephEngine, EngineConfig
        
        config = EngineConfig(
            entropy_threshold=0.5,
            max_iterations=100
        )
        engine = AlephEngine(config=config)
        
        # Manually set high entropy
        engine.state.entropy = 2.0
        engine.state.prime_state = PrimeState.uniform()
        
        await engine.step()
        
        assert engine.state.halted
        assert engine.state.halt_reason == 'entropy_threshold'