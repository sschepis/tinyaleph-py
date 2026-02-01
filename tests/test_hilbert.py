"""
Tests for Hilbert space operations and quantum operators.
"""
import pytest
import math
from tinyaleph.core.complex import Complex
from tinyaleph.hilbert.operators import (
    PrimeState, Operator, IdentityOperator, PrimeShiftOperator,
    ResonanceOperator, PhaseOperator, ProjectionOperator, CollapseOperator,
    HadamardLikeOperator, TimeEvolutionOperator, GoldenPhaseOperator,
    CompositeOperator, shift, resonance, collapse, phase, project,
    hadamard, evolve, golden_phase, identity
)
from tinyaleph.core.primes import is_prime, nth_prime


class TestPrimeState:
    """Tests for PrimeState class."""
    
    def test_basis_state_creation(self):
        state = PrimeState.basis(2)
        assert 2 in state.amplitudes
        assert state.amplitudes[2].magnitude() == 1.0
    
    def test_basis_state_validation(self):
        with pytest.raises(ValueError):
            PrimeState.basis(4)  # 4 is not prime
    
    def test_superposition(self):
        primes = [2, 3, 5]
        state = PrimeState.superposition(primes)
        
        # Check all primes have amplitudes
        for p in primes:
            assert p in state.amplitudes
        
        # Check normalization
        assert abs(state.norm_squared() - 1.0) < 1e-10
    
    def test_superposition_with_phases(self):
        primes = [2, 3]
        phases = [0.0, math.pi / 2]
        state = PrimeState.superposition(primes, phases)
        
        # First amplitude should be real
        assert abs(state.amplitudes[2].imag) < 1e-10
        # Second amplitude should be purely imaginary
        assert abs(state.amplitudes[3].real) < 1e-10
    
    def test_addition(self):
        state1 = PrimeState.basis(2)
        state2 = PrimeState.basis(3)
        combined = state1 + state2
        
        assert 2 in combined.amplitudes
        assert 3 in combined.amplitudes
    
    def test_scalar_multiplication(self):
        state = PrimeState.basis(2)
        scaled = state * Complex(2.0, 0.0)
        
        assert scaled.amplitudes[2].magnitude() == 2.0
    
    def test_normalization(self):
        amplitudes = {2: Complex(3.0, 0.0), 3: Complex(4.0, 0.0)}
        state = PrimeState(amplitudes=amplitudes)
        
        normalized = state.normalize()
        assert abs(normalized.norm_squared() - 1.0) < 1e-10
    
    def test_inner_product(self):
        state1 = PrimeState.basis(2)
        state2 = PrimeState.basis(2)
        state3 = PrimeState.basis(3)
        
        # Same basis state should give 1
        ip12 = state1.inner_product(state2)
        assert abs(ip12.magnitude() - 1.0) < 1e-10
        
        # Different basis states should give 0
        ip13 = state1.inner_product(state3)
        assert abs(ip13.magnitude()) < 1e-10
    
    def test_probabilities(self):
        state = PrimeState.superposition([2, 3, 5])
        probs = state.probabilities()
        
        # Equal superposition should have equal probabilities
        for prob in probs.values():
            assert abs(prob - 1/3) < 1e-10
        
        # Probabilities should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-10
    
    def test_entropy(self):
        # Single basis state has zero entropy
        pure_state = PrimeState.basis(2)
        assert abs(pure_state.entropy()) < 1e-10
        
        # Equal superposition has maximum entropy
        mixed_state = PrimeState.superposition([2, 3])
        assert mixed_state.entropy() > 0


class TestIdentityOperator:
    """Tests for Identity operator."""
    
    def test_identity_preserves_state(self):
        state = PrimeState.superposition([2, 3, 5])
        op = IdentityOperator()
        result = op.apply(state)
        
        # Should be identical
        for p in state.amplitudes:
            assert abs(state.amplitudes[p].magnitude() - 
                      result.amplitudes[p].magnitude()) < 1e-10


class TestPrimeShiftOperator:
    """Tests for Prime Shift operator."""
    
    def test_shift_up(self):
        state = PrimeState.basis(2)  # |2⟩
        op = PrimeShiftOperator(shift=1)
        result = op.apply(state)
        
        # Should shift 2 -> 3
        assert 3 in result.amplitudes
    
    def test_shift_multiple(self):
        state = PrimeState.basis(2)
        op = PrimeShiftOperator(shift=3)
        result = op.apply(state)
        
        # 2 is 1st prime, shifted by 3 -> 4th prime = 7
        assert 7 in result.amplitudes
    
    def test_shift_preserves_normalization(self):
        state = PrimeState.superposition([2, 3, 5])
        op = PrimeShiftOperator(shift=2)
        result = op.apply(state)
        
        assert abs(result.norm_squared() - 1.0) < 1e-10


class TestResonanceOperator:
    """Tests for Resonance operator."""
    
    def test_resonance_applies_phase(self):
        state = PrimeState.superposition([2, 3, 5])
        op = ResonanceOperator(frequency=1.0, coupling=0.1)
        result = op.apply(state)
        
        # Should preserve normalization
        assert abs(result.norm_squared() - 1.0) < 1e-10
    
    def test_zero_coupling(self):
        state = PrimeState.superposition([2, 3])
        op = ResonanceOperator(frequency=1.0, coupling=0.0)
        result = op.apply(state)
        
        # Zero coupling should preserve state
        for p in state.amplitudes:
            assert abs(state.amplitudes[p].magnitude() - 
                      result.amplitudes[p].magnitude()) < 1e-10


class TestPhaseOperator:
    """Tests for Phase operator."""
    
    def test_phase_on_target_prime(self):
        state = PrimeState.basis(2)
        op = PhaseOperator(prime=2, phase=math.pi)
        result = op.apply(state)
        
        # Phase of π should flip sign
        assert abs(result.amplitudes[2].real + 1.0) < 1e-10
    
    def test_phase_on_other_prime(self):
        state = PrimeState.basis(2)
        op = PhaseOperator(prime=3, phase=math.pi)
        result = op.apply(state)
        
        # Should not affect state with only |2⟩
        assert abs(result.amplitudes[2].real - 1.0) < 1e-10
    
    def test_phase_validation(self):
        with pytest.raises(ValueError):
            PhaseOperator(prime=4, phase=0.0)  # 4 is not prime


class TestProjectionOperator:
    """Tests for Projection operator."""
    
    def test_projection(self):
        state = PrimeState.superposition([2, 3, 5, 7])
        op = ProjectionOperator(primes=[2, 3])
        result = op.apply(state)
        
        # Should only contain projected primes
        assert 2 in result.amplitudes
        assert 3 in result.amplitudes
        assert 5 not in result.amplitudes
        assert 7 not in result.amplitudes
    
    def test_projection_normalization(self):
        state = PrimeState.superposition([2, 3, 5, 7])
        op = ProjectionOperator(primes=[2, 3])
        result = op.apply(state)
        
        # Should be normalized
        assert abs(result.norm_squared() - 1.0) < 1e-10


class TestCollapseOperator:
    """Tests for Collapse operator."""
    
    def test_deterministic_collapse(self):
        # Create state with one dominant amplitude
        amplitudes = {
            2: Complex(0.9, 0.0),
            3: Complex(0.1, 0.0)
        }
        state = PrimeState(amplitudes=amplitudes).normalize()
        
        op = CollapseOperator(deterministic=True)
        result = op.apply(state)
        
        # Should collapse to highest probability
        assert 2 in result.amplitudes
        assert len(result.amplitudes) == 1
    
    def test_collapse_produces_basis_state(self):
        state = PrimeState.superposition([2, 3, 5])
        op = CollapseOperator(seed=42)
        result = op.apply(state)
        
        # Should collapse to single basis state
        assert len(result.amplitudes) == 1
        
        # That state should have unit amplitude
        the_prime = list(result.amplitudes.keys())[0]
        assert abs(result.amplitudes[the_prime].magnitude() - 1.0) < 1e-10


class TestHadamardLikeOperator:
    """Tests for Hadamard-like operator."""
    
    def test_hadamard_creates_superposition(self):
        state = PrimeState.basis(2)
        op = HadamardLikeOperator()
        result = op.apply(state)
        
        # Should create superposition of 2 and 3 (next prime)
        assert 2 in result.amplitudes
        assert 3 in result.amplitudes
    
    def test_hadamard_normalization(self):
        state = PrimeState.basis(2)
        op = HadamardLikeOperator()
        result = op.apply(state)
        
        assert abs(result.norm_squared() - 1.0) < 1e-10


class TestTimeEvolutionOperator:
    """Tests for Time Evolution operator."""
    
    def test_zero_time(self):
        state = PrimeState.superposition([2, 3])
        op = TimeEvolutionOperator(time=0.0)
        result = op.apply(state)
        
        # Zero time should preserve magnitudes
        for p in state.amplitudes:
            assert abs(state.amplitudes[p].magnitude() - 
                      result.amplitudes[p].magnitude()) < 1e-10
    
    def test_evolution_preserves_normalization(self):
        state = PrimeState.superposition([2, 3, 5])
        op = TimeEvolutionOperator(time=1.5, frequency=0.5)
        result = op.apply(state)
        
        assert abs(result.norm_squared() - 1.0) < 1e-10


class TestGoldenPhaseOperator:
    """Tests for Golden Phase operator."""
    
    def test_golden_phase_normalization(self):
        state = PrimeState.superposition([2, 3, 5, 7])
        op = GoldenPhaseOperator()
        result = op.apply(state)
        
        assert abs(result.norm_squared() - 1.0) < 1e-10


class TestOperatorComposition:
    """Tests for operator composition."""
    
    def test_composition(self):
        state = PrimeState.basis(2)
        
        # Compose shift and phase
        shift_op = PrimeShiftOperator(shift=1)
        phase_op = PhaseOperator(prime=3, phase=math.pi)
        
        composite = CompositeOperator([phase_op, shift_op])
        result = composite.apply(state)
        
        # First shift 2->3, then apply phase to 3
        assert 3 in result.amplitudes
    
    def test_matmul_composition(self):
        op1 = IdentityOperator()
        op2 = IdentityOperator()
        
        composite = op1 @ op2
        assert isinstance(composite, CompositeOperator)


class TestConvenienceFunctions:
    """Tests for convenience factory functions."""
    
    def test_shift(self):
        op = shift(2)
        assert isinstance(op, PrimeShiftOperator)
        assert op.shift == 2
    
    def test_resonance_factory(self):
        op = resonance(frequency=2.0, coupling=0.5)
        assert isinstance(op, ResonanceOperator)
        assert op.frequency == 2.0
        assert op.coupling == 0.5
    
    def test_collapse_factory(self):
        op = collapse(deterministic=True)
        assert isinstance(op, CollapseOperator)
        assert op.deterministic == True
    
    def test_phase_factory(self):
        op = phase(prime=7, theta=math.pi)
        assert isinstance(op, PhaseOperator)
        assert op.prime == 7
    
    def test_project_factory(self):
        op = project([2, 3, 5])
        assert isinstance(op, ProjectionOperator)
        assert op.primes == [2, 3, 5]
    
    def test_hadamard_factory(self):
        op = hadamard()
        assert isinstance(op, HadamardLikeOperator)
    
    def test_evolve_factory(self):
        op = evolve(time=1.0, frequency=2.0)
        assert isinstance(op, TimeEvolutionOperator)
        assert op.time == 1.0
        assert op.frequency == 2.0
    
    def test_golden_phase_factory(self):
        op = golden_phase()
        assert isinstance(op, GoldenPhaseOperator)
    
    def test_identity_factory(self):
        op = identity()
        assert isinstance(op, IdentityOperator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])