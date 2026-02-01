"""
Resonant Attention Mechanism

Implements attention with prime-resonance weighting and coherence gating.
Inspired by transformer attention but operating in the prime Hilbert space H_Q.

Core concepts:
1. Golden ratio attention: Uses Φ-based scaling for optimal information spread
2. Resonant kernels: Attention weights based on prime resonance patterns
3. Coherence gating: Halts computation when coherence threshold reached (ACT)

Mathematical Foundation:
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
    
    Resonant modification:
    ResonantAttention(Q, K, V) = softmax(R(Q, K) / √d_k) V
    
    where R(Q, K) = Σ_p α_p(Q) · β_p(K) · resonance(p)
    
    Coherence gate:
    output = Σ_t h_t · (1 - Σ_{τ<t} h_τ) · s_t
    
    where h_t is halting probability and s_t is step output.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
import math
from functools import lru_cache

from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.primes import is_prime, nth_prime, prime_sieve
from tinyaleph.core.constants import (
    PHI, LAMBDA_STABILITY_THRESHOLD, COHERENCE_THRESHOLD
)
from tinyaleph.ml.sparse_state import SparsePrimeState


# Golden ratio constants
def golden_ratio() -> float:
    """Return the golden ratio Φ = (1 + √5) / 2."""
    return PHI


def inverse_golden() -> float:
    """Return 1/Φ = Φ - 1."""
    return PHI - 1


def golden_angle() -> float:
    """Return golden angle in radians: 2π/Φ²."""
    return 2 * math.pi / (PHI ** 2)


@lru_cache(maxsize=1000)
def prime_resonance_weight(p: int) -> float:
    """
    Compute resonance weight for prime p.
    
    Uses logarithmic scaling with golden ratio modulation:
    w(p) = 1 / (1 + log(p)/log(Φ))
    
    This gives higher weight to smaller primes while maintaining
    smooth decay controlled by the golden ratio.
    """
    if p < 2:
        return 0.0
    return 1.0 / (1.0 + math.log(p) / math.log(PHI))


def resonance_kernel(p1: int, p2: int) -> float:
    """
    Compute resonance between two primes.
    
    Based on:
    - GCD structure (coprimality detection)
    - Golden ratio phase relationship
    - Prime gap statistics
    """
    if p1 == p2:
        return 1.0
    
    # Check coprimality (always true for distinct primes, but useful structure)
    # Smaller prime gap = higher resonance
    gap = abs(p2 - p1)
    
    # Log-scaled gap with golden modulation
    gap_factor = 1.0 / (1.0 + math.log(1 + gap) / math.log(PHI))
    
    # Product weight (larger products = weaker resonance)
    product = p1 * p2
    product_factor = 1.0 / (1.0 + math.log(product) / 10.0)
    
    return gap_factor * product_factor


@dataclass
class AttentionHead:
    """
    Single attention head operating on SparsePrimeState.
    
    Projects input through Q, K, V transformations implemented
    as prime-local quaternion rotations.
    """
    
    dim: int  # Number of primes to consider
    axis_q: Quaternion = field(default_factory=lambda: Quaternion(1, 0, 0, 0))
    axis_k: Quaternion = field(default_factory=lambda: Quaternion(1, 0, 0, 0))  
    axis_v: Quaternion = field(default_factory=lambda: Quaternion(1, 0, 0, 0))
    angle_q: float = 0.0
    angle_k: float = 0.0
    angle_v: float = 0.0
    
    def project_q(self, state: SparsePrimeState) -> SparsePrimeState:
        """Project state through query transformation."""
        return state.apply_rotation(self.axis_q, self.angle_q)
    
    def project_k(self, state: SparsePrimeState) -> SparsePrimeState:
        """Project state through key transformation."""
        return state.apply_rotation(self.axis_k, self.angle_k)
    
    def project_v(self, state: SparsePrimeState) -> SparsePrimeState:
        """Project state through value transformation."""
        return state.apply_rotation(self.axis_v, self.angle_v)


def softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """
    Numerically stable softmax with temperature.
    
    softmax(x_i) = exp(x_i/T) / Σ exp(x_j/T)
    """
    if not values:
        return []
    
    # Subtract max for numerical stability
    max_val = max(values)
    scaled = [(v - max_val) / temperature for v in values]
    
    exp_vals = [math.exp(v) for v in scaled]
    total = sum(exp_vals)
    
    if total < 1e-10:
        return [1.0 / len(values)] * len(values)
    
    return [e / total for e in exp_vals]


def resonant_attention(
    query: SparsePrimeState,
    keys: List[SparsePrimeState],
    values: List[SparsePrimeState],
    temperature: float = 1.0,
    use_resonance: bool = True
) -> SparsePrimeState:
    """
    Compute resonant attention over sparse prime states.
    
    Args:
        query: Query state |q⟩
        keys: List of key states [|k_i⟩]
        values: List of value states [|v_i⟩]
        temperature: Softmax temperature (lower = sharper attention)
        use_resonance: Whether to apply prime resonance weighting
        
    Returns:
        Attended state: Σ_i α_i |v_i⟩
        
    The attention weights α_i are computed as:
        α_i = softmax(score(q, k_i))
        
    where score uses quaternionic inner product with resonance modulation.
    """
    if len(keys) != len(values):
        raise ValueError("Keys and values must have same length")
    
    if not keys:
        return SparsePrimeState.vacuum()
    
    # Compute attention scores
    scores = []
    for key in keys:
        # Quaternionic inner product
        ip = query.inner_product(key)
        score = ip.norm()
        
        if use_resonance:
            # Modulate by prime resonance
            resonance_total = 0.0
            count = 0
            
            for p1 in query.amplitudes.keys():
                for p2 in key.amplitudes.keys():
                    resonance_total += resonance_kernel(p1, p2)
                    count += 1
            
            if count > 0:
                score *= (1.0 + resonance_total / count)
        
        scores.append(score)
    
    # Apply softmax to get attention weights
    weights = softmax(scores, temperature)
    
    # Weighted combination of values
    result = SparsePrimeState()
    for weight, value in zip(weights, values):
        if weight > 1e-10:
            scaled = value * weight
            for p, q in scaled.amplitudes.items():
                if p in result.amplitudes:
                    result.amplitudes[p] = result.amplitudes[p] + q
                else:
                    result.amplitudes[p] = q
    
    return result._normalize()


@dataclass
class MultiHeadResonantAttention:
    """
    Multi-head attention with prime resonance.
    
    Combines multiple attention heads with golden-ratio spacing
    of rotation axes for optimal coverage of the quaternionic space.
    """
    
    num_heads: int
    dim: int
    heads: List[AttentionHead] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize heads with golden-spaced rotations."""
        if not self.heads:
            ga = golden_angle()
            for i in range(self.num_heads):
                # Golden-spaced rotation axes
                theta = i * ga
                axis = Quaternion(0, math.cos(theta), math.sin(theta), 0)
                
                head = AttentionHead(
                    dim=self.dim,
                    axis_q=axis,
                    axis_k=axis.conjugate(),
                    axis_v=Quaternion(0, 0, math.cos(theta), math.sin(theta)),
                    angle_q=theta,
                    angle_k=theta * inverse_golden(),
                    angle_v=theta * PHI
                )
                self.heads.append(head)
    
    def forward(
        self,
        query: SparsePrimeState,
        keys: List[SparsePrimeState],
        values: List[SparsePrimeState],
        temperature: float = 1.0
    ) -> SparsePrimeState:
        """
        Apply multi-head attention.
        
        Each head computes attention with its own Q/K/V projections,
        then results are combined with golden-ratio weighting.
        """
        head_outputs = []
        
        for head in self.heads:
            # Project through head-specific transformations
            q_proj = head.project_q(query)
            k_projs = [head.project_k(k) for k in keys]
            v_projs = [head.project_v(v) for v in values]
            
            # Compute attention for this head
            output = resonant_attention(q_proj, k_projs, v_projs, temperature)
            head_outputs.append(output)
        
        # Combine heads with golden-ratio weighting
        result = SparsePrimeState()
        weights = [1.0 / (PHI ** i) for i in range(len(head_outputs))]
        total_weight = sum(weights)
        
        for weight, output in zip(weights, head_outputs):
            norm_weight = weight / total_weight
            for p, q in output.amplitudes.items():
                scaled = q * norm_weight
                if p in result.amplitudes:
                    result.amplitudes[p] = result.amplitudes[p] + scaled
                else:
                    result.amplitudes[p] = scaled
        
        return result._normalize()


@dataclass
class CoherenceGatedComputation:
    """
    Adaptive Computation Time (ACT) with coherence-based halting.
    
    Performs iterative refinement until coherence threshold is reached
    or maximum steps exceeded. Implements:
    
        output = Σ_t h_t · (1 - Σ_{τ<t} h_τ) · s_t
        
    where h_t = σ(coherence(s_t) - threshold) is the halting probability
    and s_t is the output at step t.
    """
    
    max_steps: int = 10
    coherence_threshold: float = COHERENCE_THRESHOLD
    halt_epsilon: float = 0.01
    
    def compute(
        self,
        initial_state: SparsePrimeState,
        step_fn: Callable[[SparsePrimeState, int], SparsePrimeState],
        halt_fn: Optional[Callable[[SparsePrimeState], float]] = None
    ) -> Tuple[SparsePrimeState, int, float]:
        """
        Run computation with coherence-gated halting.
        
        Args:
            initial_state: Starting state
            step_fn: Function(state, step) -> new_state for each refinement
            halt_fn: Function(state) -> halt_probability (default: entropy-based)
            
        Returns:
            (final_state, steps_taken, total_halt_probability)
        """
        if halt_fn is None:
            halt_fn = self._default_halt_probability
        
        state = initial_state
        accumulated_output = SparsePrimeState()
        remainder = 1.0  # Probability mass not yet halted
        total_steps = 0
        
        for t in range(self.max_steps):
            # Compute step output
            step_output = step_fn(state, t)
            
            # Compute halting probability
            h_t = halt_fn(step_output)
            h_t = max(0.0, min(1.0, h_t))  # Clamp to [0, 1]
            
            # If this is the last step, force halt
            if t == self.max_steps - 1:
                h_t = 1.0
            
            # Weight for this step
            weight = h_t * remainder
            
            # Accumulate weighted output
            for p, q in step_output.amplitudes.items():
                scaled = q * weight
                if p in accumulated_output.amplitudes:
                    accumulated_output.amplitudes[p] = (
                        accumulated_output.amplitudes[p] + scaled
                    )
                else:
                    accumulated_output.amplitudes[p] = scaled
            
            # Update remainder
            remainder *= (1.0 - h_t)
            total_steps = t + 1
            
            # Check for early halt
            if remainder < self.halt_epsilon:
                break
            
            # Update state for next iteration
            state = step_output
        
        final_output = accumulated_output._normalize()
        final_output.coherence = 1.0 - remainder
        
        return final_output, total_steps, 1.0 - remainder
    
    def _default_halt_probability(self, state: SparsePrimeState) -> float:
        """
        Default halting based on state coherence (inverse entropy).
        
        High entropy = low coherence = keep computing
        Low entropy = high coherence = halt
        """
        entropy = state.entropy()
        
        # Sigmoid transformation centered at threshold
        x = (self.coherence_threshold - entropy) * 5.0  # Scale factor
        halt_prob = 1.0 / (1.0 + math.exp(-x))
        
        return halt_prob


def golden_ratio_attention_weights(n: int) -> List[float]:
    """
    Generate n attention weights with golden ratio spacing.
    
    Uses Fibonacci-based weights that sum to 1.
    This provides optimal coverage without clustering.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    
    # Generate Fibonacci-like weights
    weights = [1.0]
    for i in range(1, n):
        weights.append(weights[-1] / PHI)
    
    # Normalize
    total = sum(weights)
    return [w / total for w in weights]


@dataclass
class ResonantTransformerBlock:
    """
    Transformer block operating on SparsePrimeState.
    
    Combines:
    1. Multi-head resonant attention
    2. Coherence-gated feed-forward
    3. Residual connections with golden ratio mixing
    """
    
    attention: MultiHeadResonantAttention
    coherence_gate: CoherenceGatedComputation
    hidden_primes: int = 50  # Number of primes in hidden layer
    
    def feed_forward(self, state: SparsePrimeState) -> SparsePrimeState:
        """
        Feed-forward network as prime expansion and contraction.
        
        Expands state to larger prime space, applies nonlinearity,
        then contracts back.
        """
        # Expand: add higher primes with golden-ratio decay
        expanded = SparsePrimeState(
            amplitudes=dict(state.amplitudes),
            coherence=state.coherence
        )
        
        base_primes = list(state.amplitudes.keys())
        if base_primes:
            max_prime = max(base_primes)
            
            # Add nearby higher primes
            p = max_prime + 1
            count = 0
            while count < 10:  # Add up to 10 higher primes
                while not is_prime(p):
                    p += 1
                
                # Golden-ratio decayed amplitude from nearest existing prime
                nearest = min(base_primes, key=lambda x: abs(x - p))
                decay = 1.0 / (PHI ** (count + 1))
                new_amp = state.amplitudes[nearest] * decay
                expanded.amplitudes[p] = new_amp
                
                p += 1
                count += 1
        
        # Apply nonlinearity (quaternion tanh-like activation)
        activated = SparsePrimeState(coherence=expanded.coherence)
        for p, q in expanded.amplitudes.items():
            # Component-wise tanh
            activated.amplitudes[p] = Quaternion(
                math.tanh(q.w),
                math.tanh(q.i),
                math.tanh(q.j),
                math.tanh(q.k)
            )
        
        # Contract: truncate to top primes by probability
        return activated.truncate(threshold=0.01)
    
    def forward(
        self,
        state: SparsePrimeState,
        context: Optional[List[SparsePrimeState]] = None
    ) -> SparsePrimeState:
        """
        Forward pass through transformer block.
        
        Args:
            state: Input state
            context: Optional context states for cross-attention
            
        Returns:
            Transformed state
        """
        # Self-attention (or cross-attention if context provided)
        if context is None:
            context = [state]
        
        attention_output = self.attention.forward(
            query=state,
            keys=context,
            values=context
        )
        
        # Residual connection with golden ratio mixing
        alpha = 1.0 / PHI  # ≈ 0.618
        residual1 = state * alpha + attention_output * (1 - alpha)
        
        # Feed-forward with coherence gating
        def ff_step(s: SparsePrimeState, step: int) -> SparsePrimeState:
            return self.feed_forward(s)
        
        ff_output, steps, halt_prob = self.coherence_gate.compute(
            residual1, ff_step
        )
        
        # Second residual
        output = residual1 * alpha + ff_output * (1 - alpha)
        output.coherence = halt_prob
        
        return output


def create_resonant_transformer(
    num_layers: int = 6,
    num_heads: int = 8,
    dim: int = 64,
    max_steps: int = 5
) -> List[ResonantTransformerBlock]:
    """
    Create a stack of resonant transformer blocks.
    
    Uses golden-ratio spacing for layer initialization.
    """
    blocks = []
    
    for layer in range(num_layers):
        attention = MultiHeadResonantAttention(
            num_heads=num_heads,
            dim=dim
        )
        
        # Layer-dependent coherence threshold
        coherence_threshold = COHERENCE_THRESHOLD * (1.0 + layer * 0.1)
        
        gate = CoherenceGatedComputation(
            max_steps=max_steps,
            coherence_threshold=coherence_threshold
        )
        
        block = ResonantTransformerBlock(
            attention=attention,
            coherence_gate=gate,
            hidden_primes=dim
        )
        
        blocks.append(block)
    
    return blocks