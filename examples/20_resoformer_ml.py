#!/usr/bin/env python3
"""
Example 20: ResoFormer - ML Architecture

Demonstrates the ML module:
- SparsePrimeState for efficient state representation
- ResoFormer layers (QuaternionDense, Attention, etc.)
- ResoFormerModel for sequence processing
"""

import sys
import math
import random
sys.path.insert(0, '..')

from tinyaleph.ml import (
    # Sparse representations
    SparsePrimeState,
    coherent_superposition,
    golden_superposition,
    # Tensor operations
    Tensor,
    zeros,
    ones,
    randn,
    glorot_uniform,
    quaternion_normalize,
    # Layers
    Layer,
    Dense,
    LayerNorm,
    Dropout,
    QuaternionDense,
    SparsePrimeEmbedding,
    ResonantAttentionLayer,
    CoherenceGatingLayer,
    EntropyCollapseLayer,
    ResonanceOperator,
    ResoFormerBlock,
    # Model builders
    ResoFormerConfig,
    ResoFormerModel,
    create_resoformer_model,
    create_resoformer_classifier,
    create_resoformer_embedder,
)


def demonstrate_sparse_state():
    """Demonstrate SparsePrimeState."""
    print("=" * 60)
    print("SPARSE PRIME STATE")
    print("=" * 60)
    
    print("\nSparsePrimeState: H_Q = H_P ⊗ ℍ (Prime-Quaternion tensor)")
    
    # Create sparse state using from_primes classmethod
    print("\n1. CREATING SPARSE STATES:")
    
    state = SparsePrimeState.from_primes([2, 3, 5, 7])
    print(f"   SparsePrimeState.from_primes([2, 3, 5, 7])")
    print(f"   Number of amplitudes: {len(state)}")
    
    # Show quaternion amplitudes
    print("\n2. QUATERNION AMPLITUDES:")
    for prime, q in state:
        print(f"   |{prime}⟩: ({q.w:.3f}, {q.i:.3f}i, {q.j:.3f}j, {q.k:.3f}k)")
    
    # Single prime state
    print("\n3. SINGLE PRIME EIGENSTATE:")
    single = SparsePrimeState.single_prime(7)
    print(f"   |7⟩ = {single}")
    
    # First n superposition
    print("\n4. FIRST N SUPERPOSITION:")
    first_5 = SparsePrimeState.first_n_superposition(5)
    print(f"   First 5 primes: {list(first_5.amplitudes.keys())}")
    
    # Entropy
    print("\n5. ENTROPY:")
    entropy = state.entropy()
    print(f"   Entropy: {entropy:.4f} bits")
    print(f"   Is coherent: {state.is_coherent()}")
    
    # Prime spectrum (probability distribution)
    print("\n6. PRIME SPECTRUM:")
    spectrum = state.prime_spectrum()
    for p, prob in spectrum.items():
        print(f"   P({p}) = {prob:.4f}")
    
    # Top-k primes
    print("\n7. TOP-K PRIMES:")
    top2 = state.top_k_primes(2)
    print(f"   Top 2 primes: {top2}")
    
    return state


def demonstrate_coherent_states():
    """Demonstrate coherent and golden superpositions."""
    print("\n" + "=" * 60)
    print("COHERENT SUPERPOSITIONS")
    print("=" * 60)
    
    print("\n1. COHERENT SUPERPOSITION WITH PHASES:")
    phases = [0, math.pi/4, math.pi/2, 3*math.pi/4]
    coherent = coherent_superposition([2, 3, 5, 7], phases)
    print(f"   Primes: [2, 3, 5, 7]")
    print(f"   Phases: [0, π/4, π/2, 3π/4]")
    for p, q in coherent:
        print(f"   |{p}⟩: ({q.w:.3f}, {q.i:.3f}i)")
    
    print("\n2. GOLDEN SUPERPOSITION:")
    golden = golden_superposition(5)
    print(f"   Golden ratio spacing of first 5 primes")
    print(f"   Using golden angle: 2π/φ² ≈ 137.5°")
    for p, q in golden:
        print(f"   |{p}⟩: ({q.w:.3f}, {q.i:.3f}i)")
    
    return coherent, golden


def demonstrate_tensor_operations():
    """Demonstrate tensor operations."""
    print("\n" + "=" * 60)
    print("TENSOR OPERATIONS")
    print("=" * 60)
    
    print("\n1. TENSOR CREATION:")
    
    # Zeros
    z = zeros((3, 4))
    print(f"   zeros((3, 4)): shape = {z.shape}")
    
    # Ones
    o = ones((2, 3))
    print(f"   ones((2, 3)): shape = {o.shape}")
    
    # Random normal
    r = randn((4, 4))
    print(f"   randn((4, 4)): shape = {r.shape}")
    
    # Glorot initialization
    g = glorot_uniform((32, 64))
    print(f"   glorot_uniform((32, 64)): shape = {g.shape}")
    
    print("\n2. TENSOR OPERATIONS:")
    
    # Reshape
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"   Original: shape = {t.shape}")
    
    t_reshaped = t.reshape((3, 2))
    print(f"   After reshape((3, 2)): shape = {t_reshaped.shape}")
    
    print("\n3. MATHEMATICAL OPERATIONS:")
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    b = Tensor([0.5, 0.5, 0.5, 0.5])
    
    print(f"   a = {a.data}")
    print(f"   b = {b.data}")
    print(f"   a + b = {(a + b).data}")
    print(f"   a * b = {(a * b).data}")
    print(f"   a.sum() = {a.sum().data}")
    print(f"   a.mean() = {a.mean().data}")
    
    print("\n4. ACTIVATION FUNCTIONS:")
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"   x = {x.data}")
    print(f"   relu(x) = {x.relu().data}")
    print(f"   sigmoid(x) = {[round(v, 3) for v in x.sigmoid().data]}")
    print(f"   tanh(x) = {[round(v, 3) for v in x.tanh().data]}")
    
    print("\n5. QUATERNION NORMALIZATION:")
    q = Tensor([1.0, 2.0, 3.0, 4.0])
    q_norm = quaternion_normalize(q)
    norm_val = sum(v**2 for v in q_norm.data)**0.5
    print(f"   Input: {q.data}")
    print(f"   Normalized: {[round(v, 4) for v in q_norm.data]}")
    print(f"   Norm: {norm_val:.4f}")
    
    return z, o, r


def demonstrate_dense_layers():
    """Demonstrate dense layers."""
    print("\n" + "=" * 60)
    print("DENSE LAYERS")
    print("=" * 60)
    
    print("\n1. STANDARD DENSE LAYER:")
    dense = Dense(units=16, activation="relu", name="dense1")
    
    x = randn((4, 8))  # batch=4, input_dim=8
    y = dense(x)
    print(f"   Input: shape = {x.shape}")
    print(f"   Dense(units=16, activation='relu')")
    print(f"   Output: shape = {y.shape}")
    
    print("\n2. QUATERNION DENSE LAYER:")
    q_dense = QuaternionDense(units=8, name="q_dense1")
    
    x_q = randn((4, 32))  # batch=4, input_dim=32
    y_q = q_dense(x_q)
    print(f"   Input: shape = {x_q.shape}")
    print(f"   QuaternionDense(units=8)")
    print(f"   Output: shape = {y_q.shape} (units * 4 quaternion components)")
    
    print("\n3. LAYER NORMALIZATION:")
    norm = LayerNorm(name="layernorm1")
    
    x_norm = randn((4, 64))
    y_norm = norm(x_norm)
    print(f"   Input: shape = {x_norm.shape}")
    print(f"   LayerNorm")
    print(f"   Output: shape = {y_norm.shape}")
    
    print("\n4. DROPOUT:")
    dropout = Dropout(rate=0.5, name="dropout1")
    
    x_drop = ones((4, 16))
    y_drop_training = dropout(x_drop, training=True)
    y_drop_inference = dropout(x_drop, training=False)
    zeros_training = sum(1 for v in y_drop_training.data if v == 0)
    zeros_inference = sum(1 for v in y_drop_inference.data if v == 0)
    print(f"   Input: shape = {x_drop.shape} (all ones)")
    print(f"   Dropout(rate=0.5)")
    print(f"   Training mode zeros: {zeros_training}/{len(x_drop.data)}")
    print(f"   Inference mode zeros: {zeros_inference}/{len(x_drop.data)}")
    
    return dense, q_dense


def demonstrate_embedding_layer():
    """Demonstrate sparse prime embedding layer."""
    print("\n" + "=" * 60)
    print("SPARSE PRIME EMBEDDING")
    print("=" * 60)
    
    embed = SparsePrimeEmbedding(
        num_primes=1000,
        k=8,
        embedding_dim=32,
        name="sparse_embed"
    )
    
    print("\n1. CONFIGURATION:")
    print(f"   num_primes: 1000 (vocabulary size)")
    print(f"   k: 8 (top-k active primes per token)")
    print(f"   embedding_dim: 32")
    
    print("\n2. FORWARD PASS:")
    # Input embeddings (simulating already embedded tokens)
    x = randn((4, 6, 32))  # batch=4, seq=6, embed=32
    result = embed(x)
    
    print(f"   Input: shape = (4, 6, 32)")
    print(f"   SparsePrimeEmbedding(1000, k=8)")
    print(f"   Outputs:")
    print(f"     - indices: shape = {result['indices'].shape}")
    print(f"     - amplitudes: shape = {result['amplitudes'].shape}")
    print(f"     - logits: shape = {result['logits'].shape}")
    
    return embed


def demonstrate_attention():
    """Demonstrate attention mechanisms."""
    print("\n" + "=" * 60)
    print("RESONANT ATTENTION LAYER")
    print("=" * 60)
    
    attn = ResonantAttentionLayer(
        num_heads=4,
        key_dim=16,
        dropout=0.1,
        name="res_attn"
    )
    
    print("\n1. CONFIGURATION:")
    print(f"   num_heads: 4")
    print(f"   key_dim: 16 (per head)")
    print(f"   total_dim: {4 * 16}")
    
    print("\n2. FORWARD PASS:")
    x = randn((2, 8, 64))  # batch=2, seq=8, embed=64
    y = attn(x)
    print(f"   Input: shape = {x.shape}")
    print(f"   ResonantAttentionLayer(4 heads, key_dim=16)")
    print(f"   Output: shape = {y.shape}")
    
    return attn


def demonstrate_special_layers():
    """Demonstrate coherence gating and entropy collapse."""
    print("\n" + "=" * 60)
    print("SPECIAL LAYERS")
    print("=" * 60)
    
    print("\n1. COHERENCE GATING LAYER:")
    gating = CoherenceGatingLayer(threshold=0.7, name="coh_gate")
    
    x = randn((4, 32))
    result = gating(x)
    print(f"   Input: shape = {x.shape}")
    print(f"   CoherenceGatingLayer(threshold=0.7)")
    print(f"   Outputs:")
    print(f"     - output: shape = {result['output'].shape}")
    print(f"     - coherence: shape = {result['coherence'].shape}")
    print(f"     - gate: shape = {result['gate'].shape}")
    
    print("\n2. ENTROPY COLLAPSE LAYER:")
    collapse = EntropyCollapseLayer(
        num_attractors=16,
        target_entropy=4.0,
        temperature=0.5,
        name="entropy_collapse"
    )
    
    x = randn((4, 32))
    result_train = collapse(x, training=True)
    result_infer = collapse(x, training=False)
    print(f"   Input: shape = {x.shape}")
    print(f"   EntropyCollapseLayer(16 attractors)")
    print(f"   Training mode:")
    print(f"     - output: shape = {result_train['output'].shape}")
    print(f"     - probs: shape = {result_train['probs'].shape}")
    print(f"     - entropy: {[round(e, 3) for e in result_train['entropy'].data]}")
    print(f"   Inference mode:")
    print(f"     - indices (hard assignment): {[int(i) for i in result_infer['indices'].data]}")
    
    print("\n3. RESONANCE OPERATOR:")
    res_op = ResonanceOperator(name="res_op")
    
    x = randn((4, 32))
    y = res_op(x)
    print(f"   Input: shape = {x.shape}")
    print(f"   ResonanceOperator: R̂(n)|p⟩ = e^(2πi log(n))|p⟩")
    print(f"   Output: shape = {y.shape}")
    
    return gating, collapse, res_op


def demonstrate_resoformer_block():
    """Demonstrate ResoFormer block."""
    print("\n" + "=" * 60)
    print("RESOFORMER BLOCK")
    print("=" * 60)
    
    block = ResoFormerBlock(
        dim=64,
        num_heads=4,
        ffn_dim=256,
        dropout_rate=0.1,
        use_collapse=True,
        name="resoblock1"
    )
    
    print("\n1. BLOCK CONFIGURATION:")
    print("   dim: 64")
    print("   num_heads: 4")
    print("   ffn_dim: 256")
    print("   dropout_rate: 0.1")
    print("   use_collapse: True")
    
    print("\n2. BLOCK ARCHITECTURE:")
    print("   Pre-norm: LayerNorm → Attention → Residual")
    print("   Resonance: ResonanceOperator")
    print("   FFN: LayerNorm → Dense(256) → GELU → Dense(64) → Residual")
    print("   Gate: CoherenceGatingLayer")
    print("   Collapse: EntropyCollapseLayer (64 attractors)")
    
    print("\n3. FORWARD PASS:")
    x = randn((2, 8, 64))  # batch=2, seq=8, dim=64
    y = block(x, training=True)
    print(f"   Input: shape = {x.shape}")
    print(f"   ResoFormerBlock")
    print(f"   Output: shape = {y.shape}")
    
    return block


def demonstrate_resoformer_model():
    """Demonstrate complete ResoFormer model."""
    print("\n" + "=" * 60)
    print("RESOFORMER MODEL")
    print("=" * 60)
    
    config = ResoFormerConfig(
        vocab_size=1000,
        seq_len=32,
        dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        dropout=0.1
    )
    
    print("\n1. CONFIGURATION:")
    print(f"   vocab_size: {config.vocab_size}")
    print(f"   seq_len: {config.seq_len}")
    print(f"   dim: {config.dim}")
    print(f"   num_layers: {config.num_layers}")
    print(f"   num_heads: {config.num_heads}")
    print(f"   ffn_dim: {config.ffn_dim}")
    print(f"   dropout: {config.dropout}")
    
    print("\n2. LANGUAGE MODEL:")
    lm = create_resoformer_model(
        vocab_size=1000,
        seq_len=32,
        dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        dropout=0.1
    )
    
    # Create token input
    tokens = Tensor([list(range(8))], (1, 8))  # batch=1, seq=8
    output = lm(tokens, training=False)
    print(f"   Input tokens: shape = {tokens.shape}")
    print(f"   Output logits: shape = {output.shape}")
    
    print("\n3. CLASSIFIER:")
    clf = create_resoformer_classifier(
        vocab_size=1000,
        seq_len=32,
        dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        num_classes=10,
        dropout=0.1
    )
    output_clf = clf(tokens, training=False)
    print(f"   Classifier output: shape = {output_clf.shape}")
    
    print("\n4. EMBEDDER:")
    emb = create_resoformer_embedder(
        vocab_size=1000,
        seq_len=32,
        dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        embedding_dim=32,
        dropout=0.1
    )
    output_emb = emb(tokens, training=False)
    print(f"   Embedder output: shape = {output_emb.shape}")
    
    return lm, clf, emb


def main():
    """Run all ResoFormer demonstrations."""
    print("ALEPH PRIME - RESOFORMER ML EXAMPLES")
    print("=" * 60)
    
    demonstrate_sparse_state()
    demonstrate_coherent_states()
    demonstrate_tensor_operations()
    demonstrate_dense_layers()
    demonstrate_embedding_layer()
    demonstrate_attention()
    demonstrate_special_layers()
    demonstrate_resoformer_block()
    demonstrate_resoformer_model()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
ResoFormer provides:

1. SPARSE REPRESENTATIONS
   - SparsePrimeState: H_Q = H_P ⊗ ℍ
   - Quaternion components per prime
   - Entropy and coherence metrics
   - Golden ratio phase spacing

2. TENSOR OPERATIONS
   - Pure Python tensor implementation
   - Glorot initialization
   - Quaternion normalization
   - Standard activations (relu, sigmoid, tanh, gelu)

3. LAYERS
   - Dense and QuaternionDense
   - LayerNorm and Dropout
   - SparsePrimeEmbedding
   - ResonantAttentionLayer (multi-head)
   - CoherenceGatingLayer
   - EntropyCollapseLayer (VQ-style)
   - ResonanceOperator (phase rotation)

4. RESOFORMER ARCHITECTURE
   - ResoFormerBlock (attention + FFN + gate + collapse)
   - ResoFormerModel (full transformer stack)
   - Model variants: LM, Classifier, Embedder

5. SPECIAL FEATURES
   - Coherence-gated computation
   - Entropy collapse to 64 attractors
   - Quaternion-based representations
   - Prime-indexed sparse states

Applications:
- Prime sequence modeling
- Semantic embeddings
- Classification tasks
- Resonance pattern recognition
""")


if __name__ == "__main__":
    main()