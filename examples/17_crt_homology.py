#!/usr/bin/env python3
"""
Example 17: CRT-Homology - Chinese Remainder Theorem and Topological Analysis

Demonstrates the CRT-Homology module:
- ResidueEncoder and CRTReconstructor
- BirkhoffProjector for doubly stochastic matrices
- HomologyLoss for topological constraints
- CRTModularLayer and CRTFusedAttention
"""

import sys
sys.path.insert(0, '..')

from tinyaleph.semantic import (
    # CRT components
    ResidueEncoding, ResidueEncoder, CRTReconstructor,
    DoublyStochasticMatrix, BirkhoffProjector, HomologyLoss,
    CRTModularLayer, CRTFusedAttention, CoprimeSelector,
    # Utility functions
    create_semantic_crt_encoder, crt_embed_sequence, crt_similarity,
    homology_regularizer, extended_gcd, mod_inverse, are_coprime, softmax
)


def demonstrate_number_theory_utils():
    """Demonstrate number-theoretic utilities."""
    print("=" * 60)
    print("NUMBER-THEORETIC UTILITIES")
    print("=" * 60)
    
    # Extended GCD
    print("\nExtended Euclidean Algorithm:")
    pairs = [(15, 25), (35, 15), (101, 103), (12, 18)]
    
    for a, b in pairs:
        gcd, x, y = extended_gcd(a, b)
        print(f"  gcd({a}, {b}) = {gcd}")
        print(f"    {a}×{x} + {b}×{y} = {gcd}")
    
    # Modular inverse
    print("\nModular inverse:")
    cases = [(3, 7), (5, 11), (7, 13), (2, 9)]
    
    for a, m in cases:
        inv = mod_inverse(a, m)
        print(f"  {a}⁻¹ mod {m} = {inv} (verify: {a}×{inv} mod {m} = {(a * inv) % m})")
    
    # Coprimality check
    print("\nCoprimality check:")
    pairs = [(3, 5), (4, 6), (7, 11), (15, 25)]
    
    for a, b in pairs:
        coprime = are_coprime(a, b)
        print(f"  gcd({a}, {b}) = 1? {coprime}")


def demonstrate_residue_encoding():
    """Demonstrate residue encoding via CRT."""
    print("\n" + "=" * 60)
    print("RESIDUE ENCODING (CRT Forward Direction)")
    print("=" * 60)
    
    # Create encoder with first 5 primes as moduli
    encoder = ResidueEncoder(num_channels=5)
    
    print(f"\nEncoder configuration:")
    print(f"  Moduli: {encoder.moduli}")
    print(f"  Product modulus: {encoder.product_modulus}")
    
    # Encode some values
    values = [42, 137, 256, 1000, 12345]
    
    print("\nEncoding values to residues:")
    
    for value in values:
        encoding = encoder.encode(value)
        print(f"\n  Value: {value}")
        print(f"  Residues: {encoding.residues}")
        print(f"  As dict: {encoding.to_dict()}")
        
        # Decode back
        decoded = encoder.decode(encoding)
        print(f"  Decoded: {decoded}")
        print(f"  Correct: {decoded == value % encoder.product_modulus}")
    
    return encoder


def demonstrate_crt_reconstruction():
    """Demonstrate CRT reconstruction."""
    print("\n" + "=" * 60)
    print("CRT RECONSTRUCTION")
    print("=" * 60)
    
    # Create reconstructor
    moduli = [3, 5, 7]  # Pairwise coprime
    reconstructor = CRTReconstructor(moduli)
    
    print(f"\nModuli: {moduli}")
    print(f"Product: {reconstructor.product_modulus}")
    
    # Example from Chinese Remainder Theorem
    print("\nClassic CRT problem:")
    print("  Find x where:")
    print("    x ≡ 2 (mod 3)")
    print("    x ≡ 3 (mod 5)")
    print("    x ≡ 2 (mod 7)")
    
    residues = [2, 3, 2]
    x = reconstructor.reconstruct(residues)
    
    print(f"\n  Solution: x = {x}")
    print(f"  Verify: {x} mod 3 = {x % 3}, {x} mod 5 = {x % 5}, {x} mod 7 = {x % 7}")
    
    # Garner's algorithm (more efficient)
    x_garner = reconstructor.reconstruct_garner(residues)
    print(f"\n  Garner's algorithm: x = {x_garner}")
    print(f"  Same result: {x == x_garner}")
    
    # Partial reconstruction
    print("\nPartial reconstruction:")
    partial = {3: 2, 7: 2}  # Only 2 of 3 moduli
    value, partial_mod = reconstructor.partial_reconstruct(partial)
    print(f"  From residues mod 3 and 7: x ≡ {value} (mod {partial_mod})")
    
    return reconstructor


def demonstrate_crt_arithmetic():
    """Demonstrate arithmetic in CRT representation."""
    print("\n" + "=" * 60)
    print("CRT ARITHMETIC")
    print("=" * 60)
    
    encoder = ResidueEncoder([3, 5, 7, 11])
    
    # Addition in CRT
    print("\nAddition in CRT representation:")
    a, b = 42, 57
    
    enc_a = encoder.encode(a)
    enc_b = encoder.encode(b)
    
    enc_sum = encoder.add_encoded(enc_a, enc_b)
    decoded_sum = encoder.decode(enc_sum)
    
    print(f"  {a} + {b} = {(a + b) % encoder.product_modulus}")
    print(f"  In CRT: {enc_a.residues} + {enc_b.residues} = {enc_sum.residues}")
    print(f"  Decoded: {decoded_sum}")
    
    # Multiplication in CRT
    print("\nMultiplication in CRT representation:")
    
    enc_prod = encoder.mul_encoded(enc_a, enc_b)
    decoded_prod = encoder.decode(enc_prod)
    
    print(f"  {a} × {b} = {(a * b) % encoder.product_modulus}")
    print(f"  In CRT: {enc_a.residues} × {enc_b.residues} = {enc_prod.residues}")
    print(f"  Decoded: {decoded_prod}")
    
    # Scalar multiplication
    print("\nScalar multiplication:")
    scalar = 5
    enc_scaled = encoder.scale_encoded(enc_a, scalar)
    decoded_scaled = encoder.decode(enc_scaled)
    
    print(f"  {scalar} × {a} = {(scalar * a) % encoder.product_modulus}")
    print(f"  Decoded: {decoded_scaled}")


def demonstrate_birkhoff_projector():
    """Demonstrate Birkhoff projector for doubly stochastic matrices."""
    print("\n" + "=" * 60)
    print("BIRKHOFF PROJECTOR")
    print("=" * 60)
    
    projector = BirkhoffProjector()
    
    print("\nBirkhoff-von Neumann Theorem:")
    print("  Every doubly stochastic matrix is a convex combination of permutations.")
    
    # Project a matrix onto Birkhoff polytope
    print("\nProjecting matrix onto Birkhoff polytope (Sinkhorn-Knopp):")
    
    matrix = [
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 2.0],
        [1.0, 2.0, 1.0],
    ]
    
    print("\nInput matrix:")
    for row in matrix:
        print(f"  {row}")
    
    ds_matrix = projector.project(matrix)
    
    print("\nProjected doubly stochastic matrix:")
    for row in ds_matrix.data:
        print(f"  [{', '.join(f'{v:.4f}' for v in row)}]")
    
    print(f"\nRow sums: {[f'{s:.4f}' for s in ds_matrix.row_sums()]}")
    print(f"Col sums: {[f'{s:.4f}' for s in ds_matrix.col_sums()]}")
    print(f"Is valid: {ds_matrix.is_valid()}")
    
    # Decompose into permutations
    print("\nBirkhoff decomposition:")
    decomp = projector.decompose_birkhoff(ds_matrix, max_perms=5)
    
    for weight, perm in decomp[:3]:
        print(f"  Weight {weight:.4f}: permutation {perm}")
    
    # Entropy
    entropy = projector.entropy(ds_matrix)
    print(f"\nMatrix entropy: {entropy:.4f}")
    print("  (Higher = more uniform, Lower = more permutation-like)")
    
    return projector


def demonstrate_homology_loss():
    """Demonstrate homology-based loss function."""
    print("\n" + "=" * 60)
    print("HOMOLOGY LOSS")
    print("=" * 60)
    
    loss_fn = HomologyLoss(lambda_boundary=1.0, lambda_cycle=0.5)
    
    print("\nHomology measures topological features:")
    print("  β₀ (Betti 0): Number of connected components")
    print("  β₁ (Betti 1): Number of 1-dimensional holes (cycles)")
    
    # Test graph: vertices 0-4, various edge configurations
    vertices = 5
    
    # Disconnected graph
    edges1 = [(0, 1), (2, 3)]
    beta0_1 = loss_fn.compute_betti_0(edges1, vertices)
    print(f"\nDisconnected graph (edges {edges1}):")
    print(f"  β₀ = {beta0_1} (3 components: [0,1], [2,3], [4])")
    
    # Connected tree
    edges2 = [(0, 1), (1, 2), (2, 3), (3, 4)]
    beta0_2 = loss_fn.compute_betti_0(edges2, vertices)
    print(f"\nPath graph (edges {edges2}):")
    print(f"  β₀ = {beta0_2} (1 component)")
    
    # Graph with cycle
    edges3 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    beta0_3 = loss_fn.compute_betti_0(edges3, vertices)
    beta1_3 = loss_fn.compute_betti_1(edges3, vertices)
    print(f"\nCycle graph (edges {edges3}):")
    print(f"  β₀ = {beta0_3}, β₁ = {beta1_3}")
    
    # Compute losses
    print("\nHomology losses (target: 1 component, 0 cycles):")
    for name, edges in [("Disconnected", edges1), ("Path", edges2), ("Cycle", edges3)]:
        loss = loss_fn.total_loss(edges, vertices, target_components=1, target_cycles=0)
        print(f"  {name}: loss = {loss:.4f}")
    
    return loss_fn


def demonstrate_crt_neural_layer():
    """Demonstrate CRT-based neural layer."""
    print("\n" + "=" * 60)
    print("CRT MODULAR LAYER")
    print("=" * 60)
    
    layer = CRTModularLayer(
        input_dim=4,
        output_dim=3,
        num_channels=5
    )
    
    print(f"\nLayer configuration:")
    print(f"  Input dim: {layer.input_dim}")
    print(f"  Output dim: {layer.output_dim}")
    print(f"  CRT channels: {layer.num_channels}")
    print(f"  Moduli: {layer.moduli}")
    
    # Forward pass
    x = [10, 20, 30, 40]
    
    print(f"\nInput: {x}")
    
    # Per-channel processing
    print("\nPer-channel outputs:")
    for c in range(min(3, layer.num_channels)):
        enc_x = [xi % layer.moduli[c] for xi in x]
        channel_out = layer.forward_channel(enc_x, c)
        print(f"  Channel {c} (mod {layer.moduli[c]}): {channel_out}")
    
    # Full forward pass with CRT
    output = layer.forward(x)
    print(f"\nCRT-reconstructed output: {output}")
    
    return layer


def demonstrate_crt_attention():
    """Demonstrate CRT-based attention mechanism."""
    print("\n" + "=" * 60)
    print("CRT FUSED ATTENTION")
    print("=" * 60)
    
    attention = CRTFusedAttention(
        embed_dim=8,
        num_heads=4
    )
    
    print(f"\nAttention configuration:")
    print(f"  Embedding dim: {attention.embed_dim}")
    print(f"  Heads: {attention.num_heads}")
    print(f"  Head dim: {attention.head_dim}")
    print(f"  CRT moduli: {attention.moduli}")
    
    # Create simple sequence
    seq_len = 3
    query = [[i * 10 + j for j in range(8)] for i in range(seq_len)]
    key = query
    value = query
    
    print(f"\nInput sequence (length {seq_len}):")
    for i, q in enumerate(query):
        print(f"  Position {i}: {q[:4]}...")
    
    # Compute attention with Birkhoff projection
    print("\nBirkhoff-projected attention matrix:")
    attn_matrix = attention.attention_with_birkhoff(query, key)
    
    for i, row in enumerate(attn_matrix.data):
        print(f"  [{', '.join(f'{v:.3f}' for v in row)}]")
    
    print(f"\nMatrix is doubly stochastic: {attn_matrix.is_valid()}")
    
    return attention


def demonstrate_coprime_selection():
    """Demonstrate optimal coprime moduli selection."""
    print("\n" + "=" * 60)
    print("COPRIME MODULI SELECTION")
    print("=" * 60)
    
    selector = CoprimeSelector()
    
    # Select primes
    print("\nFirst 8 primes (optimal coprimality):")
    primes = selector.select_primes(8)
    print(f"  {primes}")
    
    # Select for range
    print("\nMinimal primes for range 10000:")
    for_range = selector.select_for_range(10000)
    product = 1
    for p in for_range:
        product *= p
    print(f"  Primes: {for_range}")
    print(f"  Product: {product}")
    
    # Balanced selection
    print("\nBalanced moduli (8 bits each):")
    balanced = selector.select_balanced(num_channels=4, bits_per_channel=8)
    print(f"  {balanced}")
    
    # Analysis
    print("\nModuli analysis:")
    analysis = selector.analyze_moduli([2, 3, 5, 7, 11])
    for key, value in analysis.items():
        if key != 'channel_bits':
            print(f"  {key}: {value}")


def main():
    """Run all CRT-Homology demonstrations."""
    print("ALEPH PRIME - CRT-HOMOLOGY EXAMPLES")
    print("=" * 60)
    
    demonstrate_number_theory_utils()
    demonstrate_residue_encoding()
    demonstrate_crt_reconstruction()
    demonstrate_crt_arithmetic()
    demonstrate_birkhoff_projector()
    demonstrate_homology_loss()
    demonstrate_crt_neural_layer()
    demonstrate_crt_attention()
    demonstrate_coprime_selection()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
CRT-Homology module provides:
- Chinese Remainder Theorem encoding/decoding
- Arithmetic in residue representation
- Birkhoff projection (Sinkhorn-Knopp)
- Homological loss functions
- CRT-based neural layers
- Multi-scale CRT attention

Key properties:
- Parallel computation in modular channels
- Error detection via redundancy
- Topological constraints via homology
- Soft permutation learning

Applications:
- Distributed computation
- Attention normalization
- Topological regularization
- Integer arithmetic ML
""")


if __name__ == "__main__":
    main()