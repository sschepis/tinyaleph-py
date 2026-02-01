#!/usr/bin/env python3
"""
Example 15: Reduction Semantics and Proof Generation

Demonstrates the reduction semantics module:
- Prime operators (NextPrime, Modular, Resonance)
- Reduction system with step-by-step evaluation
- Strong normalization proofs
- Confluence testing
"""

import sys
sys.path.insert(0, '..')

from tinyaleph.semantic import (
    # Reduction components
    PrimeOperator, NextPrimeOperator, ModularPrimeOperator,
    ResonancePrimeOperator, IdentityPrimeOperator,
    ReductionStep, ReductionTrace, ReductionSystem,
    FusionCanonicalizer, NormalFormVerifier,
    ProofTrace, ProofGenerator,
    # Terms
    N, A, FUSE, CHAIN,
    # Utilities
    is_normal_form, is_reducible, term_size, term_depth
)


def demonstrate_prime_operators():
    """Demonstrate different prime operators."""
    print("=" * 60)
    print("PRIME OPERATORS")
    print("=" * 60)
    
    # NextPrimeOperator: p ⊕ q = next prime after p + q
    print("\nNextPrimeOperator (p ⊕ q = next prime after p + q):")
    next_op = NextPrimeOperator()
    
    test_pairs = [(2, 3), (2, 5), (3, 5), (5, 7)]
    for p, q in test_pairs:
        if next_op.can_apply(p, q):
            result = next_op.apply(p, q)
            print(f"  {p} ⊕ {q} = {result}")
        else:
            print(f"  {p} ⊕ {q} = (cannot apply: p must be < q)")
    
    # ModularPrimeOperator: Modular arithmetic on primes
    print("\nModularPrimeOperator (modular structure):")
    mod_op = ModularPrimeOperator(base=1000)
    
    for p, q in [(2, 3), (3, 7), (5, 11)]:
        if mod_op.can_apply(p, q):
            result = mod_op.apply(p, q)
            print(f"  {p} ⊕_mod {q} = {result}")
    
    # ResonancePrimeOperator: Resonance-based transformation
    print("\nResonancePrimeOperator (resonance transform):")
    res_op = ResonancePrimeOperator()
    
    for p, q in [(2, 3), (3, 7), (5, 13)]:
        if res_op.can_apply(p, q):
            result = res_op.apply(p, q)
            print(f"  {p} ⊕_res {q} = {result}")
    
    # IdentityPrimeOperator: Returns q unchanged
    print("\nIdentityPrimeOperator (returns q):")
    id_op = IdentityPrimeOperator()
    
    for p, q in [(2, 3), (3, 7), (5, 11)]:
        if id_op.can_apply(p, q):
            result = id_op.apply(p, q)
            print(f"  {p} ⊕_id {q} = {result}")
    
    return next_op, mod_op, res_op


def demonstrate_term_construction():
    """Demonstrate term construction."""
    print("\n" + "=" * 60)
    print("TERM CONSTRUCTION")
    print("=" * 60)
    
    # Noun terms
    print("\nNoun terms (N(p)):")
    n2 = N(2)
    n5 = N(5)
    print(f"  N(2) = {n2}, signature = {n2.signature()}")
    print(f"  N(5) = {n5}, signature = {n5.signature()}")
    
    # Adjective terms
    print("\nAdjective terms (A(p)):")
    a3 = A(3)
    a7 = A(7)
    print(f"  A(3) = {a3}, signature = {a3.signature()}")
    print(f"  A(7) = {a7}, signature = {a7.signature()}")
    
    # Chain terms: CHAIN(operators, noun)
    print("\nChain terms (A(p)...A(q)N(r)):")
    chain1 = CHAIN([3], 7)  # A(3)N(7)
    chain2 = CHAIN([2, 3], 11)  # A(2)A(3)N(11)
    print(f"  A(3)N(7) = {chain1}")
    print(f"  A(2)A(3)N(11) = {chain2}")
    
    # Fusion terms
    print("\nFusion terms (FUSE(p,q,r)):")
    fuse1 = FUSE(3, 5, 11)  # 3+5+11 = 19 (prime)
    fuse2 = FUSE(5, 7, 11)  # 5+7+11 = 23 (prime)
    print(f"  FUSE(3,5,11) = {fuse1}")
    print(f"    Well-formed: {fuse1.is_well_formed()}")
    print(f"    Fused prime: {fuse1.get_fused_prime()}")
    print(f"  FUSE(5,7,11) = {fuse2}")
    print(f"    Well-formed: {fuse2.is_well_formed()}")
    print(f"    Fused prime: {fuse2.get_fused_prime()}")
    
    return chain1, chain2, fuse1


def demonstrate_reduction_steps():
    """Demonstrate reduction steps."""
    print("\n" + "=" * 60)
    print("REDUCTION STEPS")
    print("=" * 60)
    
    # Fusion reduction: FUSE(p,q,r) → N(p+q+r)
    print("\nFusion reduction: FUSE(p,q,r) → N(p+q+r)")
    
    fuse = FUSE(3, 5, 11)  # 3+5+11 = 19
    reduced = fuse.to_noun_term()
    
    print(f"  Before: {fuse}")
    print(f"  After:  {reduced}")
    print(f"  Rule: FUSION-REDUCTION")
    
    # Chain reduction: A(p)N(q) → N(p ⊕ q)
    print("\nChain reduction: A(p)N(q) → N(p ⊕ q)")
    
    # Create a chain and reduce it step by step
    chain = CHAIN([2, 3], 7)  # A(2)A(3)N(7)
    print(f"  Initial: {chain}")
    
    system = ReductionSystem()
    trace = system.normalize(chain)
    
    print(f"  Steps: {trace.length}")
    for i, step in enumerate(trace.steps):
        print(f"    {i+1}. {step}")
    print(f"  Final: {trace.final}")
    
    return trace


def demonstrate_reduction_system():
    """Demonstrate the full reduction system."""
    print("\n" + "=" * 60)
    print("REDUCTION SYSTEM")
    print("=" * 60)
    
    # Create reduction system
    system = ReductionSystem()
    
    # Test various terms
    terms = [
        ("Simple noun N(2)", N(2)),
        ("Simple adj A(3)", A(3)),
        ("Chain A(2)N(5)", CHAIN([2], 5)),
        ("Fusion FUSE(3,5,11)", FUSE(3, 5, 11)),
        ("Nested chain A(2)A(3)N(7)", CHAIN([2, 3], 7)),
    ]
    
    print("\nReducing terms to normal form:")
    
    for name, term in terms:
        print(f"\n  {name}")
        print(f"    Term: {term}")
        
        # Check if already in normal form
        if is_normal_form(term):
            print(f"    Already in normal form")
            continue
        
        if not is_reducible(term):
            print(f"    Not reducible (stuck term)")
            continue
        
        # Reduce
        trace = system.normalize(term)
        
        print(f"    Steps: {trace.length}")
        print(f"    Result: {trace.final}")
        print(f"    Is normal form: {is_normal_form(trace.final)}")
    
    return system


def demonstrate_proof_generation():
    """Demonstrate proof generation."""
    print("\n" + "=" * 60)
    print("PROOF GENERATION")
    print("=" * 60)
    
    # Create proof generator
    generator = ProofGenerator()
    
    # Generate proof for fusion reduction
    term = FUSE(3, 5, 11)
    
    print(f"\nGenerating proof for: FUSE(3, 5, 11) → N(19)")
    
    proof = generator.generate_proof(term)
    
    print("\nProof trace:")
    for step in proof.steps:
        print(f"  {step.index}. [{step.rule}]")
        print(f"     {step.before} → {step.after}")
        print(f"     Size: {step.size_before} → {step.size_after}")
    
    # Verify size decrease
    verification = proof.verify_size_decrease()
    print(f"\nSize decrease verified: {verification['valid']}")
    
    # Generate certificate
    cert = proof.to_certificate()
    print(f"\nProof certificate:")
    print(f"  Initial: {cert['initial']}")
    print(f"  Final: {cert['final']}")
    print(f"  Total steps: {cert['metrics']['total_steps']}")
    
    return generator


def demonstrate_normalization():
    """Demonstrate strong normalization property."""
    print("\n" + "=" * 60)
    print("STRONG NORMALIZATION")
    print("=" * 60)
    
    print("""
Strong Normalization Theorem:
  Every well-typed term has a normal form, and
  every reduction sequence terminates.
  
  The proof uses a size measure:
    size(N(p)) = 1
    size(A(p)) = 1
    size(CHAIN(ops, n)) = |ops| + 1
    size(FUSE(p,q,r)) = 1
    
  Key property: Every reduction step decreases size.
""")
    
    # Demonstrate size decrease
    print("Size decrease during reduction:")
    
    from tinyaleph.semantic.reduction import demonstrate_strong_normalization
    
    terms = [
        FUSE(3, 5, 11),
        CHAIN([2], 7),
        CHAIN([2, 3, 5], 11),
    ]
    
    for term in terms:
        result = demonstrate_strong_normalization(term)
        print(f"\n  Term: {result['term']}")
        print(f"  Sizes: {result['sizes']}")
        print(f"  Strictly decreasing: {result['strictly_decreasing']}")
        print(f"  Verified: {result['verified']}")


def demonstrate_confluence():
    """Demonstrate confluence (Church-Rosser property)."""
    print("\n" + "=" * 60)
    print("CONFLUENCE (CHURCH-ROSSER)")
    print("=" * 60)
    
    print("""
Local Confluence:
  If t →₁ s₁ and t →₂ s₂ via different one-step reductions,
  then there exists u such that s₁ →* u and s₂ →* u.
  
  Combined with strong normalization, this gives:
    All reduction paths lead to the same normal form.
""")
    
    from tinyaleph.semantic.reduction import test_local_confluence
    
    result = test_local_confluence()
    
    print("Testing local confluence:")
    for tc in result['test_cases']:
        print(f"\n  Term: {tc['term']}")
        print(f"  Normal form: {tc['normal_form']}")
        print(f"  Confluent: {tc['confluent']}")
    
    print(f"\nAll confluent: {result['all_confluent']}")


def demonstrate_fusion_canonicalization():
    """Demonstrate fusion canonicalization."""
    print("\n" + "=" * 60)
    print("FUSION CANONICALIZATION")
    print("=" * 60)
    
    canonicalizer = FusionCanonicalizer()
    
    # Find triads for various primes
    print("\nFinding fusion triads for primes:")
    
    for P in [19, 23, 29, 31, 37]:
        triads = canonicalizer.get_triads(P)
        print(f"\n  D({P}) - triads summing to {P}:")
        if triads:
            for t in triads[:3]:  # Show first 3
                print(f"    FUSE({t.p}, {t.q}, {t.r})")
            if len(triads) > 3:
                print(f"    ... and {len(triads) - 3} more")
        else:
            print(f"    (none found)")
    
    # Select canonical triad
    print("\nCanonical triad selection:")
    for P in [19, 23, 29]:
        canonical = canonicalizer.select_canonical(P)
        if canonical:
            score = canonicalizer.resonance_score(canonical)
            print(f"  d*({P}) = FUSE({canonical.p}, {canonical.q}, {canonical.r})")
            print(f"    Resonance score: {score:.4f}")


def demonstrate_normal_form_verification():
    """Demonstrate normal form verification."""
    print("\n" + "=" * 60)
    print("NORMAL FORM VERIFICATION")
    print("=" * 60)
    
    verifier = NormalFormVerifier()
    system = ReductionSystem()
    
    terms_and_claims = [
        (N(2), N(2), "N(2) = N(2)"),
        (FUSE(3, 5, 11), N(19), "FUSE(3,5,11) = N(19)"),
        (CHAIN([2], 5), 7, "A(2)N(5) = 7?"),  # Depends on operator
    ]
    
    print("\nVerifying normal form claims:")
    
    for term, claimed, description in terms_and_claims:
        is_verified = verifier.verify(term, claimed)
        status = "✓" if is_verified else "✗"
        print(f"  {status} {description}")
        
        # Show certificate
        cert = verifier.certificate(term, claimed)
        print(f"    Actual result: {cert['actual']}")
        print(f"    Steps: {cert['steps']}")


def main():
    """Run all reduction semantics demonstrations."""
    print("ALEPH PRIME - REDUCTION SEMANTICS EXAMPLES")
    print("=" * 60)
    
    demonstrate_prime_operators()
    demonstrate_term_construction()
    demonstrate_reduction_steps()
    demonstrate_reduction_system()
    demonstrate_proof_generation()
    demonstrate_normalization()
    demonstrate_confluence()
    demonstrate_fusion_canonicalization()
    demonstrate_normal_form_verification()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Reduction semantics provides:
- Prime operators for transforming indices
- Step-by-step reduction with traces
- Automatic proof generation
- Strong normalization guarantee
- Confluence (Church-Rosser property)
- Fusion canonicalization

Key properties:
- All terms reach normal form
- Normal forms are unique
- Reduction terminates
- Proofs are constructive

Applications:
- Semantic simplification
- Equivalence checking
- Term normalization
- Formal verification
""")


if __name__ == "__main__":
    main()