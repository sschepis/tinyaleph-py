#!/usr/bin/env python3
"""
Example 14: Semantic Type System

Demonstrates the formal type system for prime-indexed semantics:
- NounTerm, AdjTerm, ChainTerm, FusionTerm
- Sentence structures (NounSentence, SeqSentence, ImplSentence)
- Type checking and inference
- Builder functions for term construction
"""

import sys
sys.path.insert(0, '..')

from tinyaleph.semantic import (
    # Terms
    Term, NounTerm, AdjTerm, ChainTerm, FusionTerm,
    # Sentences
    NounSentence, SeqSentence, ImplSentence,
    # Type system
    TypeChecker, TypingContext, TypingJudgment,
    # Builders
    N, A, FUSE, CHAIN, SENTENCE, SEQ, IMPL
)


def demonstrate_basic_terms():
    """Demonstrate basic term construction."""
    print("=" * 60)
    print("BASIC TERM CONSTRUCTION")
    print("=" * 60)
    
    # NounTerm: Atomic concepts indexed by primes
    print("\nNounTerms (atomic concepts):")
    
    nouns = [
        N(2),   # First prime
        N(3),   # Second prime
        N(5),   # Third prime
        N(7),   # Fourth prime
        N(11),  # Fifth prime
    ]
    
    for noun in nouns:
        print(f"  N({noun.prime}): represents concept indexed by prime {noun.prime}")
    
    # AdjTerm: Modifiers indexed by primes
    print("\nAdjTerms (modifiers):")
    
    adjs = [
        A(13),  # Modifier 1
        A(17),  # Modifier 2
        A(19),  # Modifier 3
    ]
    
    for adj in adjs:
        print(f"  A({adj.prime}): modifier indexed by prime {adj.prime}")
    
    return nouns, adjs


def demonstrate_compound_terms():
    """Demonstrate compound term construction."""
    print("\n" + "=" * 60)
    print("COMPOUND TERM CONSTRUCTION")
    print("=" * 60)
    
    # ChainTerm: Noun with applied modifiers
    print("\nChainTerms (modified nouns):")
    
    # Using larger prime for noun so adjectives can apply (adj.prime < noun.prime constraint)
    noun = N(11)
    mod1 = A(3)
    mod2 = A(5)
    
    # Single modifier - CHAIN(operators, noun) format
    chain1 = CHAIN([mod1], noun)
    print(f"  CHAIN([A(3)], N(11)): noun 11 modified by adj 3")
    print(f"    Primes in chain: {chain1.get_all_primes()}")
    
    # Multiple modifiers (chain application)
    chain2 = CHAIN([mod1, mod2], noun)
    print(f"  CHAIN([A(3), A(5)], N(11)): noun 11 modified by adjs 3,5")
    print(f"    Primes in chain: {chain2.get_all_primes()}")
    
    # FusionTerm: Emergent concept from prime fusion
    print("\nFusionTerms (emergent concepts):")
    
    # Valid fusion requires: distinct odd primes and sum is prime
    # 3+5+11=19 (prime), 3+7+13=23 (prime)
    fusion1 = FUSE(3, 5, 11)
    print(f"  FUSE(3, 5, 11): fusion of primes 3,5,11")
    if fusion1.is_well_formed():
        print(f"    Fused prime: {fusion1.get_fused_prime()} (= 3+5+11 = 19)")
    else:
        print(f"    Sum: {fusion1.p + fusion1.q + fusion1.r}")
    
    fusion2 = FUSE(3, 7, 13)
    print(f"  FUSE(3, 7, 13): fusion of primes 3,7,13")
    if fusion2.is_well_formed():
        print(f"    Fused prime: {fusion2.get_fused_prime()} (= 3+7+13 = 23)")
    else:
        print(f"    Sum: {fusion2.p + fusion2.q + fusion2.r}")
    
    return chain1, chain2, fusion1, fusion2


def demonstrate_sentences():
    """Demonstrate sentence construction."""
    print("\n" + "=" * 60)
    print("SENTENCE CONSTRUCTION")
    print("=" * 60)
    
    # NounSentence: Simple assertion
    print("\nNounSentence (simple assertion):")
    
    noun = N(2)
    sent1 = SENTENCE(noun)
    print(f"  SENTENCE(N(2)): 'Concept 2 is the case'")
    print(f"    Type: NounSentence")
    
    # SeqSentence: Sequential composition
    print("\nSeqSentence (sequential composition):")
    
    s1 = SENTENCE(N(2))
    s2 = SENTENCE(N(3))
    seq = SEQ(s1, s2)
    print(f"  SEQ(S1, S2): 'S1 and then S2'")
    print(f"    S1: sentence about N(2)")
    print(f"    S2: sentence about N(3)")
    
    # ImplSentence: Implication
    print("\nImplSentence (implication):")
    
    antecedent = SENTENCE(N(5))
    consequent = SENTENCE(N(7))
    impl = IMPL(antecedent, consequent)
    print(f"  IMPL(A, C): 'If A then C'")
    print(f"    Antecedent: sentence about N(5)")
    print(f"    Consequent: sentence about N(7)")
    
    return sent1, seq, impl


def demonstrate_type_checking():
    """Demonstrate type checking and inference."""
    print("\n" + "=" * 60)
    print("TYPE CHECKING AND INFERENCE")
    print("=" * 60)
    
    # Create type checker
    checker = TypeChecker()
    
    # Create context with type bindings using the bind method
    from tinyaleph.semantic.types import Types
    ctx = TypingContext()
    
    # Add type bindings
    print("\nType bindings in context:")
    ctx.bind('x', Types.NOUN, N(2))
    ctx.bind('y', Types.NOUN, N(3))
    ctx.bind('p', Types.ADJECTIVE, A(5))
    print("  x: Noun (bound to N(2))")
    print("  y: Noun (bound to N(3))")
    print("  p: Adjective (bound to A(5))")
    
    # Infer types of terms
    print("\nType inference:")
    
    terms = [
        N(2),
        A(5),
        CHAIN([A(3)], N(7)),  # Use valid primes (3 < 7)
        FUSE(3, 5, 11),  # 3+5+11=19 is prime
        SENTENCE(N(2)),
    ]
    
    for term in terms:
        inferred_type = checker.infer_type(term)
        if inferred_type:
            print(f"  {term}: {inferred_type.value}")
        else:
            print(f"  {term}: ill-typed")
    
    # Full derivation
    print("\nFull derivation:")
    
    complex_term = CHAIN([A(3), A(5)], N(11))  # 3 < 5 < 11
    judgment = checker.derive(complex_term)
    print(f"  Term: CHAIN([A(3), A(5)], N(11))")
    if judgment:
        print(f"    Type: {judgment.term_type.value}")
        print(f"    Valid: {judgment.is_valid()}")
        print(f"    Well-formed: {complex_term.is_well_formed()}")
    
    return checker, ctx


def demonstrate_typing_rules():
    """Demonstrate formal typing rules."""
    print("\n" + "=" * 60)
    print("FORMAL TYPING RULES")
    print("=" * 60)
    
    print("""
The type system follows these rules:

1. NOUN INTRODUCTION
   ─────────────────────────
   Γ ⊢ N(p) : Noun
   
   Any prime p introduces a noun term.

2. ADJ INTRODUCTION
   ─────────────────────────
   Γ ⊢ A(p) : Adj
   
   Any prime p introduces an adjective term.

3. CHAIN FORMATION
   Γ ⊢ n : Noun    Γ ⊢ a₁ : Adj ... Γ ⊢ aₖ : Adj
   ───────────────────────────────────────────────
   Γ ⊢ CHAIN(n, [a₁,...,aₖ]) : Noun
   
   Adjectives modify nouns to produce nouns.

4. FUSION FORMATION
   p, q, r are primes
   ─────────────────────────────
   Γ ⊢ FUSE(p, q, r) : Noun
   
   Three primes fuse to form an emergent noun.

5. SENTENCE FORMATION
   Γ ⊢ e : Noun
   ─────────────────────────
   Γ ⊢ SENTENCE(e) : Sent
   
   Nouns can be lifted to sentences.

6. SEQUENCE FORMATION
   Γ ⊢ s₁ : Sent    Γ ⊢ s₂ : Sent
   ─────────────────────────────────
   Γ ⊢ SEQ(s₁, s₂) : Sent
   
   Sentences can be sequenced.

7. IMPLICATION FORMATION
   Γ ⊢ s₁ : Sent    Γ ⊢ s₂ : Sent
   ─────────────────────────────────
   Γ ⊢ IMPL(s₁, s₂) : Sent
   
   Sentences can form implications.
""")


def demonstrate_complex_structures():
    """Demonstrate complex nested structures."""
    print("\n" + "=" * 60)
    print("COMPLEX NESTED STRUCTURES")
    print("=" * 60)
    
    # Build a complex expression step by step
    print("\nBuilding complex semantic structure:")
    
    # Base concepts
    entity1 = N(2)  # e.g., "thing"
    entity2 = N(3)  # e.g., "person"
    
    # Properties
    prop1 = A(5)    # e.g., "big"
    prop2 = A(7)    # e.g., "red"
    
    # Modified entity
    modified = CHAIN([prop1, prop2], entity1)  # Note: operators first, noun second
    print(f"  1. Modified entity: CHAIN(N(2), [A(5), A(7)])")
    print(f"     'Big red thing'")
    
    # Fusion concept
    fusion = FUSE(2, 3, 5)
    print(f"  2. Fusion concept: FUSE(2, 3, 5)")
    print(f"     Emergent meaning from combined primes")
    
    # Sentences
    s1 = SENTENCE(modified)
    s2 = SENTENCE(N(entity2.prime))
    print(f"  3. Sentences about modified entity and entity2")
    
    # Implication
    result = IMPL(s1, s2)
    print(f"  4. Implication: 'If big-red-thing then person'")
    
    # Complex sequence
    s3 = SENTENCE(fusion)
    complex_seq = SEQ(result, s3)
    print(f"  5. Sequence: (implication) and-then (fusion-sentence)")
    
    # Analyze structure
    print("\nStructure analysis:")
    print(f"  Depth: {compute_depth(complex_seq)}")
    print(f"  Prime count: {count_primes(complex_seq)}")
    print(f"  Term count: {count_terms(complex_seq)}")
    
    return complex_seq


# Helper functions
def compute_depth(term, depth=0):
    """Compute nesting depth of term."""
    if isinstance(term, (NounTerm, AdjTerm)):
        return depth
    elif isinstance(term, ChainTerm):
        return compute_depth(term.noun, depth + 1)
    elif isinstance(term, FusionTerm):
        return depth + 1
    elif isinstance(term, NounSentence):
        return compute_depth(term.expr, depth + 1)
    elif isinstance(term, (SeqSentence, ImplSentence)):
        left = compute_depth(term.left if hasattr(term, 'left') else term.antecedent, depth + 1)
        right = compute_depth(term.right if hasattr(term, 'right') else term.consequent, depth + 1)
        return max(left, right)
    return depth


def count_primes(term):
    """Count distinct primes in term."""
    primes = set()
    def collect(t):
        if isinstance(t, NounTerm):
            primes.add(t.prime)
        elif isinstance(t, AdjTerm):
            primes.add(t.prime)
        elif isinstance(t, ChainTerm):
            collect(t.noun)
            for op in t.operators:
                collect(op)
        elif isinstance(t, FusionTerm):
            primes.add(t.p)
            primes.add(t.q)
            primes.add(t.r)
        elif isinstance(t, NounSentence):
            collect(t.expr)
        elif isinstance(t, SeqSentence):
            collect(t.left)
            collect(t.right)
        elif isinstance(t, ImplSentence):
            collect(t.antecedent)
            collect(t.consequent)
    collect(term)
    return len(primes)


def count_terms(term):
    """Count total terms in structure."""
    if isinstance(term, (NounTerm, AdjTerm)):
        return 1
    elif isinstance(term, ChainTerm):
        return 1 + count_terms(term.noun) + sum(count_terms(op) for op in term.operators)
    elif isinstance(term, FusionTerm):
        return 1
    elif isinstance(term, NounSentence):
        return 1 + count_terms(term.expr)
    elif isinstance(term, SeqSentence):
        return 1 + count_terms(term.left) + count_terms(term.right)
    elif isinstance(term, ImplSentence):
        return 1 + count_terms(term.antecedent) + count_terms(term.consequent)
    return 1


def main():
    """Run all type system demonstrations."""
    print("ALEPH PRIME - SEMANTIC TYPE SYSTEM EXAMPLES")
    print("=" * 60)
    
    nouns, adjs = demonstrate_basic_terms()
    chains = demonstrate_compound_terms()
    sentences = demonstrate_sentences()
    checker, ctx = demonstrate_type_checking()
    demonstrate_typing_rules()
    complex_struct = demonstrate_complex_structures()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The semantic type system provides:
- Prime-indexed atomic terms (Noun, Adj)
- Compound structures (Chain, Fusion)
- Sentence types (Simple, Sequence, Implication)
- Formal type checking and inference

Key properties:
- Every term has a unique type
- Types guide valid compositions
- Primes provide unique indices
- Fusion creates emergent meanings

Applications:
- Formal semantic analysis
- Type-safe semantic composition
- Structured knowledge representation
""")


if __name__ == "__main__":
    main()