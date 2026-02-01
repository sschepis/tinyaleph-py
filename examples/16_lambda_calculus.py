#!/usr/bin/env python3
"""
Example 16: Lambda Calculus Translation and Evaluation

Demonstrates the lambda calculus module:
- Translation from typed terms to λ-expressions
- λ-calculus evaluation (β-reduction)
- PRQS lexicon for semantic primes
- Concept interpretation
"""

import sys
sys.path.insert(0, '..')

from tinyaleph.semantic import (
    # Lambda expressions
    VarExpr, ConstExpr, LamExpr, AppExpr, PairExpr, ImplExpr, PrimOpExpr,
    # Translation
    Translator, TypeDirectedTranslator, LambdaEvaluator, Semantics,
    # Concept interpretation
    ConceptInterpreter, PRQS_LEXICON, classify_prime,
    # Terms to translate
    N, A, FUSE, CHAIN
)


def demonstrate_lambda_expressions():
    """Demonstrate lambda expression construction."""
    print("=" * 60)
    print("LAMBDA EXPRESSION CONSTRUCTION")
    print("=" * 60)
    
    # Variable expression
    x = VarExpr("x")
    y = VarExpr("y")
    print(f"\nVariables:")
    print(f"  x = {x}")
    print(f"  y = {y}")
    
    # Constant expression (primes)
    c2 = ConstExpr(2)
    c5 = ConstExpr(5)
    print(f"\nConstants (primes):")
    print(f"  2 = {c2}")
    print(f"  5 = {c5}")
    
    # Lambda abstraction
    identity = LamExpr("x", x)
    print(f"\nLambda abstractions:")
    print(f"  λx.x (identity) = {identity}")
    
    const_fn = LamExpr("x", LamExpr("y", VarExpr("x")))
    print(f"  λx.λy.x (const) = {const_fn}")
    
    # Application
    app = AppExpr(identity, c2)
    print(f"\nApplications:")
    print(f"  (λx.x) 2 = {app}")
    
    # Pair expression
    pair = PairExpr(c2, c5)
    print(f"\nPairs:")
    print(f"  ⟨2, 5⟩ = {pair}")
    
    # Implication expression
    impl = ImplExpr(c2, c5)
    print(f"\nImplications:")
    print(f"  2 → 5 = {impl}")
    
    # Prime operation expression
    primop = PrimOpExpr("⊕", c2, c5)
    print(f"\nPrime operations:")
    print(f"  2 ⊕ 5 = {primop}")
    
    return identity, const_fn, app


def demonstrate_translation():
    """Demonstrate translation from typed terms to lambda."""
    print("\n" + "=" * 60)
    print("TERM TO LAMBDA TRANSLATION")
    print("=" * 60)
    
    translator = Translator()
    
    # Translate various terms
    terms = [
        ("Noun N(2)", N(2)),
        ("Adj A(3)", A(3)),
        ("Chain A(2)N(5)", CHAIN([2], 5)),
        ("Fusion FUSE(3,5,11)", FUSE(3, 5, 11)),
    ]
    
    print("\nTranslating typed terms to λ-expressions:")
    print("\nτ translation function:")
    print("  τ(N(p)) = p (constant)")
    print("  τ(A(p)) = λx.⊕(p, x)")
    print("  τ(CHAIN) = function application")
    print("  τ(FUSE(p,q,r)) = p+q+r")
    
    for name, term in terms:
        lambda_expr = translator.translate(term)
        print(f"\n  {name}:")
        print(f"    Term: {term}")
        print(f"    τ(term) = {lambda_expr}")
    
    return translator


def demonstrate_type_directed_translation():
    """Demonstrate type-directed translation."""
    print("\n" + "=" * 60)
    print("TYPE-DIRECTED TRANSLATION")
    print("=" * 60)
    
    translator = TypeDirectedTranslator()
    
    print("""
The translation τ preserves types:
  Γ ⊢ e : T  implies  τ(Γ) ⊢ τ(e) : τ(T)

Type mappings:
  τ(Noun) = const type
  τ(Adj) = function type
  τ(Chain) = application type
""")
    
    # Demonstrate with examples
    terms = [
        N(2),
        A(3),
        CHAIN([3], 7),
        CHAIN([2, 3], 11),
        FUSE(3, 5, 11),
    ]
    
    print("Type-directed translations:")
    
    for term in terms:
        result = translator.translate_typed(term)
        print(f"\n  τ({term})")
        print(f"    = {result['expr']}")
        print(f"    Source type: {result['source_type']}")
        print(f"    Target type: {result['target_type']}")
    
    return translator


def demonstrate_evaluation():
    """Demonstrate lambda evaluation (β-reduction)."""
    print("\n" + "=" * 60)
    print("LAMBDA EVALUATION (β-REDUCTION)")
    print("=" * 60)
    
    evaluator = LambdaEvaluator()
    
    print("\nβ-reduction rule:")
    print("  (λx.M) N → M[x := N]")
    print("  (substitute N for x in M)")
    
    # Build expressions to evaluate
    expressions = []
    
    # Identity applied to constant
    identity = LamExpr("x", VarExpr("x"))
    expr1 = AppExpr(identity, ConstExpr(2))
    expressions.append(("(λx.x) 2", expr1))
    
    # Constant function
    const = LamExpr("x", LamExpr("y", VarExpr("x")))
    expr2 = AppExpr(AppExpr(const, ConstExpr(3)), ConstExpr(5))
    expressions.append(("(λx.λy.x) 3 5", expr2))
    
    # Prime operation
    expr3 = PrimOpExpr("⊕", ConstExpr(2), ConstExpr(5))
    expressions.append(("2 ⊕ 5", expr3))
    
    print("\nEvaluating expressions:")
    
    for name, expr in expressions:
        print(f"\n  {name}")
        print(f"    Expression: {expr}")
        
        try:
            result = evaluator.evaluate(expr)
            print(f"    Result: {result['result']}")
            print(f"    Steps: {result['steps']}")
            print(f"    Is value: {result['is_value']}")
        except Exception as e:
            print(f"    Error: {type(e).__name__}: {e}")
    
    return evaluator


def demonstrate_prqs_lexicon():
    """Demonstrate the PRQS semantic lexicon."""
    print("\n" + "=" * 60)
    print("PRQS LEXICON - Semantic Prime Mappings")
    print("=" * 60)
    
    print("\nPRQS (Prime-indexed Resonant Quantum Semantics)")
    print("Maps primes to semantic concepts:")
    
    print("\nFirst 15 noun concepts:")
    print("-" * 50)
    
    nouns = PRQS_LEXICON.get('nouns', {})
    for i, (p, data) in enumerate(sorted(nouns.items())[:15]):
        print(f"  {p:4d}: {data['concept']:15s} ({data['category']})")
    
    print("\nFirst 15 adjective concepts:")
    print("-" * 50)
    
    adjs = PRQS_LEXICON.get('adjectives', {})
    for i, (p, data) in enumerate(sorted(adjs.items())[:15]):
        print(f"  {p:4d}: {data['concept']:15s} ({data['category']})")
    
    # Show prime classification
    print("\n" + "-" * 40)
    print("\nPrime classification (by mod 4 and mod 6):")
    
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        cat = classify_prime(p)
        print(f"  {p:2d}: primary={cat['primary']}, secondary={cat['secondary']}")
    
    return PRQS_LEXICON


def demonstrate_concept_interpretation():
    """Demonstrate concept interpretation."""
    print("\n" + "=" * 60)
    print("CONCEPT INTERPRETATION")
    print("=" * 60)
    
    interpreter = ConceptInterpreter()
    
    # Interpret primes as concepts
    print("\nPrime → Concept interpretation:")
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    for p in primes:
        noun_interp = interpreter.interpret_noun(N(p))
        adj_interp = interpreter.interpret_adj(A(p))
        print(f"  {p:2d}: noun='{noun_interp}', adj='{adj_interp}'")
    
    # Interpret chain as phrase
    print("\nChain → Phrase interpretation:")
    
    chains = [
        CHAIN([3], 7),
        CHAIN([5, 7], 11),
        CHAIN([3, 5, 7], 2),
    ]
    
    for chain in chains:
        phrase = interpreter.interpret_chain(chain)
        print(f"  {chain} → '{phrase}'")
    
    # Full interpretation with metadata
    print("\nFull noun interpretation with metadata:")
    
    for p in [2, 5, 11]:
        full = interpreter.interpret_noun_full(N(p))
        print(f"\n  N({p}):")
        print(f"    Concept: {full['concept']}")
        print(f"    Category: {full['category']}")
        print(f"    Is core: {full['is_core']}")
    
    # Interpret fusion semantically
    print("\nFusion → Emergent meaning:")
    
    fusions = [
        (3, 5, 11),
        (5, 7, 11),
        (2, 5, 11),
    ]
    
    for p, q, r in fusions:
        meaning = interpreter.interpret_fusion_semantic(p, q, r)
        print(f"\n  FUSE({p},{q},{r})")
        print(f"    Components: {meaning['components']}")
        print(f"    Dominant: {meaning['dominant']}")
        print(f"    Description: {meaning['description']}")
    
    return interpreter


def demonstrate_compatibility_analysis():
    """Demonstrate semantic compatibility analysis."""
    print("\n" + "=" * 60)
    print("SEMANTIC COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    interpreter = ConceptInterpreter()
    
    print("\nAnalyzing compatibility between primes:")
    print("(Higher score = more compatible for composition)")
    
    pairs = [
        (2, 3),
        (3, 5),
        (5, 7),
        (7, 11),
        (2, 7),
        (3, 11),
    ]
    
    for p1, p2 in pairs:
        compat = interpreter.analyze_compatibility(p1, p2)
        print(f"\n  {p1} and {p2}:")
        print(f"    Complementary: {compat['complementary']}")
        print(f"    Same secondary: {compat['same_secondary']}")
        print(f"    Score: {compat['score']:.2f}")
        print(f"    Interpretation: {compat['interpretation']}")


def demonstrate_semantics_class():
    """Demonstrate the Semantics class for denotational semantics."""
    print("\n" + "=" * 60)
    print("DENOTATIONAL SEMANTICS")
    print("=" * 60)
    
    semantics = Semantics()
    
    print("\nDenotational semantics assigns meanings to expressions:")
    print("  [[E]] = meaning of expression E")
    
    # Compute denotations
    expressions = [
        ConstExpr(2),
        ConstExpr(5),
        PairExpr(ConstExpr(2), ConstExpr(3)),
        LamExpr("x", VarExpr("x")),
    ]
    
    print("\nExpression denotations:")
    
    for expr in expressions:
        denotation = semantics.denote(expr)
        print(f"  [[{expr}]] = {denotation}")
    
    # Semantic equivalence
    print("\nSemantic equivalence:")
    
    # (λx.x) 5 ≡ 5
    e1 = AppExpr(LamExpr("x", VarExpr("x")), ConstExpr(5))
    e2 = ConstExpr(5)
    
    equiv = semantics.equivalent(e1, e2)
    print(f"  (λx.x) 5 ≡ 5 : {equiv}")
    
    # Verify operational = denotational
    print("\nVerifying operational = denotational semantics:")
    
    term = CHAIN([2], 5)
    verification = semantics.verify_semantic_equivalence(term)
    print(f"  Term: {verification['term']}")
    print(f"  Operational: {verification['operational']}")
    print(f"  Denotational: {verification['denotational']}")
    print(f"  Equivalent: {verification['equivalent']}")
    
    return semantics


def main():
    """Run all lambda calculus demonstrations."""
    print("ALEPH PRIME - LAMBDA CALCULUS EXAMPLES")
    print("=" * 60)
    
    demonstrate_lambda_expressions()
    demonstrate_translation()
    demonstrate_type_directed_translation()
    demonstrate_evaluation()
    demonstrate_prqs_lexicon()
    demonstrate_concept_interpretation()
    demonstrate_compatibility_analysis()
    demonstrate_semantics_class()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Lambda calculus module provides:
- Full λ-expression representation
- Type-directed translation τ
- β-reduction evaluation
- PRQS semantic lexicon (65+ primes)
- Concept interpretation
- Compatibility analysis

Key translations:
  τ(N(p)) = p (constant)
  τ(A(p)) = λx.⊕(p, x) (function)
  τ(CHAIN) = function composition
  τ(FUSE) = fusion to sum

Applications:
- Semantic analysis
- Concept composition
- Meaning computation
- Natural language semantics
""")


if __name__ == "__main__":
    main()