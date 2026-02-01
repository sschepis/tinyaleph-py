#!/usr/bin/env python3
"""
Example 19: Semantic Inference and Entity Extraction

Demonstrates the inference module:
- CompoundBuilder for semantic composition
- SemanticInference for reasoning
- EntityExtractor for NLP-style extraction
- Wierzbicka semantic primes (SEMANTIC_PRIMES)
"""

import sys
sys.path.insert(0, '..')

from tinyaleph.semantic import (
    # Semantic primitives
    SemanticCategory, SemanticPrimitive, SEMANTIC_PRIMES,
    # Compound building
    CompoundConcept, CompoundBuilder,
    # Inference
    InferenceRule, SemanticInference,
    # Entity extraction
    ExtractedEntity, ExtractedRelation, EntityExtractor,
    # Utility functions
    build_compound, analyze_semantic_value, infer_from_knowledge,
    extract_semantic_structure, semantic_similarity
)


def demonstrate_semantic_primes():
    """Demonstrate Wierzbicka's semantic primes."""
    print("=" * 60)
    print("SEMANTIC PRIMES (Wierzbicka's NSM)")
    print("=" * 60)
    
    print("\nNatural Semantic Metalanguage proposes ~65 semantic primes")
    print("that are universal across all human languages.")
    
    # Group by category
    categories = {}
    for prime, prim in SEMANTIC_PRIMES.items():
        cat = prim.category.value
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((prime, prim.label))
    
    print("\nSemantic primes by category:")
    print("-" * 50)
    
    for cat in ['entity', 'action', 'property', 'relation', 'quantifier', 'modifier', 'connective']:
        if cat in categories:
            primes = categories[cat]
            labels = [f"{label}({p})" for p, label in primes[:5]]
            print(f"\n  {cat.upper()}:")
            print(f"    {', '.join(labels)}")
            if len(primes) > 5:
                print(f"    ... and {len(primes) - 5} more")
    
    print(f"\nTotal semantic primes defined: {len(SEMANTIC_PRIMES)}")
    
    return SEMANTIC_PRIMES


def demonstrate_compound_building():
    """Demonstrate building compound concepts."""
    print("\n" + "=" * 60)
    print("COMPOUND BUILDING")
    print("=" * 60)
    
    builder = CompoundBuilder()
    
    # Build from primitives
    print("\n1. FROM PRIMITIVES")
    
    # Single primitive
    thing = builder.primitive("SOMETHING")
    print(f"   'something' = {thing.structure}")
    print(f"   Value: {thing.value}")
    
    # Conjunction
    print("\n2. CONJUNCTION (A ∧ B)")
    
    good = builder.primitive("GOOD")
    big = builder.primitive("BIG")
    compound = builder.conjoin(thing, good, big)
    
    print(f"   'something good big' = {compound.structure}")
    print(f"   Value: {compound.value}")
    print(f"   Complexity: {compound.complexity}")
    
    # Modification
    print("\n3. MODIFICATION (Mod(A))")
    
    small_thing = builder.modify(thing, "SMALL")
    print(f"   'small something' = {small_thing.structure}")
    print(f"   Value: {small_thing.value}")
    
    # Relation
    print("\n4. RELATION (R(A, B))")
    
    person = builder.primitive("SOMEONE")
    other = builder.primitive("OTHER")
    above_rel = builder.relate(person, "ABOVE", other)
    
    print(f"   'someone above other' = {above_rel.structure}")
    print(f"   Value: {above_rel.value}")
    
    # Negation
    print("\n5. NEGATION (¬A)")
    
    not_good = builder.negate(good)
    print(f"   'not good' = {not_good.structure}")
    print(f"   Value: {not_good.value}")
    
    # Conditional
    print("\n6. CONDITIONAL (IF A THEN B)")
    
    think = builder.primitive("THINK")
    know = builder.primitive("KNOW")
    conditional = builder.conditional(think, know)
    
    print(f"   'if think then know' = {conditional.structure}")
    print(f"   Value: {conditional.value}")
    
    # Quantification
    print("\n7. QUANTIFICATION (Q(A))")
    
    all_things = builder.quantify("ALL", thing)
    print(f"   'all something' = {all_things.structure}")
    print(f"   Value: {all_things.value}")
    
    return builder


def demonstrate_compound_analysis():
    """Demonstrate analyzing compound values."""
    print("\n" + "=" * 60)
    print("COMPOUND ANALYSIS")
    print("=" * 60)
    
    builder = CompoundBuilder()
    
    # Analyze arbitrary values
    values = [
        2,      # I
        6,      # I × YOU (2 × 3)
        35,     # SOMEONE × SOMETHING (5 × 7)
        191,    # GOOD (prime)
        1337,   # 7 × 191
        9699690, # 2×3×5×7×11×13×17×19
    ]
    
    print("\nAnalyzing semantic values:")
    print("-" * 50)
    
    for value in values:
        compound = builder.analyze(value)
        print(f"\n  Value: {value}")
        print(f"  Structure: {compound.structure}")
        print(f"  Complexity: {compound.complexity}")
        if compound.components:
            labels = [c.label for c in compound.components]
            print(f"  Components: {labels}")
    
    # Semantic similarity
    print("\n" + "-" * 40)
    print("\nSemantic similarity (via shared factors):")
    
    c1 = builder.conjoin(builder.primitive("GOOD"), builder.primitive("SOMETHING"))
    c2 = builder.conjoin(builder.primitive("BIG"), builder.primitive("SOMETHING"))
    c3 = builder.conjoin(builder.primitive("BAD"), builder.primitive("OTHER"))
    
    pairs = [(c1, c2), (c1, c3), (c2, c3)]
    
    for a, b in pairs:
        sim = builder.similarity(a, b)
        print(f"  sim({a.structure}, {b.structure}) = {sim:.3f}")
    
    return builder


def demonstrate_inference_engine():
    """Demonstrate semantic inference."""
    print("\n" + "=" * 60)
    print("SEMANTIC INFERENCE ENGINE")
    print("=" * 60)
    
    engine = SemanticInference()
    
    # Assert knowledge
    print("\n1. ASSERTING KNOWLEDGE")
    
    # Good action exists
    good = 191  # GOOD
    do = 79     # DO
    engine.assert_value(good * do)
    print(f"   Asserted: GOOD × DO = {good * do}")
    
    # Something happens
    something = 7
    happen = 83
    engine.assert_value(something * happen)
    print(f"   Asserted: SOMETHING × HAPPEN = {something * happen}")
    
    # Someone exists
    someone = 5
    engine.assert_value(someone)
    print(f"   Asserted: SOMEONE = {someone}")
    
    print(f"\n   Total knowledge: {len(engine.knowledge)} items")
    
    # Query
    print("\n2. QUERYING")
    
    queries = [
        (good, "Contains GOOD?"),
        (do, "Contains DO?"),
        (good * do, "Contains GOOD×DO?"),
        (113, "Contains DIE?"),
    ]
    
    for pattern, question in queries:
        matches = engine.query(pattern)
        print(f"   {question}: {len(matches)} matches")
    
    # Subsumption
    print("\n3. SUBSUMPTION")
    
    print("   A subsumes B if A's factors ⊆ B's factors (A divides B)")
    
    tests = [
        (good, good * do),
        (do, good * do),
        (good * do, good),
        (something, something * happen),
    ]
    
    for general, specific in tests:
        subsumes = engine.subsumes(general, specific)
        print(f"   {general} subsumes {specific}? {subsumes}")
    
    # Forward chaining
    print("\n4. FORWARD CHAINING")
    
    print(f"   Running inference rules...")
    new_inferences = engine.forward_chain(max_steps=10)
    
    print(f"   New inferences: {len(new_inferences)}")
    for inf in list(new_inferences)[:3]:
        print(f"     {inf}")
    
    # Abduction
    print("\n5. ABDUCTIVE REASONING")
    
    observation = good * do * someone
    explanations = engine.abductive_explain(observation)
    
    print(f"   Observation: {observation}")
    print(f"   Possible explanations:")
    for exp, conf in explanations[:3]:
        print(f"     {exp} (confidence: {conf:.2f})")
    
    # Generalization
    print("\n6. GENERALIZATION")
    
    instances = [
        good * something * do,
        good * someone * do,
        good * do * happen,
    ]
    
    general = engine.generalize(instances)
    print(f"   Instances: {instances}")
    print(f"   Generalization (GCD): {general}")
    print(f"   = GOOD × DO (shared across all)")
    
    return engine


def demonstrate_entity_extraction():
    """Demonstrate entity and relation extraction."""
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION")
    print("=" * 60)
    
    extractor = EntityExtractor()
    
    # Sample texts
    texts = [
        "The cat is big.",
        "John sees Mary.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    print("\nExtracting entities from text:")
    print("-" * 50)
    
    for text in texts:
        print(f"\n  Text: '{text}'")
        
        entities = extractor.extract_entities(text)
        print(f"  Entities ({len(entities)}):")
        for e in entities[:3]:
            print(f"    '{e.text}' ({e.entity_type}): prime={e.prime_encoding}")
        
        relations = extractor.extract_relations(text, entities)
        print(f"  Relations ({len(relations)}):")
        for r in relations[:2]:
            print(f"    {r.subject.text} --{r.predicate}--> {r.object_.text}")
    
    return extractor


def demonstrate_semantic_graph():
    """Demonstrate semantic graph construction."""
    print("\n" + "=" * 60)
    print("SEMANTIC GRAPH CONSTRUCTION")
    print("=" * 60)
    
    extractor = EntityExtractor()
    
    text = "Alice sees Bob. Bob wants the apple. The apple is red."
    
    print(f"\nText: '{text}'")
    print("\nSemantic graph:")
    
    graph = extractor.to_semantic_graph(text)
    
    print("\n  Nodes:")
    for node in graph['nodes'][:5]:
        print(f"    [{node['id']}] {node['text']} ({node['type']})")
    
    print("\n  Edges:")
    for edge in graph['edges'][:5]:
        print(f"    {edge['source']} --{edge['relation']}--> {edge['target']}")
    
    # Fingerprint
    fingerprint = extractor.semantic_fingerprint(text)
    print(f"\n  Semantic fingerprint: {fingerprint}")
    
    return graph


def demonstrate_semantic_similarity():
    """Demonstrate semantic similarity computation."""
    print("\n" + "=" * 60)
    print("SEMANTIC SIMILARITY")
    print("=" * 60)
    
    texts = [
        "The cat is big.",
        "The dog is big.",
        "The small bird flies.",
        "Mathematics is beautiful.",
    ]
    
    print("\nComparing text pairs:")
    print("-" * 50)
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = semantic_similarity(texts[i], texts[j])
            print(f"\n  Text 1: '{texts[i]}'")
            print(f"  Text 2: '{texts[j]}'")
            print(f"  Similarity: {sim:.4f}")


def demonstrate_full_pipeline():
    """Demonstrate full semantic analysis pipeline."""
    print("\n" + "=" * 60)
    print("FULL SEMANTIC ANALYSIS PIPELINE")
    print("=" * 60)
    
    text = "The wise teacher knows all things."
    
    print(f"\nInput text: '{text}'")
    print("\nRunning full analysis...")
    
    analysis = extract_semantic_structure(text)
    
    print("\n1. ENTITIES:")
    for e in analysis['entities'][:4]:
        print(f"   {e['text']} ({e['type']}): {e['prime']}")
    
    print("\n2. RELATIONS:")
    for r in analysis['relations'][:3]:
        print(f"   {r['subject']} --{r['predicate']}--> {r['object']}")
    
    print("\n3. GRAPH:")
    print(f"   Nodes: {len(analysis['graph']['nodes'])}")
    print(f"   Edges: {len(analysis['graph']['edges'])}")
    
    print("\n4. FINGERPRINT:")
    print(f"   Value: {analysis['fingerprint']}")
    if analysis['fingerprint_analysis']:
        print(f"   Structure: {analysis['fingerprint_analysis']['structure']}")
    
    return analysis


def main():
    """Run all inference demonstrations."""
    print("ALEPH PRIME - SEMANTIC INFERENCE EXAMPLES")
    print("=" * 60)
    
    demonstrate_semantic_primes()
    demonstrate_compound_building()
    demonstrate_compound_analysis()
    demonstrate_inference_engine()
    demonstrate_entity_extraction()
    demonstrate_semantic_graph()
    demonstrate_semantic_similarity()
    demonstrate_full_pipeline()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Semantic inference module provides:
- ~50 Wierzbicka semantic primes
- Compound concept construction
- Forward/backward chaining inference
- Subsumption and generalization
- Entity and relation extraction
- Semantic similarity computation

Key operations:
- Conjunction: A ∧ B = A × B (product)
- Modification: Mod(A) = modifier × A
- Relation: R(A,B) = A × R × B
- Negation: ¬A = NOT × A

Applications:
- Knowledge representation
- Semantic reasoning
- Text analysis
- Similarity computation
""")


if __name__ == "__main__":
    main()