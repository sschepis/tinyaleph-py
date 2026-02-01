#!/usr/bin/env python3
"""
Example 11: Symbolic SMF and Hexagram Classification

Demonstrates the symbolic layer capabilities:
- SymbolicSMF for symbol-grounded memory field
- Hexagram classification (64 I Ching archetypes)
- SymbolicTemporalLayer for time-aware symbolic processing
- Symbol-meaning relationships and resonance
"""

import sys
sys.path.insert(0, '..')

from tinyaleph.observer import (
    # Symbolic SMF
    SymbolicSMF,
    SMFSymbolMapper,
    AXIS_SYMBOL_MAPPING,
    TAG_TO_AXIS,
    
    # Symbolic Temporal
    SymbolicMoment,
    SymbolicTemporalLayer,
    SymbolicPatternDetector,
    EntropyCollapseHead,
    HEXAGRAM_ARCHETYPES,
    FIRST_64_PRIMES,
    
    # Symbols
    Symbol,
    SymbolCategory,
    SymbolDatabase,
    symbol_database,
)


def demonstrate_hexagram_archetypes():
    """Demonstrate the 64 hexagram archetypes."""
    print("=" * 60)
    print("HEXAGRAM ARCHETYPES - 64 I Ching Moment Types")
    print("=" * 60)
    
    print(f"\nTotal archetypes defined: {len(HEXAGRAM_ARCHETYPES)}")
    
    print("\nFirst 8 hexagrams (trigram doubles):")
    for i in range(8):
        arch = HEXAGRAM_ARCHETYPES.get(i)
        if arch:
            print(f"  #{i}: {arch['name']} ({arch['symbol']})")
            print(f"       Tags: {', '.join(arch['tags'])}")
    
    print("\nTransformation hexagrams (8-15):")
    for i in range(8, 16):
        arch = HEXAGRAM_ARCHETYPES.get(i)
        if arch:
            print(f"  #{i}: {arch['name']} - {arch['symbol']}")
    
    print("\nFirst 64 primes for encoding:")
    print(f"  {FIRST_64_PRIMES[:16]}...")
    
    return HEXAGRAM_ARCHETYPES


def demonstrate_entropy_collapse():
    """Demonstrate the entropy collapse head."""
    print("\n" + "=" * 60)
    print("ENTROPY COLLAPSE HEAD")
    print("=" * 60)
    
    collapse = EntropyCollapseHead(target_entropy=5.99, num_attractors=64)
    
    print(f"\n64-attractor codebook initialized")
    print(f"Target entropy: {collapse.target_entropy}")
    
    # Test different state vectors
    print("\nClassifying state vectors:")
    
    test_vectors = [
        [1.0] + [0.0] * 15,           # Axis 0 dominant
        [0.0, 1.0] + [0.0] * 14,      # Axis 1 dominant
        [0.5] * 16,                    # Uniform
        [0.1 * i for i in range(16)], # Gradient
    ]
    
    labels = ["Axis 0 only", "Axis 1 only", "Uniform", "Gradient"]
    
    for label, vec in zip(labels, test_vectors):
        result = collapse.hard_assign(vec)
        arch = HEXAGRAM_ARCHETYPES.get(result['index'], {})
        print(f"\n  {label}:")
        print(f"    Hexagram #{result['index']}: {arch.get('name', '?')}")
        print(f"    Confidence: {result['confidence']:.4f}")
    
    return collapse


def demonstrate_symbolic_smf():
    """Demonstrate the SymbolicSMF field."""
    print("\n" + "=" * 60)
    print("SYMBOLIC SMF - Symbol-Grounded Memory Field")
    print("=" * 60)
    
    smf = SymbolicSMF()
    
    print(f"\nAxisSymbol Mapping:")
    for axis_idx in range(4):
        mapping = AXIS_SYMBOL_MAPPING.get(axis_idx, {})
        print(f"  Axis {axis_idx}: {mapping.get('category', '?')} → {mapping.get('archetypes', [])[:2]}")
    
    print(f"\nTag to Axis mapping (samples):")
    for tag in ['unity', 'duality', 'transformation', 'wisdom']:
        axis = TAG_TO_AXIS.get(tag, -1)
        print(f"  '{tag}' → Axis {axis}")
    
    # Ground SMF in symbols
    print("\nGrounding SMF orientation in symbols:")
    
    # Excite specific axes
    smf.s[0] = 0.9  # coherence
    smf.s[7] = 0.8  # wisdom
    smf.s[4] = 0.5  # change
    smf.normalize()
    
    grounded = smf.ground_in_symbols(3)
    for g in grounded:
        print(f"  Axis {g['axis']}: {g['symbol'].name} (alignment: {g['alignment']:.3f})")
    
    # Symbol resonance
    print("\nSymbol resonance with current state:")
    resonant = smf.find_resonant_symbols(5)
    for r in resonant[:3]:
        print(f"  {r['symbol'].name}: {r['resonance']:.4f}")
    
    return smf


def demonstrate_symbolic_temporal():
    """Demonstrate the SymbolicTemporalLayer."""
    print("\n" + "=" * 60)
    print("SYMBOLIC TEMPORAL LAYER")
    print("=" * 60)
    
    transitions = []
    
    def on_transition(event):
        transitions.append(event)
    
    temporal = SymbolicTemporalLayer(
        coherence_threshold=0.7,
        on_hexagram_transition=on_transition
    )
    
    print("\nProcessing moment sequence:")
    
    # Create series of moments with different states
    states = [
        {'coherence': 0.9, 'entropy': 0.1, 'active_primes': [2, 3, 5]},
        {'coherence': 0.8, 'entropy': 0.3, 'active_primes': [7, 11, 13]},
        {'coherence': 0.5, 'entropy': 0.6, 'active_primes': [17, 19, 23]},
        {'coherence': 0.7, 'entropy': 0.4, 'active_primes': [29, 31, 37]},
        {'coherence': 0.85, 'entropy': 0.2, 'active_primes': [41, 43, 47]},
    ]
    
    triggers = ['observation', 'analysis', 'transition', 'integration', 'completion']
    
    for trigger, state in zip(triggers, states):
        moment = temporal.create_moment(trigger, state)
        arch = moment.archetype['name'] if moment.archetype else 'unknown'
        print(f"  {trigger}: Hex #{moment.hexagram_index} ({arch})")
        print(f"    φ-resonance: {moment.phi_resonance:.4f}, confidence: {moment.classification_confidence:.4f}")
    
    print(f"\nHexagram transitions: {len(transitions)}")
    
    # Get dominant archetypes
    print("\nDominant archetypes:")
    dominant = temporal.get_dominant_archetypes(3)
    for d in dominant:
        print(f"  {d['name']}: {d['count']} ({d['frequency']:.1%})")
    
    # Prediction
    prediction = temporal.predict_next_archetype()
    if prediction['predicted']:
        print(f"\nNext archetype prediction: {prediction['predicted']}")
        print(f"  Confidence: {prediction['confidence']:.4f}")
    
    return temporal


def demonstrate_iching_reading():
    """Demonstrate I Ching style reading."""
    print("\n" + "=" * 60)
    print("I CHING READING")
    print("=" * 60)
    
    temporal = SymbolicTemporalLayer()
    
    # Create a moment
    state = {
        'coherence': 0.75,
        'entropy': 0.35,
        'active_primes': [2, 7, 17, 31],
    }
    
    temporal.create_moment('divination', state)
    
    reading = temporal.get_iching_reading()
    
    if reading:
        print(f"\nCurrent Hexagram: #{reading['current']['number']}")
        print(f"  Name: {reading['current']['name']}")
        print(f"  Symbol: {reading['current']['symbol']}")
        print(f"  Tags: {', '.join(reading['current']['tags'])}")
        
        print(f"\nConfidence: {reading['confidence']:.4f}")
        
        print(f"\nResonance:")
        print(f"  PHI: {reading['resonance']['phi']:.4f}")
        print(f"  Prime: {reading['resonance']['prime']:.4f}")
        print(f"  Harmonic: {reading['resonance']['harmonic']}")
        
        if reading['related_symbols']:
            print(f"\nRelated symbols:")
            for sym in reading['related_symbols'][:3]:
                print(f"  {sym['name']} (via {sym['match_type']})")
    
    return reading


def demonstrate_smf_mapper():
    """Demonstrate SMFSymbolMapper."""
    print("\n" + "=" * 60)
    print("SMF SYMBOL MAPPER")
    print("=" * 60)
    
    mapper = SMFSymbolMapper()
    
    # Get some symbols
    symbols = symbol_database.get_all_symbols()[:5]
    
    print("\nMapping symbols to SMF orientations:")
    
    for sym in symbols:
        smf = mapper.symbol_to_smf(sym)
        dominant = smf.dominant_axes(2)
        axes = [d['name'] for d in dominant]
        print(f"  {sym.name}: dominant axes = {axes}")
    
    # Symbolic distance
    if len(symbols) >= 2:
        print("\nSymbolic distances:")
        for i in range(min(3, len(symbols))):
            for j in range(i + 1, min(4, len(symbols))):
                smf1 = mapper.symbol_to_smf(symbols[i])
                smf2 = mapper.symbol_to_smf(symbols[j])
                dist = mapper.symbolic_distance(smf1, smf2)
                print(f"  {symbols[i].name} ↔ {symbols[j].name}: {dist:.4f}")
    
    return mapper


def demonstrate_pattern_detection():
    """Demonstrate symbolic pattern detection."""
    print("\n" + "=" * 60)
    print("SYMBOLIC PATTERN DETECTION")
    print("=" * 60)
    
    detector = SymbolicPatternDetector(window_size=10)
    temporal = SymbolicTemporalLayer()
    
    # Generate a sequence of moments
    states = [
        {'coherence': 0.9, 'active_primes': [2, 3]},
        {'coherence': 0.7, 'active_primes': [5, 7]},
        {'coherence': 0.5, 'active_primes': [11, 13]},
        {'coherence': 0.3, 'active_primes': [17, 19]},
        {'coherence': 0.5, 'active_primes': [23, 29]},
        {'coherence': 0.7, 'active_primes': [31, 37]},
        {'coherence': 0.9, 'active_primes': [41, 43]},
    ]
    
    moments = []
    for i, state in enumerate(states):
        m = temporal.create_moment(f'event_{i}', state)
        moments.append(m)
    
    # Detect patterns
    patterns = detector.detect_patterns(moments)
    print(f"\nDetected temporal patterns: {len(patterns)}")
    for p in patterns[:3]:
        print(f"  Period: {p.get('period', '?')}, Strength: {p.get('strength', 0):.4f}")
    
    # Detect narrative patterns
    narratives = detector.detect_narrative_patterns(moments)
    print(f"\nNarrative patterns: {len(narratives)}")
    for n in narratives:
        print(f"  Type: {n['type']}, Occurrences: {n['occurrences']}")
    
    return detector


def main():
    """Run all symbolic demonstrations."""
    print("ALEPH PRIME - SYMBOLIC LAYER EXAMPLES")
    print("=" * 60)
    
    demonstrate_hexagram_archetypes()
    demonstrate_entropy_collapse()
    demonstrate_symbolic_smf()
    demonstrate_symbolic_temporal()
    demonstrate_iching_reading()
    demonstrate_smf_mapper()
    demonstrate_pattern_detection()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The symbolic layer provides:

1. HEXAGRAM ARCHETYPES
   - 64 I Ching hexagram moment types
   - Cultural tags for each archetype
   - Prime-based encoding

2. ENTROPY COLLAPSE HEAD
   - Classifies continuous states into discrete attractors
   - Soft and hard assignment modes
   - Confidence scoring

3. SYMBOLIC SMF
   - Grounds 16D memory field in archetypal symbols
   - Axis-symbol mappings
   - Symbol resonance calculation

4. SYMBOLIC TEMPORAL LAYER
   - Time-aware symbolic processing
   - Hexagram transition detection
   - Archetype sequence analysis
   - I Ching style readings

5. PATTERN DETECTION
   - Narrative pattern recognition
   - Temporal rhythm detection
   - Archetype sequence matching

Key applications:
- Moment classification and divination
- Narrative structure analysis
- Cultural archetype grounding
- Temporal symbolic sequences
""")


if __name__ == "__main__":
    main()