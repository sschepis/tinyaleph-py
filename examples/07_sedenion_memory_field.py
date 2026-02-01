#!/usr/bin/env python3
"""
Example 07: Sedenion Memory Field (SMF)

This example demonstrates the 16-dimensional holographic memory:
- Encoding memories in sedenions
- Temporal decay and coherence
- Memory recall and retrieval
- Interference patterns
- Memory consolidation

Memory Encoding Formula:
    M = Σ_t α_t · S_t · e^(-λ(t_now - t))

where:
- S_t is a sedenion at time t
- α_t is the amplitude/importance
- λ is the decay rate
"""

import numpy as np
from tinyaleph.observer.smf import SedenionMemoryField, MemoryMoment

def main():
    print("=" * 60)
    print("TinyAleph: Sedenion Memory Field (SMF)")
    print("=" * 60)
    print()
    
    # ===== PART 1: Creating the Memory Field =====
    print("PART 1: Creating the Memory Field")
    print("-" * 40)
    
    smf = SedenionMemoryField(
        decay_rate=0.01,
        max_moments=100
    )
    
    print(f"Created Sedenion Memory Field:")
    print(f"  Dimension: {smf.DIM} (sedenion = 16D)")
    print(f"  Decay rate: {smf.decay_rate}")
    print(f"  Max memories: {smf.max_moments}")
    print(f"  Current time: {smf.current_time}")
    print(f"  Size: {smf.size} memories")
    print()
    
    # ===== PART 2: Encoding Memories =====
    print("PART 2: Encoding Memories")
    print("-" * 40)
    
    # Encode several memories with different importance
    memories = [
        ("Important fact about primes", 0.9),
        ("A casual observation", 0.3),
        ("Critical theorem", 1.0),
        ("Random thought", 0.1),
        ("Interesting pattern", 0.6),
    ]
    
    moments = []
    for content, importance in memories:
        moment = smf.encode(content, importance=importance)
        moments.append(moment)
        print(f"Encoded: '{content[:30]}...'")
        print(f"  Importance: {importance}")
        print(f"  Coherence: {moment.coherence:.4f}")
        print(f"  Entropy: {moment.entropy:.4f}")
    
    print(f"\nTotal memories: {smf.size}")
    print()
    
    # ===== PART 3: Memory Properties =====
    print("PART 3: Memory Properties")
    print("-" * 40)
    
    moment = moments[0]  # "Important fact about primes"
    
    print(f"Memory: '{moment.content[:30]}...'")
    print(f"  Timestamp: {moment.timestamp}")
    print(f"  Coherence: {moment.coherence:.4f}")
    print(f"  Entropy: {moment.entropy:.4f}")
    print()
    
    # Sedenion structure
    print(f"Sedenion representation:")
    print(f"  Dimension: {moment.sedenion.dim}")
    print(f"  First 4 components: {moment.sedenion.c[:4]}")
    print(f"  Norm: {moment.sedenion.norm():.4f}")
    print()
    
    # ===== PART 4: Recall by Similarity =====
    print("PART 4: Recall by Similarity")
    print("-" * 40)
    
    query = "primes theorem"
    print(f"Query: '{query}'")
    
    results = smf.recall(query, top_k=3)
    print(f"Top 3 matches:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. '{result.content}' (coherence={result.coherence:.4f})")
    print()
    
    # ===== PART 5: Recall by Content =====
    print("PART 5: Recall by Content (Pattern Matching)")
    print("-" * 40)
    
    pattern = "fact"
    matches = smf.recall_by_content(pattern)
    print(f"Memories containing '{pattern}':")
    for match in matches:
        print(f"  - '{match.content}'")
    print()
    
    # ===== PART 6: Time Evolution =====
    print("PART 6: Time Evolution and Decay")
    print("-" * 40)
    
    print("Initial state:")
    print(f"  Current time: {smf.current_time}")
    print(f"  Total entropy: {smf.total_entropy:.4f}")
    print(f"  Mean coherence: {smf.mean_coherence:.4f}")
    
    # Advance time
    for i in range(5):
        smf.step(dt=10.0)
        print(f"\nAfter t={smf.current_time:.1f}:")
        print(f"  Mean coherence: {smf.mean_coherence:.4f}")
        
        # Add a new memory at each step
        smf.encode(f"Memory at time {smf.current_time}", importance=0.5)
    
    print(f"\nFinal size: {smf.size} memories")
    print()
    
    # ===== PART 7: Decay and Pruning =====
    print("PART 7: Decay and Pruning")
    print("-" * 40)
    
    # Create fresh SMF for demo
    smf2 = SedenionMemoryField(decay_rate=0.1)  # Faster decay
    
    # Add memories
    for i in range(5):
        smf2.encode(f"Memory {i}", importance=0.5)
        smf2.step(dt=1.0)
    
    print(f"Before decay: {smf2.size} memories")
    print(f"Mean coherence: {smf2.mean_coherence:.4f}")
    
    # Advance time significantly
    smf2.step(dt=50.0)
    smf2.decay_all()  # Apply decay
    
    print(f"After 50 time units and decay:")
    print(f"  Memories remaining: {smf2.size}")
    print(f"  Mean coherence: {smf2.mean_coherence:.4f}")
    print()
    
    # ===== PART 8: Memory Interference =====
    print("PART 8: Memory Interference")
    print("-" * 40)
    
    smf3 = SedenionMemoryField()
    m1 = smf3.encode("Alpha pattern", importance=0.8)
    m2 = smf3.encode("Beta pattern", importance=0.8)
    
    print(f"Memory 1: '{m1.content}'")
    print(f"Memory 2: '{m2.content}'")
    
    # Create interference
    interference = smf3.interference(m1, m2)
    print(f"\nInterference pattern:")
    print(f"  Dimension: {interference.dim}")
    print(f"  Norm: {interference.norm():.4f}")
    print(f"  First 4 components: {interference.c[:4]}")
    print()
    
    # ===== PART 9: Superposition =====
    print("PART 9: Memory Superposition")
    print("-" * 40)
    
    # Create weighted superposition of memories
    weights = [0.5, 0.3, 0.2]
    selected_moments = moments[:3]
    
    superposed = smf.superpose(selected_moments, weights)
    print(f"Superposed {len(selected_moments)} memories with weights {weights}")
    print(f"  Result dimension: {superposed.dim}")
    print(f"  Result norm: {superposed.norm():.4f}")
    print()
    
    # ===== PART 10: Consolidation =====
    print("PART 10: Memory Consolidation")
    print("-" * 40)
    
    # Consolidate all memories into single sedenion
    consolidated = smf.consolidate()
    print(f"Consolidated {smf.size} memories into single sedenion:")
    print(f"  Dimension: {consolidated.dim}")
    print(f"  Norm: {consolidated.norm():.4f}")
    print(f"  Entropy: {consolidated.entropy():.4f}")
    print()
    
    # ===== PART 11: Memory Statistics =====
    print("PART 11: Memory Statistics")
    print("-" * 40)
    
    print(f"Memory field statistics:")
    print(f"  Total memories: {smf.size}")
    print(f"  Current time: {smf.current_time:.1f}")
    print(f"  Total entropy: {smf.total_entropy:.4f}")
    print(f"  Mean coherence: {smf.mean_coherence:.4f}")
    print()
    
    # Age distribution
    print("Memory age distribution:")
    for moment in list(smf.moments)[:5]:
        age = smf.current_time - moment.timestamp
        print(f"  '{moment.content[:20]}...': age={age:.1f}, coherence={moment.coherence:.4f}")
    print()
    
    # ===== PART 12: Clear and Reset =====
    print("PART 12: Clear and Reset")
    print("-" * 40)
    
    print(f"Before clear: {smf.size} memories, time={smf.current_time:.1f}")
    
    smf.clear()
    smf.reset_time()
    
    print(f"After clear: {smf.size} memories, time={smf.current_time:.1f}")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Sedenion Memory Field")
    print("=" * 60)
    print("""
Sedenion Memory Field (SMF):

Structure:
- 16-dimensional hypercomplex representation
- Each memory is a MemoryMoment with:
  * Sedenion (16D vector)
  * Timestamp
  * Entropy
  * Coherence
  * Content

Key Operations:
1. encode(): String → MemoryMoment
2. recall(): Find similar memories
3. step(): Advance time
4. decay_all(): Apply temporal decay
5. consolidate(): Merge all memories
6. superpose(): Weighted combination

Memory Dynamics:
- Coherence decays with time: c *= exp(-λt)
- Low coherence memories are pruned
- Importance affects initial coherence
- Entropy measures information content

Applications:
- Episodic memory storage
- Associative recall
- Pattern completion
- Temporal context modeling

Connection to Primes:
- Sedenions extend quaternions (16D vs 4D)
- Used for holographic prime encoding
- Supports interference patterns
    """)

if __name__ == "__main__":
    main()