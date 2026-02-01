#!/usr/bin/env python3
"""
Example 10: AlephEngine Runtime

This example demonstrates the unified runtime engine:
- AlephEngine: Central execution context
- Hooks and callbacks
- Configuration management
- Async execution
- State history and checkpoints
- Full integration of all components

AlephEngine is the heart of TinyAleph, orchestrating all
subsystems into a coherent computational environment.
"""

import asyncio
import time
from tinyaleph.runtime.engine import (
    AlephEngine,
    AlephConfig,
    EngineHooks,
    ExecutionPhase,
)
from tinyaleph.hilbert.state import PrimeState
from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.constants import PHI
import math

def main():
    print("=" * 60)
    print("TinyAleph: AlephEngine Runtime")
    print("=" * 60)
    print()
    
    # ===== PART 1: Engine Creation =====
    print("PART 1: Engine Creation")
    print("-" * 40)
    
    # Default engine
    engine = AlephEngine()
    print(f"Created default AlephEngine:")
    print(f"  State: {engine.state}")
    print(f"  Phase: {engine.phase}")
    print(f"  Coherence: {engine.coherence:.4f}")
    print()
    
    # Named engine with custom config
    config = AlephConfig(
        name="quantum_processor",
        max_history=100,
        coherence_threshold=0.7,
        default_primes=[2, 3, 5, 7, 11],
    )
    engine = AlephEngine(config=config)
    print(f"Created named engine: {engine.config.name}")
    print(f"  Max history: {engine.config.max_history}")
    print(f"  Coherence threshold: {engine.config.coherence_threshold}")
    print(f"  Default primes: {engine.config.default_primes}")
    print()
    
    # ===== PART 2: State Management =====
    print("PART 2: State Management")
    print("-" * 40)
    
    # Initialize with state
    initial_state = PrimeState.uniform_superposition([2, 3, 5])
    engine = AlephEngine(initial_state=initial_state)
    print(f"Initial state: {engine.state}")
    
    # Set new state
    new_state = PrimeState.single_prime(7)
    engine.set_state(new_state)
    print(f"After set_state(|7⟩): {engine.state}")
    
    # State is tracked in history
    print(f"History length: {len(engine.history)}")
    print()
    
    # ===== PART 3: Operators and Evolution =====
    print("PART 3: Operators and Evolution")
    print("-" * 40)
    
    engine = AlephEngine()
    engine.set_state(PrimeState.uniform_superposition([2, 3, 5]))
    
    print(f"Before: {engine.state}")
    print(f"  Entropy: {engine.state.entropy():.4f}")
    
    # Apply phase shift
    engine.apply_phase_shift(2, math.pi / 4)
    print(f"\nAfter phase shift on |2⟩:")
    print(f"  {engine.state}")
    
    # Evolve state
    steps = 5
    for i in range(steps):
        engine.evolve(dt=0.1)
    print(f"\nAfter {steps} evolution steps:")
    print(f"  {engine.state}")
    print(f"  Coherence: {engine.coherence:.4f}")
    print()
    
    # ===== PART 4: Execution Phases =====
    print("PART 4: Execution Phases")
    print("-" * 40)
    
    engine = AlephEngine()
    
    print("Execution phases:")
    for phase in ExecutionPhase:
        print(f"  {phase.name}: {phase.value}")
    
    print(f"\nCurrent phase: {engine.phase.name}")
    
    # Transition through phases
    engine.transition_phase(ExecutionPhase.INITIALIZING)
    print(f"After INITIALIZING: {engine.phase.name}")
    
    engine.transition_phase(ExecutionPhase.PROCESSING)
    print(f"After PROCESSING: {engine.phase.name}")
    
    engine.transition_phase(ExecutionPhase.COLLAPSING)
    print(f"After COLLAPSING: {engine.phase.name}")
    
    engine.transition_phase(ExecutionPhase.IDLE)
    print(f"After IDLE: {engine.phase.name}")
    print()
    
    # ===== PART 5: Hooks and Callbacks =====
    print("PART 5: Hooks and Callbacks")
    print("-" * 40)
    
    # Create hooks
    hook_log = []
    
    def on_state_change(old, new):
        hook_log.append(f"State: {len(old.amplitudes)} -> {len(new.amplitudes)} primes")
    
    def on_phase_change(old, new):
        hook_log.append(f"Phase: {old.name} -> {new.name}")
    
    def on_collapse(state, result):
        hook_log.append(f"Collapse: measured |{result}⟩")
    
    hooks = EngineHooks(
        on_state_change=on_state_change,
        on_phase_change=on_phase_change,
        on_collapse=on_collapse,
    )
    
    engine = AlephEngine(hooks=hooks)
    
    # Trigger hooks
    engine.set_state(PrimeState.uniform_superposition([2, 3, 5, 7]))
    engine.transition_phase(ExecutionPhase.COLLAPSING)
    
    # Collapse triggers hook
    result = engine.collapse()
    
    print("Hook log:")
    for entry in hook_log:
        print(f"  {entry}")
    print()
    
    # ===== PART 6: Checkpoint and Restore =====
    print("PART 6: Checkpoint and Restore")
    print("-" * 40)
    
    engine = AlephEngine()
    engine.set_state(PrimeState.uniform_superposition([2, 3, 5]))
    
    print(f"Initial: {engine.state}")
    
    # Create checkpoint
    checkpoint = engine.checkpoint()
    print(f"Checkpoint created at step {checkpoint['step']}")
    
    # Modify state
    engine.set_state(PrimeState.single_prime(7))
    engine.evolve(dt=0.5)
    print(f"After modification: {engine.state}")
    
    # Restore from checkpoint
    engine.restore(checkpoint)
    print(f"After restore: {engine.state}")
    print()
    
    # ===== PART 7: History and Metrics =====
    print("PART 7: History and Metrics")
    print("-" * 40)
    
    engine = AlephEngine()
    
    # Generate some history
    for i in range(5):
        primes = [2, 3, 5, 7, 11][:i+1]
        engine.set_state(PrimeState.uniform_superposition(primes))
        engine.evolve(dt=0.1)
    
    print(f"History length: {len(engine.history)}")
    
    # Get metrics
    metrics = engine.metrics()
    print(f"\nEngine metrics:")
    print(f"  Total steps: {metrics['total_steps']}")
    print(f"  Current coherence: {metrics['coherence']:.4f}")
    print(f"  Phase: {metrics['phase']}")
    print(f"  History size: {metrics['history_size']}")
    print()
    
    # ===== PART 8: Resonance Operations =====
    print("PART 8: Resonance Operations")
    print("-" * 40)
    
    engine = AlephEngine()
    
    # Store resonant fragment
    pattern = [0.5, 0.3, 0.2]
    engine.store_fragment("concept_alpha", pattern)
    print(f"Stored fragment: concept_alpha")
    
    # Query fragment
    query = [0.5, 0.3, 0.2]
    result = engine.query_fragment("concept_alpha", query)
    print(f"Query similarity: {result:.4f}")
    
    # Store multiple fragments
    engine.store_fragment("concept_beta", [0.1, 0.8, 0.1])
    engine.store_fragment("concept_gamma", [0.2, 0.2, 0.6])
    
    # Find best match
    best = engine.find_best_fragment(query)
    print(f"Best match: {best}")
    print()
    
    # ===== PART 9: Entanglement Network =====
    print("PART 9: Entanglement Network")
    print("-" * 40)
    
    engine = AlephEngine()
    
    # Create entangled pairs
    engine.entangle_primes(2, 3)
    engine.entangle_primes(5, 7)
    engine.entangle_primes(2, 5)
    
    print("Created entanglement network:")
    network = engine.get_entanglement_network()
    print(f"  Edges: {list(network.edges())}")
    
    # Check entanglement
    print(f"\n2-3 entangled: {engine.are_entangled(2, 3)}")
    print(f"2-7 entangled: {engine.are_entangled(2, 7)}")
    
    # Path through network
    path = engine.entanglement_path(2, 7)
    print(f"Path 2 -> 7: {path}")
    print()
    
    # ===== PART 10: Batch Processing =====
    print("PART 10: Batch Processing")
    print("-" * 40)
    
    engine = AlephEngine()
    
    # Batch state operations
    states = [
        PrimeState.single_prime(p)
        for p in [2, 3, 5, 7, 11]
    ]
    
    print("Processing batch of states:")
    results = engine.process_batch(states)
    
    for i, (state, result) in enumerate(zip(states, results)):
        print(f"  State {i}: coherence = {result['coherence']:.4f}")
    print()
    
    # ===== PART 11: Async Execution =====
    print("PART 11: Async Execution")
    print("-" * 40)
    
    async def async_demo():
        engine = AlephEngine()
        
        print("Starting async evolution...")
        
        # Async evolve
        start = time.time()
        await engine.async_evolve(steps=10, dt=0.1)
        elapsed = time.time() - start
        
        print(f"Completed in {elapsed*1000:.2f}ms")
        print(f"Final coherence: {engine.coherence:.4f}")
        
        # Parallel processing
        states = [
            PrimeState.uniform_superposition([2, 3, 5]),
            PrimeState.uniform_superposition([7, 11, 13]),
            PrimeState.uniform_superposition([17, 19, 23]),
        ]
        
        results = await engine.async_process_batch(states)
        print(f"\nParallel batch results: {len(results)} processed")
    
    asyncio.run(async_demo())
    print()
    
    # ===== PART 12: Full Pipeline =====
    print("PART 12: Full Pipeline Integration")
    print("-" * 40)
    
    # Create engine with all subsystems
    config = AlephConfig(
        name="full_pipeline",
        max_history=50,
        coherence_threshold=0.5,
        default_primes=[2, 3, 5, 7, 11, 13, 17],
    )
    
    pipeline_log = []
    
    def log_state(old, new):
        pipeline_log.append(f"State updated")
    
    hooks = EngineHooks(on_state_change=log_state)
    engine = AlephEngine(config=config, hooks=hooks)
    
    # Initialize state
    engine.set_state(PrimeState.uniform_superposition([2, 3, 5]))
    engine.transition_phase(ExecutionPhase.INITIALIZING)
    
    # Store semantic fragments
    engine.store_fragment("query", [0.6, 0.3, 0.1])
    engine.store_fragment("key", [0.5, 0.4, 0.1])
    engine.store_fragment("value", [0.2, 0.3, 0.5])
    
    # Create entanglement structure
    engine.entangle_primes(2, 3)
    engine.entangle_primes(3, 5)
    
    # Processing phase
    engine.transition_phase(ExecutionPhase.PROCESSING)
    
    for i in range(10):
        engine.evolve(dt=0.1)
        
        # Check coherence
        if engine.coherence < engine.config.coherence_threshold:
            engine.transition_phase(ExecutionPhase.COLLAPSING)
            result = engine.collapse()
            print(f"  Collapsed at step {i}: |{result}⟩")
            break
    
    # Final state
    engine.transition_phase(ExecutionPhase.IDLE)
    
    metrics = engine.metrics()
    print(f"\nFinal pipeline metrics:")
    print(f"  Total steps: {metrics['total_steps']}")
    print(f"  Coherence: {metrics['coherence']:.4f}")
    print(f"  Phase: {metrics['phase']}")
    print(f"  State updates logged: {len(pipeline_log)}")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: AlephEngine Runtime")
    print("=" * 60)
    print("""
AlephEngine is the unified runtime for TinyAleph:

Core Components:
    AlephConfig      - Configuration management
    AlephEngine      - Central execution context
    EngineHooks      - Callback system for events
    ExecutionPhase   - State machine for processing

State Management:
    - PrimeState storage and evolution
    - Checkpoint/restore for fault tolerance
    - History tracking for analysis

Operations:
    - Phase shifts and unitary evolution
    - Resonant fragment storage/query
    - Entanglement network management
    - Batch and async processing

Execution Phases:
    IDLE         - Engine ready, no active computation
    INITIALIZING - Setting up initial state
    PROCESSING   - Active computation
    COLLAPSING   - Measurement in progress
    COMPLETE     - Computation finished

Hooks:
    on_state_change  - Called when state updates
    on_phase_change  - Called when phase transitions
    on_collapse      - Called when measurement occurs
    on_error         - Called when errors occur

Usage Pattern:
    1. Create engine with config
    2. Initialize state
    3. Store semantic fragments
    4. Create entanglement structure
    5. Process (evolve, transform)
    6. Collapse when coherent
    7. Extract results

AlephEngine unifies all subsystems into a coherent
computational environment for prime-based quantum
symbolic computing.
    """)

if __name__ == "__main__":
    main()