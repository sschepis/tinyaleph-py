"""
Main execution engine for TinyAleph.

The AlephEngine coordinates prime resonance computing, managing
state evolution, memory fields, and semantic coherence.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from tinyaleph.hilbert.state import PrimeState
from tinyaleph.core.constants import (
    COHERENCE_THRESHOLD,
    ENTROPY_THRESHOLD,
    MEMORY_DECAY_RATE,
)


@dataclass
class EngineConfig:
    """
    Configuration for the AlephEngine.
    
    Attributes:
        coherence_threshold: Minimum coherence before halting
        entropy_threshold: Maximum entropy before halting
        max_iterations: Maximum computation steps
        memory_decay: Decay rate for memory field
        enable_distributed: Enable distributed computing
    """
    
    coherence_threshold: float = COHERENCE_THRESHOLD
    entropy_threshold: float = ENTROPY_THRESHOLD
    max_iterations: int = 100
    memory_decay: float = MEMORY_DECAY_RATE
    enable_distributed: bool = False


@dataclass
class EngineState:
    """
    Current state of the AlephEngine.
    
    Attributes:
        iteration: Current iteration number
        coherence: Current coherence level
        entropy: Current entropy level
        halted: Whether the engine has halted
        halt_reason: Reason for halting (if halted)
        prime_state: Current prime state (if any)
    """
    
    iteration: int = 0
    coherence: float = 1.0
    entropy: float = 0.0
    halted: bool = False
    halt_reason: Optional[str] = None
    prime_state: Optional[PrimeState] = None


class AlephEngine:
    """
    Main execution engine for prime resonance computing.
    
    The engine manages:
    - PrimeState evolution
    - Semantic coherence tracking
    - Memory field operations
    - Halting conditions based on coherence/entropy
    
    Examples:
        >>> async def main():
        ...     engine = AlephEngine()
        ...     engine.bind_concept("math", PrimeState.composite(6))
        ...     result = await engine.run(PrimeState.uniform())
        ...     print(f"Final coherence: {result.coherence}")
        >>> asyncio.run(main())
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize the AlephEngine.
        
        Args:
            config: Engine configuration (uses defaults if None)
        """
        self.config = config or EngineConfig()
        self.state = EngineState()
        
        # Semantic bindings: concept -> (PrimeState, strength, coherence)
        self._bindings: Dict[str, tuple[PrimeState, float, float]] = {}
        
        # Event hooks
        self.hooks: Dict[str, List[Callable]] = {
            'pre_step': [],
            'post_step': [],
            'on_halt': [],
            'on_collapse': [],
            'on_bind': [],
        }
        
        # Current time for memory operations
        self._time: float = 0.0
    
    def register_hook(self, event: str, callback: Callable) -> None:
        """
        Register callback for engine events.
        
        Available events:
        - pre_step: Called before each step
        - post_step: Called after each step
        - on_halt: Called when engine halts
        - on_collapse: Called when state collapses
        - on_bind: Called when concept is bound
        
        Args:
            event: Event name
            callback: Function to call (receives engine and kwargs)
        """
        if event in self.hooks:
            self.hooks[event].append(callback)
    
    def unregister_hook(self, event: str, callback: Callable) -> None:
        """Remove a registered callback."""
        if event in self.hooks and callback in self.hooks[event]:
            self.hooks[event].remove(callback)
    
    def _fire_hooks(self, event: str, **kwargs: Any) -> None:
        """Fire all callbacks for an event."""
        for callback in self.hooks.get(event, []):
            try:
                callback(self, **kwargs)
            except Exception as e:
                # Log but don't crash on hook errors
                pass
    
    def bind_concept(self, concept: str, state: PrimeState, strength: float = 1.0) -> None:
        """
        Bind a semantic concept to a prime state.
        
        Args:
            concept: Name of the concept
            state: PrimeState representing the concept
            strength: Binding strength [0, 1]
        """
        coherence = state.coherence()
        self._bindings[concept] = (state, strength, coherence)
        self._fire_hooks('on_bind', concept=concept, state=state)
    
    def unbind_concept(self, concept: str) -> None:
        """Remove a concept binding."""
        if concept in self._bindings:
            del self._bindings[concept]
    
    def get_binding(self, concept: str) -> Optional[PrimeState]:
        """Get the PrimeState bound to a concept."""
        if concept in self._bindings:
            return self._bindings[concept][0]
        return None
    
    def compose_concepts(self, concepts: List[str]) -> PrimeState:
        """
        Compose multiple concepts into a unified state.
        
        Uses interference composition:
        |ψ_composed⟩ = Σ s_i |ψ_i⟩ (normalized)
        
        Args:
            concepts: List of concept names to compose
            
        Returns:
            Normalized composite PrimeState
        """
        if not concepts:
            return PrimeState.uniform()
        
        result: Optional[PrimeState] = None
        
        for concept in concepts:
            binding = self._bindings.get(concept)
            if binding is None:
                continue
            
            state, strength, _ = binding
            
            if result is None:
                result = state * strength
            else:
                result = result + (state * strength)
        
        if result is None:
            return PrimeState.uniform()
        
        return result.normalize()
    
    @property
    def global_coherence(self) -> float:
        """Compute global coherence from all bindings."""
        if not self._bindings:
            return 1.0
        
        coherences = [c for _, _, c in self._bindings.values()]
        return float(sum(coherences) / len(coherences))
    
    async def step(self) -> EngineState:
        """
        Execute single computation step.
        
        Checks halting conditions and evolves state.
        
        Returns:
            Current engine state after step
        """
        self._fire_hooks('pre_step', state=self.state)
        
        # Check coherence threshold
        if self.state.coherence < self.config.coherence_threshold:
            self.state.halted = True
            self.state.halt_reason = 'coherence_threshold'
            self._fire_hooks('on_halt', reason='coherence_threshold')
            return self.state
        
        # Check entropy threshold
        if self.state.entropy > self.config.entropy_threshold:
            self.state.halted = True
            self.state.halt_reason = 'entropy_threshold'
            self._fire_hooks('on_halt', reason='entropy_threshold')
            return self.state
        
        # Check iteration limit
        if self.state.iteration >= self.config.max_iterations:
            self.state.halted = True
            self.state.halt_reason = 'max_iterations'
            self._fire_hooks('on_halt', reason='max_iterations')
            return self.state
        
        # Evolve prime state
        if self.state.prime_state is not None:
            self.state.entropy = self.state.prime_state.entropy()
        
        # Update global coherence
        self.state.coherence = self.global_coherence
        
        # Advance time
        self._time += 1.0
        self.state.iteration += 1
        
        self._fire_hooks('post_step', state=self.state)
        
        return self.state
    
    async def run(self, initial_state: Optional[PrimeState] = None) -> EngineState:
        """
        Run engine until halting condition.
        
        Args:
            initial_state: Initial prime state (defaults to uniform)
            
        Returns:
            Final engine state
        """
        self.state = EngineState(
            prime_state=initial_state or PrimeState.uniform()
        )
        
        while not self.state.halted:
            await self.step()
            # Allow other tasks to run
            await asyncio.sleep(0)
        
        return self.state
    
    def run_sync(self, initial_state: Optional[PrimeState] = None) -> EngineState:
        """
        Synchronous version of run().
        
        Args:
            initial_state: Initial prime state
            
        Returns:
            Final engine state
        """
        return asyncio.get_event_loop().run_until_complete(self.run(initial_state))
    
    def collapse(self) -> tuple[int, float]:
        """
        Collapse current state to measurement.
        
        Returns:
            (measured_prime, probability)
        """
        if self.state.prime_state is None:
            return (2, 1.0)
        
        result = self.state.prime_state.measure()
        self._fire_hooks('on_collapse', result=result)
        return result
    
    def reset(self) -> None:
        """Reset engine to initial state."""
        self.state = EngineState()
        self._time = 0.0
    
    @property
    def time(self) -> float:
        """Current engine time."""
        return self._time
    
    @property
    def is_running(self) -> bool:
        """Check if engine is currently running."""
        return not self.state.halted
    
    def __repr__(self) -> str:
        status = "halted" if self.state.halted else "running"
        return f"AlephEngine(iter={self.state.iteration}, {status}, coherence={self.state.coherence:.3f})"