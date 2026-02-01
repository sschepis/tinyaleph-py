"""
Quantum Entanglement for Distributed Prime Networks

Implements entanglement protocols for networked prime-state systems.
Enables non-local correlations between distributed nodes while 
maintaining coherence across the network.

Core Concepts:
1. Bell States: Maximally entangled prime pairs
2. Entanglement Swapping: Extend entanglement through intermediate nodes
3. Teleportation: Transfer prime states using entanglement + classical bits
4. Entanglement Distillation: Purify noisy entangled states

Mathematical Foundation:
    Bell basis for primes p, q:
    |Φ+⟩ = (1/√2)(|p,p⟩ + |q,q⟩)
    |Φ-⟩ = (1/√2)(|p,p⟩ - |q,q⟩)  
    |Ψ+⟩ = (1/√2)(|p,q⟩ + |q,p⟩)
    |Ψ-⟩ = (1/√2)(|p,q⟩ - |q,p⟩)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import math
import random
from uuid import uuid4

from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.complex import Complex
from tinyaleph.core.primes import is_prime, nth_prime
from tinyaleph.core.constants import PHI


class BellState(Enum):
    """The four Bell states for prime pairs."""
    PHI_PLUS = "Φ+"   # (|pp⟩ + |qq⟩)/√2
    PHI_MINUS = "Φ-"  # (|pp⟩ - |qq⟩)/√2
    PSI_PLUS = "Ψ+"   # (|pq⟩ + |qp⟩)/√2
    PSI_MINUS = "Ψ-"  # (|pq⟩ - |qp⟩)/√2


@dataclass
class EntangledPair:
    """
    Represents an entangled pair of prime states.
    
    The pair is in a Bell state, meaning measurements on
    one particle instantly determine the state of the other.
    
    Attributes:
        id: Unique identifier for this entangled pair
        prime_a: First prime in the basis
        prime_b: Second prime in the basis
        bell_state: Which Bell state the pair is in
        fidelity: Quality of entanglement (1.0 = perfect)
        node_a: ID of node holding first particle
        node_b: ID of node holding second particle
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    prime_a: int = 2
    prime_b: int = 3
    bell_state: BellState = BellState.PHI_PLUS
    fidelity: float = 1.0
    node_a: Optional[str] = None
    node_b: Optional[str] = None
    
    def __post_init__(self):
        if not is_prime(self.prime_a):
            raise ValueError(f"{self.prime_a} is not prime")
        if not is_prime(self.prime_b):
            raise ValueError(f"{self.prime_b} is not prime")
        if self.prime_a == self.prime_b:
            raise ValueError("Entangled primes must be distinct")
    
    def measure_a(self) -> Tuple[int, int]:
        """
        Measure particle A in computational basis.
        
        Returns (outcome_a, predicted_b) based on Bell state.
        Collapses the entanglement.
        """
        # Equal probability for each outcome in computational basis
        if random.random() < 0.5:
            outcome_a = self.prime_a
            if self.bell_state in (BellState.PHI_PLUS, BellState.PHI_MINUS):
                predicted_b = self.prime_a
            else:
                predicted_b = self.prime_b
        else:
            outcome_a = self.prime_b
            if self.bell_state in (BellState.PHI_PLUS, BellState.PHI_MINUS):
                predicted_b = self.prime_b
            else:
                predicted_b = self.prime_a
        
        # Include sign from phase
        if self.bell_state in (BellState.PHI_MINUS, BellState.PSI_MINUS):
            # Phase information would be in quaternionic amplitude
            pass
        
        return outcome_a, predicted_b
    
    def measure_b(self) -> Tuple[int, int]:
        """
        Measure particle B in computational basis.
        
        Returns (outcome_b, predicted_a) based on Bell state.
        """
        outcome_b, predicted_a = self.measure_a()
        return outcome_b, predicted_a
    
    def apply_local_rotation_a(self, angle: float) -> EntangledPair:
        """Apply local phase rotation to particle A."""
        # Phase rotations can change between Bell states
        # This is a simplified model
        return EntangledPair(
            id=self.id,
            prime_a=self.prime_a,
            prime_b=self.prime_b,
            bell_state=self.bell_state,
            fidelity=self.fidelity * math.cos(angle / 2),
            node_a=self.node_a,
            node_b=self.node_b
        )
    
    def is_maximally_entangled(self, threshold: float = 0.99) -> bool:
        """Check if fidelity indicates maximal entanglement."""
        return self.fidelity >= threshold


@dataclass
class EntanglementSource:
    """
    Source that produces entangled prime pairs.
    
    Models a heralded entanglement source that can generate
    Bell pairs on demand.
    """
    
    base_fidelity: float = 0.95
    success_probability: float = 0.8
    default_primes: Tuple[int, int] = (2, 3)
    generated_count: int = 0
    
    def generate(
        self,
        prime_a: Optional[int] = None,
        prime_b: Optional[int] = None,
        bell_state: BellState = BellState.PHI_PLUS
    ) -> Optional[EntangledPair]:
        """
        Attempt to generate an entangled pair.
        
        Returns None if generation fails (probabilistic).
        """
        if random.random() > self.success_probability:
            return None
        
        pa = prime_a if prime_a else self.default_primes[0]
        pb = prime_b if prime_b else self.default_primes[1]
        
        # Add noise to fidelity
        noise = random.gauss(0, 0.02)
        fidelity = max(0.5, min(1.0, self.base_fidelity + noise))
        
        self.generated_count += 1
        
        return EntangledPair(
            prime_a=pa,
            prime_b=pb,
            bell_state=bell_state,
            fidelity=fidelity
        )
    
    def generate_n(
        self,
        n: int,
        prime_a: Optional[int] = None,
        prime_b: Optional[int] = None
    ) -> List[EntangledPair]:
        """Generate up to n entangled pairs."""
        pairs = []
        for _ in range(n):
            pair = self.generate(prime_a, prime_b)
            if pair:
                pairs.append(pair)
        return pairs


@dataclass
class EntanglementSwapper:
    """
    Implements entanglement swapping between nodes.
    
    If A-B are entangled and B-C are entangled,
    a Bell measurement on B can entangle A-C.
    """
    
    success_probability: float = 0.5
    fidelity_loss: float = 0.1
    
    def swap(
        self,
        pair_ab: EntangledPair,
        pair_bc: EntangledPair
    ) -> Optional[EntangledPair]:
        """
        Swap entanglement from A-B and B-C to create A-C.
        
        Requires that pair_ab and pair_bc share a common node (B).
        
        Returns new A-C pair or None if swap fails.
        """
        if random.random() > self.success_probability:
            return None
        
        # New fidelity is product of original fidelities minus loss
        new_fidelity = (
            pair_ab.fidelity * pair_bc.fidelity * (1 - self.fidelity_loss)
        )
        
        # Determine Bell state of result (simplified)
        # In reality, depends on Bell measurement outcome
        if pair_ab.bell_state == pair_bc.bell_state:
            new_bell = BellState.PHI_PLUS
        else:
            new_bell = BellState.PSI_PLUS
        
        return EntangledPair(
            prime_a=pair_ab.prime_a,
            prime_b=pair_bc.prime_b,
            bell_state=new_bell,
            fidelity=new_fidelity,
            node_a=pair_ab.node_a,
            node_b=pair_bc.node_b
        )


@dataclass
class Teleporter:
    """
    Quantum teleportation of prime states.
    
    Uses entanglement + classical communication to transfer
    a prime state from sender to receiver.
    """
    
    def teleport(
        self,
        state_amplitudes: Dict[int, Complex],
        entangled_pair: EntangledPair
    ) -> Tuple[Dict[int, Complex], Tuple[int, int]]:
        """
        Teleport a prime state using shared entanglement.
        
        Args:
            state_amplitudes: The state |ψ⟩ to teleport
            entangled_pair: Shared entanglement between sender/receiver
            
        Returns:
            (teleported_state, classical_bits)
            
        The classical bits are the Bell measurement outcome that
        must be sent to the receiver for proper reconstruction.
        """
        if not state_amplitudes:
            return {}, (0, 0)
        
        # Simplified teleportation:
        # 1. Bell measurement at sender
        # 2. Classical communication (the return value)
        # 3. Correction at receiver (applied to output)
        
        # Bell measurement outcome (random in this simulation)
        bell_outcome = (random.randint(0, 1), random.randint(0, 1))
        
        # Apply fidelity as noise
        teleported = {}
        fidelity = entangled_pair.fidelity
        
        for p, amp in state_amplitudes.items():
            # Apply noise based on fidelity
            noise_r = random.gauss(0, 0.1 * (1 - fidelity))
            noise_i = random.gauss(0, 0.1 * (1 - fidelity))
            
            new_amp = Complex(
                amp.real * fidelity + noise_r,
                amp.imag * fidelity + noise_i
            )
            
            # Apply correction based on Bell outcome
            if bell_outcome == (0, 1):
                # X correction (swap amplitudes within pairs)
                pass
            elif bell_outcome == (1, 0):
                # Z correction (phase flip)
                new_amp = Complex(new_amp.real, -new_amp.imag)
            elif bell_outcome == (1, 1):
                # XZ correction
                new_amp = Complex(new_amp.real, -new_amp.imag)
            
            teleported[p] = new_amp
        
        # Normalize
        total = sum(a.magnitude_squared() for a in teleported.values())
        if total > 1e-10:
            scale = 1.0 / math.sqrt(total)
            teleported = {p: a * Complex(scale, 0) for p, a in teleported.items()}
        
        return teleported, bell_outcome


@dataclass
class EntanglementDistiller:
    """
    Distillation of noisy entanglement into high-fidelity pairs.
    
    Uses multiple low-fidelity pairs to produce fewer high-fidelity pairs.
    Implements simplified DEJMPS protocol.
    """
    
    target_fidelity: float = 0.99
    
    def distill(
        self,
        pairs: List[EntangledPair]
    ) -> Optional[EntangledPair]:
        """
        Distill multiple pairs into one higher-fidelity pair.
        
        Requires at least 2 pairs. Success is probabilistic.
        
        Returns high-fidelity pair or None if distillation fails.
        """
        if len(pairs) < 2:
            return None
        
        # Sort by fidelity
        sorted_pairs = sorted(pairs, key=lambda p: p.fidelity, reverse=True)
        
        # Use two best pairs
        p1, p2 = sorted_pairs[0], sorted_pairs[1]
        
        # Distillation success probability depends on fidelities
        f1, f2 = p1.fidelity, p2.fidelity
        success_prob = f1 * f2 + (1 - f1) * (1 - f2)
        
        if random.random() > success_prob:
            return None
        
        # New fidelity (simplified formula)
        new_fidelity = (f1 * f2) / (f1 * f2 + (1 - f1) * (1 - f2))
        new_fidelity = min(1.0, new_fidelity)
        
        return EntangledPair(
            prime_a=p1.prime_a,
            prime_b=p1.prime_b,
            bell_state=p1.bell_state,
            fidelity=new_fidelity,
            node_a=p1.node_a,
            node_b=p1.node_b
        )
    
    def distill_to_target(
        self,
        pairs: List[EntangledPair],
        max_rounds: int = 10
    ) -> Optional[EntangledPair]:
        """
        Iteratively distill until target fidelity reached.
        """
        current_pairs = list(pairs)
        
        for _ in range(max_rounds):
            if not current_pairs:
                return None
            
            # Check if any pair meets target
            for pair in current_pairs:
                if pair.fidelity >= self.target_fidelity:
                    return pair
            
            # Need at least 2 pairs to continue
            if len(current_pairs) < 2:
                return current_pairs[0] if current_pairs else None
            
            # Distill pairs pairwise
            new_pairs = []
            for i in range(0, len(current_pairs) - 1, 2):
                result = self.distill([current_pairs[i], current_pairs[i + 1]])
                if result:
                    new_pairs.append(result)
            
            current_pairs = new_pairs
        
        return current_pairs[0] if current_pairs else None


@dataclass
class EntanglementNetwork:
    """
    Network of entangled nodes.
    
    Maintains a graph of entanglement links between nodes.
    """
    
    nodes: Set[str] = field(default_factory=set)
    pairs: Dict[str, EntangledPair] = field(default_factory=dict)
    source: EntanglementSource = field(default_factory=EntanglementSource)
    swapper: EntanglementSwapper = field(default_factory=EntanglementSwapper)
    
    def add_node(self, node_id: str) -> None:
        """Add a node to the network."""
        self.nodes.add(node_id)
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its entanglement links."""
        self.nodes.discard(node_id)
        
        # Remove pairs involving this node
        to_remove = [
            pair_id for pair_id, pair in self.pairs.items()
            if pair.node_a == node_id or pair.node_b == node_id
        ]
        for pair_id in to_remove:
            del self.pairs[pair_id]
    
    def establish_link(
        self,
        node_a: str,
        node_b: str,
        prime_a: int = 2,
        prime_b: int = 3
    ) -> Optional[EntangledPair]:
        """
        Establish entanglement between two nodes.
        
        Returns the created pair or None if generation fails.
        """
        if node_a not in self.nodes or node_b not in self.nodes:
            raise ValueError("Both nodes must be in network")
        
        pair = self.source.generate(prime_a, prime_b)
        if pair:
            pair.node_a = node_a
            pair.node_b = node_b
            self.pairs[pair.id] = pair
        
        return pair
    
    def get_links(self, node_id: str) -> List[EntangledPair]:
        """Get all entanglement links for a node."""
        return [
            pair for pair in self.pairs.values()
            if pair.node_a == node_id or pair.node_b == node_id
        ]
    
    def are_entangled(self, node_a: str, node_b: str) -> bool:
        """Check if two nodes share entanglement."""
        for pair in self.pairs.values():
            if ((pair.node_a == node_a and pair.node_b == node_b) or
                (pair.node_a == node_b and pair.node_b == node_a)):
                return True
        return False
    
    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find a path of entanglement links from source to target.
        
        Uses BFS to find shortest path.
        """
        if source == target:
            return [source]
        
        if source not in self.nodes or target not in self.nodes:
            return None
        
        # Build adjacency from pairs
        adjacency: Dict[str, List[str]] = {node: [] for node in self.nodes}
        for pair in self.pairs.values():
            if pair.node_a and pair.node_b:
                adjacency[pair.node_a].append(pair.node_b)
                adjacency[pair.node_b].append(pair.node_a)
        
        # BFS
        visited = {source}
        queue = [(source, [source])]
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in adjacency[current]:
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def establish_long_distance(
        self,
        source: str,
        target: str
    ) -> Optional[EntangledPair]:
        """
        Establish entanglement between non-adjacent nodes via swapping.
        
        Finds path and performs successive swaps.
        """
        path = self.find_path(source, target)
        
        if not path or len(path) < 2:
            return None
        
        if len(path) == 2:
            # Already adjacent, just get existing link
            for pair in self.pairs.values():
                if ((pair.node_a == source and pair.node_b == target) or
                    (pair.node_a == target and pair.node_b == source)):
                    return pair
            # No link exists, create one
            return self.establish_link(source, target)
        
        # Need to swap through intermediate nodes
        # Get pairs along path
        path_pairs = []
        for i in range(len(path) - 1):
            for pair in self.pairs.values():
                if ((pair.node_a == path[i] and pair.node_b == path[i + 1]) or
                    (pair.node_a == path[i + 1] and pair.node_b == path[i])):
                    path_pairs.append(pair)
                    break
        
        if len(path_pairs) < len(path) - 1:
            return None  # Missing links
        
        # Successive swapping
        current_pair = path_pairs[0]
        for i in range(1, len(path_pairs)):
            current_pair = self.swapper.swap(current_pair, path_pairs[i])
            if not current_pair:
                return None
        
        current_pair.node_a = source
        current_pair.node_b = target
        self.pairs[current_pair.id] = current_pair
        
        return current_pair
    
    def total_entanglement(self) -> float:
        """Compute total entanglement in network (sum of fidelities)."""
        return sum(pair.fidelity for pair in self.pairs.values())
    
    def average_fidelity(self) -> float:
        """Compute average fidelity of all pairs."""
        if not self.pairs:
            return 0.0
        return self.total_entanglement() / len(self.pairs)
    
    def prune_low_fidelity(self, threshold: float = 0.5) -> int:
        """Remove pairs below fidelity threshold. Returns count removed."""
        to_remove = [
            pair_id for pair_id, pair in self.pairs.items()
            if pair.fidelity < threshold
        ]
        for pair_id in to_remove:
            del self.pairs[pair_id]
        return len(to_remove)


def create_ghz_state(primes: List[int]) -> Dict[Tuple[int, ...], Complex]:
    """
    Create GHZ (Greenberger-Horne-Zeilinger) state for multiple primes.
    
    |GHZ⟩ = (1/√2)(|p₁p₁...p₁⟩ + |p₂p₂...p₂⟩)
    
    where p₁ and p₂ are the first two distinct primes.
    """
    if len(primes) < 2:
        raise ValueError("Need at least 2 primes for GHZ state")
    
    n = len(primes)
    p1, p2 = primes[0], primes[1]
    
    amplitude = Complex(1.0 / math.sqrt(2), 0)
    
    state = {
        tuple([p1] * n): amplitude,
        tuple([p2] * n): amplitude
    }
    
    return state


def create_w_state(primes: List[int]) -> Dict[Tuple[int, ...], Complex]:
    """
    Create W state for multiple primes.
    
    |W⟩ = (1/√n)(|p₂p₁p₁...⟩ + |p₁p₂p₁...⟩ + ... + |p₁p₁...p₂⟩)
    
    Each term has exactly one p₂ and rest p₁.
    """
    if len(primes) < 2:
        raise ValueError("Need at least 2 primes for W state")
    
    n = len(primes)
    p1, p2 = primes[0], primes[1]
    
    amplitude = Complex(1.0 / math.sqrt(n), 0)
    
    state = {}
    for i in range(n):
        # Create tuple with p2 at position i, p1 elsewhere
        config = [p1] * n
        config[i] = p2
        state[tuple(config)] = amplitude
    
    return state


def entanglement_entropy(
    bipartite_state: Dict[Tuple[int, int], Complex]
) -> float:
    """
    Compute entanglement entropy of a bipartite state.
    
    Uses von Neumann entropy of reduced density matrix.
    
    S = -Tr(ρ_A log₂ ρ_A)
    
    For pure bipartite states, this equals the entropy of
    the Schmidt coefficients.
    """
    if not bipartite_state:
        return 0.0
    
    # Compute reduced density matrix for first subsystem
    # ρ_A = Tr_B(|ψ⟩⟨ψ|)
    
    # Get marginal probabilities for first prime
    marginal: Dict[int, float] = {}
    for (p1, p2), amp in bipartite_state.items():
        prob = amp.magnitude_squared()
        marginal[p1] = marginal.get(p1, 0.0) + prob
    
    # Normalize
    total = sum(marginal.values())
    if total < 1e-10:
        return 0.0
    
    marginal = {p: v / total for p, v in marginal.items()}
    
    # Compute entropy
    entropy = 0.0
    for prob in marginal.values():
        if prob > 1e-10:
            entropy -= prob * math.log2(prob)
    
    return entropy