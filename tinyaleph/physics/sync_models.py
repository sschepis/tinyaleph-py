"""
Extended Synchronization Models

Advanced Kuramoto variants with network topology and plasticity:
- NetworkKuramoto: General adjacency matrix coupling
- AdaptiveKuramoto: Hebbian plasticity (concepts that sync link together)
- SakaguchiKuramoto: Phase frustration for chimera states
- SmallWorldKuramoto: Watts-Strogatz small-world networks
- MultiSystemCoupling: Cross-system synchronization
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import random

import numpy as np
from numpy.typing import NDArray
from tinyaleph.physics.kuramoto import KuramotoModel


# =============================================================================
# NETWORK KURAMOTO
# =============================================================================

class NetworkKuramoto(KuramotoModel):
    """
    NetworkKuramoto - Topology-aware Kuramoto model
    
    Uses adjacency matrix A for network structure:
        dθᵢ/dt = ωᵢ + K Σⱼ Aᵢⱼ sin(θⱼ - θᵢ)
    
    Instead of all-to-all mean-field coupling.
    """
    
    def __init__(
        self,
        frequencies: List[float],
        adjacency: Optional[List[List[float]]] = None,
        coupling: float = 0.3
    ):
        """
        Initialize NetworkKuramoto model.
        
        Args:
            frequencies: Natural frequencies ωᵢ
            adjacency: Adjacency matrix A[i][j] (None = all-to-all)
            coupling: Base coupling K
        """
        n = len(frequencies)
        freq_array = np.array(frequencies)
        
        # Initialize parent with explicit parameters
        object.__setattr__(self, 'n_oscillators', n)
        object.__setattr__(self, 'coupling', coupling)
        object.__setattr__(self, 'frequencies', freq_array)
        object.__setattr__(self, 'phases', np.random.uniform(0, 2 * np.pi, n))
        
        # Build adjacency matrix
        if adjacency is None:
            # Default to all-to-all
            self.adjacency = np.ones((n, n)) - np.eye(n)
        else:
            self.adjacency = np.array(adjacency)
        
        # Compute degree (sum of edge weights) for normalization
        self.degree = np.sum(self.adjacency, axis=1)
    
    def step(self, dt: float = 0.01) -> None:
        """
        Advance system by one time step with network topology.
        
        Uses adjacency-weighted coupling instead of mean-field.
        """
        N = self.n_oscillators
        
        # Compute phase differences
        phase_diff = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        
        # Weighted coupling
        coupling_matrix = self.adjacency * np.sin(phase_diff)
        coupling_term = np.sum(coupling_matrix, axis=1)
        
        # Normalize by degree
        with np.errstate(divide='ignore', invalid='ignore'):
            coupling_term = np.where(self.degree > 0, coupling_term / self.degree, 0)
        
        # Euler step
        self.phases += (self.frequencies + self.coupling * coupling_term) * dt
        self.phases = self.phases % (2 * np.pi)
    
    def tick(self, dt: float = 0.01) -> None:
        """Alias for step()."""
        self.step(dt)
    
    def get_topology_properties(self) -> Dict:
        """Get network topology statistics."""
        n_edges = int(np.sum(self.adjacency > 0) / 2)
        avg_degree = float(np.mean(self.degree))
        max_degree = float(np.max(self.degree))
        
        return {
            "n_nodes": self.n_oscillators,
            "n_edges": n_edges,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "density": 2 * n_edges / (self.n_oscillators * (self.n_oscillators - 1))
            if self.n_oscillators > 1 else 0
        }
    
    def get_network_state(self) -> Dict:
        """Get current network state."""
        return {
            "phases": list(self.phases),
            "frequencies": list(self.frequencies),
            "order_parameter": float(abs(self.order_parameter())),
            "mean_phase": float(self.mean_phase())
        }


# =============================================================================
# ADAPTIVE KURAMOTO
# =============================================================================

class AdaptiveKuramoto(NetworkKuramoto):
    """
    AdaptiveKuramoto - Hebbian Plasticity
    
    Coupling strengths evolve based on synchronization:
        dθᵢ/dt = ωᵢ + (1/N) Σⱼ Kᵢⱼ sin(θⱼ - θᵢ)
        dKᵢⱼ/dt = ε(cos(θⱼ - θᵢ) - Kᵢⱼ)
    
    "Concepts that sync together link together"
    """
    
    def __init__(
        self,
        frequencies: List[float],
        coupling: float = 0.3,
        plasticity_rate: float = 0.01
    ):
        """
        Initialize AdaptiveKuramoto model.
        
        Args:
            frequencies: Natural frequencies
            coupling: Initial coupling strength
            plasticity_rate: Hebbian learning rate ε
        """
        super().__init__(frequencies, adjacency=None, coupling=1.0)
        
        self.epsilon = plasticity_rate
        N = self.n_oscillators
        
        # Adaptive coupling matrix K_ij (starts uniform)
        self.coupling_matrix = np.ones((N, N)) * coupling
        np.fill_diagonal(self.coupling_matrix, 0.0)
        
        # Update adjacency to use coupling matrix
        self.adjacency = self.coupling_matrix.copy()
        self.degree = np.sum(self.adjacency, axis=1)
    
    def step(self, dt: float = 0.01) -> None:
        """
        Advance with adaptive coupling.
        
        Includes Hebbian plasticity: couplings strengthen when oscillators sync.
        """
        N = self.n_oscillators
        
        # Compute phase differences
        phase_diff = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        
        # Weighted coupling using current coupling matrix
        coupling_matrix = self.coupling_matrix * np.sin(phase_diff)
        coupling_term = np.mean(coupling_matrix, axis=1)
        
        # Euler step for phases
        self.phases += (self.frequencies + coupling_term) * dt
        self.phases = self.phases % (2 * np.pi)
        
        # Hebbian plasticity update
        # dK_ij/dt = ε(cos(θⱼ - θᵢ) - K_ij)
        coherence = np.cos(phase_diff)
        dK = self.epsilon * (coherence - self.coupling_matrix) * dt
        self.coupling_matrix += dK
        self.coupling_matrix = np.clip(self.coupling_matrix, 0.0, 2.0)
        np.fill_diagonal(self.coupling_matrix, 0.0)
        
        # Update adjacency/degree
        self.adjacency = self.coupling_matrix.copy()
        self.degree = np.sum(self.adjacency, axis=1)
    
    def mean_coupling(self) -> float:
        """Get average coupling strength."""
        N = self.n_oscillators
        mask = ~np.eye(N, dtype=bool)
        return float(np.mean(self.coupling_matrix[mask]))
    
    def coupling_variance(self) -> float:
        """Get variance of coupling strengths."""
        N = self.n_oscillators
        mask = ~np.eye(N, dtype=bool)
        return float(np.var(self.coupling_matrix[mask]))
    
    def get_coupling_matrix(self) -> np.ndarray:
        """Get current coupling matrix."""
        return self.coupling_matrix.copy()
    
    def get_strong_links(self, threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """Get links with coupling above threshold."""
        N = self.n_oscillators
        links = []
        for i in range(N):
            for j in range(N):
                if i != j and self.coupling_matrix[i, j] > threshold:
                    links.append((i, j, float(self.coupling_matrix[i, j])))
        return sorted(links, key=lambda x: x[2], reverse=True)


# =============================================================================
# SAKAGUCHI KURAMOTO
# =============================================================================

class SakaguchiKuramoto(KuramotoModel):
    """
    SakaguchiKuramoto - Phase Frustration Model
    
    Adds phase lag α to enable chimera states:
        dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ - α)
    
    Non-zero α breaks symmetry and allows partial synchronization patterns.
    """
    
    def __init__(
        self,
        frequencies: List[float],
        coupling: float = 0.3,
        alpha: float = 0.0
    ):
        """
        Initialize SakaguchiKuramoto model.
        
        Args:
            frequencies: Natural frequencies
            coupling: Coupling strength K
            alpha: Phase lag parameter (0 = standard Kuramoto)
        """
        n = len(frequencies)
        freq_array = np.array(frequencies)
        
        object.__setattr__(self, 'n_oscillators', n)
        object.__setattr__(self, 'coupling', coupling)
        object.__setattr__(self, 'frequencies', freq_array)
        object.__setattr__(self, 'phases', np.random.uniform(0, 2 * np.pi, n))
        
        self.alpha = alpha
    
    def set_alpha(self, alpha: float) -> None:
        """Set phase lag parameter."""
        self.alpha = alpha
    
    def step(self, dt: float = 0.01) -> None:
        """Advance with phase-frustrated coupling."""
        # Compute pairwise phase differences with frustration
        phase_diff = self.phases[:, np.newaxis] - self.phases[np.newaxis, :] - self.alpha
        
        # Mean-field coupling with phase lag
        coupling_term = np.mean(np.sin(phase_diff), axis=1)
        
        # Euler step
        self.phases += (self.frequencies + self.coupling * coupling_term) * dt
        self.phases = self.phases % (2 * np.pi)
    
    def tick(self, dt: float = 0.01) -> None:
        """Alias for step()."""
        self.step(dt)
    
    def detect_chimera(self, threshold: float = 0.3) -> Dict:
        """
        Detect chimera states (coexistence of coherent/incoherent).
        
        Chimera = some oscillators synchronized, others not.
        """
        # Local order parameter for each oscillator
        local_order = np.zeros(self.n_oscillators)
        
        for i in range(self.n_oscillators):
            # Compute order with neighbors
            neighbors = np.exp(1j * self.phases)
            local_order[i] = abs(np.mean(neighbors))
        
        # Classify as coherent or incoherent
        coherent_mask = local_order > threshold
        n_coherent = np.sum(coherent_mask)
        n_incoherent = self.n_oscillators - n_coherent
        
        is_chimera = (n_coherent > 0.2 * self.n_oscillators and 
                     n_incoherent > 0.2 * self.n_oscillators)
        
        return {
            "is_chimera": is_chimera,
            "n_coherent": int(n_coherent),
            "n_incoherent": int(n_incoherent),
            "local_order": local_order.tolist(),
            "global_order": float(abs(self.order_parameter()))
        }


# =============================================================================
# SMALL WORLD KURAMOTO
# =============================================================================

class SmallWorldKuramoto(NetworkKuramoto):
    """
    SmallWorldKuramoto - Watts-Strogatz Topology
    
    Small-world networks have:
    - High clustering (like regular lattices)
    - Short path lengths (like random graphs)
    
    This enables efficient information propagation with local coherence.
    """
    
    def __init__(
        self,
        frequencies: List[float],
        coupling: float = 0.3,
        k_neighbors: int = 4,
        rewiring_prob: float = 0.1
    ):
        """
        Initialize SmallWorldKuramoto model.
        
        Args:
            frequencies: Natural frequencies
            coupling: Coupling strength
            k_neighbors: Each node connects to k nearest neighbors
            rewiring_prob: Probability of rewiring each edge (0=ring, 1=random)
        """
        n = len(frequencies)
        self.k_neighbors = k_neighbors
        self.rewiring_prob = rewiring_prob
        
        # Generate Watts-Strogatz small-world adjacency
        adjacency = self._generate_small_world(n, k_neighbors, rewiring_prob)
        
        super().__init__(frequencies, adjacency.tolist(), coupling)
    
    def _generate_small_world(self, n: int, k: int, p: float) -> np.ndarray:
        """
        Generate Watts-Strogatz small-world network.
        
        Args:
            n: Number of nodes
            k: Number of neighbors on each side
            p: Rewiring probability
            
        Returns:
            Adjacency matrix
        """
        # Start with ring lattice
        adjacency = np.zeros((n, n))
        
        for i in range(n):
            for j in range(1, k // 2 + 1):
                # Connect to k/2 neighbors on each side
                right = (i + j) % n
                left = (i - j) % n
                adjacency[i, right] = 1
                adjacency[right, i] = 1
                adjacency[i, left] = 1
                adjacency[left, i] = 1
        
        # Rewiring
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if random.random() < p:
                    right = (i + j) % n
                    
                    if adjacency[i, right] > 0:
                        # Find new target
                        candidates = [x for x in range(n) 
                                     if x != i and adjacency[i, x] == 0]
                        if candidates:
                            new_target = random.choice(candidates)
                            adjacency[i, right] = 0
                            adjacency[right, i] = 0
                            adjacency[i, new_target] = 1
                            adjacency[new_target, i] = 1
        
        return adjacency
    
    def clustering_coefficient(self) -> float:
        """
        Compute average clustering coefficient.
        
        High clustering = nodes' neighbors are likely connected.
        """
        n = self.n_oscillators
        cc_sum = 0.0
        
        for i in range(n):
            neighbors = np.where(self.adjacency[i] > 0)[0]
            k_i = len(neighbors)
            
            if k_i < 2:
                continue
            
            # Count edges between neighbors
            edges = 0
            for ni in neighbors:
                for nj in neighbors:
                    if ni < nj and self.adjacency[ni, nj] > 0:
                        edges += 1
            
            # Maximum possible edges
            max_edges = k_i * (k_i - 1) / 2
            cc_sum += edges / max_edges if max_edges > 0 else 0
        
        return cc_sum / n if n > 0 else 0.0
    
    def is_small_world(self, random_samples: int = 10) -> Dict:
        """
        Check if network exhibits small-world properties.
        
        Small-world: high clustering + short paths
        """
        C = self.clustering_coefficient()
        
        # Compare to random graph with same density
        n = self.n_oscillators
        density = np.sum(self.adjacency > 0) / (n * (n - 1))
        C_random = density  # Expected clustering for random graph
        
        # Estimate path length (simplified)
        L = self._estimate_path_length()
        L_random = np.log(n) / np.log(np.mean(self.degree)) if np.mean(self.degree) > 1 else n
        
        # Small-world coefficient
        sigma = (C / C_random) / (L / L_random) if C_random > 0 and L_random > 0 else 0
        
        return {
            "clustering": C,
            "clustering_random": C_random,
            "path_length": L,
            "path_length_random": L_random,
            "small_world_coefficient": sigma,
            "is_small_world": sigma > 1
        }
    
    def _estimate_path_length(self) -> float:
        """Estimate average shortest path length via BFS sampling."""
        n = self.n_oscillators
        if n < 2:
            return 0.0
        
        samples = min(10, n)
        total_length = 0
        count = 0
        
        for start in random.sample(range(n), samples):
            distances = self._bfs_distances(start)
            for d in distances:
                if d > 0 and d < n:
                    total_length += d
                    count += 1
        
        return total_length / count if count > 0 else n
    
    def _bfs_distances(self, start: int) -> List[int]:
        """BFS to compute distances from start node."""
        n = self.n_oscillators
        distances = [-1] * n
        distances[start] = 0
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for neighbor in np.where(self.adjacency[current] > 0)[0]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        return distances


# =============================================================================
# MULTI-SYSTEM COUPLING
# =============================================================================

class MultiSystemCoupling:
    """
    MultiSystemCoupling - Cross-system synchronization
    
    Couples multiple Kuramoto systems together:
        dθᵢ^(a)/dt = ωᵢ^(a) + internal_coupling + Σ_b G_ab external_coupling
    
    Used for multi-agent alignment and hierarchical synchronization.
    """
    
    def __init__(self, systems: List[KuramotoModel]):
        """
        Initialize multi-system coupling.
        
        Args:
            systems: List of Kuramoto systems to couple
        """
        self.systems = systems
        n_systems = len(systems)
        
        # Inter-system coupling matrix
        self.coupling_matrix = np.zeros((n_systems, n_systems))
    
    def set_coupling(self, i: int, j: int, strength: float) -> None:
        """Set coupling strength between systems i and j."""
        self.coupling_matrix[i, j] = strength
        self.coupling_matrix[j, i] = strength
    
    def step(self, dt: float = 0.01) -> None:
        """
        Advance all systems with inter-system coupling.
        
        Each system feels the mean-field of other systems.
        """
        n_systems = len(self.systems)
        
        # Compute order parameters for each system
        order_params = [complex(sys.order_parameter()) for sys in self.systems]
        
        # Advance each system
        for a, sys in enumerate(self.systems):
            # Internal evolution
            sys.step(dt)
            
            # External coupling from other systems
            external = 0j
            for b in range(n_systems):
                if a != b and self.coupling_matrix[a, b] > 0:
                    external += self.coupling_matrix[a, b] * order_params[b]
            
            # Apply external field to phases
            if abs(external) > 0:
                ext_phase = np.angle(external)
                ext_strength = abs(external)
                sys.phases += ext_strength * np.sin(ext_phase - sys.phases) * dt
                sys.phases = sys.phases % (2 * np.pi)
    
    def tick(self, dt: float = 0.01) -> None:
        """Alias for step()."""
        self.step(dt)
    
    def global_order_parameter(self) -> complex:
        """Compute order parameter across all systems."""
        all_phases = np.concatenate([sys.phases for sys in self.systems])
        return complex(np.mean(np.exp(1j * all_phases)))
    
    def system_orders(self) -> List[float]:
        """Get order parameters for each system."""
        return [float(abs(sys.order_parameter())) for sys in self.systems]
    
    def get_state(self) -> Dict:
        """Get multi-system state."""
        return {
            "n_systems": len(self.systems),
            "system_orders": self.system_orders(),
            "global_order": float(abs(self.global_order_parameter())),
            "coupling_matrix": self.coupling_matrix.tolist()
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_hierarchical_coupling(
    n_layers: int,
    oscillators_per_layer: int,
    intra_coupling: float = 1.0,
    inter_coupling: float = 0.3
) -> MultiSystemCoupling:
    """
    Create hierarchical multi-layer Kuramoto system.
    
    Args:
        n_layers: Number of hierarchy layers
        oscillators_per_layer: Oscillators per layer
        intra_coupling: Coupling within each layer
        inter_coupling: Coupling between adjacent layers
        
    Returns:
        Configured MultiSystemCoupling
    """
    systems = []
    for layer in range(n_layers):
        freq_center = layer * 0.5  # Higher layers = higher frequencies
        frequencies = list(np.random.normal(freq_center, 0.1, oscillators_per_layer))
        sys = KuramotoModel(
            n_oscillators=oscillators_per_layer,
            coupling=intra_coupling,
            frequencies=np.array(frequencies)
        )
        systems.append(sys)
    
    multi = MultiSystemCoupling(systems)
    
    # Connect adjacent layers
    for i in range(n_layers - 1):
        multi.set_coupling(i, i + 1, inter_coupling)
    
    return multi


def create_peer_coupling(
    n_peers: int,
    oscillators_per_peer: int,
    peer_coupling: float = 0.1
) -> MultiSystemCoupling:
    """
    Create peer-to-peer Kuramoto coupling.
    
    All systems equally coupled.
    
    Args:
        n_peers: Number of peer systems
        oscillators_per_peer: Oscillators per peer
        peer_coupling: Coupling between peers
        
    Returns:
        Configured MultiSystemCoupling
    """
    systems = []
    for peer in range(n_peers):
        frequencies = list(np.random.normal(0, 1, oscillators_per_peer))
        sys = KuramotoModel(
            n_oscillators=oscillators_per_peer,
            coupling=1.0,
            frequencies=np.array(frequencies)
        )
        systems.append(sys)
    
    multi = MultiSystemCoupling(systems)
    
    # All-to-all peer coupling
    for i in range(n_peers):
        for j in range(i + 1, n_peers):
            multi.set_coupling(i, j, peer_coupling)
    
    return multi


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NetworkKuramoto",
    "AdaptiveKuramoto",
    "SakaguchiKuramoto",
    "SmallWorldKuramoto",
    "MultiSystemCoupling",
    "create_hierarchical_coupling",
    "create_peer_coupling",
]