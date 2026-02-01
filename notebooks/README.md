# TinyAleph Jupyter Notebooks

Interactive tutorials for exploring the TinyAleph library.

## Notebooks

| # | Notebook | Topics Covered |
|---|----------|----------------|
| 01 | [Getting Started](01_getting_started.ipynb) | Prime states, superposition, entropy, measurement |
| 02 | [Quaternions and Rotations](02_quaternions_and_rotations.ipynb) | Quaternion arithmetic, 3D rotations, SLERP, octonions |
| 03 | [Kuramoto Synchronization](03_kuramoto_synchronization.ipynb) | Coupled oscillators, order parameter, phase transitions |
| 04 | [Semantic Types](04_semantic_types.ipynb) | Type system, well-formedness, PRQS lexicon |
| 05 | [AlephEngine](05_aleph_engine.ipynb) | Runtime engine, hooks, checkpoints, entanglement |
| 06 | [ResoFormer ML](06_resoformer_ml.ipynb) | Sparse states, tensors, attention, transformer |

## Prerequisites

```bash
# Install TinyAleph
pip install -e ..

# Install Jupyter
pip install jupyter

# Optional: visualization
pip install matplotlib numpy
```

## Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

## Notebook Overview

### 01: Getting Started
Introduction to the Prime Hilbert Space:
- Creating prime eigenstates `|p⟩`
- Uniform superpositions
- Entropy and coherence measures
- Measurement and collapse
- Complex number operations

### 02: Quaternions and Rotations
Hypercomplex algebras for 3D geometry:
- Hamilton quaternions: `q = w + xi + yj + zk`
- Non-commutative multiplication
- Axis-angle rotations
- Rotation composition
- SLERP interpolation
- Octonions and sedenions

### 03: Kuramoto Synchronization
Coupled oscillator physics:
- Kuramoto model: `dθ/dt = ω + (K/N)Σsin(θⱼ - θᵢ)`
- Order parameter `r` for synchronization
- Phase transitions at critical coupling
- Stochastic dynamics with noise
- Lyapunov stability analysis

### 04: Semantic Types
Prime-indexed type system:
- NounTerm `N(p)` and AdjTerm `A(p)`
- ChainTerm well-formedness: adjectives < noun
- FusionTerm: `p + q + r` is prime
- Prime-preserving operators
- PRQS lexicon of semantic primitives

### 05: AlephEngine
Unified runtime environment:
- AlephConfig for customization
- ExecutionPhase state machine
- EngineHooks for callbacks
- Checkpoint and restore
- Resonant fragment storage
- Entanglement networks

### 06: ResoFormer ML
Machine learning architecture:
- SparsePrimeState (H_P ⊗ ℍ)
- Pure Python tensor operations
- Dense and QuaternionDense layers
- ResonantAttentionLayer
- CoherenceGatingLayer
- EntropyCollapseLayer
- Complete ResoFormer model

## Key Concepts

### Golden Ratio (φ ≈ 1.618)
The golden ratio appears throughout TinyAleph:
- Coherence threshold: `1/φ ≈ 0.618`
- Entropy threshold: `ln(φ) ≈ 0.481`
- Golden angle: `2π/φ² ≈ 137.5°`

### Prime Hilbert Space
```
H_P = {|ψ⟩ = Σ_p α_p|p⟩ : Σ|α_p|² = 1}
```
States as superpositions over prime basis vectors.

### Extended Space with Quaternions
```
H_Q = H_P ⊗ ℍ
```
Each prime has a 4D quaternionic amplitude.

## Tips

1. **Run cells in order** - notebooks build on previous cells
2. **Experiment** - modify parameters and observe results
3. **Visualizations** - matplotlib plots help understanding
4. **Check imports** - ensure `sys.path.insert(0, '..')` runs first

## Further Reading

- [User's Guide](../docs/users_guide.md) - comprehensive tutorial
- [Reference Guide](../docs/reference_guide.md) - API documentation
- [Examples](../examples/) - Python script examples