# AlephPrime User's Guide

Welcome to AlephPrime, a framework for **Prime-Resonant Quantum Computing**. This library unifies concepts from hypercomplex algebra, prime number theory, and quantum physics to create novel AI architectures.

## Overview

AlephPrime provides three main applications built on the core `tinyaleph` library:

1.  **[Reso-LLM](reso_llm_reference.md)**: A standalone Large Language Model designed from the ground up with resonance principles. It features self-directed agency, 16D holographic memory, and chaotic stability monitoring.
2.  **[LLM Fusion](llm_fusion_reference.md)**: A "grafting" kit that adds resonance layers to existing transformers (like GPT-2, Llama). This allows you to enhance standard models with prime semantics and stability controls.
3.  **[Semantic Premodel](semantic_premodel_reference.md)**: A tool for building deterministic semantic landscapes. It maps concepts to prime numbers using triadic fusion ($p+q+r=T$), creating a stable semantic foundation for the other models.

---

## Getting Started

### 1. Installation

```bash
pip install -e .
```

### 2. Choose Your Path

*   **If you want to train a new, physics-inspired model:**
    Go to **Reso-LLM**. This gives you the full experience of the Prime Hilbert Space architecture.
    
*   **If you have a pre-trained model (e.g., Llama 2) and want to enhance it:**
    Go to **LLM Fusion**. You can inject resonance layers to add stability monitoring and compositional semantics without retraining the base model.

*   **If you want to explore the mathematical semantics:**
    Start with **Semantic Premodel**. Build a landscape and explore how concepts fuse together mathematically.

---

## Core Concepts

### Prime Hilbert Space
Unlike standard vector spaces where dimensions are arbitrary, AlephPrime operates in a space where basis vectors are indexed by prime numbers.
*   **Orthogonality**: Primes are naturally orthogonal (unique factorization).
*   **Composition**: Multiplying primes combines concepts ($2 \cdot 3 = 6$).

### Resonance & Synchronization
The framework uses the **Kuramoto Model** to synchronize attention. Tokens are treated as oscillators; when they "resonate" (phase-lock), they form coherent thoughts.

### Stability & Chaos
We use **Lyapunov Exponents** to measure the stability of the model's generation.
*   $\lambda < 0$: Stable (Convergent)
*   $\lambda > 0$: Chaotic (Divergent/Hallucination)

---

## Documentation Index

*   [Reso-LLM Reference](reso_llm_reference.md)
*   [LLM Fusion Reference](llm_fusion_reference.md)
*   [Semantic Premodel Reference](semantic_premodel_reference.md)
*   [TinyAleph Core Reference](reference_guide.md) (Existing)
