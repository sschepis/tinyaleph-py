# Semantic Premodel Reference Guide

The Semantic Premodel application builds a deterministic semantic landscape where concepts are mapped to prime numbers. This landscape serves as the semantic substrate for the PRSC (Prime Resonance Semantic Coherence) layer in Reso-LLM and LLM Fusion.

## Core Concepts

### Prime Basis
Every concept is assigned a unique prime number or a composite of primes. This leverages the Fundamental Theorem of Arithmetic (unique factorization) to ensure that combined concepts have unique representations.

### Triadic Fusion
The core mechanism for deriving new meanings is **Triadic Fusion**:
$$p + q + r = T$$
Where:
*   $p, q, r$ are prime numbers representing constituent concepts.
*   $T$ is the target number (which may be prime or composite).

If $p, q, r$ satisfy specific resonance criteria (defined in `tinyaleph.semantic.reduction`), they "fuse" to form the emergent meaning of $T$.

### Mirror Operator
*   **Prime 2** is reserved as the "Mirror" or "Dual" operator.
*   It functions as a phase inverter or negation operator in the semantic space.
*   It is excluded from standard fusion triads to prevent trivial symmetries.

---

## Workflow

The process consists of two phases: **Build** and **Refine**.

### 1. Build Phase (Deterministic)
Seeds the landscape from a base lexicon (PRQS Lexicon) and computes all valid fusion routes.

**Command:**
```bash
python -m apps.semantic_premodel.cli build \
    --output runs/landscape.json \
    --num-primes 200 \
    --max-routes 5
```

**Process:**
1.  Load seeded nouns/adjectives from `PRQS_LEXICON`.
2.  For each non-seeded prime $T$:
    *   Find all valid triads $(p, q, r)$ such that $p+q+r=T$.
    *   Score triads based on resonance.
    *   Select the canonical (highest scoring) triad.
    *   Derive emergent meaning from the interpreter.
3.  Compute entropy based on the number of competing routes (more routes = higher ambiguity).

### 2. Refine Phase (Optional)
Allows manual or automated refinement of meanings without altering the underlying mathematical structure.

**Command:**
```bash
python -m apps.semantic_premodel.cli refine \
    --input runs/landscape.json \
    --output runs/landscape.refined.json \
    --map runs/refine_map.json
```

**Refinement Map Format:**
```json
{
  "127": "sacred_bounded_context",
  "131": "reflective_growth"
}
```

---

## Data Structures

### SemanticLandscape
The top-level container.
*   `metadata`: Configuration and creation info.
*   `entries`: Dictionary mapping Prime $\to$ `PrimeEntry`.
*   `stats`: Global statistics (entropy, coverage).

### PrimeEntry
Represents a single concept.
*   `prime`: The integer value.
*   `meaning`: String representation (e.g., "fire", "structure").
*   `origin`: "seeded", "fusion", or "fallback".
*   `routes`: List of `FusionRoute`s that lead to this prime.
*   `coherence`: $1.0 - \text{Entropy}$. High coherence means the concept is unambiguous.

### FusionRoute
A specific path to creating a concept.
*   `p, q, r`: The constituent primes.
*   `score`: Resonance score.
*   `canonical`: Boolean, true if this is the primary definition.

---

## Integration

The generated JSON landscape is used by `ResoLLM` and `LLM Fusion` to initialize their PRSC layers.

```python
# In ResoLLMConfig
config.prsc.enabled = True
config.prsc.landscape_path = "path/to/landscape.json"
```

When loaded, the model binds the semantic vectors of the primes to the corresponding tokens or concepts, creating a "semantic prior" that guides the model's reasoning.
