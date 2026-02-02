# Semantic Premodel

Build a deterministic semantic landscape from the PRQS lexicon and triadic fusion routes, with an optional refinement phase.

## Goals
- Phase 1 (deterministic): seed from PRQS, compute triadic fusion routes, canonicalize, and derive a stable landscape.
- Phase 2 (optional): refine meanings without changing the base mapping.
- Reserve prime 2 as a mirror/dual operator (phase inversion), not a fusion input.

## Usage

### Build
```bash
python -m apps.semantic_premodel.cli build --output runs/landscape.json --num-primes 200 --max-routes 5
```
Use `--no-adjectives` to omit adjective metadata.

### Refine
```bash
python -m apps.semantic_premodel.cli refine --input runs/landscape.json --output runs/landscape.refined.json --map runs/refine_map.json
```

`refine_map.json` format:
```json
{
  "127": "sacred_bounded_context",
  "131": "reflective_growth"
}
```

## Mirror/Dual Operator
The mirror operator applies a phase inversion to a prime state.

```python
from apps.semantic_premodel.mirror import mirror_state
from tinyaleph.hilbert.state import PrimeState

state = PrimeState.basis(7)
mirrored = mirror_state(state)
```

## Integration Note
This app is designed to feed a precomputed landscape into `apps/reso_llm` as a prior semantic substrate.
