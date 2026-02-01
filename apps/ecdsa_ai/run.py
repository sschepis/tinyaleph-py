"""CLI entrypoint for key-derivation exploration."""
from __future__ import annotations

import argparse
import time

from apps.ecdsa_ai.agent import KeyDerivationExplorer
from apps.ecdsa_ai.datasets import ensure_dataset
from apps.ecdsa_ai.logging import JsonlLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Key derivation exploration")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite", type=float, default=0.25)
    parser.add_argument("--mut-temp", type=float, default=0.25)
    parser.add_argument("--memory", type=int, default=200)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--hnp-signatures", type=int, default=20)
    parser.add_argument("--hnp-bits", type=int, default=53)
    parser.add_argument("--hnp-mode", type=str, choices=["low", "full"], default="low")
    parser.add_argument("--ecdsa-weight", type=float, default=1.0)
    parser.add_argument("--hnp-weight", type=float, default=1.0)
    parser.add_argument("--residue-primes", type=int, default=32)
    parser.add_argument("--log", type=str, default=None)

    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    dataset = ensure_dataset(
        path=args.dataset,
        seed=seed,
        ecdsa_samples=args.samples,
        hnp_signatures=args.hnp_signatures,
        hnp_bits=args.hnp_bits,
        hnp_mode=args.hnp_mode,
    )
    logger = JsonlLogger(args.log) if args.log else None

    explorer = KeyDerivationExplorer(
        population_size=args.population,
        elite_ratio=args.elite,
        mutation_temp=args.mut_temp,
        memory_size=args.memory,
        sample_size=args.samples,
        dataset=dataset,
        ecdsa_weight=args.ecdsa_weight,
        hnp_weight=args.hnp_weight,
        residue_primes=args.residue_primes,
        logger=logger,
    )

    best = explorer.run(generations=args.generations)
    last_result = best.last_result

    print("Best encoding signature:", best.encoding.signature())
    print("Best score:", f"{best.score:.6f}")
    print("HNP mode:", dataset.hnp_mode)
    print("HNP bits:", dataset.hnp_bits)
    if last_result:
        print("Last estimate:", last_result.estimate)
        if last_result.error is not None:
            print("Last error:", f"{last_result.error:.6f}")
        print("Metrics:", last_result.metrics)
    if best.hnp_metrics is not None:
        print("HNP match ratio:", f"{best.hnp_metrics['match_ratio']:.6f}")
        print("HNP low-bit error:", f"{best.hnp_metrics['low_error']:.6f}")


if __name__ == "__main__":
    main()
