"""
CLI for building and refining semantic premodels.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict

from .builder import build_landscape
from .config import LandscapeConfig
from .io import save_landscape, load_landscape
from .refine import refine_with_mapping


def _load_mapping(path: str) -> Dict[int, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raw = {}
    mapping: Dict[int, str] = {}
    for key, value in raw.items():
        mapping[int(key)] = str(value)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(prog="semantic_premodel")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build a deterministic semantic landscape")
    build.add_argument("--output", required=True, help="Output JSON path")
    build.add_argument("--num-primes", type=int, default=200)
    build.add_argument("--max-routes", type=int, default=5)
    build.add_argument("--canonical-only", action="store_true")
    build.add_argument("--no-adjectives", action="store_false", dest="include_adjectives")
    build.add_argument("--min-prime", type=int, default=2)

    refine = sub.add_parser("refine", help="Refine a landscape with a mapping file")
    refine.add_argument("--input", required=True, help="Input JSON path")
    refine.add_argument("--output", required=True, help="Output JSON path")
    refine.add_argument("--map", required=True, help="JSON mapping path {prime: meaning}")

    args = parser.parse_args()

    if args.cmd == "build":
        config = LandscapeConfig(
            num_primes=args.num_primes,
            max_routes_per_prime=args.max_routes,
            canonical_only=args.canonical_only,
            include_adjectives=args.include_adjectives,
            min_prime=args.min_prime,
        )
        landscape = build_landscape(config)
        save_landscape(landscape, args.output)
        return

    if args.cmd == "refine":
        landscape = load_landscape(args.input)
        mapping = _load_mapping(args.map)
        refined = refine_with_mapping(landscape, mapping)
        save_landscape(refined, args.output)
        return


if __name__ == "__main__":
    main()
