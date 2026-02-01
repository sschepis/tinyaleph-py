"""Automated sweep runner for key-derivation exploration."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

from apps.ecdsa_ai.agent import KeyDerivationExplorer
from apps.ecdsa_ai.datasets import ensure_dataset
from apps.ecdsa_ai.logging import JsonlLogger


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _select_bits(mode: str, bits_low: List[int], bits_full: List[int]) -> List[int]:
    return bits_low if mode == "low" else bits_full


@dataclass
class RunConfig:
    mode: str
    bits: int
    population: int
    generations: int
    hnp_weight: float
    ecdsa_weight: float
    seed: int


def run() -> None:
    parser = argparse.ArgumentParser(description="Automate ECDSA AI sweeps")
    parser.add_argument("--output-dir", default="examples/ecdsa/runs")
    parser.add_argument("--modes", default="low,full")
    parser.add_argument("--hnp-bits-low", default="16,24,32,40,53")
    parser.add_argument("--hnp-bits-full", default="256")
    parser.add_argument("--populations", default="32,64")
    parser.add_argument("--generations", default="10,25")
    parser.add_argument("--hnp-weights", default="1,3,5")
    parser.add_argument("--ecdsa-weight", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hnp-signatures", type=int, default=50)
    parser.add_argument("--residue-primes", type=int, default=32)
    parser.add_argument("--elite", type=float, default=0.25)
    parser.add_argument("--mut-temp", type=float, default=0.25)
    parser.add_argument("--seeds", default="1")

    args = parser.parse_args()

    modes = _parse_str_list(args.modes)
    bits_low = _parse_int_list(args.hnp_bits_low)
    bits_full = _parse_int_list(args.hnp_bits_full)
    populations = _parse_int_list(args.populations)
    generations = _parse_int_list(args.generations)
    hnp_weights = _parse_float_list(args.hnp_weights)
    seeds = _parse_int_list(args.seeds)

    run_root = os.path.join(args.output_dir, _timestamp_slug())
    datasets_dir = os.path.join(run_root, "datasets")
    logs_dir = os.path.join(run_root, "logs")
    runs_dir = os.path.join(run_root, "runs")
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    config_payload = {
        "modes": modes,
        "hnp_bits_low": bits_low,
        "hnp_bits_full": bits_full,
        "populations": populations,
        "generations": generations,
        "hnp_weights": hnp_weights,
        "ecdsa_weight": args.ecdsa_weight,
        "samples": args.samples,
        "hnp_signatures": args.hnp_signatures,
        "residue_primes": args.residue_primes,
        "elite": args.elite,
        "mut_temp": args.mut_temp,
        "seeds": seeds,
        "created_at": time.time(),
    }
    _write_json(os.path.join(run_root, "config.json"), config_payload)

    run_records: List[Dict[str, Any]] = []

    for mode in modes:
        bits_list = _select_bits(mode, bits_low, bits_full)
        for bits in bits_list:
            dataset_path = os.path.join(datasets_dir, f"dataset_{mode}_{bits}.json")
            dataset = ensure_dataset(
                path=dataset_path,
                seed=seeds[0],
                ecdsa_samples=args.samples,
                hnp_signatures=args.hnp_signatures,
                hnp_bits=bits,
                hnp_mode=mode,
            )

            for population in populations:
                for gen in generations:
                    for hnp_weight in hnp_weights:
                        for seed in seeds:
                            _seed_everything(seed)
                            run_cfg = RunConfig(
                                mode=mode,
                                bits=bits,
                                population=population,
                                generations=gen,
                                hnp_weight=hnp_weight,
                                ecdsa_weight=args.ecdsa_weight,
                                seed=seed,
                            )
                            log_name = (
                                f"run_{mode}_{bits}_pop{population}_gen{gen}_"
                                f"w{hnp_weight}_seed{seed}.jsonl"
                            )
                            log_path = os.path.join(logs_dir, log_name)
                            logger = JsonlLogger(log_path)

                            explorer = KeyDerivationExplorer(
                                population_size=population,
                                elite_ratio=args.elite,
                                mutation_temp=args.mut_temp,
                                memory_size=200,
                                sample_size=args.samples,
                                dataset=dataset,
                                ecdsa_weight=args.ecdsa_weight,
                                hnp_weight=hnp_weight,
                                residue_primes=args.residue_primes,
                                logger=logger,
                            )

                            start_time = time.time()
                            best = explorer.run(generations=gen)
                            elapsed = time.time() - start_time

                            record = {
                                "mode": mode,
                                "bits": bits,
                                "population": population,
                                "generations": gen,
                                "hnp_weight": hnp_weight,
                                "ecdsa_weight": args.ecdsa_weight,
                                "seed": seed,
                                "dataset": dataset_path,
                                "log": log_path,
                                "elapsed_sec": elapsed,
                                "best_score": best.score,
                                "best_signature": best.encoding.signature(),
                                "best_error": best.last_result.error if best.last_result else None,
                                "hnp_match_ratio": best.hnp_metrics["match_ratio"] if best.hnp_metrics else None,
                                "hnp_low_error": best.hnp_metrics["low_error"] if best.hnp_metrics else None,
                                "hnp_best_candidate": best.hnp_metrics.get("best_candidate") if best.hnp_metrics else None,
                            }

                            record_path = os.path.join(
                                runs_dir,
                                log_name.replace(".jsonl", ".json"),
                            )
                            _write_json(record_path, record)
                            run_records.append(record)

    run_records.sort(key=lambda r: r["best_score"], reverse=True)
    summary = {
        "run_root": run_root,
        "total_runs": len(run_records),
        "top_by_score": run_records[:10],
        "top_by_hnp_error": sorted(run_records, key=lambda r: (r["hnp_low_error"] is None, r["hnp_low_error"]))[:10],
    }
    _write_json(os.path.join(run_root, "summary.json"), summary)


if __name__ == "__main__":
    run()
