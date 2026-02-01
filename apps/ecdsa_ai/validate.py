"""Validate top configs from a previous sweep on fresh seeds/datasets."""
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


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


@dataclass
class RunConfig:
    mode: str
    bits: int
    population: int
    generations: int
    hnp_weight: float
    ecdsa_weight: float
    residue_primes: int


def _load_run_records(run_root: str) -> List[Dict[str, Any]]:
    runs_dir = os.path.join(run_root, "runs")
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")

    records: List[Dict[str, Any]] = []
    for name in os.listdir(runs_dir):
        if not name.endswith(".json"):
            continue
        records.append(_load_json(os.path.join(runs_dir, name)))
    return records


def run() -> None:
    parser = argparse.ArgumentParser(description="Validate top sweep configs")
    parser.add_argument("run_root", help="Path to sweep run directory")
    parser.add_argument("--output-dir", default="examples/ecdsa/runs")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--metric", choices=["score", "hnp_error"], default="score")
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hnp-signatures", type=int, default=50)
    parser.add_argument("--hnp-bits", type=int, default=None)
    parser.add_argument("--hnp-mode", choices=["low", "full"], default=None)
    parser.add_argument("--residue-primes", type=int, default=32)
    parser.add_argument("--elite", type=float, default=0.25)
    parser.add_argument("--mut-temp", type=float, default=0.25)

    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    records = _load_run_records(args.run_root)
    if not records:
        raise RuntimeError("No run records found to validate")

    if args.metric == "hnp_error":
        records.sort(key=lambda r: (r["hnp_low_error"] is None, r["hnp_low_error"]))
    else:
        records.sort(key=lambda r: r["best_score"], reverse=True)

    selected = records[:args.top]

    run_root = os.path.join(args.output_dir, f"validation_{_timestamp_slug()}")
    datasets_dir = os.path.join(run_root, "datasets")
    logs_dir = os.path.join(run_root, "logs")
    runs_dir = os.path.join(run_root, "runs")
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    config_payload = {
        "source_run_root": args.run_root,
        "metric": args.metric,
        "top": args.top,
        "seeds": seeds,
        "samples": args.samples,
        "hnp_signatures": args.hnp_signatures,
        "hnp_bits_override": args.hnp_bits,
        "hnp_mode_override": args.hnp_mode,
        "residue_primes": args.residue_primes,
        "elite": args.elite,
        "mut_temp": args.mut_temp,
        "selected": selected,
        "created_at": time.time(),
    }
    _write_json(os.path.join(run_root, "config.json"), config_payload)

    run_records: List[Dict[str, Any]] = []
    total_runs = len(selected) * len(seeds)
    run_index = 0

    for record in selected:
        cfg = RunConfig(
            mode=record["mode"],
            bits=record["bits"],
            population=record["population"],
            generations=record["generations"],
            hnp_weight=record["hnp_weight"],
            ecdsa_weight=record["ecdsa_weight"],
            residue_primes=args.residue_primes,
        )

        effective_bits = args.hnp_bits if args.hnp_bits is not None else cfg.bits
        effective_mode = args.hnp_mode if args.hnp_mode is not None else cfg.mode

        for seed in seeds:
            run_index += 1
            print(
                f"[validate] {run_index}/{total_runs} mode={effective_mode} bits={effective_bits} "
                f"pop={cfg.population} gen={cfg.generations} w={cfg.hnp_weight} seed={seed}",
                flush=True,
            )
            _seed_everything(seed)
            dataset_path = os.path.join(
                datasets_dir,
                f"dataset_{effective_mode}_{effective_bits}_seed{seed}.json",
            )
            dataset = ensure_dataset(
                path=dataset_path,
                seed=seed,
                ecdsa_samples=args.samples,
                hnp_signatures=args.hnp_signatures,
                hnp_bits=effective_bits,
                hnp_mode=effective_mode,
            )

            log_name = (
                f"validate_{effective_mode}_{effective_bits}_pop{cfg.population}_gen{cfg.generations}_"
                f"w{cfg.hnp_weight}_seed{seed}.jsonl"
            )
            log_path = os.path.join(logs_dir, log_name)
            logger = JsonlLogger(log_path)

            explorer = KeyDerivationExplorer(
                population_size=cfg.population,
                elite_ratio=args.elite,
                mutation_temp=args.mut_temp,
                memory_size=200,
                sample_size=args.samples,
                dataset=dataset,
                ecdsa_weight=cfg.ecdsa_weight,
                hnp_weight=cfg.hnp_weight,
                residue_primes=cfg.residue_primes,
                logger=logger,
            )

            start_time = time.time()
            best = explorer.run(generations=cfg.generations)
            elapsed = time.time() - start_time
            print(f"[validate] completed in {elapsed:.1f}s", flush=True)

            run_record = {
                "source_record": record,
                "mode": effective_mode,
                "bits": effective_bits,
                "source_mode": cfg.mode,
                "source_bits": cfg.bits,
                "population": cfg.population,
                "generations": cfg.generations,
                "hnp_weight": cfg.hnp_weight,
                "ecdsa_weight": cfg.ecdsa_weight,
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

            record_path = os.path.join(runs_dir, log_name.replace(".jsonl", ".json"))
            _write_json(record_path, run_record)
            run_records.append(run_record)

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
