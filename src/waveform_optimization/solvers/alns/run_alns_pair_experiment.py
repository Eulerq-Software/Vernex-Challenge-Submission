#!/usr/bin/env python3
"""Run the controlled ALNS destroy-repair pair experiment for binary phases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

from waveform_optimization.core.binary_objective import BinaryObjectiveState
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.alns.solver import solve_alns


SIZES = (5, 7, 11, 20, 30, 50, 100, 150, 200, 300, 500, 1000)
SEEDS = (42, 123, 456, 789, 2026)
DEFAULT_PAIRS = (
    ("best_flip", "greedy"),
    ("correlation_guided", "greedy"),
    ("random", "randomized_greedy"),
    ("block", "randomized_greedy"),
    ("best_flip", "randomized_greedy"),
    ("correlation_guided", "randomized_greedy")
)
FIELDNAMES = (
    "N", "Q", "seed", "destroy_operator", "repair_operator",
    "initial_solution_id", "time_limit", "acceptance_method",
    "destroy_fraction", "exact_max_neighborhood_size", "initial_energy",
    "best_energy", "reference_objective", "barker_ratio", "best_sequence",
    "phase_string", "actual_runtime", "iterations", "iterations_per_second",
    "accepted_moves", "rejected_moves", "improving_moves", "new_best_moves",
    "status", "error_message",
)


def time_limit_for(N: int) -> float:
    if N <= 30:
        return 10.0
    if N <= 200:
        return 15.0
    if N == 300:
        return 30.0
    if N == 500:
        return 40.0
    if N == 1000:
        return 90.0
    raise ValueError(f"No experiment time limit configured for N={N}.")


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_pairs(value: str | None) -> tuple[tuple[str, str], ...]:
    if value is None:
        return DEFAULT_PAIRS
    pairs = []
    for item in value.split(","):
        destroy, separator, repair = item.strip().partition("/")
        if not separator or not destroy or not repair:
            raise ValueError("Pairs must use destroy/repair, separated by commas.")
        pairs.append((destroy, repair))
    return tuple(pairs)


def solution_id(N: int, seed: int) -> str:
    return f"N{N}_seed{seed}"


def make_manifest(path: Path, sizes: tuple[int, ...], seeds: tuple[int, ...]) -> dict[str, dict]:
    """Create once, then reuse verbatim across every assigned pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
        for N in sizes:
            for seed in seeds:
                sequence = np.random.default_rng(seed).integers(0, 2, size=N, dtype=np.int8)
                state = BinaryObjectiveState(sequence)
                rows.append({
                    "initial_solution_id": solution_id(N, seed),
                    "N": N,
                    "Q": 2,
                    "seed": seed,
                    "initial_phase_string": "".join(str(int(bit)) for bit in sequence),
                    "initial_energy": state.energy,
                })
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    manifest = {row["initial_solution_id"]: row for row in rows}
    expected = {solution_id(N, seed) for N in sizes for seed in seeds}
    missing = expected - set(manifest)
    if missing:
        raise ValueError(f"Shared manifest is incomplete; missing {sorted(missing)}.")
    for row in manifest.values():
        N = int(row["N"])
        phase_string = row["initial_phase_string"]
        if len(phase_string) != N or set(phase_string) - {"0", "1"}:
            raise ValueError(f"Invalid shared initial solution {row['initial_solution_id']}.")
    return manifest


def existing_run_ids(results_path: Path) -> set[tuple[str, str, str, str]]:
    if not results_path.exists():
        return set()
    with results_path.open(newline="") as handle:
        return {
            (row["initial_solution_id"], row["destroy_operator"], row["repair_operator"], row["status"])
            for row in csv.DictReader(handle)
        }


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(history, handle, indent=2)


def run_one(manifest_row: dict, destroy: str, repair: str, exact_cap: int) -> tuple[dict, list[dict] | None]:
    N = int(manifest_row["N"])
    seed = int(manifest_row["seed"])
    initial_string = manifest_row["initial_phase_string"]
    state = BinaryObjectiveState([int(bit) for bit in initial_string])
    limit = time_limit_for(N)
    base = {
        "N": N, "Q": 2, "seed": seed, "destroy_operator": destroy,
        "repair_operator": repair, "initial_solution_id": manifest_row["initial_solution_id"],
        "time_limit": limit, "acceptance_method": "simulated_annealing",
        "destroy_fraction": 0.10, "exact_max_neighborhood_size": exact_cap,
        "initial_energy": state.energy,
    }
    try:
        result = solve_alns(
            N, initial_state=state, time_limit=limit, seed=seed,
            destroy_methods=(destroy,), repair_methods=(repair,),
            acceptance_method="simulated_annealing", destroy_fraction=0.10,
            exact_max_neighborhood_size=exact_cap,
        )
        phase_string = "".join(str(int(bit)) for bit in result.best_sequence)
        reference = float(islr_from_phase_indices(result.best_sequence, Q=2))
        reference_int = int(round(reference))
        mismatch = not math.isclose(reference, result.best_energy, abs_tol=1e-9)
        runtime = result.elapsed_time
        base.update({
            "best_energy": result.best_energy,
            "reference_objective": reference,
            "barker_ratio": result.best_energy / ((N - 1) / 2),
            "best_sequence": phase_string,
            "phase_string": phase_string,
            "actual_runtime": runtime,
            "iterations": result.iterations,
            "iterations_per_second": result.iterations / runtime if runtime > 0 else "",
            "accepted_moves": result.accepted_moves,
            "rejected_moves": result.rejected_moves,
            "improving_moves": result.improving_moves,
            "new_best_moves": result.new_best_moves,
            "status": "objective_mismatch" if mismatch else "ok",
            "error_message": (
                f"reference={reference_int}, internal={result.best_energy}" if mismatch else ""
            ),
        })
        return base, result.history
    except Exception as exc:  # Keep the batch running and make failures identifiable.
        base.update({
            "best_energy": "", "reference_objective": "", "barker_ratio": "",
            "best_sequence": "", "phase_string": "", "actual_runtime": "",
            "iterations": "", "iterations_per_second": "", "accepted_moves": "",
            "rejected_moves": "", "improving_moves": "", "new_best_moves": "",
            "status": "error", "error_message": f"{type(exc).__name__}: {exc}",
        })
        traceback.print_exc(file=sys.stderr)
        return base, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/alns_pair_experiments"))
    parser.add_argument("--owner", default="surya")
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--pairs", help="Comma-separated destroy/repair pairs.")
    parser.add_argument("--exact-max-neighborhood-size", type=int, default=10)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)
    pairs = parse_pairs(args.pairs)
    if set(sizes) - set(SIZES) or not sizes:
        raise ValueError(f"Sizes must be selected from {SIZES}.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    if args.exact_max_neighborhood_size < 1:
        raise ValueError("exact-max-neighborhood-size must be at least 1.")

    manifest_path = args.output_root / "shared_initial_solutions.csv"
    manifest = make_manifest(manifest_path, sizes, seeds)
    owner_dir = args.output_root / args.owner
    results_path = owner_dir / "results.csv"
    owner_dir.mkdir(parents=True, exist_ok=True)
    completed = existing_run_ids(results_path) if args.resume else set()
    write_header = not results_path.exists() or results_path.stat().st_size == 0

    with results_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            handle.flush()
        for destroy, repair in pairs:
            for N in sizes:
                for seed in seeds:
                    run_id = (solution_id(N, seed), destroy, repair)
                    if args.resume and any(entry[:3] == run_id for entry in completed):
                        print(f"SKIP {run_id}", flush=True)
                        continue
                    row, history = run_one(manifest[run_id[0]], destroy, repair, args.exact_max_neighborhood_size)
                    writer.writerow(row)
                    handle.flush()
                    print(f"{row['status'].upper()} N={N} seed={seed} {destroy}/{repair} best={row['best_energy']}", flush=True)
                    if history is not None and not args.no_history:
                        write_history(owner_dir / "history" / f"N{N}_seed{seed}_{destroy}_{repair}.json", history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
