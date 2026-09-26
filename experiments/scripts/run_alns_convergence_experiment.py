#!/usr/bin/env python3
"""Run Maruthu's long-run ALNS convergence experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import secrets
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from waveform_optimization.core.binary_objective import BinaryObjectiveState
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.alns.initialization import generate_initial_solution
from waveform_optimization.solvers.alns.solver import ALNSConfig, ALNSTelemetry, ALNSSolver

OWNER = "maruthu"
Q = 2
REPAIR_OPERATOR = "randomized_greedy"
OUTER_ACCEPTANCE = "simulated_annealing"
DESTROY_OPERATORS = ("random", "best_flip", "correlation_guided", "mixed")
SCHEDULE = {
    20: (30.0, (0, 5, 10, 20, 30)),
    50: (60.0, (0, 10, 20, 30, 45, 60)),
    100: (120.0, (0, 10, 30, 60, 90, 120)),
    150: (180.0, (0, 10, 30, 60, 90, 120, 150, 180)),
    200: (240.0, (0, 10, 30, 60, 90, 120, 180, 240)),
}
SEEDS = (42, 123, 456)
RUN_FIELDS = (
    "run_id", "owner", "timestamp", "git_commit", "machine_cpu", "N", "Q", "seed",
    "destroy_operator", "repair_operator", "outer_acceptance", "destroy_fraction",
    "selected_neighborhood_size", "initialization_method", "initial_sequence_hash",
    "initial_energy", "initialization_seconds", "search_time_limit_seconds",
    "actual_search_runtime_seconds", "best_energy", "barker_reference_ratio",
    "merit_factor", "best_sequence", "total_iterations", "iterations_per_second",
    "accepted_moves", "rejected_moves", "improving_moves", "new_best_moves",
    "last_strict_best_iteration", "last_strict_best_time_seconds", "stagnation_seconds",
    "stagnation_fraction", "unchanged_candidate_count", "equal_energy_changed_sequence_count",
    "accepted_worsening_count", "rejected_worsening_count", "improving_candidate_count",
    "strict_new_best_count", "status", "error_message", "config_json",
)
CHECKPOINT_FIELDS = (
    "run_id", "N", "seed", "destroy_operator", "repair_operator", "checkpoint_seconds",
    "best_energy_at_checkpoint", "current_energy_at_checkpoint",
    "iterations_completed_by_checkpoint", "accepted_moves_by_checkpoint",
    "rejected_moves_by_checkpoint", "improving_moves_by_checkpoint", "new_best_moves_by_checkpoint",
    "actual_observation_time_seconds", "is_final",
)
EVENT_FIELDS = (
    "run_id", "event_elapsed_seconds", "event_iteration", "previous_best_energy",
    "new_best_energy", "best_sequence",
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine_cpu() -> str:
    return platform.processor() or platform.machine() or socket.gethostname()


def phase_string(sequence: Any, N: int) -> str:
    values = [int(value) for value in sequence]
    result = "".join(str(value) for value in values)
    if len(result) != N or set(result) - {"0", "1"}:
        raise ValueError("Sequence must contain exactly N binary phase indices.")
    return result


def sequence_hash(sequence: Any, N: int) -> str:
    return hashlib.sha256(phase_string(sequence, N).encode("ascii")).hexdigest()


def write_row(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()


def cache_initial_state(root: Path, N: int, seed: int,
                        initialization_time_limit: float) -> dict[str, Any]:
    path = root / "initial_states" / f"N_{N}_seed_{seed}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        sequence = record["original_sequence"]
        if sequence_hash(sequence, N) != record["sequence_sha256"]:
            raise ValueError(f"Cached initial state hash mismatch: {path}")
        return record
    started = time.perf_counter()
    state = generate_initial_solution(
        N=N, method="ils", time_limit=initialization_time_limit, seed=seed
    )
    elapsed = time.perf_counter() - started
    sequence = [int(value) for value in state.sequence]
    record = {
        "N": N, "Q": Q, "seed": seed, "initialization_method": "ils",
        "original_sequence": sequence, "initial_energy": int(state.energy),
        "initialization_seconds": elapsed, "sequence_sha256": sequence_hash(sequence, N),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def checkpoint_row(run_id: str, N: int, seed: int, destroy: str,
                   row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id, "N": N, "seed": seed, "destroy_operator": destroy,
        "repair_operator": REPAIR_OPERATOR, "checkpoint_seconds": row["checkpoint_seconds"],
        "best_energy_at_checkpoint": row["best_energy"],
        "current_energy_at_checkpoint": row["current_energy"],
        "iterations_completed_by_checkpoint": row["iterations_completed"],
        "accepted_moves_by_checkpoint": row["accepted_moves"],
        "rejected_moves_by_checkpoint": row["rejected_moves"],
        "improving_moves_by_checkpoint": row["improving_moves"],
        "new_best_moves_by_checkpoint": row["new_best_moves"],
        "actual_observation_time_seconds": row["actual_observation_time_seconds"],
        "is_final": row["is_final"],
    }


def run_one(root: Path, N: int, seed: int, destroy: str,
            search_time_limit: float, checkpoints: tuple[int, ...],
            initial: dict[str, Any], config_json: str) -> dict[str, Any]:
    run_id = f"N{N}_seed{seed}_{destroy}_{REPAIR_OPERATOR}"
    sequence = initial["original_sequence"]
    initial_string = phase_string(sequence, N)
    initial_energy = int(initial["initial_energy"])
    initial_hash = initial["sequence_sha256"]
    selected_size = min(N, max(2, int((0.10 * N) + 0.999999)))
    started = time.perf_counter()
    status = "ok"
    error_message = ""
    result = None
    telemetry = ALNSTelemetry(checkpoint_seconds=tuple(float(value) for value in checkpoints if value > 0))
    config = {
        "N": N, "Q": Q, "seed": seed, "destroy_operator": destroy,
        "repair_operator": REPAIR_OPERATOR, "outer_acceptance": OUTER_ACCEPTANCE,
        "destroy_fraction": 0.10, "search_time_limit_seconds": search_time_limit,
        "checkpoints": checkpoints, "initialization_method": "ils",
    }
    try:
        solver_config = ALNSConfig(
            time_limit=search_time_limit, seed=seed, initialization_method="ils",
            destroy_fraction=0.10, destroy_methods=(destroy,),
            repair_methods=(REPAIR_OPERATOR,), acceptance_method=OUTER_ACCEPTANCE,
        )
        result = ALNSSolver(solver_config).solve(
            N, initial_state=BinaryObjectiveState(sequence).copy(), telemetry=telemetry
        )
        best_sequence = [int(value) for value in result.best_sequence]
        best_string = phase_string(best_sequence, N)
        recomputed = float(islr_from_phase_indices(best_sequence, Q=Q))
        if recomputed != result.best_energy:
            raise ValueError(f"Best objective mismatch: shared={recomputed}, ALNS={result.best_energy}")
        if result.iterations > 0 and telemetry.counters["unchanged_candidate_count"] == result.iterations:
            raise RuntimeError(
                "Every ALNS candidate was unchanged; destroy/repair movement validation failed."
            )
        actual_runtime = float(result.elapsed_time)
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
        best_string = ""
        if result is not None:
            best_string = phase_string(result.best_sequence, N)
        actual_runtime = time.perf_counter() - started

    if result is None:
        counters = telemetry.counters
        iterations = accepted = rejected = improving = new_best = 0
        best_energy = ""
        last_iteration = ""
        last_time = ""
    else:
        counters = telemetry.counters
        iterations = result.iterations
        accepted, rejected = result.accepted_moves, result.rejected_moves
        improving, new_best = result.improving_moves, result.new_best_moves
        best_energy = result.best_energy
        last_event = telemetry.new_best_events[-1] if telemetry.new_best_events else None
        last_iteration = last_event["event_iteration"] if last_event else 0
        last_time = last_event["event_elapsed_seconds"] if last_event else 0.0

    event_path = root / OWNER / "new_best_events.csv"
    write_row(event_path, EVENT_FIELDS, {
        "run_id": run_id, "event_elapsed_seconds": 0.0, "event_iteration": 0,
        "previous_best_energy": "", "new_best_energy": initial_energy,
        "best_sequence": initial_string,
    })
    for event in telemetry.new_best_events:
        write_row(event_path, EVENT_FIELDS, {
            **event, "run_id": run_id,
            "best_sequence": phase_string(event["best_sequence"], N),
        })

    checkpoint_path = root / OWNER / "checkpoints.csv"
    if result is not None:
        for checkpoint in telemetry.checkpoints:
            write_row(checkpoint_path, CHECKPOINT_FIELDS,
                      checkpoint_row(run_id, N, seed, destroy, checkpoint))
        if telemetry.final_checkpoint is not None:
            write_row(checkpoint_path, CHECKPOINT_FIELDS,
                      checkpoint_row(run_id, N, seed, destroy, telemetry.final_checkpoint))

    last_best_time = float(last_time or 0.0)
    stagnation = max(0.0, actual_runtime - last_best_time)
    run_row = {
        "run_id": run_id, "owner": OWNER,
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "machine_cpu": machine_cpu(), "N": N, "Q": Q, "seed": seed,
        "destroy_operator": destroy, "repair_operator": REPAIR_OPERATOR,
        "outer_acceptance": OUTER_ACCEPTANCE, "destroy_fraction": 0.10,
        "selected_neighborhood_size": selected_size, "initialization_method": "ils",
        "initial_sequence_hash": initial_hash, "initial_energy": initial_energy,
        "initialization_seconds": initial["initialization_seconds"],
        "search_time_limit_seconds": search_time_limit,
        "actual_search_runtime_seconds": actual_runtime, "best_energy": best_energy,
        "barker_reference_ratio": (best_energy / ((N - 1) / 2) if best_energy != "" else ""),
        "merit_factor": ((N * N) / (2.0 * best_energy) if best_energy != "" else ""),
        "best_sequence": best_string, "total_iterations": iterations,
        "iterations_per_second": (iterations / actual_runtime if actual_runtime > 0 else ""),
        "accepted_moves": accepted, "rejected_moves": rejected,
        "improving_moves": improving, "new_best_moves": new_best,
        "last_strict_best_iteration": last_iteration,
        "last_strict_best_time_seconds": last_time,
        "stagnation_seconds": stagnation,
        "stagnation_fraction": stagnation / actual_runtime if actual_runtime > 0 else "",
        "unchanged_candidate_count": counters.get("unchanged_candidate_count", 0),
        "equal_energy_changed_sequence_count": counters.get("equal_energy_changed_sequence_count", 0),
        "accepted_worsening_count": counters.get("accepted_worsening_count", 0),
        "rejected_worsening_count": counters.get("rejected_worsening_count", 0),
        "improving_candidate_count": counters.get("improving_candidate_count", 0),
        "strict_new_best_count": counters.get("strict_new_best_count", 0),
        "status": status, "error_message": error_message, "config_json": config_json,
    }
    write_row(root / OWNER / "run_results.csv", RUN_FIELDS, run_row)
    sequence_dir = root / OWNER / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    (sequence_dir / f"{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "best_sequence": best_string}, indent=2),
        encoding="utf-8",
    )
    (root / OWNER / "logs").mkdir(parents=True, exist_ok=True)
    (root / OWNER / "logs" / f"{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "status": status, "error_message": error_message}, indent=2),
        encoding="utf-8",
    )
    return run_row


def smoke_assertions(rows: list[dict[str, Any]]) -> None:
    """Reject a smoke run unless every operator demonstrates real movement."""
    if len(rows) != 4:
        raise RuntimeError(f"Smoke expected four operator rows, got {len(rows)}.")
    for row in rows:
        changed = int(row["total_iterations"]) - int(row["unchanged_candidate_count"])
        if changed <= 100:
            raise RuntimeError(
                f"Smoke movement check failed for {row['destroy_operator']}: "
                f"changed candidates={changed}, expected > 100."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--smoke", action="store_true",
                        help="Run N=20, seed=42, all destroy operators for 5 seconds")
    parser.add_argument("--N", dest="sizes", help="Comma-separated N values for smoke runs")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--destroy-operators", default=",".join(DESTROY_OPERATORS))
    parser.add_argument("--time-limit", type=float, help="Override every selected N budget")
    parser.add_argument("--initialization-time-limit", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.owner != OWNER:
        raise ValueError("This experiment is assigned to owner maruthu.")
    if args.smoke:
        sizes = (20,)
        seeds = (42,)
        destroys = DESTROY_OPERATORS
        search_time_limit = 5.0
        output_root = args.output_root or Path("results/alns_convergence_smoke")
    else:
        sizes = tuple(int(value) for value in args.sizes.split(",")) if args.sizes else tuple(SCHEDULE)
        seeds = tuple(int(value) for value in args.seeds.split(","))
        destroys = tuple(value.strip() for value in args.destroy_operators.split(","))
        search_time_limit = args.time_limit
        output_root = args.output_root or Path("results/alns_convergence")
    if set(sizes) - set(SCHEDULE) or set(destroys) - set(DESTROY_OPERATORS):
        raise ValueError("Requested sizes/operators are outside the official experiment scope.")
    config_json = json.dumps({"owner": OWNER, "Q": Q, "repair_operator": REPAIR_OPERATOR,
                              "destroy_operators": destroys, "schedule": SCHEDULE,
                              "seeds": seeds}, sort_keys=True)
    smoke_rows = []
    for N in sizes:
        for seed in seeds:
            initial = cache_initial_state(
                output_root, N, seed, args.initialization_time_limit
            )
            for destroy in destroys:
                limit, checkpoints = SCHEDULE[N]
                row = run_one(
                    output_root, N, seed, destroy,
                    search_time_limit if search_time_limit is not None else limit,
                    checkpoints, initial, config_json,
                )
                if args.smoke:
                    smoke_rows.append(row)
    if args.smoke:
        smoke_assertions(smoke_rows)
    owner_dir = output_root / OWNER
    (owner_dir / "config.json").parent.mkdir(parents=True, exist_ok=True)
    (owner_dir / "config.json").write_text(config_json, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
