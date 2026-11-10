#!/usr/bin/env python3
"""
Run one Python solver on one (N, Q) instance and dump a JSON record.

Used by scripts/xlang_benchmark.sh. Kept deliberately generic: the solver is
given as "module.path:ClassName" so new solvers need no change here, only a new
line in the SOLVERS registry of the bash script.

The solver is called through the common interface defined in
src/waveform_optimization/solvers/base.py:

    solver.solve(N=N, Q=Q, time_limit=None, seed=seed)

and the returned SolverResult is read with attribute access, never dict access.

--params (optional): path to a JSON file of extra solver kwargs (e.g. GA
population_size, mutation_prob, ...). The SAME file is read by the MATLAB
side (xlang_run_solver.m), so both languages run with identical
hyperparameters -- this only matters for parameterized solvers like the
genetic algorithm; solvers without a matching keyword argument simply
ignore it via **kwargs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time

# JSON keys are named to match the MATLAB params struct (see
# genetic_algorithm_solver.m). The Python solver uses "_rate" instead of
# "_prob" for the same two knobs, so they are renamed on the way in.
_PARAM_RENAME = {
    "crossover_prob": "crossover_rate",
    "mutation_prob": "mutation_rate",
}


def parse_seed(raw: str | None) -> int | None:
    if raw is None or raw.strip().lower() in {"", "none", "null", "na"}:
        return None
    return int(raw)


def load_params(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {_PARAM_RENAME.get(k, k): v for k, v in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True, help="module.path:ClassName")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--Q", type=int, required=True)
    ap.add_argument("--seed", default="none")
    ap.add_argument("--out", required=True, help="path of the JSON record to write")
    ap.add_argument(
        "--params",
        default=None,
        help="optional path to a JSON file of extra solver kwargs, shared "
        "with the MATLAB run of the same solver",
    )
    args = ap.parse_args()

    payload: dict = {
        "language": "python",
        "solver_spec": args.solver,
        "N": args.N,
        "Q": args.Q,
        "status": "error",
        "error": "",
    }

    try:
        module_name, class_name = args.solver.split(":")
        module = importlib.import_module(module_name)
        solver_cls = getattr(module, class_name)

        # Independent recomputation uses the shared mathematical core, so a
        # solver that silently reports an inconsistent objective is caught.
        from waveform_optimization.core.objective import islr_from_phase_indices

        seed = parse_seed(args.seed)
        extra_kwargs = load_params(args.params)
        if extra_kwargs:
            payload["params"] = extra_kwargs
        solver = solver_cls()

        wall_start = time.perf_counter()
        result = solver.solve(
            N=args.N, Q=args.Q, time_limit=None, seed=seed, **extra_kwargs
        )
        wall_seconds = time.perf_counter() - wall_start

        q = [int(v) for v in result.best_phase_indices]
        recomputed = float(islr_from_phase_indices(q=q, Q=args.Q))
        valid = len(q) == args.N and all(0 <= v < args.Q for v in q)

        payload.update(
            {
                "status": str(result.status),
                "algorithm": str(result.algorithm),
                "N": int(result.N),
                "Q": int(result.Q),
                "seed": seed,
                "best_objective": float(result.best_objective),
                "recomputed_objective": recomputed,
                "objective_evaluations": int(result.objective_evaluations),
                "runtime_seconds": float(result.runtime_seconds),
                "wall_seconds": float(wall_seconds),
                "best_phase_indices": q,
                "valid": bool(valid),
                "error": "",
            }
        )
    except Exception as exc:  # noqa: BLE001 - the record must always be written
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    return 0 if payload["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())