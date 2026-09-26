"""Run reproducible baseline waveform-optimization benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import secrets
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from waveform_optimization.benchmarking.validation import validate_solver_result
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.weak_baselines.coordinate_descent import CoordinateDescentSolver
from waveform_optimization.solvers.weak_baselines.greedy import GreedySolver
from waveform_optimization.solvers.weak_baselines.random_search import RandomSearchSolver

DEFAULT_CONFIG = {
    "algorithms": ["greedy", "random_search", "coordinate_descent"],
    "N_values": [5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 100, 150,
                  200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "Q": 2, "runs": 10, "time_limit_seconds_by_N": {},
    "runs_by_N": {}, "seed": None, "solver_parameters": {},
    "normalization": {"barker_reference_scale": 1.0},
}
SOLVERS = {"greedy": GreedySolver, "random_search": RandomSearchSolver,
           "coordinate_descent": CoordinateDescentSolver}
CSV_FIELDS = ["algorithm", "N", "Q", "run_id", "seed", "time_limit_seconds",
              "runtime_seconds", "best_objective", "normalized_objective",
              "objective_evaluations", "status", "json_path",
              "normalization_label", "phase_string"]
CSV_KEY_FIELDS = ["algorithm", "N", "Q", "run_id", "seed",
                  "time_limit_seconds"]


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip("'\"")


def _fallback_yaml(text: str) -> dict[str, Any]:
    """Parse the mapping/list subset used by the benchmark config."""
    result: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, result)]
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("Unsupported YAML list placement")
            parent.append(_scalar(content[2:]))
            continue
        key, separator, value = content.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML line: {raw_line}")
        if value.strip():
            parent[_scalar(key.strip())] = _scalar(value)
        else:
            next_content = ""
            for candidate in lines[index + 1:]:
                if candidate.strip():
                    next_content = candidate.strip()
                    break
            child: Any = [] if next_content.startswith("-") else {}
            parent[_scalar(key.strip())] = child
            stack.append((indent, child))
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and apply defaults for omitted settings."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        loaded: dict[str, Any] = {}
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            loaded = _fallback_yaml(text)
        else:
            loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Benchmark configuration must be a mapping.")
    merged = {**DEFAULT_CONFIG, **loaded}
    merged["normalization"] = {**DEFAULT_CONFIG["normalization"],
                                **loaded.get("normalization", {})}
    return merged


def _code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _machine_details() -> dict[str, str]:
    return {"platform": platform.platform(), "python": platform.python_version(),
            "hostname": socket.gethostname()}


def _record_base(algorithm: str, N: int, Q: int, run_id: int, seed: int | None,
                 limit: float | None, json_path: str,
                 parameters: dict[str, Any]) -> dict[str, Any]:
    return {"algorithm": algorithm, "N": N, "Q": Q, "run_id": run_id,
            "seed": seed, "time_limit_seconds": limit, "runtime_seconds": None,
            "best_objective": None, "normalized_objective": None,
            "objective_evaluations": None, "status": "failed", "json_path": json_path,
            "history": [], "solver_parameters": parameters,
            "code_revision": _code_revision(), "machine_details": _machine_details(),
            "normalization_label": "Barker reference scale",
            "best_phase_indices": None, "phase_string": None,
            "validation_error": None}


def _phase_string(best_phase_indices: list[int], N: int, Q: int) -> str:
    """Validate and encode a binary phase-index solution."""
    if Q != 2:
        raise ValueError("phase_string is only supported for Q=2.")
    phase_string = "".join(str(index) for index in best_phase_indices)
    if len(phase_string) != N:
        raise ValueError("phase_string must contain exactly N characters.")
    if any(character not in {"0", "1"} for character in phase_string):
        raise ValueError("phase_string may contain only '0' and '1'.")
    return phase_string


def _compact_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only objective improvements from per-iteration solver history."""
    compacted: list[dict[str, Any]] = []
    best_objective = float("inf")
    for entry in history:
        objective = entry.get("objective")
        if isinstance(objective, (int, float)) and objective < best_objective:
            compacted.append(entry)
            best_objective = float(objective)
    return compacted


def _time_limit_token(time_limit: float | None) -> str:
    if time_limit is None:
        return "unlimited"
    return f"{time_limit:g}s"


def _csv_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        "" if record.get(field) is None else str(record.get(field))
        for field in CSV_KEY_FIELDS
    )


def _write_csv(csv_path: Path, records: list[dict[str, Any]]) -> None:
    existing: dict[tuple[Any, ...], dict[str, Any]] = {}
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing[_csv_key(row)] = row
    for record in records:
        existing[_csv_key(record)] = record
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in existing.values():
            writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})


def rebuild_csv_from_json(json_root: str | Path = "results/binary_baseline/raw",
                          csv_path: str | Path = "results/binary_baseline/binary_baseline_raw_results.csv") -> list[dict[str, Any]]:
    """Rebuild the flat CSV from saved JSON records without running solvers."""
    root = Path(json_root)
    records: list[dict[str, Any]] = []
    for json_file in sorted(root.glob("**/*.json")):
        record = json.loads(json_file.read_text(encoding="utf-8"))
        phase_string = _phase_string(
            record["best_phase_indices"], int(record["N"]), int(record["Q"])
        )
        record["phase_string"] = phase_string
        record["json_path"] = str(json_file.relative_to(Path(csv_path).parent))
        records.append(record)
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_file, records)
    return records


def _schedule_value(config: dict[str, Any], key: str, N: int,
                    fallback: Any) -> Any:
    values = config.get(key, {})
    if not isinstance(values, dict):
        return fallback
    return values.get(N, values.get(str(N), fallback))


def run_benchmark(config: dict[str, Any], output_root: str | Path = "results/binary_baseline",
                  algorithms: list[str] | None = None, N_values: list[int] | None = None,
                  Q: int | None = None, runs: int | None = None,
                  time_limit: float | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    algorithms = algorithms or list(config["algorithms"])
    N_values = N_values or list(config["N_values"])
    Q = config["Q"] if Q is None else Q
    runs = config["runs"] if runs is None else runs
    seed = config.get("seed") if seed is None else seed
    unknown = set(algorithms) - SOLVERS.keys()
    if unknown:
        raise ValueError(f"Unknown algorithms: {sorted(unknown)}")
    root = Path(output_root)
    csv_path = root / "binary_baseline_raw_results.csv"
    records: list[dict[str, Any]] = []
    for algorithm in algorithms:
        parameters = dict(config.get("solver_parameters", {}).get(algorithm, {}))
        for N in N_values:
            configured_limit = _schedule_value(
                config, "time_limit_seconds_by_N", N,
                config.get("time_limit_seconds"),
            )
            configured_runs = _schedule_value(config, "runs_by_N", N, runs)
            algorithm_limit = time_limit if time_limit is not None else (
                None if algorithm == "greedy" else configured_limit
            )
            algorithm_runs = 1 if algorithm == "greedy" else configured_runs
            for run_id in range(1, algorithm_runs + 1):
                if algorithm == "greedy":
                    run_seed = None
                elif seed is None:
                    run_seed = secrets.randbits(63)
                else:
                    run_seed = seed + run_id - 1
                budget = _time_limit_token(algorithm_limit)
                relative_json = (Path("raw") / algorithm / f"N_{N}" / budget /
                                 f"run_{run_id}.json")
                record = _record_base(algorithm, N, Q, run_id, run_seed, algorithm_limit,
                                      str(relative_json), parameters)
                started = time.perf_counter()
                try:
                    result = SOLVERS[algorithm]().solve(N=N, Q=Q, time_limit=algorithm_limit,
                                                        seed=run_seed, **parameters)
                    result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
                    validate_solver_result(result_dict)
                    recomputed = islr_from_phase_indices(result_dict["best_phase_indices"], Q)
                    if abs(float(result_dict["best_objective"]) - recomputed) > 1e-10:
                        raise ValueError("Solver objective does not match the shared objective.")
                    scale = float(config.get("normalization", {}).get("barker_reference_scale", 1.0))
                    if scale <= 0:
                        raise ValueError("Barker reference scale must be positive.")
                    record.update(result_dict)
                    record["phase_string"] = _phase_string(
                        result_dict["best_phase_indices"], N, Q
                    )
                    record["runtime_seconds"] = time.perf_counter() - started
                    record["normalized_objective"] = recomputed / ((N - 1) / 2)
                    record["normalization_label"] = "Barker reference scale"
                    record["history"] = _compact_history(record.get("history", []))
                except (TypeError, ValueError, KeyError) as error:
                    record["runtime_seconds"] = time.perf_counter() - started
                    record["validation_error"] = str(error)
                    record["status"] = "validation_failed"
                target = root / relative_json
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
                records.append(record)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, records)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/benchmark.yaml"))
    parser.add_argument("--algorithms", help="Comma-separated algorithm names")
    parser.add_argument("--N", dest="N_values", help="Comma-separated N values")
    parser.add_argument("--Q", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("results/binary_baseline"))
    parser.add_argument("--rebuild-csv", action="store_true",
                        help="Rebuild the CSV from existing JSON files only")
    args = parser.parse_args(argv)
    if args.rebuild_csv:
        rebuild_csv_from_json(
            args.output_root / "raw",
            args.output_root / "binary_baseline_raw_results.csv",
        )
        return 0
    run_benchmark(load_config(args.config), output_root=args.output_root,
                  algorithms=args.algorithms.split(",") if args.algorithms else None,
                  N_values=[int(value) for value in args.N_values.split(",")] if args.N_values else None,
                  Q=args.Q, runs=args.runs, time_limit=args.time_limit, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())