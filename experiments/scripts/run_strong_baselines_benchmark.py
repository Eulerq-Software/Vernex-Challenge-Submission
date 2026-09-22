from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from waveform_optimization.benchmarking.validation import validate_solver_result
from waveform_optimization.solvers.strong_baselines.genetic_algorithm import (
    GeneticAlgorithmSolver,
)
from waveform_optimization.solvers.strong_baselines.simulated_annealing import (
    SimulatedAnnealingSolver,
)
from waveform_optimization.solvers.strong_baselines.tabu_search import TabuSearchSolver
from waveform_optimization.solvers.strong_baselines.variable_neighborhood_search import (
    VariableNeighborhoodSearchSolver,
)

INSTANCES = [
    (13, 2),
    (21, 4),
    (31, 4),
    (41, 6),
]

SEEDS = [1, 2, 3]
TIME_LIMIT_SECONDS = 1.0

SOLVER_FACTORIES = {
    "simulated_annealing": SimulatedAnnealingSolver,
    "genetic_algorithm": GeneticAlgorithmSolver,
    "tabu_search": TabuSearchSolver,
    "variable_neighborhood_search": VariableNeighborhoodSearchSolver,
}

SOLVER_KWARGS = {
    "simulated_annealing": {"max_iterations": 10_000_000},
    "genetic_algorithm": {"max_generations": 10_000_000},
    "tabu_search": {"max_iterations": 10_000_000},
    "variable_neighborhood_search": {"max_iterations": 10_000_000},
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results"

RAW_FIELDNAMES = [
    "algorithm",
    "N",
    "Q",
    "seed",
    "best_objective",
    "runtime_seconds",
    "objective_evaluations",
    "status",
]


def run_all() -> list[dict]:
    rows = []

    for N, Q in INSTANCES:
        for name, factory in SOLVER_FACTORIES.items():
            for seed in SEEDS:
                solver = factory()
                result = solver.solve(
                    N=N,
                    Q=Q,
                    time_limit=TIME_LIMIT_SECONDS,
                    seed=seed,
                    **SOLVER_KWARGS[name],
                )
                validate_solver_result(result.to_dict())

                rows.append(
                    {
                        "algorithm": result.algorithm,
                        "N": result.N,
                        "Q": result.Q,
                        "seed": seed,
                        "best_objective": result.best_objective,
                        "runtime_seconds": result.runtime_seconds,
                        "objective_evaluations": result.objective_evaluations,
                        "status": result.status,
                    }
                )

    return rows


def write_raw_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int, str], list[dict]] = {}
    for row in rows:
        key = (row["N"], row["Q"], row["algorithm"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (N, Q, algorithm), group in sorted(groups.items()):
        objectives = [r["best_objective"] for r in group]
        runtimes = [r["runtime_seconds"] for r in group]
        evaluations = [r["objective_evaluations"] for r in group]

        summary.append(
            {
                "N": N,
                "Q": Q,
                "algorithm": algorithm,
                "runs": len(group),
                "best_objective": min(objectives),
                "mean_objective": statistics.mean(objectives),
                "median_objective": statistics.median(objectives),
                "stdev_objective": statistics.pstdev(objectives) if len(objectives) > 1 else 0.0,
                "mean_runtime_seconds": statistics.mean(runtimes),
                "mean_objective_evaluations": statistics.mean(evaluations),
            }
        )

    return summary


def write_summary_markdown(summary: list[dict], path: Path) -> None:
    lines = [
        "# Strong Baselines - Initial Benchmark Results",
        "",
        f"Time limit per run: {TIME_LIMIT_SECONDS} seconds. Seeds per (instance, algorithm): {SEEDS}.",
        "All other parameters are the solver defaults (no tuning applied).",
        "",
    ]

    grouped_by_instance: dict[tuple[int, int], list[dict]] = {}
    for row in summary:
        grouped_by_instance.setdefault((row["N"], row["Q"]), []).append(row)

    for (N, Q), rows in sorted(grouped_by_instance.items()):
        lines.append(f"## N = {N}, Q = {Q}")
        lines.append("")
        lines.append(
            "| Algorithm | Best | Mean | Median | Stdev | Mean runtime (s) | Mean evaluations |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in sorted(rows, key=lambda r: r["mean_objective"]):
            lines.append(
                f"| {row['algorithm']} "
                f"| {row['best_objective']:.3f} "
                f"| {row['mean_objective']:.3f} "
                f"| {row['median_objective']:.3f} "
                f"| {row['stdev_objective']:.3f} "
                f"| {row['mean_runtime_seconds']:.3f} "
                f"| {row['mean_objective_evaluations']:.0f} |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    start = time.perf_counter()
    rows = run_all()
    elapsed = time.perf_counter() - start

    raw_path = OUTPUT_DIR / "strong_baselines_raw.csv"
    summary_path = OUTPUT_DIR / "strong_baselines_summary.md"

    write_raw_csv(rows, raw_path)
    summary = aggregate(rows)
    write_summary_markdown(summary, summary_path)

    print(f"Completed {len(rows)} runs in {elapsed:.1f} seconds.")
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()