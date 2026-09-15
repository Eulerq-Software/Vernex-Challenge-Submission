
from waveform_optimization.solvers.enumeration.solver import EnumerationSolver


def main() -> None:
    for N in [5, 7]:
        solver = EnumerationSolver()
        result = solver.solve(N=N, Q=2)

        print(f"N = {N}")
        print(f"Best phase indices: {result.best_phase_indices}")
        print(f"Best objective: {result.best_objective}")
        print(f"Evaluations: {result.objective_evaluations}")
        print(f"Runtime: {result.runtime_seconds:.4f} seconds")
        print("-" * 40)


if __name__ == "__main__":
    main()