from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.benchmarking.validation import (
    validate_solver_result,
)
from waveform_optimization.solvers.enumeration import EnumerationSolver


def test_solver_result_contains_single_best_phase_index_list() -> None:
    solver = EnumerationSolver()
    result = solver.solve(N=5, Q=2)

    assert isinstance(result, SolverResult)
    assert isinstance(result.best_phase_indices, list)
    assert len(result.best_phase_indices) == 5
    assert all(phase in {0, 1} for phase in result.best_phase_indices)

    assert result.best_objective == 2.0


def test_solver_result_has_expected_metadata() -> None:
    solver = EnumerationSolver()
    result = solver.solve(N=3, Q=2)

    assert result.algorithm == "enumeration"
    assert result.N == 3
    assert result.Q == 2
    assert result.status == "completed"
    assert result.objective_evaluations == 2**3
    assert result.runtime_seconds >= 0.0


def test_solver_result_can_be_converted_to_dictionary() -> None:
    solver = EnumerationSolver()
    result = solver.solve(N=3, Q=2)
    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert result_dict["best_phase_indices"] == result.best_phase_indices
    assert result_dict["best_objective"] == result.best_objective

    assert "top_solutions" not in result_dict
    assert "top_k" not in result_dict


def test_solver_result_passes_validation() -> None:
    solver = EnumerationSolver()
    result = solver.solve(N=3, Q=2)

    validate_solver_result(result.to_dict())


def test_invalid_solver_result_is_rejected() -> None:
    invalid_result = {
        "algorithm": "enumeration",
        "N": 3,
        "Q": 2,
        "best_phase_indices": [0, 1],
        "best_objective": 1.0,
        "runtime_seconds": 0.1,
        "objective_evaluations": 8,
        "status": "completed",
    }

    try:
        validate_solver_result(invalid_result)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(
            "Invalid phase-index length should have been rejected."
        )